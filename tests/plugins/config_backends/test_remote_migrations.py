"""Remote config: in-memory migrations (D12), unknown keys and the managed scope."""
from __future__ import annotations

import json

import pytest

from hermes_cli.config_backend import (
    ConfigValueError,
    get_config_backend,
    write_config_key,
)
from plugins.config_backends import remote as remote_pkg
from plugins.config_backends.remote import backend as backend_mod

from .conftest import _config_cmd


def test_old_writer_version_migrates_in_memory_only(plane):
    from hermes_cli.config import read_raw_config
    from hermes_cli.config_migrations import SUPPORT_FLOOR_VERSION
    latest = backend_mod._latest_config_version()
    old = max(SUPPORT_FLOOR_VERSION, latest - 1)
    plane.profile("default").update(values={"display": {"personality": "pirate"}}, version=1, writer=old)

    doc = read_raw_config()

    assert doc["_config_version"] == latest
    assert doc["display"]["personality"] == "pirate"
    assert plane.patches() == []  # D12: never written back
    assert not (plane.home / "config.yaml").exists()


def test_migration_steps_persist_into_memory(plane, monkeypatch):
    """A migration step's ``_persist_migration`` lands in the in-memory layer, not on disk/plane."""
    from hermes_cli import config as config_mod
    from hermes_cli import config_migrations
    latest = backend_mod._latest_config_version()
    plane.profile("default").update(values={"old_key": 1}, version=1, writer=latest - 1)

    def fake_run_migrations(current, results, quiet, **kwargs):
        doc = config_mod.read_raw_config()
        doc["new_key"] = doc.pop("old_key")
        config_mod._persist_migration(doc)

    monkeypatch.setattr(config_migrations, "run_migrations", fake_run_migrations)
    doc = config_mod.read_raw_config()
    assert doc.get("new_key") == 1 and "old_key" not in doc
    assert plane.patches() == []


def test_unknown_top_level_key_warns_once(plane, caplog):
    from hermes_cli.config import load_config
    plane.upper = {"from_the_future": {"x": 1}}
    with caplog.at_level("WARNING"):
        load_config()
        load_config()
    assert caplog.text.count("unknown config key 'from_the_future'") == 1


def test_managed_config_yaml_is_ignored_and_flagged(plane, tmp_path, monkeypatch, capsys):
    from hermes_cli import managed_scope
    from hermes_cli.config import load_config
    from hermes_cli.env_loader import load_hermes_dotenv
    managed = tmp_path / "etc-hermes"
    managed.mkdir()
    (managed / "config.yaml").write_text(json.dumps({"display": {"personality": "managed"}}))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope._CONFIG_CACHE.clear()
    plane.upper = {"display": {"personality": "remote"}}

    load_hermes_dotenv(hermes_home=plane.home)

    assert "IGNORED" in capsys.readouterr().err
    assert managed_scope.load_managed_config() == {}
    assert load_config()["display"]["personality"] == "remote"


def _old_compression_profile(plane):
    from hermes_cli.config import read_raw_config
    assert backend_mod._latest_config_version() > 46
    plane.upper = {"compression": {"threshold_tokens": 256000}}  # the old default the 46->47 step removes
    plane.profile("default")["writer"] = 46
    doc = read_raw_config()
    assert "threshold_tokens" not in (doc.get("compression") or {}), "precondition: migrated in memory"


def test_migration_is_not_sent_by_an_unrelated_write(plane):
    from hermes_cli.config import set_config_value
    _old_compression_profile(plane)

    set_config_value("display.personality", "pirate")

    (patch,) = plane.patches()
    assert patch["body"]["set"] == {"display": {"personality": "pirate"}}
    assert "unset" not in patch["body"]  # D12: the migration's removal is never written back


def test_migration_is_not_sent_after_a_cas_reread(plane):
    from hermes_cli.config import set_config_value
    _old_compression_profile(plane)
    plane.profile("default")["version"] = 3  # another writer moved the profile level: first PATCH 409s

    set_config_value("display.personality", "pirate")

    patches = plane.patches()
    assert [p["body"]["expectedVersion"] for p in patches] == [0, 3]
    for p in patches:
        assert p["body"]["set"] == {"display": {"personality": "pirate"}}
        assert "unset" not in p["body"]


def test_bulk_save_after_migration_sends_only_the_edit(plane):
    from hermes_cli.config import read_raw_config, save_config
    _old_compression_profile(plane)
    doc = read_raw_config()
    doc.setdefault("display", {})["personality"] = "pirate"

    save_config(doc)

    (p,) = plane.patches()
    assert p["body"]["set"] == {"display": {"personality": "pirate"}}
    assert "unset" not in p["body"]
    assert "compression" not in json.dumps(p["body"])


def test_unrelated_write_keeps_an_old_profile_stamp_so_later_reads_still_migrate(plane):
    """Orchestrator ruling (option 1): the profile level stamped 46 still stores a pre-migration
    key; an unrelated write must not re-stamp it, or every later reader would skip 46->47."""
    from hermes_cli.config import read_raw_config, set_config_value
    assert backend_mod._latest_config_version() > 46
    plane.profile("default").update(values={"compression": {"threshold_tokens": 256000}}, writer=46)
    assert "threshold_tokens" not in (read_raw_config().get("compression") or {}), "precondition"

    set_config_value("display.personality", "pirate")

    (patch,) = plane.patches()
    assert "writerConfigVersion" not in patch["body"]
    assert plane.profile("default")["writer"] == 46
    assert plane.profile("default")["values"]["compression"] == {"threshold_tokens": 256000}

    remote_pkg._reset_for_tests()  # a later process reads the same profile level
    doc = read_raw_config()
    assert "threshold_tokens" not in (doc.get("compression") or {})  # 46->47 still ran
    assert doc["display"]["personality"] == "pirate"
    assert len(plane.patches()) == 1  # D12: the later read wrote nothing back


def test_write_never_lowers_a_newer_profile_stamp(plane):
    from hermes_cli.config import set_config_value
    newer = backend_mod._latest_config_version() + 1
    plane.profile("default")["writer"] = newer

    set_config_value("display.personality", "pirate")

    (patch,) = plane.patches()
    assert "writerConfigVersion" not in patch["body"]
    assert plane.profile("default")["writer"] == newer


def test_migration_of_a_replaced_doc_does_not_touch_the_replacement(plane, monkeypatch):
    """A read starts migrating the stored-46 doc; before the migration begins, a poll installs the
    current-version doc that keeps compression.threshold_tokens on purpose. The stale 46->47 step
    must not run against it (the replacement stays intact, nothing is written back)."""
    import threading

    from hermes_cli.config import read_raw_config
    latest = backend_mod._latest_config_version()
    assert latest > 46
    plane.profile("default").update(values={"compression": {"threshold_tokens": 256000}}, writer=46)
    backend = get_config_backend()
    st = backend._state(plane.home)
    entered, resume = threading.Event(), threading.Event()
    real_migrate = backend._migrate_in_memory

    def paused(*args, **kwargs):
        entered.set()
        assert resume.wait(10)
        return real_migrate(*args, **kwargs)

    monkeypatch.setattr(backend, "_migrate_in_memory", paused)
    failures = []

    def read():
        try:
            read_raw_config()
        except BaseException as exc:  # noqa: BLE001 — surfaced below
            failures.append(exc)

    reader = threading.Thread(target=read)
    reader.start()
    assert entered.wait(10)
    plane.profile("default").update(writer=latest, version=1)
    assert backend.poll_one(st)
    assert st.doc["compression"] == {"threshold_tokens": 256000}
    resume.set()
    reader.join(10)
    assert not reader.is_alive() and not failures, failures

    doc = read_raw_config()
    assert doc["_config_version"] == latest
    assert doc["compression"] == {"threshold_tokens": 256000}
    assert plane.patches() == []


def test_edit_during_an_in_memory_migration_never_sends_the_migration(plane, monkeypatch):
    """F2: another thread's in-memory migration has run its steps but not finished when a public
    edit reads and saves. The edit must diff against the doc it read, so the migration's removal
    of compression.threshold_tokens is not sent (D12)."""
    import threading

    from hermes_cli import config_migrations
    from hermes_cli.config import read_raw_config, set_config_value
    from hermes_cli.config_backend import read_config_doc
    assert backend_mod._latest_config_version() > 46
    plane.profile("default").update(values={"compression": {"threshold_tokens": 256000}}, version=1, writer=46)
    get_config_backend()._state(plane.home)
    steps_ran, resume = threading.Event(), threading.Event()
    real_run = config_migrations.run_migrations

    def paused(*args, **kwargs):
        real_run(*args, **kwargs)
        if threading.current_thread().name == "migrator":  # the edit's own checks run unpaused
            steps_ran.set()
            assert resume.wait(10), "the edit waited for the migration"

    monkeypatch.setattr(config_migrations, "run_migrations", paused)
    failures = []

    def migrate():  # a lock-free reader (read_config_doc) starts the migration
        try:
            read_config_doc(plane.home / "config.yaml")
        except BaseException as exc:  # noqa: BLE001 — surfaced below
            failures.append(exc)

    migrator = threading.Thread(target=migrate, name="migrator")
    migrator.start()
    try:
        assert steps_ran.wait(10)
        set_config_value("display.personality", "pirate")
    finally:
        resume.set()
        migrator.join(10)
    assert not migrator.is_alive() and not failures, failures

    (patch,) = plane.patches()
    assert "unset" not in patch["body"] and "compression" not in json.dumps(patch["body"])
    assert plane.profile("default")["values"]["compression"] == {"threshold_tokens": 256000}
    later = read_raw_config()
    assert later["display"]["personality"] == "pirate"
    assert "threshold_tokens" not in (later.get("compression") or {})  # still migrated, in memory


def test_setting_a_value_that_old_schema_migration_rewrites_is_refused(plane, capsys):
    """F8: on a level stamped 46 (option 1 keeps that stamp), 256000 is exactly what the 46->47
    step removes on every read: acknowledging it would hide it. Refused, nothing sent. A value
    the migration keeps is saved as usual."""
    from hermes_cli.config import read_raw_config
    assert backend_mod._latest_config_version() > 46
    plane.profile("default").update(values={"display": {"personality": "old"}}, version=1, writer=46)
    read_raw_config()

    code, err = _config_cmd(capsys, "set", "compression.threshold_tokens", "256000")

    assert code == 1 and "compression.threshold_tokens" in err and "schema v46" in err
    assert plane.patches() == []
    with pytest.raises(ConfigValueError) as exc:
        write_config_key(plane.home / "config.yaml", "compression.threshold_tokens", 256000)
    assert exc.value.code == "config_migration_conflict"

    write_config_key(plane.home / "config.yaml", "compression.threshold_tokens", 300000)
    assert plane.profile("default")["values"]["compression"] == {"threshold_tokens": 300000}
    remote_pkg._reset_for_tests()  # a fresh process reads it back
    assert read_raw_config()["compression"]["threshold_tokens"] == 300000


_LEGACY_SOUL = "# Beta identity\n\n## Messaging other agents\nKEEP THIS PROFILE SECTION\n"


def _sibling_with_legacy_soul(home):
    beta = home / "profiles" / "beta"
    beta.mkdir(parents=True)
    (beta / "SOUL.md").write_text(_LEGACY_SOUL)
    return beta / "SOUL.md"


def test_remote_read_migration_changes_no_local_file(plane):
    """Round 4 #3: reading default's remote config stamped 40 runs the 40->41 step in memory. That
    step's only effect is on profile SOUL.md files; a config read must not rewrite a sibling's."""
    from hermes_cli.config_backend import read_config_doc
    soul = _sibling_with_legacy_soul(plane.home)
    plane.profile("default").update(values={"display": {"personality": "old"}}, writer=40, version=1)

    doc = read_config_doc(plane.home / "config.yaml")

    assert doc["_config_version"] == backend_mod._latest_config_version()  # it did migrate
    assert soul.read_text() == _LEGACY_SOUL
    assert plane.patches() == []


def test_refused_write_validation_changes_no_local_file(plane):
    """Round 4 #3: the config_migration_conflict check simulates the migration of the stored doc;
    a refused edit must leave every local file as it was."""
    from hermes_cli.config import read_raw_config
    plane.profile("default").update(values={}, writer=40, version=1)
    read_raw_config()
    soul = _sibling_with_legacy_soul(plane.home)  # created after the read's own migration

    with pytest.raises(ConfigValueError) as exc:
        write_config_key(plane.home / "config.yaml", "compression.threshold_tokens", 256000)

    assert exc.value.code == "config_migration_conflict"
    assert soul.read_text() == _LEGACY_SOUL
    assert plane.patches() == []


def _home_files(root):
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
            for p in sorted(root.rglob("*"))}


@pytest.mark.parametrize("config_only", [True, False])
def test_config_only_migration_ladder_touches_no_file(tmp_path, monkeypatch, config_only):
    """The whole ladder from the support floor, config-only, over a home whose files every
    file-touching step would change (.env dead/legacy values, a legacy SOUL.md section, no logs/):
    the only output is the config document. Control: a file-mode run does change them."""
    from hermes_cli import config as config_mod
    from hermes_cli import config_migrations
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("HERMES_CONFIG_BACKEND", "LLM_MODEL", "OPENAI_MODEL", "TERMINAL_VERCEL_RUNTIME"):
        monkeypatch.delenv(name, raising=False)
    from hermes_cli.config_defaults import LEGACY_VERCEL_RUNTIME
    (home / ".env").write_text(f"LLM_MODEL=old-model\nTERMINAL_VERCEL_RUNTIME={LEGACY_VERCEL_RUNTIME}\n")
    (home / "SOUL.md").write_text(_LEGACY_SOUL)
    _sibling_with_legacy_soul(home)
    doc = {"_config_version": config_migrations.SUPPORT_FLOOR_VERSION, "model": {"default": "m"}}
    persisted = []
    monkeypatch.setattr(config_mod, "read_raw_config", lambda: dict(persisted[-1] if persisted else doc))
    monkeypatch.setattr(config_mod, "_persist_migration", lambda cfg: persisted.append(dict(cfg)))
    before = _home_files(home)

    config_migrations.run_migrations(config_migrations.SUPPORT_FLOOR_VERSION,
                                     {"env_added": [], "config_added": [], "warnings": []}, True,
                                     config_only=config_only)

    assert persisted  # the document itself was migrated either way
    assert (_home_files(home) == before) is config_only
