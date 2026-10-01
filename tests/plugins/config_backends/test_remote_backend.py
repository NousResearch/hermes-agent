"""Remote config backend against a contract-faithful stub plane (``stub_plane.py``).

Covers the P4 behaviours: boot fetch fails closed with bounded retry, reads come from the plane and
never from a local config.yaml, writes are key-level diffs with CAS, locks and secret literals are
refused client-side (nothing sent), the poller keeps the in-memory document through an outage,
migrations run in memory only, and the managed config.yaml is ignored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from utils import fast_safe_load

from plugins.config_backends import remote as remote_pkg
from plugins.config_backends.remote import backend as backend_mod
from plugins.config_backends.remote import credentials as cred_mod
from hermes_cli.config_backend import (
    ConfigBackendUnavailable, ConfigLockedError, ConfigValueError, get_config_backend, write_config_key)

from .stub_plane import INSTANCE, TOKEN, StubPlane, remote_env


@pytest.fixture
def plane(tmp_path, monkeypatch):
    home = Path(tmp_path) / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with StubPlane() as p:
        for k, v in remote_env(p).items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(backend_mod, "BOOT_RETRY_DELAYS", (0.0, 0.0))
        remote_pkg._reset_for_tests()
        cred_mod._IDP_CACHE.clear()
        p.home = home
        yield p
        remote_pkg._reset_for_tests()
        cred_mod._IDP_CACHE.clear()


def _gets(plane):
    return [r for r in plane.requests if r["method"] == "GET"]


def _config_cmd(capsys, *argv):
    from hermes_cli.config import config_command
    ns = argparse.Namespace(config_command=argv[0], key=argv[1], value=argv[2] if len(argv) > 2 else None,
                            force=False)
    with pytest.raises(SystemExit) as exc:
        config_command(ns)
    return exc.value.code, capsys.readouterr().err


# --- selection and reads ------------------------------------------------------------------

def test_selected_from_env_and_file_tooling_off(plane):
    backend = get_config_backend()
    assert backend.name == "remote"
    assert backend.supports_file_tooling() is False
    assert backend.honors_managed_config() is False
    assert "GATEWAY_RELAY_IDP_CLIENT_SECRET" in backend.protected_env_names()


def test_reads_come_from_plane_not_local_file(plane):
    from hermes_cli.config import load_config
    plane.upper = {"model": {"default": "hermes-4", "provider": "nous"}, "terminal": {"timeout": 180}}
    local = plane.home / "config.yaml"
    local.write_text(json.dumps({"model": {"default": "LOCAL"}, "terminal": {"timeout": 1}}))

    cfg = load_config()

    assert cfg["model"]["default"] == "hermes-4"
    assert cfg["terminal"]["timeout"] == 180
    assert "display" in cfg  # DEFAULT_CONFIG stays under the remote user layer
    assert fast_safe_load(local.read_text())["model"]["default"] == "LOCAL"  # untouched
    gets = _gets(plane)
    assert gets and all(r["auth"] == f"Bearer {TOKEN}" and r["instance"] == INSTANCE for r in gets)
    assert gets[0]["profile"] == "default"


def test_boot_fetch_runs_inside_load_hermes_dotenv(plane):
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=plane.home)
    assert len(_gets(plane)) == 1
    load_hermes_dotenv(hermes_home=plane.home)  # idempotent: one fetch per process per profile
    assert len(_gets(plane)) == 1


# --- fail-closed boot ---------------------------------------------------------------------

def test_boot_retries_then_exits_on_outage(plane):
    from hermes_cli.env_loader import load_hermes_dotenv
    plane.fail_status = 503
    with pytest.raises(ConfigBackendUnavailable) as exc:
        load_hermes_dotenv(hermes_home=plane.home)
    assert len(_gets(plane)) == 3  # attempts at t=0, ~10 s, ~30 s (delays patched to 0)
    assert "does not start" in str(exc.value.code)
    assert isinstance(exc.value, SystemExit)


def test_boot_does_not_retry_a_refusal(plane, monkeypatch):
    from hermes_cli.env_loader import load_hermes_dotenv
    monkeypatch.setenv("HERMES_CONFIG_INSTANCE_ID", "someone-else")
    with pytest.raises(ConfigBackendUnavailable) as exc:
        load_hermes_dotenv(hermes_home=plane.home)
    assert len(_gets(plane)) == 1
    assert "config_agent_unknown" in str(exc.value.code)


def test_boot_requires_instance_id(plane, monkeypatch):
    from hermes_cli.config import load_config
    monkeypatch.delenv("HERMES_CONFIG_INSTANCE_ID")
    with pytest.raises(ConfigBackendUnavailable, match="HERMES_CONFIG_INSTANCE_ID"):
        load_config()
    assert plane.requests == []


def test_bad_idp_credentials_fail_closed(plane, monkeypatch):
    from hermes_cli.config import load_config
    monkeypatch.setenv("GATEWAY_RELAY_IDP_CLIENT_SECRET", "wrong")
    with pytest.raises(ConfigBackendUnavailable, match="IdP token request"):
        load_config()
    assert plane.requests == []  # no /self call without a token


def test_partial_idp_config_is_not_retried(plane, monkeypatch):
    from hermes_cli.config import load_config
    monkeypatch.delenv("GATEWAY_RELAY_IDP_CLIENT_SECRET")
    with pytest.raises(ConfigBackendUnavailable, match="CLIENT_SECRET"):
        load_config()
    assert plane.token_requests == 0


def test_secret_source_supplying_plane_credential_refuses_start(plane):
    from types import SimpleNamespace

    from hermes_cli.env_loader import _refuse_protected_env_from_sources
    report = SimpleNamespace(provenance={"GATEWAY_RELAY_IDP_CLIENT_SECRET": object()},
                             sources=[SimpleNamespace(skipped_existing=["HERMES_CONFIG_REMOTE_URL"])])
    with pytest.raises(ConfigBackendUnavailable) as exc:
        _refuse_protected_env_from_sources(report)
    assert "GATEWAY_RELAY_IDP_CLIENT_SECRET" in str(exc.value.code)
    assert "HERMES_CONFIG_REMOTE_URL" in str(exc.value.code)
    ok = SimpleNamespace(provenance={"OPENROUTER_API_KEY": object()}, sources=[])
    _refuse_protected_env_from_sources(ok)  # unrelated names pass


def test_boot_refuses_remote_secrets_source_that_maps_plane_credential(plane, monkeypatch):
    """End to end through load_hermes_dotenv: the REMOTE secrets: section drives the sources (D32)."""
    from types import SimpleNamespace

    from agent.secret_sources import registry
    from hermes_cli import env_loader
    plane.upper = {"secrets": {"command": {"enabled": True}}}
    seen = {}

    def fake_apply_all(cfg, home_path, environ=None):
        seen["cfg"] = cfg
        return SimpleNamespace(sources=[SimpleNamespace(skipped_existing=[])], applied_any=True,
                               provenance={"HERMES_CONFIG_REMOTE_URL": SimpleNamespace(source="command")})

    monkeypatch.setattr(registry, "apply_all", fake_apply_all)
    monkeypatch.setattr(env_loader, "_APPLIED_HOMES", set())
    with pytest.raises(ConfigBackendUnavailable, match="HERMES_CONFIG_REMOTE_URL"):
        env_loader.load_hermes_dotenv(hermes_home=plane.home)
    assert seen["cfg"] == {"command": {"enabled": True}}


def test_file_backend_has_no_protected_names(monkeypatch):
    from types import SimpleNamespace

    from hermes_cli.env_loader import _refuse_protected_env_from_sources
    monkeypatch.delenv("HERMES_CONFIG_BACKEND", raising=False)
    _refuse_protected_env_from_sources(
        SimpleNamespace(provenance={"GATEWAY_RELAY_IDP_CLIENT_SECRET": object()}, sources=[]))


# --- writes -------------------------------------------------------------------------------

def test_config_set_sends_only_the_changed_key(plane):
    from hermes_cli.config import load_config, set_config_value
    plane.upper = {"model": {"default": "hermes-4", "provider": "nous"}, "terminal": {"timeout": 180}}

    set_config_value("display.personality", "pirate")

    patches = plane.patches()
    assert len(patches) == 1
    body = patches[0]["body"]
    assert body["set"] == {"display": {"personality": "pirate"}}  # absent from the effective doc
    assert "unset" not in body
    assert body["expectedVersion"] == 0
    assert isinstance(body["writerConfigVersion"], int)
    assert plane.profile("default")["values"] == {"display": {"personality": "pirate"}}
    assert load_config()["display"]["personality"] == "pirate"
    assert not (plane.home / "config.yaml").exists()


def test_change_under_existing_section_is_a_leaf_set(plane):
    from hermes_cli.config import set_config_value
    plane.upper = {"display": {"personality": "concise", "compact": False}}
    set_config_value("display.personality", "pirate")
    assert plane.patches()[0]["body"]["set"] == {"display.personality": "pirate"}
    assert plane.profile("default")["values"] == {"display": {"personality": "pirate"}}


def test_write_conflict_rereads_and_retries_once(plane):
    from hermes_cli.config import load_config
    load_config()
    plane.profile("default")["version"] = 5  # another writer moved the profile level
    write_config_key(plane.home / "config.yaml", "display.personality", "pirate")
    patches = plane.patches()
    assert [p["body"]["expectedVersion"] for p in patches] == [0, 5]
    assert plane.profile("default")["version"] == 6


def test_locked_key_refused_by_config_set_without_sending(plane, capsys):
    plane.upper = {"terminal": {"timeout": 180}}
    plane.upper_locks = [{"path": "terminal", "level": "tenant"}]

    code, err = _config_cmd(capsys, "set", "terminal.timeout", "5")

    assert code == 1
    assert "locked by the tenant level" in err
    assert plane.patches() == []


def test_locked_key_refused_by_backend_single_key_write(plane):
    plane.upper_locks = [{"path": "terminal", "level": "tenant"}]
    with pytest.raises(ConfigLockedError) as exc:
        write_config_key(plane.home / "config.yaml", "terminal.backend", "local")
    assert (exc.value.path, exc.value.locked_by) == ("terminal", "tenant")
    assert plane.patches() == []


def test_r7_scalar_over_locked_subtree_refused(plane):
    plane.upper_locks = [{"path": "a.b", "level": "tenant"}]
    with pytest.raises(ConfigLockedError):
        write_config_key(plane.home / "config.yaml", "a", "scalar")
    assert plane.patches() == []


def test_bulk_save_strips_locked_keys_and_sends_the_rest(plane, capsys):
    from hermes_cli.config import read_raw_config, save_config
    plane.upper = {"terminal": {"timeout": 180}, "model": {"default": "hermes-4"}}
    plane.upper_locks = [{"path": "terminal", "level": "tenant"}]
    doc = read_raw_config()
    doc["terminal"]["timeout"] = 5
    doc["display"] = {"personality": "pirate"}

    save_config(doc)

    patches = plane.patches()
    assert len(patches) == 1
    assert patches[0]["body"]["set"] == {"display": {"personality": "pirate"}}
    assert "terminal.timeout" not in json.dumps(patches[0]["body"])
    assert "locked by Remote Config were not saved: terminal" in capsys.readouterr().err


def test_secret_literal_refused_client_side(plane, capsys):
    with pytest.raises(ConfigValueError) as exc:
        write_config_key(plane.home / "config.yaml", "mcp_servers.github.env.GITHUB_TOKEN", "ghp_live")
    assert exc.value.code == "config_secret_literal"
    code, err = _config_cmd(capsys, "set", "delegation.api_key", "sk-live-abc")
    assert code == 1 and "secret-shaped" in err
    assert plane.patches() == []
    write_config_key(plane.home / "config.yaml", "delegation.api_key", "${OPENROUTER_API_KEY}")
    assert plane.patches()[-1]["body"]["set"] == {"delegation": {"api_key": "${OPENROUTER_API_KEY}"}}


def test_unset_of_inherited_key_warns(plane, caplog):
    plane.upper = {"display": {"personality": "concise"}}
    plane.profile("default")["values"] = {"display": {"personality": "pirate"}}
    with caplog.at_level("WARNING"):
        write_config_key(plane.home / "config.yaml", "display.personality", None)
    assert plane.patches()[0]["body"]["unset"] == ["display.personality"]
    assert "still set by an upper level" in caplog.text


def test_yaml_only_values_are_converted_before_the_diff(plane):
    import datetime
    write_config_key(plane.home / "config.yaml", "x.when", datetime.date(2026, 9, 25))
    write_config_key(plane.home / "config.yaml", "x.big", 2**60)
    sent = [p["body"]["set"] for p in plane.patches()]
    assert sent == [{"x": {"when": "2026-09-25"}}, {"x.big": str(2**60)}]
    with pytest.raises(ConfigValueError, match="NaN"):
        write_config_key(plane.home / "config.yaml", "x.nan", float("nan"))


# --- poll, outage, versions ---------------------------------------------------------------

def test_poll_picks_up_changes_and_invalidates_caches(plane):
    from hermes_cli.config import load_config
    plane.upper = {"display": {"personality": "concise"}}
    assert load_config()["display"]["personality"] == "concise"
    backend = remote_pkg.get_remote_backend()
    v1 = backend.version(plane.home)

    backend.poll_all()  # unchanged → 304
    assert plane.requests[-1]["if_none_match"] and backend.version(plane.home) == v1

    plane.upper = {"display": {"personality": "pirate"}}
    backend.poll_all()
    assert backend.version(plane.home) != v1
    assert load_config()["display"]["personality"] == "pirate"


def test_outage_while_running_keeps_in_memory_doc(plane):
    from hermes_cli.config import load_config
    plane.upper = {"display": {"personality": "concise"}}
    load_config()
    backend = remote_pkg.get_remote_backend()
    plane.fail_status = 503

    backend.poll_all()  # never raises, never exits

    assert load_config()["display"]["personality"] == "concise"
    assert "503" in backend.describe(plane.home)
    plane.fail_status = None
    backend.poll_all()
    assert "last poll failed" not in backend.describe(plane.home)


def test_poll_refusal_logs_error_but_keeps_running(plane, caplog):
    from hermes_cli.config import load_config
    load_config()
    plane.fail_status = 403
    with caplog.at_level("WARNING"):
        remote_pkg.get_remote_backend().poll_all()
    assert [r.levelname for r in caplog.records if "poll for profile" in r.getMessage()] == ["ERROR"]
    plane.fail_status = 503
    caplog.clear()
    with caplog.at_level("WARNING"):
        remote_pkg.get_remote_backend().poll_all()
    assert [r.levelname for r in caplog.records if "poll for profile" in r.getMessage()] == ["WARNING"]


def test_forked_child_rearms_its_poller(plane):
    import os

    from hermes_cli.config import load_config
    load_config()
    backend = remote_pkg.get_remote_backend()
    first = backend._poller
    backend._poller_pid = -1  # as in a child after fork: the parent's thread did not survive
    load_config()
    assert backend._poller_pid == os.getpid() and backend._poller is not first
    assert backend._poller is not None and backend._poller.is_alive()


def test_poll_interval_floor(monkeypatch):
    monkeypatch.setenv("HERMES_CONFIG_REMOTE_POLL_SECONDS", "1")
    assert backend_mod.poll_interval() == backend_mod.MIN_POLL_SECONDS
    monkeypatch.setenv("HERMES_CONFIG_REMOTE_POLL_SECONDS", "600")
    assert backend_mod.poll_interval() == 600


def test_named_profile_fetches_its_own_layer(plane):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    from hermes_cli.config import load_config
    work = plane.home / "profiles" / "work"
    work.mkdir(parents=True)
    plane.profile("work")["values"] = {"display": {"personality": "formal"}}
    token = set_hermes_home_override(work)
    try:
        assert load_config()["display"]["personality"] == "formal"
    finally:
        reset_hermes_home_override(token)
    assert {r["profile"] for r in _gets(plane)} == {"work"}


# --- migrations, unknown keys, managed scope ----------------------------------------------

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

    def fake_run_migrations(current, results, quiet):
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


# --- tooling ------------------------------------------------------------------------------

def test_file_tooling_refused(plane):
    from hermes_cli.config_backend import require_file_tooling
    with pytest.raises(ConfigBackendUnavailable, match="Profile clone"):
        require_file_tooling("Profile clone")


def test_doctor_reports_backend_status(plane, capsys):
    from hermes_cli.doctor_config import _check_config_backend
    from hermes_cli.doctor_report import Finding
    f = Finding()
    assert _check_config_backend(get_config_backend(), plane.home, f) is True
    assert "Remote Config" in capsys.readouterr().out
    plane.fail_status = 503
    remote_pkg._reset_for_tests()
    f2 = Finding()
    assert _check_config_backend(get_config_backend(), plane.home, f2) is False
    assert f2.issues


def test_general_plugin_manager_never_loads_config_backends(plane, monkeypatch):
    from hermes_cli.plugins import PluginManager
    monkeypatch.setenv("HERMES_CONFIG_BACKEND", "file")
    mgr = PluginManager()
    mgr.discover_and_load()
    assert not any("config_backends" in key or key == "remote" for key in mgr._plugins)


# --- review round 1 regressions -----------------------------------------------------------

_BACKEND_FLIP_SOURCE = {"command": {"enabled": True, "override_existing": True,
                                    "command": "printf 'HERMES_CONFIG_BACKEND=file\\n'"}}


def test_secret_source_cannot_switch_remote_mode_off_at_boot(plane, monkeypatch):
    """D32 through the REAL startup path and a real command source: a source that writes
    HERMES_CONFIG_BACKEND=file must not disarm remote mode (and with it the protected-name check)
    and let the next read fall back to the local config.yaml."""
    from hermes_cli import env_loader
    monkeypatch.setattr(env_loader, "_APPLIED_HOMES", set())
    (plane.home / "config.yaml").write_text(json.dumps({"display": {"personality": "local"}}))
    plane.upper = {"display": {"personality": "remote"}, "secrets": _BACKEND_FLIP_SOURCE}

    with pytest.raises(ConfigBackendUnavailable, match="HERMES_CONFIG_BACKEND"):
        env_loader.load_hermes_dotenv(hermes_home=plane.home)

    import os
    assert os.environ["HERMES_CONFIG_BACKEND"] == "remote"  # the source's write was reverted
    assert get_config_backend().name == "remote"


def test_file_mode_ignores_a_source_that_selects_the_backend(monkeypatch, capsys):
    """The selector is never a source's to change under the file backend either: reverted, dropped
    from the report (so no later snapshot re-applies it), and startup continues."""
    import os
    from types import SimpleNamespace

    from hermes_cli.config_backend import get_config_backend as gcb
    from hermes_cli.env_loader import _refuse_protected_env_from_sources
    monkeypatch.delenv("HERMES_CONFIG_BACKEND", raising=False)
    before = dict(os.environ)
    file_backend = gcb()
    src = SimpleNamespace(applied=["HERMES_CONFIG_BACKEND", "OPENROUTER_API_KEY"], skipped_existing=[])
    report = SimpleNamespace(provenance={"HERMES_CONFIG_BACKEND": object(), "OPENROUTER_API_KEY": object()},
                             sources=[src])
    monkeypatch.setenv("HERMES_CONFIG_BACKEND", "remote")  # what the source wrote

    _refuse_protected_env_from_sources(report, file_backend, before)

    assert "HERMES_CONFIG_BACKEND" not in os.environ
    assert set(report.provenance) == {"OPENROUTER_API_KEY"} and src.applied == ["OPENROUTER_API_KEY"]
    assert "ignored HERMES_CONFIG_BACKEND" in capsys.readouterr().err


def _hold_gets(monkeypatch):
    """Pause the FIRST GET after its response arrived and before the backend installs it, until
    ``release`` is set; every later request passes straight through."""
    import threading

    from plugins.config_backends.remote import client
    fetched, release = threading.Event(), threading.Event()
    original = client.request

    def held(method, *args, **kwargs):
        resp = original(method, *args, **kwargs)
        if method == "GET" and not fetched.is_set():
            fetched.set()
            assert release.wait(10)
        return resp

    monkeypatch.setattr(client, "request", held)
    return fetched, release


def test_poll_in_flight_does_not_overwrite_an_acknowledged_write(plane, monkeypatch):
    import threading
    plane.upper = {"display": {"personality": "old"}}
    backend = get_config_backend()
    st = backend._state(plane.home)
    plane.upper["display"]["personality"] = "upper-change"
    fetched, release = _hold_gets(monkeypatch)

    poll = threading.Thread(target=backend.poll_one, args=(st,))
    poll.start()
    assert fetched.wait(10)  # the poll's GET (profile v0) has its response, not yet installed
    write_config_key(plane.home / "config.yaml", "display.personality", "my-write")
    assert (st.profile_version, st.doc["display"]["personality"]) == (1, "my-write")
    release.set()
    poll.join(10)
    assert not poll.is_alive()

    assert (st.profile_version, st.doc["display"]["personality"]) == (1, "my-write")
    assert plane.profile("default")["values"] == {"display": {"personality": "my-write"}}


def test_stale_fetch_with_same_profile_version_is_dropped(plane, monkeypatch):
    """Only an upper level changed, so both responses carry profileVersion 0: the guard must be
    'which fetch was installed last', not a profileVersion comparison."""
    import threading
    plane.upper = {"display": {"personality": "old"}}
    backend = get_config_backend()
    st = backend._state(plane.home)
    plane.upper["display"]["personality"] = "v1"
    fetched, release = _hold_gets(monkeypatch)

    slow = threading.Thread(target=backend.poll_one, args=(st,))
    slow.start()
    assert fetched.wait(10)  # holds an upper=v1 response
    plane.upper["display"]["personality"] = "v2"
    assert backend.poll_one(st) is True  # a newer fetch, started later, lands first
    assert st.doc["display"]["personality"] == "v2"
    release.set()
    slow.join(10)
    assert not slow.is_alive()

    assert st.profile_version == 0
    assert st.doc["display"]["personality"] == "v2"


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


@pytest.mark.parametrize("stored", ["latest", None])
def test_write_stamps_a_current_or_unstamped_profile_level(plane, stored):
    from hermes_cli.config import set_config_value
    latest = backend_mod._latest_config_version()
    plane.profile("default")["writer"] = latest if stored == "latest" else None

    set_config_value("display.personality", "pirate")

    (patch,) = plane.patches()
    assert patch["body"]["writerConfigVersion"] == latest
    assert plane.profile("default")["writer"] == latest


def test_write_never_lowers_a_newer_profile_stamp(plane):
    from hermes_cli.config import set_config_value
    newer = backend_mod._latest_config_version() + 1
    plane.profile("default")["writer"] = newer

    set_config_value("display.personality", "pirate")

    (patch,) = plane.patches()
    assert "writerConfigVersion" not in patch["body"]
    assert plane.profile("default")["writer"] == newer


def _dashboard_put(config):
    import asyncio

    from fastapi import HTTPException

    from hermes_cli.web_routers.config_env import ConfigUpdate, update_config
    try:
        asyncio.run(update_config(ConfigUpdate(config=config)))
    except HTTPException as exc:
        return exc.status_code, exc.detail
    return 200, None


def test_dashboard_save_of_a_newly_locked_key_returns_the_lock_message(plane):
    """The lock arrived on the plane after the last poll, so only the server refuses (403)."""
    from hermes_cli.config import read_raw_config
    plane.upper = {"display": {"personality": "old"}}
    read_raw_config()
    plane.upper_locks = [{"path": "display.personality", "level": "tenant"}]

    status, detail = _dashboard_put({"display": {"personality": "new"}})

    assert status == 403
    assert "display.personality" in detail and "tenant" in detail
    assert plane.profile("default")["values"] == {}


def test_dashboard_save_secret_literal_is_a_400(plane):
    status, detail = _dashboard_put({"model": {"api_key": "sk-live-not-a-ref-1234567890"}})
    assert status == 400
    assert "secret-shaped" in detail and "sk-live" not in detail
    assert plane.patches() == []


def test_dashboard_unexpected_config_failure_stays_opaque(plane):
    from hermes_cli.config import read_raw_config
    read_raw_config()
    plane.fail_status = 500  # the write reaches the plane, which fails: not an expected refusal
    status, detail = _dashboard_put({"display": {"personality": "new"}})
    assert (status, detail) == (500, "Internal server error")


# --- review round 2 regressions -----------------------------------------------------------

def test_refused_source_plane_url_is_never_visible_to_a_concurrent_request(plane, monkeypatch):
    """D32 credential boundary: a real command source supplies HERMES_CONFIG_REMOTE_URL pointing
    at another plane. A poll scheduled right after the sources ran (before the refusal) must still
    go to the configured plane: the forbidden URL is never published, so the bearer and instance id
    never reach the other server."""
    import os
    import threading

    from agent.secret_sources import registry
    from hermes_cli import env_loader
    monkeypatch.setattr(env_loader, "_APPLIED_HOMES", set())
    with StubPlane() as other:
        plane.upper = {"secrets": {"command": {
            "enabled": True, "override_existing": True,
            "command": f"printf 'HERMES_CONFIG_REMOTE_URL={other.url}\\n'"}}}
        backend = get_config_backend()
        st = backend._state(plane.home)
        real_apply_all = registry.apply_all
        seen = {}

        def apply_then_poll(*args, **kwargs):
            report = real_apply_all(*args, **kwargs)
            seen["url_after_sources"] = os.environ.get("HERMES_CONFIG_REMOTE_URL")
            poll = threading.Thread(target=backend.poll_one, args=(st,))
            poll.start()
            poll.join(10)
            assert not poll.is_alive()
            return report

        monkeypatch.setattr(registry, "apply_all", apply_then_poll)
        gets_before = len(_gets(plane))
        with pytest.raises(ConfigBackendUnavailable, match="HERMES_CONFIG_REMOTE_URL"):
            env_loader.load_hermes_dotenv(hermes_home=plane.home)

        assert "url_after_sources" in seen, "precondition: the source ran"
        assert other.requests == []  # nothing — least of all the bearer — reached the other plane
        assert seen["url_after_sources"] == plane.url
        assert len(_gets(plane)) > gets_before  # the gap poll went to the configured plane
        assert os.environ["HERMES_CONFIG_REMOTE_URL"] == plane.url


def test_file_mode_publishes_permitted_source_values(monkeypatch, tmp_path):
    """Staging must not lose ordinary secrets: permitted names are published after the check."""
    import os

    from hermes_cli import env_loader
    monkeypatch.delenv("HERMES_CONFIG_BACKEND", raising=False)
    monkeypatch.setenv("CC_STAGED_SECRET", "x")
    monkeypatch.delenv("CC_STAGED_SECRET")
    monkeypatch.setattr(env_loader, "_APPLIED_HOMES", set())
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n  command:\n    enabled: true\n    command: \"printf 'CC_STAGED_SECRET=from-source\\\\n'\"\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    env_loader._apply_external_secret_sources(home)

    assert os.environ.get("CC_STAGED_SECRET") == "from-source"


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


def test_dashboard_save_refused_by_plane_value_check_is_a_400(plane):
    """The plane's own value refusal (config_value_invalid too_deep, contract §11.1) reaches the
    dashboard as a 400 with its reason, not an opaque 500."""
    from hermes_cli.config import read_raw_config
    read_raw_config()
    value = "leaf"
    for _ in range(33):
        value = [value]

    status, detail = _dashboard_put({"display": {"custom": value}})

    assert len(plane.patches()) == 1  # client sent it; the server refused
    assert status == 400
    assert "too_deep" in detail and "display" in detail  # the plane's reason and path pass through


@pytest.mark.parametrize("status,error,expected", [
    (400, "config_value_invalid", 400), (400, "config_path_invalid", 400), (400, "config_path_reserved", 400),
    (400, "config_secret_literal", 400), (413, "config_level_too_large", 413), (413, "config_body_too_large", 413),
    (400, "config_request_invalid", None), (500, "internal", None)])
def test_plane_refusals_map_to_http(status, error, expected):
    from hermes_cli.web_routers._common import config_refusal_http
    from plugins.config_backends.remote import client
    exc = backend_mod.RemoteBackend._write_error(
        client.Response(status=status, body={"error": error, "message": "refused", "path": "a.b"},
                        etag=None, retry_after=None))
    http = config_refusal_http(exc)
    assert (http.status_code if http else None) == expected
    assert exc.code == error


def test_stub_plane_deep_merge_follows_contract_6_2():
    """The test double resolves like the plane: null over a mapping is ignored (contract §6.2)."""
    from .stub_plane import deep_merge
    upper = {"display": {"personality": "concise"}, "model": "a", "tags": [1, 2]}
    assert deep_merge(upper, {"display": None}) == upper
    assert deep_merge(upper, {"model": None})["model"] is None
    assert deep_merge(upper, {"tags": [3]})["tags"] == [3]
