"""post_update step registry: scopes, isolation, and the migrate contract.

These assert behavior contracts, not snapshots: the registries' scopes must
match what boot_bootstrap gates them with, a failing step must not stop the
rest, and step_migrate_config must restore its backups when a migration
fails or does not advance the version.
"""
from pathlib import Path

import pytest

from hermes_agent.hermes_cli import post_update
from hermes_agent.hermes_cli.post_update import (
    HOME_STEPS,
    MACHINE_STEPS,
    run_steps,
    step_migrate_config,
    step_state_db_guard,
)


# ── registry invariants ──────────────────────────────────────────────


def test_registries_are_disjoint_and_named():
    home_names = {name for name, _ in HOME_STEPS}
    machine_names = {name for name, _ in MACHINE_STEPS}
    assert home_names, "home registry must not be empty"
    assert not (home_names & machine_names)
    for name, func in (*HOME_STEPS, *MACHINE_STEPS):
        assert callable(func), name


def test_home_steps_cover_the_boot_contract():
    # boot_bootstrap gates these with the per-home record; the three
    # user-state concerns (config, skills, state.db) must all be present.
    names = {name for name, _ in HOME_STEPS}
    assert {"migrate_config", "sync_skills", "state_db_guard"} <= names


# ── run_steps isolation ──────────────────────────────────────────────


def test_run_steps_isolates_failures():
    order = []

    def ok():
        order.append("ok")
        return {"ok": True}

    def boom():
        order.append("boom")
        raise RuntimeError("nope")

    results = run_steps((("first", boom), ("second", ok)))
    assert order == ["boom", "ok"]  # failure did not stop the run
    assert results["first"]["ok"] is False
    assert "nope" in results["first"]["error"]
    assert results["second"] == {"ok": True}


# ── step_migrate_config ──────────────────────────────────────────────


def test_migrate_config_noop_when_current(monkeypatch):
    import hermes_agent.hermes_cli.config as cfg

    monkeypatch.setattr(cfg, "check_config_version", lambda: (34, 34))
    result = step_migrate_config()
    assert result == {"ok": True, "skipped": "up-to-date"}


def test_migrate_config_restores_backup_when_version_does_not_advance(
    tmp_path, monkeypatch
):
    import hermes_agent.hermes_cli.config as cfg
    import hermes_agent.hermes_cli.config_migrations as mig

    config_path = tmp_path / "config.yaml"
    config_path.write_text("_config_version: 20\n", encoding="utf-8")
    env_path = tmp_path / ".env"

    floor = getattr(mig, "SUPPORT_FLOOR_VERSION", 12)
    versions = iter([(max(20, floor), 34), (max(20, floor), 34)])
    monkeypatch.setattr(cfg, "check_config_version", lambda: next(versions))
    monkeypatch.setattr(cfg, "get_config_path", lambda: config_path)
    monkeypatch.setattr(cfg, "get_env_path", lambda: env_path)

    def fake_migrate(**kw):
        # Corrupt the file; the version check will then report no advance.
        config_path.write_text("_config_version: 20\nbroken: true\n", encoding="utf-8")

    monkeypatch.setattr(cfg, "migrate_config", lambda **kw: fake_migrate(**kw))

    with pytest.raises(RuntimeError, match="did not advance"):
        step_migrate_config()

    # Original content restored from the backup.
    assert config_path.read_text(encoding="utf-8") == "_config_version: 20\n"
    backups = list(tmp_path.glob("config.yaml.bak-*"))
    assert backups, "backup file must exist"


def test_boot_keeps_the_steps_before_a_failing_one_and_retries_it_each_start(
    tmp_path, monkeypatch, capsys
):
    """A step that raises is partial progress at boot, not a failure to undo: the steps before it
    stay (no backup restore), the stamp stops at the last good one, the failing step is retried on
    every start of the same revision, and ``hermes doctor`` says why the version is behind."""
    import json

    import hermes_agent.hermes_yaml as yaml
    from hermes_agent.hermes_cli import boot_bootstrap
    from hermes_agent.hermes_cli.config import DEFAULT_CONFIG
    from hermes_agent.hermes_cli.doctor_config import _drift_config_version
    from hermes_agent.hermes_cli.doctor_report import Finding

    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "payload"
    root.mkdir()
    (root / "install-stamp.json").write_text(json.dumps({"commit": "abcdef012345", "payload": "full"}))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(root))
    monkeypatch.setattr(post_update, "BOOT_HOME_STEPS", (("migrate_config", step_migrate_config),))
    config_path = home / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "_config_version": 16, "compression": {"summary_model": "m"}, "plugins": {"disabled": [["x"]]},
    }), encoding="utf-8")

    def boot():
        return boot_bootstrap.run_boot_bootstrap(root)["home"]

    first = boot()["migrate_config"]
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert first["ok"] is False and first["failed_step"] == 21
    assert raw["_config_version"] == 20
    assert raw["auxiliary"]["compression"]["model"] == "m", "v17 must not be restored away"

    assert boot()["migrate_config"]["failed_step"] == 21, "same revision: retried on the next start"
    assert len(list(home.glob("config.yaml.bak-*"))) == 1, "a retry that changed nothing keeps no backup"
    _drift_config_version(Finding(), False, config_path)
    assert "unhashable type" in capsys.readouterr().out, "plain doctor says why the step failed"

    raw["plugins"]["disabled"] = []
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert boot()["migrate_config"] == {"ok": True, "migrated": f"20->{DEFAULT_CONFIG['_config_version']}"}
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["_config_version"] == DEFAULT_CONFIG["_config_version"]
    assert boot() == "skipped"


def test_a_retry_that_raises_still_owes_the_step(tmp_path, monkeypatch):
    """A retry can raise before the ladder runs (a full disk while taking the backup). That says
    nothing about the failing step, so the next start must retry it again: only a retry that
    succeeded clears it."""
    import json

    import hermes_agent.hermes_yaml as yaml
    from hermes_agent.hermes_cli import boot_bootstrap
    from hermes_agent.hermes_cli.config import DEFAULT_CONFIG

    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "payload"
    root.mkdir()
    (root / "install-stamp.json").write_text(json.dumps({"commit": "abcdef012345", "payload": "full"}))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(root))
    monkeypatch.setattr(post_update, "BOOT_HOME_STEPS", (("migrate_config", step_migrate_config),))
    config_path = home / "config.yaml"
    config_path.write_text(yaml.safe_dump({"_config_version": 20, "plugins": {"disabled": [["x"]]}}),
                           encoding="utf-8")

    def boot():
        return boot_bootstrap.run_boot_bootstrap(root)["home"]

    assert boot()["migrate_config"]["failed_step"] == 21
    real_backup = post_update._backup_existing

    def full_disk(paths):
        raise OSError("No space left on device")

    monkeypatch.setattr(post_update, "_backup_existing", full_disk)
    raised = boot()["migrate_config"]
    assert raised["ok"] is False and raised["failed_step"] == 21
    assert "No space left" in raised["failed_warning"], "doctor shows why THIS retry broke"
    monkeypatch.setattr(post_update, "_backup_existing", real_backup)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["plugins"]["disabled"] = []
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert boot()["migrate_config"] == {"ok": True, "migrated": f"20->{DEFAULT_CONFIG['_config_version']}"}, \
        "the step is still owed after a retry that raised"
    assert boot() == "skipped"


# ── step_state_db_guard ──────────────────────────────────────────────


def test_state_db_guard_skips_missing_db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert step_state_db_guard() == {"ok": True, "skipped": "no-state-db"}


def test_state_db_guard_flags_corrupt_db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "state.db").write_text("this is not sqlite", encoding="utf-8")
    result = step_state_db_guard()
    assert result["ok"] is False
    assert result.get("error")


def test_state_db_guard_passes_valid_db(tmp_path, monkeypatch):
    import sqlite3

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    conn = sqlite3.connect(tmp_path / "state.db")
    conn.execute("CREATE TABLE t (x)")
    conn.commit()
    conn.close()
    assert step_state_db_guard() == {"ok": True}


# ── step_drop_live_plugin_catalog ────────────────────────────────────


def test_boot_drops_the_live_plugin_catalog_cache(tmp_path, monkeypatch):
    """Bundled-app updates skip `hermes update`'s tail; the boot step covers them (#119340)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cache = tmp_path / "cache" / "plugin-catalog.json"
    cache.parent.mkdir()
    cache.write_text("{}", encoding="utf-8")

    assert ("drop_live_plugin_catalog", post_update.step_drop_live_plugin_catalog) in post_update.BOOT_HOME_STEPS
    assert post_update.step_drop_live_plugin_catalog() == {"ok": True}
    assert not cache.exists()
    assert post_update.step_drop_live_plugin_catalog() == {"ok": True}  # absent cache is fine


# ── machine-step registry ────────────────────────────────────────────


def test_provisioning_is_the_machine_scope_driver_path():
    """cua-driver has no refresh step of its own — provisioning carries it.

    Pinned managed tools ride pm's lockfile; the provision_runtimes sweep
    moves a pin bump exactly like node or ripgrep, and a second mechanism
    would be a second authority on a tool's version.
    """
    names = [name for name, _ in post_update.MACHINE_STEPS]
    assert "provision_runtimes" in names
    assert not any("cua" in name for name in names)


def test_provisioning_does_not_use_human_diagnostics(tmp_path, monkeypatch):
    import json
    import importlib
    import pm
    from pm import paths

    engine = importlib.import_module("pm.install")
    runtime = tmp_path / "tools"
    runtime.mkdir()
    (runtime / "facts.json").write_text(json.dumps({"schema": 1, "packages": {}}))
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"schema": 1, "packages": {"node": {"version": "test"}}}))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(runtime))
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock)
    monkeypatch.setattr(engine, "sealed", lambda: False)
    monkeypatch.setattr(engine, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr(pm, "check", lambda: ["translated diagnostic without a package token"])
    ensured = []
    monkeypatch.setattr(pm, "ensure", lambda name, **kwargs: ensured.append((name, kwargs)))

    assert post_update.main(["--scope", "machine"]) == 0
    assert ensured == [("node", {"explicit": True})]


def test_provision_runtimes_is_a_noop_when_pm_is_current(monkeypatch):
    import pm

    monkeypatch.setattr(pm, "drift", dict)
    assert post_update.step_provision_runtimes() == {"ok": True, "skipped": "current"}


def test_provision_runtimes_reensures_only_what_pm_names(monkeypatch):
    import importlib

    import pm

    # pm/__init__ rebinds the name `pm.ensure` to the FUNCTION; the module
    # object (whose attrs step_provision_runtimes imports at call time)
    # comes from sys.modules.
    pm_ensure = importlib.import_module("pm.install")

    ensured = []
    monkeypatch.setattr(pm, "drift", lambda: {"node": "outdated", "venv": "out of sync"})
    monkeypatch.setattr(pm_ensure, "sealed", lambda: False)
    monkeypatch.setattr(pm_ensure, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr(pm, "ensure", lambda name, explicit=False: ensured.append((name, explicit)))
    monkeypatch.setattr(pm, "sync_venv", lambda explicit=False, evict_incompatible_plugins=False:
                        ensured.append(("venv", explicit and evict_incompatible_plugins)))

    result = post_update.step_provision_runtimes()

    assert result["ok"] is True
    assert ("node", True) in ensured and ("venv", True) in ensured


def test_provision_runtimes_respects_the_lazy_install_policy(monkeypatch):
    import importlib

    import pm

    pm_ensure = importlib.import_module("pm.install")

    monkeypatch.setattr(pm, "drift", lambda: {"node": "outdated"})
    monkeypatch.setattr(pm_ensure, "sealed", lambda: False)
    monkeypatch.setattr(pm_ensure, "lazy_installs_allowed", lambda: False)
    result = post_update.step_provision_runtimes()
    assert result == {"ok": True, "skipped": "lazy-installs-disabled"}


# ── __main__ entry ───────────────────────────────────────────────────


def test_main_reports_failure_in_exit_code(monkeypatch):
    monkeypatch.setattr(
        post_update, "HOME_STEPS",
        (("bad", lambda: (_ for _ in ()).throw(RuntimeError("x"))),),
    )
    monkeypatch.setattr(post_update, "MACHINE_STEPS", ())
    assert post_update.main(["--scope", "home"]) == 1


def test_main_scope_selects_registries(monkeypatch):
    ran = []
    monkeypatch.setattr(
        post_update, "HOME_STEPS", (("h", lambda: ran.append("h") or {"ok": True}),)
    )
    monkeypatch.setattr(
        post_update, "MACHINE_STEPS", (("m", lambda: ran.append("m") or {"ok": True}),)
    )
    assert post_update.main(["--scope", "home"]) == 0
    assert ran == ["h"]
    ran.clear()
    assert post_update.main(["--scope", "all"]) == 0
    assert ran == ["h", "m"]


def test_scope_cli_rejects_unrelated_flags_without_running_steps(monkeypatch):
    ran = []
    monkeypatch.setattr(post_update, "HOME_STEPS", (("home", lambda: ran.append("home") or {"ok": True}),))
    monkeypatch.setattr(post_update, "MACHINE_STEPS", (("machine", lambda: ran.append("machine") or {"ok": True}),))
    for flags in (
        ["--gateway-mode"],
        ["--assume-yes"],
        ["--pre-update-snapshot-id", "fixture"],
        ["--pre-update-version", "fixture"],
        ["--resumed-after-sync"],
        ["--update-phase", "--resumed-after-sync"],
    ):
        with pytest.raises(SystemExit) as exc:
            post_update.main(flags)
        assert exc.value.code == 2
        assert ran == []
