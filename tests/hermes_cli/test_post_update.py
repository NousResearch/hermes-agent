"""post_update step registry: scopes, isolation, and the migrate contract.

These assert behavior contracts, not snapshots: the registries' scopes must
match what boot_bootstrap gates them with, a failing step must not stop the
rest, and step_migrate_config must restore its backups when a migration
fails or does not advance the version.
"""
from pathlib import Path

import pytest

from hermes_cli import post_update
from hermes_cli.post_update import (
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
    import hermes_cli.config as cfg

    monkeypatch.setattr(cfg, "check_config_version", lambda: (34, 34))
    result = step_migrate_config()
    assert result == {"ok": True, "skipped": "up-to-date"}


def test_migrate_config_restores_backup_when_version_does_not_advance(
    tmp_path, monkeypatch
):
    import hermes_cli.config as cfg
    import hermes_cli.config_migrations as mig

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


def test_migrate_config_restores_backup_on_exception(tmp_path, monkeypatch):
    import hermes_cli.config as cfg
    import hermes_cli.config_migrations as mig

    config_path = tmp_path / "config.yaml"
    config_path.write_text("_config_version: 20\n", encoding="utf-8")

    floor = getattr(mig, "SUPPORT_FLOOR_VERSION", 12)
    monkeypatch.setattr(cfg, "check_config_version", lambda: (max(20, floor), 34))
    monkeypatch.setattr(cfg, "get_config_path", lambda: config_path)
    monkeypatch.setattr(cfg, "get_env_path", lambda: tmp_path / ".env")

    def exploding_migrate(**kw):
        config_path.write_text("half-written garbage", encoding="utf-8")
        raise RuntimeError("migration blew up")

    monkeypatch.setattr(cfg, "migrate_config", lambda **kw: exploding_migrate(**kw))

    with pytest.raises(RuntimeError, match="blew up"):
        step_migrate_config()
    assert config_path.read_text(encoding="utf-8") == "_config_version: 20\n"


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


def test_provision_runtimes_is_a_noop_when_pm_is_current(monkeypatch):
    import pm

    monkeypatch.setattr(pm, "check", lambda: [])
    assert post_update.step_provision_runtimes() == {"ok": True, "skipped": "current"}


def test_provision_runtimes_reensures_only_what_pm_names(monkeypatch):
    import importlib

    import pm

    # pm/__init__ rebinds the name `pm.ensure` to the FUNCTION; the module
    # object (whose attrs step_provision_runtimes imports at call time)
    # comes from sys.modules.
    pm_ensure = importlib.import_module("pm.ensure")

    ensured = []
    monkeypatch.setattr(pm, "check", lambda: ["node: not installed or outdated", "venv: out of sync with uv.lock"])
    monkeypatch.setattr(pm_ensure, "sealed", lambda: False)
    monkeypatch.setattr(pm_ensure, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr(pm, "ensure", lambda name, explicit=False: ensured.append((name, explicit)))
    monkeypatch.setattr(pm, "sync_venv", lambda explicit=False: ensured.append(("venv", explicit)))

    result = post_update.step_provision_runtimes()

    assert result["ok"] is True
    assert ("node", True) in ensured and ("venv", True) in ensured


def test_provision_runtimes_respects_the_lazy_install_policy(monkeypatch):
    import importlib

    import pm

    pm_ensure = importlib.import_module("pm.ensure")

    monkeypatch.setattr(pm, "check", lambda: ["node: not installed or outdated"])
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
