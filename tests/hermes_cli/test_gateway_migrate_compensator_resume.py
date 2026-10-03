"""The compensator's outcome, and a stranded host's recovery, under the flipped host lock.

Both scenarios are about states the PR itself can create, so the liveness these tests assert is
the REAL one: a "started" gateway writes the pid file + runtime record a live gateway writes, and
``gateway.status`` reads them back. Only the service manager is faked (the seam the migration
suite already fakes).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_constants
from gateway import migration as gm
from nous_cli import gateway_migrate as migrate_cli


def _write_live_gateway(home: Path, served: list[str] | None = None) -> None:
    record = {"pid": os.getpid(), "hermes_home": str(home), "gateway_state": "running"}
    if served is not None:
        record["served_profiles"] = served
    (home / "gateway.pid").write_text(json.dumps({"pid": os.getpid(), "hermes_home": str(home)}))
    (home / "gateway_state.json").write_text(json.dumps(record))


@pytest.fixture
def stranded(tmp_path, monkeypatch):
    """A host mid-migration: manifest on disk, both secondaries' units already uninstalled,
    nothing serving. Exactly what an apply that died in ``_wait_for_served`` leaves behind."""
    root = tmp_path / "home" / ".hermes"
    for sub in ("profiles/coder", "profiles/ops"):
        (root / sub).mkdir(parents=True)
        (root / sub / "SOUL.md").write_text("x\n", encoding="utf-8")
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n", encoding="utf-8")
    (root / ".env").write_text("TELEGRAM_BOT_TOKEN=111111:shared\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    for name in ("TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "GATEWAY_MULTIPLEX_PROFILES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    assert str(hermes_constants.get_default_hermes_root()).startswith(str(tmp_path))

    (root / gm.MANIFEST_NAME).write_text(json.dumps({
        "version": 1, "flag_was": False,
        "default": {"profile": "default", "home": str(root), "pid": None, "services": []},
        "secondaries": [
            {"profile": "coder", "home": str(root / "profiles/coder"), "pid": 4101,
             "services": [{"kind": "systemd", "system": False}]},
            {"profile": "ops", "home": str(root / "profiles/ops"), "pid": 4102,
             "services": [{"kind": "systemd", "system": False}]},
        ],
    }))

    ops: list[tuple[str, str]] = []
    state = SimpleNamespace(root=root, ops=ops, start_succeeds=True)

    def _service_op(kind, system, verb, home, *, run_as_user=None):
        name = hermes_constants.profile_name_for_home(home) or "default"
        ops.append((name, verb))
        if verb in ("start", "restart") and name == "default" and state.start_succeeds:
            _write_live_gateway(root, ["default", "coder", "ops"])

    import gateway.status as status
    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: "hermes gateway run")
    monkeypatch.setattr(gm, "_installed_services", lambda home: [])
    monkeypatch.setattr(gm, "_service_op", _service_op)
    monkeypatch.setattr(gm, "_host_supports_migration", lambda: None)
    monkeypatch.setattr(gm, "_preflight_apply", lambda plan, target, run_as_user: None)
    # raising=False so this same file runs against a tree without the constant (A/B).
    monkeypatch.setattr(gm, "_COMPENSATOR_WAIT_SECONDS", 1.0, raising=False)
    return state


def test_the_compensator_restores_one_gateway_and_never_rebuilds_the_fleet(stranded, capsys):
    """Design (a): the flipped host lock forbids N per-profile gateways, and the compensator
    clears the served record first — so every secondary it used to start would lose the flock
    race and respawn at exit 75 forever. It restores the ONE gateway the topology allows."""
    result = gm.rollback_migration_result(stranded.root)
    assert result.succeeded is True
    out = "\n".join(result.lines)

    started = [(name, verb) for name, verb in stranded.ops if verb in ("install", "start", "restart")]
    assert all(name == "default" for name, _ in started), f"a secondary gateway was rebuilt: {started}"
    assert "Restored the per-profile gateways" not in out
    assert "coder, ops" in out and "NOT reinstalled" in out
    assert not (stranded.root / gm.MANIFEST_NAME).exists()


def test_the_compensator_reports_failure_when_no_gateway_comes_up(stranded, capsys):
    """A service-manager ``start`` that returns is not proof of a live gateway: returning True on
    it printed '✓ Restored' over a unit respawning every 5s at ExecMainStatus=75."""
    stranded.start_succeeds = False
    result = gm.rollback_migration_result(stranded.root)
    assert result.succeeded is False
    out = "\n".join(result.lines)
    assert "no gateway confirmed serving this host" in out
    assert "Compensation incomplete" in out
    assert (stranded.root / gm.MANIFEST_NAME).exists(), "the manifest must survive for the next resume"


def test_a_preflight_blocker_can_no_longer_trap_a_host_that_is_mid_migration(stranded, capsys):
    """BLOCKER 3: the only documented recovery is 're-run, it resumes'. A blocker that appears
    AFTER the units are gone (here: both profiles ending up on one bot token) used to make that
    re-run exit 1 forever, with --standalone gone and no rollback command left."""
    for name in ("coder", "ops"):
        (stranded.root / "profiles" / name / ".env").write_text(
            "TELEGRAM_BOT_TOKEN=111111:shared\n", encoding="utf-8")
    plan = gm.build_migration_plan()
    assert plan.blocked and plan.manifest is not None, "the premise: blocked AND mid-migration"

    with pytest.raises(SystemExit) as exc:
        migrate_cli.cmd_migrate(SimpleNamespace(multiplex=True, dry_run=False, yes=True))
    assert exc.value.code == 0, "the blocked gate must not refuse a host that is already mid-migration"
    out = capsys.readouterr().out
    assert "resuming the migration recorded in" in out
    assert "Preflight findings" in out, "the checks still RUN and are reported"
    assert str(stranded.root / gm.MANIFEST_NAME) in out
    assert "serves 3 profiles" in out and not (stranded.root / gm.MANIFEST_NAME).exists()
