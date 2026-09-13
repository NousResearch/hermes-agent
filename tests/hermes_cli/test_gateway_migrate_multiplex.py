"""``hermes gateway migrate``: preflight verdicts, apply/rollback bookkeeping, and the update hook.

Service layer is faked through the module's ``_installed_service`` / ``_service_op`` seams (the same
shape ``hermes gateway install`` tests use); the default gateway boot is faked by writing the
``served_profiles`` record the real multiplexer writes. Blockers reuse the gateway's own credential
fingerprint and port-binding predicates, so the tests assert verdict → effect, not internal lists.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_constants
from hermes_cli import gateway_migrate as gm


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    """default + coder + ops; both secondaries run a 'live' standalone gateway with a systemd unit."""
    root = tmp_path / "hermes"
    for sub in ("profiles/coder", "profiles/ops"):
        (root / sub).mkdir(parents=True)
    (root / "config.yaml").write_text("model:\n  default: x\n", encoding="utf-8")
    (root / ".env").write_text("TELEGRAM_BOT_TOKEN=111111:default-token\n", encoding="utf-8")
    (root / "profiles/coder/.env").write_text("TELEGRAM_BOT_TOKEN=222222:coder-token\n", encoding="utf-8")
    (root / "profiles/ops/.env").write_text("DISCORD_BOT_TOKEN=ops-discord-333333\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    for name in ("TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "API_SERVER_KEY", "WEBHOOK_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)

    state = SimpleNamespace(
        services={"coder": ("systemd", False), "ops": ("systemd", False)},
        pids={"coder": 4101, "ops": 4102},
        ops=[],
        refused_at_start={},
    )

    def _service_op(kind, system, verb, home):
        name = _name(home)
        state.ops.append((name, verb))
        if verb == "start" and name != "default":
            # What the real `hermes -p <name> gateway run` checks first: is a live multiplexer
            # still recorded as serving me? (exit 78 if so — the unit is then parked for good).
            from hermes_cli.gateway import named_profile_served_by_running_multiplexer
            state.refused_at_start[name] = named_profile_served_by_running_multiplexer(name)
        if verb == "uninstall":
            state.services.pop(name, None)
        elif verb == "install":
            state.services[name] = (kind, system)
        elif verb in ("start", "restart") and name == "default":
            (root / "gateway.pid").write_text(json.dumps({"pid": os.getpid(), "hermes_home": str(root)}))
            runtime_path = root / "gateway_state.json"
            runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
            runtime.update({"pid": os.getpid(), "hermes_home": str(root), "gateway_state": "running"})
            # Multiplex startup records ownership; standalone startup historically preserved the
            # old key, which is the stale-state half of #109473's rollback failure.
            if _config_flag(root):
                runtime["served_profiles"] = ["default", "coder", "ops"]
            runtime_path.write_text(json.dumps(runtime))

    monkeypatch.setattr(gm, "_installed_service", lambda home: state.services.get(_name(home)))
    monkeypatch.setattr(gm, "_live_gateway_pid", lambda home: state.pids.get(_name(home)))
    monkeypatch.setattr(gm, "_service_op", _service_op)
    monkeypatch.setattr(gm, "_stop_gateway_process", lambda home: state.pids.pop(_name(home), None))
    monkeypatch.setattr(gm, "_host_supports_migration", lambda: None)
    state.root = root
    return state


def _name(home: Path) -> str:
    return hermes_constants.profile_name_for_home(home) or "default"


def _config_flag(root: Path):
    import yaml
    raw = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    return (raw.get("gateway") or {}).get("multiplex_profiles")


def test_dry_run_and_blocked_preflight_change_nothing(fleet, capsys):
    plan = gm.build_migration_plan()
    assert not plan.blocked and plan.eligible_for_migration()
    assert [p.name for p in plan.standalone_secondaries] == ["coder", "ops"]

    gm.cmd_migrate(SimpleNamespace(multiplex=True, standalone=False, dry_run=True, yes=True))  # returns; no exit
    assert "dry run" in capsys.readouterr().out
    # Blocked: coder reuses the default's Telegram token -> the gateway's own fingerprint says duplicate.
    (fleet.root / "profiles/coder/.env").write_text("TELEGRAM_BOT_TOKEN=111111:default-token\n", encoding="utf-8")
    blocked = gm.build_migration_plan()
    assert blocked.blocked and "profile_routes" in blocked.blockers[0] and "'coder'" in blocked.blockers[0]
    with pytest.raises(SystemExit) as exc:
        gm.cmd_migrate(SimpleNamespace(multiplex=True, standalone=False, dry_run=False, yes=True))
    assert exc.value.code == 1
    assert fleet.ops == [] and fleet.services == {"coder": ("systemd", False), "ops": ("systemd", False)}
    assert fleet.pids == {"coder": 4101, "ops": 4102} and _config_flag(fleet.root) is None
    assert not (fleet.root / gm.MANIFEST_NAME).exists()
    assert "nothing will be changed" in capsys.readouterr().out


def test_apply_records_manifest_flips_flag_and_rollback_restores(fleet, capsys):
    plan = gm.build_migration_plan()
    assert gm.apply_migration(plan, served_wait=5.0) is True
    manifest = json.loads((fleet.root / gm.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert {s["profile"] for s in manifest["secondaries"]} == {"coder", "ops"}
    assert all(s["service"] == {"kind": "systemd", "system": False} for s in manifest["secondaries"])
    assert _config_flag(fleet.root) is True
    assert "coder" not in fleet.services and "ops" not in fleet.services and fleet.pids == {}
    # The default is brought up on the SAME service manager the secondaries used.
    assert fleet.services["default"] == ("systemd", False)
    assert ("default", "install") in fleet.ops and ("default", "start") in fleet.ops
    assert "serves 3 profiles" in capsys.readouterr().out
    # Idempotent: a second run sees the live multiplexer and refuses cleanly.
    again = gm.build_migration_plan()
    assert again.already_multiplexed and gm.apply_migration(again) is True

    runtime_path = fleet.root / "gateway_state.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["platforms"] = {
        "telegram": {"state": "connected"},
        "coder:telegram": {"state": "connected"},
        "ops:discord": {"state": "connected"},
    }
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    fleet.ops.clear()
    assert gm.rollback_migration(fleet.root) is True
    assert _config_flag(fleet.root) is False
    assert fleet.services == {"default": ("systemd", False), "coder": ("systemd", False), "ops": ("systemd", False)}
    assert [op for op in fleet.ops if op[0] != "default"] == [
        ("coder", "install"), ("coder", "start"), ("ops", "install"), ("ops", "start")]
    assert fleet.ops[-1] == ("default", "restart")
    # Each secondary's own gateway must have been startable at the moment it was started.
    assert fleet.refused_at_start == {"coder": False, "ops": False}
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert runtime["served_profiles"] == []
    assert runtime["platforms"] == {"telegram": {"state": "connected"}}
    from hermes_cli.gateway import named_profile_served_by_running_multiplexer
    assert named_profile_served_by_running_multiplexer("coder") is False
    assert not (fleet.root / gm.MANIFEST_NAME).exists()


def test_rollback_with_failed_secondary_still_restarts_default_and_keeps_manifest(fleet, monkeypatch):
    assert gm.apply_migration(gm.build_migration_plan(), served_wait=5.0) is True
    fleet.ops.clear()
    real_op = gm._service_op

    def _flaky(kind, system, verb, home):
        if verb == "start" and _name(home) == "coder":
            raise RuntimeError("systemctl start failed")
        real_op(kind, system, verb, home)

    monkeypatch.setattr(gm, "_service_op", _flaky)
    assert gm.rollback_migration(fleet.root) is False
    # The flag is off, so the default must not be left multiplexing; the manifest stays for a re-run.
    assert _config_flag(fleet.root) is False
    assert fleet.ops[-1] == ("default", "restart")
    assert ("ops", "start") in fleet.ops
    assert (fleet.root / gm.MANIFEST_NAME).exists()


def test_rollback_rerun_does_not_respawn_a_running_detached_secondary(fleet, monkeypatch):
    # Manifest of a fleet with no service manager anywhere (detached gateways only).
    gm._write_manifest(fleet.root, {"version": 1, "flag_was": False, "default": {"service": None}, "secondaries": [
        {"profile": n, "home": str(fleet.root / "profiles" / n), "pid": 4100, "service": None} for n in ("coder", "ops")]})
    fleet.services.clear()
    fleet.pids = {"coder": 4101}  # coder came back up in an earlier, interrupted rollback; ops did not
    spawned = []
    monkeypatch.setattr(gm, "_spawn_detached_gateway", lambda home: spawned.append(_name(home)) or True)
    assert gm.rollback_migration(fleet.root) is True
    assert spawned == ["ops"]


def test_malformed_manifest_is_refused_before_any_mutation(fleet, capsys):
    (fleet.root / gm.MANIFEST_NAME).write_text(
        json.dumps({"version": 1, "flag_was": False, "default": [], "secondaries": [{"home": "x"}]}), encoding="utf-8")
    assert gm.rollback_migration(fleet.root) is False
    assert _config_flag(fleet.root) is None and fleet.ops == []
    assert "fix or delete the manifest" in capsys.readouterr().out
    gm.cmd_migrate(SimpleNamespace(multiplex=False, standalone=True, dry_run=True, yes=True))
    assert "fix or delete the manifest" in capsys.readouterr().out


def test_rollback_plan_and_execution_agree_on_a_secondary_with_nothing_recorded(fleet, capsys):
    gm._write_manifest(fleet.root, {"version": 1, "flag_was": False, "default": {"service": None}, "secondaries": [
        {"profile": "coder", "home": str(fleet.root / "profiles/coder"), "pid": None, "service": None}]})
    fleet.services.clear()
    fleet.pids.clear()
    plan = "\n".join(gm.format_rollback_plan(fleet.root, gm._read_manifest(fleet.root), dry_run=True))
    assert gm.rollback_migration(fleet.root) is True
    out = capsys.readouterr().out
    # The plan must not promise an action the run never performs: both name the same outcome.
    plan_line = next(line.split("coder: ", 1)[1] for line in plan.splitlines() if "coder: " in line)
    assert plan_line in out and "restore" not in plan_line.split(";")[0]
    assert fleet.ops == []


def test_standalone_dry_run_prints_rollback_plan_without_mutation(fleet, capsys):
    assert gm.apply_migration(gm.build_migration_plan(), served_wait=5.0) is True
    capsys.readouterr()
    fleet.ops.clear()
    before = {
        path: path.read_bytes()
        for path in (
            fleet.root / "config.yaml",
            fleet.root / gm.MANIFEST_NAME,
            fleet.root / "gateway_state.json",
        )
    }
    services_before = dict(fleet.services)
    pids_before = dict(fleet.pids)

    gm.cmd_migrate(SimpleNamespace(multiplex=False, standalone=True, dry_run=True, yes=True))

    out = capsys.readouterr().out
    assert "Rollback plan (dry run" in out and "coder" in out and "ops" in out
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert fleet.services == services_before and fleet.pids == pids_before and fleet.ops == []


def test_secondary_port_binder_is_notice_with_ingress_and_blocker_without(fleet, monkeypatch):
    """The verdict follows the adapter's ``serves_profile_prefix`` declaration, not a hardcoded list."""
    (fleet.root / "profiles/ops/.env").write_text(
        "DISCORD_BOT_TOKEN=ops-discord-333333\nAPI_SERVER_KEY=ops-api-key-abcdef\n", encoding="utf-8")
    (fleet.root / "profiles/ops/config.yaml").write_text(
        "platforms:\n  api_server:\n    enabled: true\n    extra:\n      port: 9999\n", encoding="utf-8")
    plan = gm.build_migration_plan()
    assert not plan.blocked, plan.blockers
    assert any("api_server" in n and "/p/ops/" in n for n in plan.notices), plan.notices

    monkeypatch.setattr(gm, "platform_serves_profile_prefix", lambda value: False)
    plan = gm.build_migration_plan()
    assert plan.blocked and "api_server" in plan.blockers[0] and "/p/ops/" in plan.blockers[0]


def test_serves_profile_prefix_is_read_from_adapter_classes():
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.platforms.webhook import WebhookAdapter
    assert APIServerAdapter.serves_profile_prefix and WebhookAdapter.serves_profile_prefix
    assert gm.platform_serves_profile_prefix("api_server") is True
    assert gm.platform_serves_profile_prefix("webhook") is True
    # Every adapter that binds through shared_ingress.bind_listener is served at /p/<profile>/
    # for a secondary, so migrate must report it as a notice, never a blocker.
    for platform in ("sms", "line", "teams", "bluebubbles", "whatsapp_cloud", "msgraph_webhook"):
        assert gm.platform_serves_profile_prefix(platform) is True, platform
    # An outbound-only adapter never declares it (and never needs to).
    assert gm.platform_serves_profile_prefix("telegram") is False


def test_update_hook_migrates_when_unblocked_and_only_warns_when_blocked(fleet, capsys):
    fleet.services["default"] = ("systemd", False)  # same service domain as the secondaries
    gm.maybe_auto_migrate_after_update()
    out = capsys.readouterr().out
    assert "Migrating per-profile gateways" in out and "serves 3 profiles" in out
    assert _config_flag(fleet.root) is True and (fleet.root / gm.MANIFEST_NAME).exists()

    # Blocked fleet: warning block with the fix + one-liner; nothing changes.
    for f in ("gateway.pid", "gateway_state.json", gm.MANIFEST_NAME):
        (fleet.root / f).unlink()
    (fleet.root / "config.yaml").write_text("model:\n  default: x\n", encoding="utf-8")
    fleet.services.update({"coder": ("systemd", False), "default": ("systemd", False)})
    fleet.pids.update({"coder": 4101}); fleet.ops.clear()
    (fleet.root / "profiles/coder/.env").write_text("TELEGRAM_BOT_TOKEN=111111:default-token\n", encoding="utf-8")
    gm.maybe_auto_migrate_after_update()
    out = capsys.readouterr().out
    assert gm.MIGRATE_COMMAND in out and "profile_routes" in out
    assert fleet.ops == [] and _config_flag(fleet.root) is None


def test_update_hook_never_touches_single_profile_or_already_multiplexed(fleet, capsys):
    fleet.services.clear(); fleet.pids.clear()  # secondaries exist but run no gateway of their own
    gm.maybe_auto_migrate_after_update()
    assert capsys.readouterr().out == "" and _config_flag(fleet.root) is None


@pytest.mark.parametrize(
    ("secondary_service", "secondary_uid", "secondary_home", "expected"),
    [
        (("systemd", True), 1000, "profiles/coder", "different service domain"),
        (("launchd", False), 1000, "profiles/coder", "different service domain"),
        (("systemd", False), 2000, "profiles/coder", "UNIX privilege boundary"),
        (("systemd", False), 1000, "external", "outside"),
    ],
)
def test_update_hook_refuses_to_cross_service_user_or_home_boundary(
    fleet, capsys, monkeypatch, secondary_service, secondary_uid, secondary_home, expected,
):
    """#109954: the unattended hook must not fold a secondary that sits behind a kernel-enforced
    boundary (other service domain, other UNIX user, HERMES_HOME outside profiles/). It prints the
    boundary + the explicit command and touches nothing; the dry-run plan shows the same finding as a
    notice and the explicit command stays available."""
    fleet.services["default"] = ("systemd", False)
    fleet.services["ops"] = secondary_service
    ops_home = fleet.root.parent / "external-ops" if secondary_home == "external" else fleet.root / secondary_home
    monkeypatch.setattr(
        gm, "_gateway_identity",
        lambda home, pid, service: (secondary_uid if _name(home) == "ops" else 1000, ops_home if _name(home) == "ops" else home),
        raising=False,  # absent on the pre-fix module: the test must then fail on behaviour, not on the seam
    )

    gm.maybe_auto_migrate_after_update()

    out = capsys.readouterr().out
    assert expected in out and gm.MIGRATE_COMMAND in out and "'ops'" in out
    assert fleet.ops == [] and fleet.pids == {"coder": 4101, "ops": 4102}
    assert fleet.services == {"default": ("systemd", False), "coder": ("systemd", False), "ops": secondary_service}
    assert _config_flag(fleet.root) is None and not (fleet.root / gm.MANIFEST_NAME).exists()

    plan = gm.build_migration_plan()
    assert not plan.blocked and any(expected in n for n in plan.notices)
    gm.cmd_migrate(SimpleNamespace(multiplex=True, standalone=False, dry_run=True, yes=True))
    assert expected in capsys.readouterr().out


def test_update_hook_still_migrates_same_user_same_scope_profiles_under_the_default_tree(fleet, capsys, monkeypatch):
    """The guard is a boundary check, not a kill switch: one user, one service domain, everything under
    profiles/ (the shape `hermes profile create` produces) still auto-migrates."""
    fleet.services["default"] = ("systemd", False)
    monkeypatch.setattr(gm, "_gateway_identity", lambda home, pid, service: (1000, home), raising=False)

    gm.maybe_auto_migrate_after_update()

    out = capsys.readouterr().out
    assert "Migrating per-profile gateways" in out and "serves 3 profiles" in out
    assert _config_flag(fleet.root) is True and ("ops", "uninstall") in fleet.ops


def test_auto_multiplex_migration_false_opts_out_of_the_update_hook_but_not_the_explicit_command(fleet, capsys):
    """``gateway.auto_multiplex_migration: false`` is a durable opt-out: an otherwise-eligible fleet is
    left alone by ``hermes update`` (no output, no ops, no flag flip), while the operator typing
    ``migrate --multiplex`` still migrates. Only the nested key counts."""
    assert gm.build_migration_plan().eligible_for_migration()  # would migrate but for the flag
    (fleet.root / "config.yaml").write_text(
        "model:\n  default: x\nauto_multiplex_migration: false\ngateway:\n  auto_multiplex_migration: false\n",
        encoding="utf-8")

    gm.maybe_auto_migrate_after_update()
    assert capsys.readouterr().out == ""
    assert fleet.ops == [] and _config_flag(fleet.root) is None
    assert fleet.services == {"coder": ("systemd", False), "ops": ("systemd", False)}
    assert fleet.pids == {"coder": 4101, "ops": 4102}
    assert not (fleet.root / gm.MANIFEST_NAME).exists()

    # A top-level alias is NOT honoured; absent and an explicit true keep the automatic behaviour.
    from hermes_cli.gateway_migrate_guards import auto_migration_opted_out
    (fleet.root / "config.yaml").write_text(
        "model:\n  default: x\nauto_multiplex_migration: false\n", encoding="utf-8")
    assert auto_migration_opted_out(fleet.root) is False
    (fleet.root / "config.yaml").write_text(
        "model:\n  default: x\ngateway:\n  auto_multiplex_migration: true\n", encoding="utf-8")
    assert auto_migration_opted_out(fleet.root) is False

    # The opt-out governs the AUTOMATIC path only: an explicit --multiplex is an explicit request.
    (fleet.root / "config.yaml").write_text(
        "model:\n  default: x\ngateway:\n  auto_multiplex_migration: false\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        gm.cmd_migrate(SimpleNamespace(multiplex=True, standalone=False, dry_run=False, yes=True))
    assert exc.value.code == 0
    assert _config_flag(fleet.root) is True and (fleet.root / gm.MANIFEST_NAME).exists()


def test_explicit_migrate_with_no_standalone_secondaries_still_flips_flag_and_restarts_default(fleet, capsys, monkeypatch):
    """The user typed --multiplex: 'nothing to migrate' + flag left off was a no-op the user did not ask
    for. The update hook keeps its no-op (previous test); the explicit command proceeds."""
    fleet.services.clear(); fleet.pids.clear()

    def _detached(home):  # no service manager anywhere -> detached start writes the served record
        fleet.services["default-detached"] = True
        (fleet.root / "gateway.pid").write_text(json.dumps({"pid": os.getpid(), "hermes_home": str(fleet.root)}))
        (fleet.root / "gateway_state.json").write_text(json.dumps({
            "pid": os.getpid(), "hermes_home": str(fleet.root), "gateway_state": "running",
            "served_profiles": ["default", "coder", "ops"]}))
        return True
    monkeypatch.setattr(gm, "_spawn_detached_gateway", _detached)
    assert not gm.build_migration_plan().standalone_secondaries
    with pytest.raises(SystemExit) as exc:
        gm.cmd_migrate(SimpleNamespace(multiplex=True, standalone=False, dry_run=False, yes=True))
    assert exc.value.code == 0
    assert fleet.services.pop("default-detached") is True
    assert _config_flag(fleet.root) is True
    manifest = json.loads((fleet.root / gm.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["secondaries"] == [] and manifest["flag_was"] is False
    assert ("default", "install") not in fleet.ops
    out = capsys.readouterr().out
    assert "serves 3 profiles" in out
    # The same manifest rolls it back: flag restored, nothing to reinstall.
    assert gm.rollback_migration(fleet.root) is True and _config_flag(fleet.root) is False
