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
    )

    def _name(home: Path) -> str:
        return hermes_constants.profile_name_for_home(home) or "default"

    def _service_op(kind, system, verb, home):
        name = _name(home)
        state.ops.append((name, verb))
        if verb == "uninstall":
            state.services.pop(name, None)
        elif verb == "install":
            state.services[name] = (kind, system)
        elif verb in ("start", "restart") and name == "default":
            # What the real multiplexer does at startup: record the served set in the default home.
            (root / "gateway.pid").write_text(json.dumps({"pid": os.getpid(), "hermes_home": str(root)}))
            (root / "gateway_state.json").write_text(json.dumps({
                "pid": os.getpid(), "hermes_home": str(root), "gateway_state": "running",
                "served_profiles": ["default", "coder", "ops"],
            }))

    monkeypatch.setattr(gm, "_installed_service", lambda home: state.services.get(_name(home)))
    monkeypatch.setattr(gm, "_live_gateway_pid", lambda home: state.pids.get(_name(home)))
    monkeypatch.setattr(gm, "_service_op", _service_op)
    monkeypatch.setattr(gm, "_stop_gateway_process", lambda home: state.pids.pop(_name(home), None))
    monkeypatch.setattr(gm, "_host_supports_migration", lambda: None)
    state.root = root
    return state


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

    fleet.ops.clear()
    assert gm.rollback_migration(fleet.root) is True
    assert _config_flag(fleet.root) is False
    assert fleet.services == {"default": ("systemd", False), "coder": ("systemd", False), "ops": ("systemd", False)}
    assert [op for op in fleet.ops if op[0] != "default"] == [
        ("coder", "install"), ("coder", "start"), ("ops", "install"), ("ops", "start")]
    assert not (fleet.root / gm.MANIFEST_NAME).exists()


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
    gm.maybe_auto_migrate_after_update()
    out = capsys.readouterr().out
    assert "Migrating per-profile gateways" in out and "serves 3 profiles" in out
    assert _config_flag(fleet.root) is True and (fleet.root / gm.MANIFEST_NAME).exists()

    # Blocked fleet: warning block with the fix + one-liner; nothing changes.
    for f in ("gateway.pid", "gateway_state.json", gm.MANIFEST_NAME):
        (fleet.root / f).unlink()
    (fleet.root / "config.yaml").write_text("model:\n  default: x\n", encoding="utf-8")
    fleet.services.update({"coder": ("systemd", False)}); fleet.services.pop("default", None)
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
