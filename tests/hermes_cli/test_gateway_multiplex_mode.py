"""``gateway.multiplex_profiles`` defaults to ON, but an UNSET flag is settled at boot by
``hermes_cli.gateway_multiplex_mode.resolve_multiplex_mode`` — the same preflight
``hermes gateway migrate --multiplex`` runs — so flipping the default can never make a default
gateway double-bind a fleet that still runs per-profile gateways.

The service layer is faked through ``gateway_migrate``'s ``_installed_services`` / ``_live_gateway_pid``
seams (the shape ``test_gateway_migrate_multiplex.py`` uses).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

import hermes_constants
from gateway.config import GatewayConfig, load_gateway_config
from hermes_cli import gateway_migrate as gm
from hermes_cli import gateway_multiplex_mode as mode
from hermes_cli.config_defaults import DEFAULT_CONFIG


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    """default + coder + ops with distinct bot tokens; nothing runs a gateway unless a test says so."""
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
    services: dict[str, list] = {}
    pids: dict[str, int] = {}
    monkeypatch.setattr(gm, "_installed_services", lambda home: services.get(_name(home), []))
    monkeypatch.setattr(gm, "_live_gateway_pid", lambda home: pids.get(_name(home)))
    monkeypatch.setattr(gm, "_host_supports_migration", lambda: None)
    return root, services, pids


def _name(home: Path) -> str:
    return hermes_constants.profile_name_for_home(home) or "default"


def test_default_config_is_on_but_the_loader_leaves_an_unset_flag_undecided():
    """DEFAULT_CONFIG says on; GatewayConfig keeps "unset" distinguishable from "chosen" so the boot
    guard can tell them apart (an explicit value must survive verbatim)."""
    assert DEFAULT_CONFIG["gateway"]["multiplex_profiles"] is True
    assert GatewayConfig.from_dict({}).multiplex_profiles is None
    assert not GatewayConfig.from_dict({}).multiplex_profiles  # readers treat undecided as off
    assert GatewayConfig.from_dict({"gateway": {"multiplex_profiles": False}}).multiplex_profiles is False
    assert GatewayConfig.from_dict({"multiplex_profiles": "true"}).multiplex_profiles is True


def test_unset_flag_multiplexes_a_quiet_fleet_and_stays_standalone_beside_a_live_secondary(fleet):
    root, services, pids = fleet
    decision = mode.resolve_multiplex_mode(cfg := load_gateway_config())
    assert decision == mode.MultiplexDecision(True, "default", decision.reason)
    assert cfg.multiplex_profiles is True

    pids["coder"] = 4101  # coder still runs its own gateway: a multiplexer would double-poll its bot
    decision = mode.resolve_multiplex_mode(cfg := load_gateway_config())
    assert not decision.enabled and decision.source == "guard"
    assert "'coder'" in decision.reason and gm.MIGRATE_COMMAND in decision.reason
    assert cfg.multiplex_profiles is False
    # The CLI side agrees: no live multiplexer record and no explicit opt-in means coder is NOT served.
    from hermes_cli.gateway import named_profile_served_by_running_multiplexer
    assert named_profile_served_by_running_multiplexer("coder") is False
    assert mode.default_gateway_multiplexes(root) is False

    pids.clear()
    services["ops"] = [("systemd", False)]  # an installed unit counts the same as a live pid
    assert mode.resolve_multiplex_mode(load_gateway_config()).source == "guard"


def test_preflight_blocker_and_single_profile_keep_the_unset_default_standalone(fleet):
    root, _services, _pids = fleet
    # coder reuses the default's bot token -> the migration preflight's duplicate-credential blocker.
    (root / "profiles/coder/.env").write_text("TELEGRAM_BOT_TOKEN=111111:default-token\n", encoding="utf-8")
    decision = mode.resolve_multiplex_mode(load_gateway_config())
    assert decision.source == "guard" and "profile_routes" in decision.reason

    for sub in ("profiles/coder", "profiles/ops"):
        shutil.rmtree(root / sub)
    decision = mode.resolve_multiplex_mode(load_gateway_config())
    assert decision == mode.MultiplexDecision(False, "guard", mode.SINGLE_PROFILE_REASON)


def test_explicit_flag_is_never_second_guessed(fleet, monkeypatch):
    root, _services, pids = fleet
    pids["coder"] = 4101
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n", encoding="utf-8")
    cfg = load_gateway_config()
    assert mode.resolve_multiplex_mode(cfg) == mode.MultiplexDecision(True, "config")
    assert cfg.multiplex_profiles is True
    assert mode.explicit_multiplex_flag(root) is True

    monkeypatch.setenv("GATEWAY_MULTIPLEX_PROFILES", "false")
    pids.clear()
    cfg = load_gateway_config()
    assert mode.resolve_multiplex_mode(cfg) == mode.MultiplexDecision(False, "config")
    assert cfg.multiplex_profiles is False
    assert mode.explicit_multiplex_flag(root) is False


def test_migration_plan_treats_the_unset_default_as_not_yet_multiplexed(fleet):
    """The fleet the boot guard refuses is exactly the one ``hermes gateway migrate --multiplex`` folds:
    an unset flag must not read as "already multiplexed" or the migration would short-circuit."""
    root, services, pids = fleet
    pids.update({"coder": 4101, "ops": 4102})
    services.update({"coder": [("systemd", False)], "ops": [("systemd", False)]})
    plan = gm.build_migration_plan()
    assert plan.multiplex_flag_on is False and not plan.already_multiplexed
    assert plan.eligible_for_migration()


def test_live_record_outranks_the_raw_flag_for_other_processes(fleet, monkeypatch):
    """A CLI process asks the LIVE default gateway (which settled the unset default itself) before
    reading config; a gateway that stayed standalone recorded an empty served set."""
    root, _services, _pids = fleet
    import gateway.status as status
    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: "hermes gateway run")
    (root / "gateway.pid").write_text(json.dumps({"pid": os.getpid(), "hermes_home": str(root)}))
    record = {"pid": os.getpid(), "hermes_home": str(root), "gateway_state": "running", "served_profiles": []}
    (root / "gateway_state.json").write_text(json.dumps(record))
    assert mode.default_gateway_multiplexes(root) is False
    record["served_profiles"] = ["default", "coder", "ops"]
    (root / "gateway_state.json").write_text(json.dumps(record))
    assert mode.default_gateway_multiplexes(root) is True


def test_guard_refusal_is_recorded_in_runtime_status_and_cleared_on_default(tmp_path, monkeypatch):
    """A guard refusal must be visible to `hermes gateway status`, not only in the boot log; a later
    boot that multiplexes clears it (a stale reason would misdescribe the live gateway)."""
    from gateway import status as gw_status
    from hermes_cli.gateway_multiplex_mode import MultiplexDecision, record_multiplex_decision
    monkeypatch.setattr(gw_status, "_get_runtime_status_path", lambda: tmp_path / "gateway_state.json")
    record_multiplex_decision(MultiplexDecision(False, "guard", "profile(s) 'coder' still run their own gateway"))
    assert "coder" in gw_status.read_runtime_status(tmp_path / "gateway_state.json")["multiplex_standalone_reason"]
    record_multiplex_decision(MultiplexDecision(True, "default", "unset; default applies"))
    assert gw_status.read_runtime_status(tmp_path / "gateway_state.json")["multiplex_standalone_reason"] is None
