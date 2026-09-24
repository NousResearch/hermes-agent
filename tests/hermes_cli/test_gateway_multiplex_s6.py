"""s6 container: the unset multiplex default resolves ON when the boot has parked every named
slot, ``hermes gateway migrate --multiplex`` folds an UP slot in-process, a guard refusal on a
multi-profile host is loud, and the resolved default is written back to config.yaml."""
from __future__ import annotations

from pathlib import Path

import pytest

import hermes_constants
from hermes_cli import gateway as gw
from hermes_cli import gateway_migrate as gm
from hermes_cli import gateway_multiplex_mode as mode
from hermes_cli import gateway_multiplex_s6 as s6
from hermes_cli.container_boot import reconcile_profile_gateways
from hermes_cli.service_manager import S6ServiceManager


class _FakeS6:
    """Slot state the way ``s6-svstat`` would report it, plus what stop/start would do to it."""

    def __init__(self, scandir: Path, up: set[str]):
        self.scandir, self.up, self.calls = scandir, set(up), []
        for slot in ("gateway-default", "gateway-alpha", "gateway-beta"):
            (scandir / slot).mkdir(parents=True)

    def is_running(self, name):
        return name in self.up

    def stop(self, name):
        self.calls.append(("stop", name)); self.up.discard(name)

    def start(self, name):
        self.calls.append(("start", name)); self.up.add(name)

    def restart(self, name):
        self.calls.append(("restart", name)); self.up.add(name)


@pytest.fixture
def s6_host(tmp_path, monkeypatch):
    root = tmp_path / "data"
    for sub in ("profiles/alpha", "profiles/beta"):
        (root / sub).mkdir(parents=True)
        (root / sub / "SOUL.md").write_text("x", encoding="utf-8")
    (root / "config.yaml").write_text("# my config\nmodel:\n  default: x\n", encoding="utf-8")
    (root / ".env").write_text("TELEGRAM_BOT_TOKEN=111111:default-token\n", encoding="utf-8")
    (root / "profiles/alpha/.env").write_text("TELEGRAM_BOT_TOKEN=222222:alpha-token\n", encoding="utf-8")
    (root / "profiles/beta/.env").write_text("DISCORD_BOT_TOKEN=beta-discord-333333\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    for name in ("GATEWAY_MULTIPLEX_PROFILES", "TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "API_SERVER_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    scandir = tmp_path / "run-service"
    fake = _FakeS6(scandir, up=set())
    monkeypatch.setattr(gw, "_running_under_s6", lambda: True)
    monkeypatch.setattr(S6ServiceManager, "__init__", lambda self, scandir=scandir: setattr(self, "scandir", scandir))
    monkeypatch.setattr(s6, "_manager", lambda: fake)
    monkeypatch.setattr(s6, "_wait_down", lambda service_dir, timeout_ms=0: None)
    monkeypatch.setattr(gm, "_live_gateway_pid", lambda home: None)
    monkeypatch.setattr(gm, "_wait_for_served", lambda home, expected, timeout: sorted(expected))
    return root, fake


def test_parked_named_slots_do_not_veto_the_unset_default(s6_host):
    """The user's exact scenario: boot parked alpha+beta down, key unset → the guard must let the
    default apply (only an UP named slot is a second gateway). Base refused with 'Restart the container'."""
    root, fake = s6_host
    from gateway.config import GatewayConfig
    decision = mode.resolve_multiplex_mode(GatewayConfig.from_dict({}))
    assert (decision.enabled, decision.source) == (True, "default"), decision
    fake.up.add("gateway-alpha")
    blocked = mode.resolve_multiplex_mode(GatewayConfig.from_dict({}))
    assert blocked.source == "guard" and "alpha" in blocked.reason


def test_migrate_folds_an_up_named_slot_in_process_and_boot_shares_the_rule(s6_host, tmp_path):
    """In-container convergence: an UP alpha slot is stopped + parked (down file), the root slot
    (re)started, no container restart. The fold rule is the ONE ``fold_named_slot_intent``."""
    root, fake = s6_host
    fake.up.update({"gateway-alpha", "gateway-default"})
    plan = gm.build_migration_plan()
    assert [p.name for p in plan.standalone_secondaries] == ["alpha"]
    assert gm.apply_migration(plan, served_wait=0) is True
    assert ("stop", "gateway-alpha") in fake.calls and ("restart", "gateway-default") in fake.calls
    assert (fake.scandir / "gateway-alpha" / "down").exists()
    assert "gateway-alpha" not in fake.up and "gateway-default" in fake.up
    assert mode.explicit_multiplex_flag(root) is True
    # Boot and migration fold through the same function: a running named intent moves to the root.
    fold = s6.fold_named_slot_intent(None, [("alpha", "running"), ("beta", "stopped")])
    assert fold == s6.FoldDecision(folded=("alpha",), root_should_start=True)
    (root / "profiles/alpha/gateway_state.json").write_text('{"desired_state": "running"}', encoding="utf-8")
    actions = {a.profile: a for a in reconcile_profile_gateways(
        hermes_home=root, scandir=tmp_path / "boot-svc", dry_run=True, container_argv=())}
    assert actions["default"].action == "started" and actions["alpha"].folded_into_root


def test_standalone_box_names_unserved_profiles_and_is_silent_for_single_profile(s6_host):
    guard = mode.MultiplexDecision(False, "guard", "profile(s) 'alpha' still run their own gateway")
    lines = mode.standalone_warning_lines(guard)
    assert lines[0].startswith("┌") and lines[-1].startswith("└")
    assert any("alpha, beta" in line for line in lines) and any("migrate --multiplex" in line for line in lines)
    assert mode.standalone_warning_lines(mode.MultiplexDecision(False, "guard", mode.SINGLE_PROFILE_REASON)) == []
    assert mode.standalone_warning_lines(mode.MultiplexDecision(True, "default", "unset")) == []


def test_resolved_default_is_written_once_and_never_on_a_guard(s6_host):
    root, _fake = s6_host
    guard = mode.MultiplexDecision(False, "guard", "profile(s) 'alpha' still run their own gateway")
    assert mode.persist_resolved_default(guard, root) is False
    assert mode.explicit_multiplex_flag(root) is None
    on = mode.MultiplexDecision(True, "default", "unset; default applies")
    assert mode.persist_resolved_default(on, root) is True
    text = (root / "config.yaml").read_text(encoding="utf-8")
    assert "# my config" in text and "multiplex_profiles: true" in text  # comment-preserving writer
    assert mode.persist_resolved_default(on, root) is False  # already explicit: once
    # A retired ``false`` is rewritten in place, never silently: the one-time notice reaches the
    # gateway start (log_multiplex_decision prints it) and the next ``hermes update`` summary.
    (root / "config.yaml").write_text("# my config\ngateway:\n  multiplex_profiles: false\n", encoding="utf-8")
    from gateway.config import GatewayConfig
    decision = mode.resolve_multiplex_mode(GatewayConfig.from_dict({"gateway": {"multiplex_profiles": False}}))
    assert decision.source == "retired-opt-out"
    mode.log_multiplex_decision(decision)
    text = (root / "config.yaml").read_text(encoding="utf-8")
    assert "# my config" in text and "multiplex_profiles: true" in text and "false" not in text
    notice = mode.consume_rewritten_notice(root)
    assert any("rewritten to true" in line for line in notice) and notice[0].startswith("┌")
    assert mode.consume_rewritten_notice(root) == []  # one-time
