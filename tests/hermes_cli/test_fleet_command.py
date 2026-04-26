"""Tests for the ``hermes fleet`` CLI command."""

from __future__ import annotations

import argparse
import os
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.fleet import (
    _cmd_add,
    _cmd_list,
    _cmd_setup,
    _cmd_status,
    add_fleet_parser,
    cmd_fleet,
)


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("TELEGRAM_FLEET_MANAGER_TOKEN", raising=False)
    return tmp_path


@pytest.fixture(autouse=True)
def _isolate():
    from gateway.telegram_fleet.coordinator import reset_coordinator
    from gateway.telegram_fleet.guardrails import reset_rate_limits

    reset_coordinator()
    reset_rate_limits()
    yield
    reset_coordinator()
    reset_rate_limits()


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


# ── status ─────────────────────────────────────────────────────────────


def test_status_warns_when_unconfigured(hermes_home, capsys):
    rc = _cmd_status(_ns(fleet_command="status"))
    out = capsys.readouterr().out
    assert "Manager token NOT set" in out
    assert "Run `hermes fleet setup`" in out
    assert rc == 1  # nonzero so shell can detect missing setup


def test_status_reports_active_count(hermes_home, monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_FLEET_MANAGER_TOKEN", "12345:ABC" + "x" * 30)
    from gateway.telegram_fleet.api import ManagedBotInfo
    from gateway.telegram_fleet.coordinator import FleetCoordinator

    api = MagicMock()
    api.get_me.return_value = {"id": 1, "username": "TestMgr", "can_manage_bots": True}
    coord = FleetCoordinator(manager_token="12345:ABC", api_client=api)
    coord.spawn_bot("active_bot")
    coord.absorb_managed_bot(
        ManagedBotInfo(token="x:tok", bot_id=1, bot_username="active_bot")
    )
    coord.spawn_bot("pending_bot")
    rc = _cmd_status(_ns(fleet_command="status"))
    out = capsys.readouterr().out
    assert "Manager token configured" in out
    assert "1 active, 1 pending" in out
    assert rc == 0


# ── list ───────────────────────────────────────────────────────────────


def test_list_empty_roster(hermes_home, capsys):
    rc = _cmd_list(_ns(fleet_command="list"))
    out = capsys.readouterr().out
    assert "Roster is empty" in out
    assert rc == 0


def test_list_pretty_prints_active_and_pending(hermes_home, monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_FLEET_MANAGER_TOKEN", "x:y")
    from gateway.telegram_fleet.api import ManagedBotInfo
    from gateway.telegram_fleet.coordinator import FleetCoordinator

    api = MagicMock()
    api.get_me.return_value = {"id": 1, "username": "TestMgr", "can_manage_bots": True}
    coord = FleetCoordinator(manager_token="12345:ABC", api_client=api)
    coord.spawn_bot("alpha_bot", persona="research lead")
    coord.absorb_managed_bot(
        ManagedBotInfo(token="x:tok", bot_id=1, bot_username="alpha_bot")
    )
    coord.spawn_bot("beta_bot", persona="market analyst")
    rc = _cmd_list(_ns(fleet_command="list"))
    out = capsys.readouterr().out
    assert "@alpha_bot" in out
    assert "research lead" in out
    assert "@beta_bot" in out
    assert "active" in out
    assert "pending" in out
    assert rc == 0


def test_list_truncates_long_personas(hermes_home, monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_FLEET_MANAGER_TOKEN", "x:y")
    from gateway.telegram_fleet.api import ManagedBotInfo
    from gateway.telegram_fleet.coordinator import FleetCoordinator

    api = MagicMock()
    api.get_me.return_value = {"id": 1, "username": "TestMgr", "can_manage_bots": True}
    coord = FleetCoordinator(manager_token="12345:ABC", api_client=api)
    coord.spawn_bot("verbose_bot", persona="x" * 500)
    rc = _cmd_list(_ns(fleet_command="list"))
    out = capsys.readouterr().out
    # Each line capped near 60 chars of persona (plus ellipsis).
    assert "verbose_bot" in out
    line = next(ln for ln in out.splitlines() if "verbose_bot" in ln)
    assert len(line) < 200


def test_list_redacts_tokens(hermes_home, monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_FLEET_MANAGER_TOKEN", "x:y")
    from gateway.telegram_fleet.api import ManagedBotInfo
    from gateway.telegram_fleet.coordinator import FleetCoordinator

    api = MagicMock()
    api.get_me.return_value = {"id": 1, "username": "TestMgr", "can_manage_bots": True}
    coord = FleetCoordinator(manager_token="12345:ABC", api_client=api)
    coord.spawn_bot("worker_bot")
    coord.absorb_managed_bot(
        ManagedBotInfo(token="123:secretToken", bot_id=1, bot_username="worker_bot")
    )
    _cmd_list(_ns(fleet_command="list"))
    out = capsys.readouterr().out
    assert "secretToken" not in out
    assert "123:" not in out


# ── add ────────────────────────────────────────────────────────────────


def test_add_rejects_when_no_token(hermes_home, capsys):
    rc = _cmd_add(_ns(username="research_bot", persona="", name=None))
    out = capsys.readouterr().out
    assert "TELEGRAM_FLEET_MANAGER_TOKEN is not set" in out
    assert rc == 1


def test_add_appends_bot_suffix(hermes_home, monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_FLEET_MANAGER_TOKEN", "12345:ABC")
    from gateway.telegram_fleet.coordinator import FleetCoordinator
    from hermes_cli import fleet as fleet_module

    api = MagicMock()
    api.get_me.return_value = {"id": 1, "username": "TestMgr", "can_manage_bots": True}
    coord = FleetCoordinator(manager_token="12345:ABC", api_client=api)
    # _cmd_add calls get_coordinator(refresh=True), so monkeypatch the function
    # itself rather than the singleton.
    monkeypatch.setattr(
        "gateway.telegram_fleet.get_coordinator", lambda *a, **k: coord
    )

    rc = _cmd_add(_ns(username="research", persona="research lead", name=None))
    out = capsys.readouterr().out
    assert "research_bot" in out
    assert "t.me/newbot/TestMgr/research_bot" in out
    assert rc == 0


def test_add_prints_deep_link(hermes_home, monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_FLEET_MANAGER_TOKEN", "12345:ABC")
    from gateway.telegram_fleet.coordinator import FleetCoordinator

    api = MagicMock()
    api.get_me.return_value = {"id": 1, "username": "HermesMgr", "can_manage_bots": True}
    coord = FleetCoordinator(manager_token="12345:ABC", api_client=api)
    monkeypatch.setattr(
        "gateway.telegram_fleet.get_coordinator", lambda *a, **k: coord
    )

    rc = _cmd_add(_ns(username="alpha_bot", persona="lead", name="Alpha"))
    out = capsys.readouterr().out
    assert "t.me/newbot/HermesMgr/alpha_bot" in out
    assert "Tap this link" in out
    assert rc == 0


# ── setup (bot-API mocked) ─────────────────────────────────────────────


def test_setup_rejects_invalid_token_shape(hermes_home, monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_FLEET_MANAGER_TOKEN", "too-short")
    rc = _cmd_setup(_ns())
    out = capsys.readouterr().out
    assert "Token doesn't look right" in out
    assert rc == 1


def test_setup_rejects_when_manager_mode_off(hermes_home, monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_FLEET_MANAGER_TOKEN", "12345:ABC" + "x" * 30)
    from gateway.telegram_fleet import api as api_module

    fake_client = MagicMock()
    fake_client.get_me.return_value = {
        "id": 1,
        "username": "MyBot",
        "can_manage_bots": False,  # the failure mode we're testing
    }
    monkeypatch.setattr(api_module, "FleetApiClient", lambda *a, **k: fake_client)
    rc = _cmd_setup(_ns())
    out = capsys.readouterr().out
    assert "Bot Manager Mode" in out
    assert "BotFather" in out
    assert rc == 1


def test_setup_succeeds_and_persists_username(hermes_home, monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_FLEET_MANAGER_TOKEN", "12345:ABC" + "x" * 30)
    from gateway.telegram_fleet import api as api_module

    fake_client = MagicMock()
    fake_client.get_me.return_value = {
        "id": 1,
        "username": "HermesMgr",
        "can_manage_bots": True,
    }
    monkeypatch.setattr(api_module, "FleetApiClient", lambda *a, **k: fake_client)

    rc = _cmd_setup(_ns())
    out = capsys.readouterr().out
    assert "Manager bot identified" in out
    assert "@HermesMgr" in out
    assert "Bot Manager Mode is enabled" in out
    assert rc == 0

    # Roster persisted with manager username.
    from gateway.telegram_fleet.roster import load_roster

    roster = load_roster()
    assert roster.manager_bot_username == "HermesMgr"


def test_setup_writes_token_to_env_file(hermes_home, monkeypatch):
    monkeypatch.setenv("TELEGRAM_FLEET_MANAGER_TOKEN", "12345:ABC" + "x" * 30)
    from gateway.telegram_fleet import api as api_module

    fake_client = MagicMock()
    fake_client.get_me.return_value = {
        "id": 1,
        "username": "HermesMgr",
        "can_manage_bots": True,
    }
    monkeypatch.setattr(api_module, "FleetApiClient", lambda *a, **k: fake_client)

    _cmd_setup(_ns())

    env_file = hermes_home / ".env"
    assert env_file.exists()
    content = env_file.read_text()
    assert "TELEGRAM_FLEET_MANAGER_TOKEN=" in content
    # mode 0600 on POSIX
    if os.name == "posix":
        import stat

        mode = stat.S_IMODE(env_file.stat().st_mode)
        assert mode == 0o600


# ── parser registration ────────────────────────────────────────────────


def test_parser_registers_all_subcommands():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    add_fleet_parser(subparsers)
    args = parser.parse_args(["fleet", "status"])
    assert args.cmd == "fleet"
    assert args.fleet_command == "status"
    args = parser.parse_args(["fleet", "add", "research_bot", "--persona", "lead"])
    assert args.fleet_command == "add"
    assert args.username == "research_bot"
    assert args.persona == "lead"


def test_cmd_fleet_dispatches_to_status_by_default(hermes_home, capsys):
    rc = cmd_fleet(_ns(fleet_command=None))
    out = capsys.readouterr().out
    # No subcommand → falls through to status
    assert "Telegram Fleet — status" in out
    # Returns 1 because no token configured.
    assert rc == 1


# ── adopt ──────────────────────────────────────────────────────────────


def test_adopt_rejects_bad_token_shape(hermes_home, capsys):
    from hermes_cli.fleet import _cmd_adopt

    rc = _cmd_adopt(
        _ns(token="too-short", persona="x", model=None, toolset=None)
    )
    out = capsys.readouterr().out
    assert "Token doesn't look right" in out
    assert rc == 1


def test_adopt_writes_active_entry(hermes_home, monkeypatch, capsys):
    """Adopting a real-looking token validates via getMe and lands as active."""
    from hermes_cli.fleet import _cmd_adopt
    from gateway.telegram_fleet import api as api_module
    from gateway.telegram_fleet.coordinator import FleetCoordinator

    fake_client = MagicMock()
    fake_client.get_me.return_value = {
        "id": 7777,
        "username": "my_existing_bot",
        "can_manage_bots": False,  # adopted bots don't need manager mode
    }
    monkeypatch.setattr(api_module, "FleetApiClient", lambda *a, **k: fake_client)

    rc = _cmd_adopt(
        _ns(
            token="12345:ABC" + "x" * 30,
            persona="research lead",
            model=None,
            toolset=None,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Adopted @my_existing_bot" in out

    # Roster updated with active bot.
    coord = FleetCoordinator()
    child = coord.find("my_existing_bot")
    assert child is not None
    assert child.is_active()
    assert child.bot_id == 7777
    assert child.persona == "research lead"


def test_adopt_surfaces_telegram_rejection(hermes_home, monkeypatch, capsys):
    """Real-shaped but invalid token gets a clean error."""
    from hermes_cli.fleet import _cmd_adopt
    from gateway.telegram_fleet import api as api_module
    from gateway.telegram_fleet.api import BotApiError

    fake_client = MagicMock()
    fake_client.get_me.side_effect = BotApiError("getMe", 401, "Unauthorized")
    monkeypatch.setattr(api_module, "FleetApiClient", lambda *a, **k: fake_client)

    rc = _cmd_adopt(
        _ns(
            token="12345:ABC" + "x" * 30,
            persona="x",
            model=None,
            toolset=None,
        )
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "Telegram rejected" in out or "rejected" in out.lower()


def test_adopt_parses_toolset_csv(hermes_home, monkeypatch, capsys):
    """--toolset web,file → list."""
    from hermes_cli.fleet import _cmd_adopt
    from gateway.telegram_fleet import api as api_module
    from gateway.telegram_fleet.coordinator import FleetCoordinator

    fake_client = MagicMock()
    fake_client.get_me.return_value = {
        "id": 1234,
        "username": "tooled_bot",
        "can_manage_bots": False,
    }
    monkeypatch.setattr(api_module, "FleetApiClient", lambda *a, **k: fake_client)

    rc = _cmd_adopt(
        _ns(
            token="12345:ABC" + "x" * 30,
            persona="",
            model=None,
            toolset="web, file, terminal",
        )
    )
    assert rc == 0
    coord = FleetCoordinator()
    child = coord.find("tooled_bot")
    assert child.toolset == ["web", "file", "terminal"]


# ── coordinator: adopt_existing_bot directly ─────────────────────────


def test_coordinator_adopt_validates_token_shape(hermes_home):
    from gateway.telegram_fleet.coordinator import FleetCoordinator
    from gateway.telegram_fleet.guardrails import FleetGuardrailError

    api = MagicMock()
    coord = FleetCoordinator(manager_token="x:y", api_client=api)
    with pytest.raises(FleetGuardrailError, match="bot token"):
        coord.adopt_existing_bot(token="not-a-token")


def test_coordinator_adopt_replaces_existing_pending_entry(hermes_home, monkeypatch):
    """Adopting overrides a stale pending entry with the same username."""
    from gateway.telegram_fleet import api as api_module
    from gateway.telegram_fleet.coordinator import FleetCoordinator

    api = MagicMock()
    api.get_me.return_value = {"id": 1, "username": "Mgr", "can_manage_bots": True}
    coord = FleetCoordinator(manager_token="x:y", api_client=api)
    coord.spawn_bot("dup_bot", persona="from spawn")  # creates pending entry

    fake_child_client = MagicMock()
    fake_child_client.get_me.return_value = {
        "id": 9999,
        "username": "dup_bot",
        "can_manage_bots": False,
    }
    monkeypatch.setattr(
        api_module, "FleetApiClient", lambda *a, **k: fake_child_client
    )

    child = coord.adopt_existing_bot(
        token="12345:ABC" + "x" * 30, persona="from adopt"
    )
    assert child.is_active()
    assert child.bot_id == 9999
    # The stale pending entry was replaced (not duplicated).
    assert len([c for c in coord.list_children() if c.username == "dup_bot"]) == 1


# ── connect ────────────────────────────────────────────────────────────


def test_connect_rejects_when_no_token(hermes_home, capsys):
    from hermes_cli.fleet import _cmd_connect

    rc = _cmd_connect(_ns(wait=0))
    out = capsys.readouterr().out
    assert "TELEGRAM_FLEET_MANAGER_TOKEN is not set" in out
    assert rc == 1


def test_connect_warns_when_no_pending_entries(hermes_home, monkeypatch, capsys):
    """Empty roster → connect is a no-op with a useful nudge."""
    from hermes_cli.fleet import _cmd_connect

    monkeypatch.setenv("TELEGRAM_FLEET_MANAGER_TOKEN", "12345:ABC")
    from gateway.telegram_fleet.coordinator import FleetCoordinator

    api = MagicMock()
    api.get_me.return_value = {"id": 1, "username": "Mgr", "can_manage_bots": True}
    coord = FleetCoordinator(manager_token="12345:ABC", api_client=api)
    monkeypatch.setattr(
        "gateway.telegram_fleet.get_coordinator", lambda *a, **k: coord
    )

    rc = _cmd_connect(_ns(wait=0))
    out = capsys.readouterr().out
    assert "No pending entries" in out
    assert rc == 0


def test_connect_promotes_pending_to_active(hermes_home, monkeypatch, capsys):
    """When the user has tapped the link, connect should drain and promote."""
    from hermes_cli.fleet import _cmd_connect
    from gateway.telegram_fleet.api import ManagedBotInfo
    from gateway.telegram_fleet.coordinator import FleetCoordinator

    monkeypatch.setenv("TELEGRAM_FLEET_MANAGER_TOKEN", "12345:ABC")
    api = MagicMock()
    api.get_me.return_value = {"id": 1, "username": "Mgr", "can_manage_bots": True}
    coord = FleetCoordinator(manager_token="12345:ABC", api_client=api)
    coord.spawn_bot("research_bot", persona="research lead")

    # Mock drain to return one resolved managed bot.
    api.drain_managed_bot_events.return_value = (
        [ManagedBotInfo(token="42:tok", bot_id=42, bot_username="research_bot")],
        100,
    )
    monkeypatch.setattr(
        "gateway.telegram_fleet.get_coordinator", lambda *a, **k: coord
    )

    rc = _cmd_connect(_ns(wait=0))
    out = capsys.readouterr().out
    assert "research_bot" in out
    assert "promoted to active" in out
    assert rc == 0
    # Roster updated.
    child = coord.find("research_bot")
    assert child.is_active()
    assert child.bot_id == 42


def test_connect_handles_no_updates(hermes_home, monkeypatch, capsys):
    """User hasn't tapped yet — connect tells them, doesn't error."""
    from hermes_cli.fleet import _cmd_connect
    from gateway.telegram_fleet.coordinator import FleetCoordinator

    monkeypatch.setenv("TELEGRAM_FLEET_MANAGER_TOKEN", "12345:ABC")
    api = MagicMock()
    api.get_me.return_value = {"id": 1, "username": "Mgr", "can_manage_bots": True}
    coord = FleetCoordinator(manager_token="12345:ABC", api_client=api)
    coord.spawn_bot("research_bot")
    api.drain_managed_bot_events.return_value = ([], None)
    monkeypatch.setattr(
        "gateway.telegram_fleet.get_coordinator", lambda *a, **k: coord
    )

    rc = _cmd_connect(_ns(wait=0))
    out = capsys.readouterr().out
    assert "No managed_bot updates queued" in out
    assert "--wait" in out  # nudges them to long-poll if they want
    assert rc == 0


# ── coordinator: absorb_pending_updates plumbing ───────────────────────


def test_coordinator_absorb_pending_updates_promotes(hermes_home):
    """Direct test of the coordinator method the CLI wraps."""
    from gateway.telegram_fleet.api import ManagedBotInfo
    from gateway.telegram_fleet.coordinator import FleetCoordinator

    api = MagicMock()
    api.get_me.return_value = {"id": 1, "username": "Mgr", "can_manage_bots": True}
    api.drain_managed_bot_events.return_value = (
        [ManagedBotInfo(token="42:tok", bot_id=42, bot_username="worker_bot")],
        7,
    )
    coord = FleetCoordinator(manager_token="12345:ABC", api_client=api)
    coord.spawn_bot("worker_bot")
    absorbed = coord.absorb_pending_updates()
    assert len(absorbed) == 1
    assert absorbed[0].username == "worker_bot"
    assert absorbed[0].is_active()


def test_coordinator_absorb_returns_empty_when_no_updates(hermes_home):
    from gateway.telegram_fleet.coordinator import FleetCoordinator

    api = MagicMock()
    api.get_me.return_value = {"id": 1, "username": "Mgr", "can_manage_bots": True}
    api.drain_managed_bot_events.return_value = ([], None)
    coord = FleetCoordinator(manager_token="12345:ABC", api_client=api)
    coord.spawn_bot("worker_bot")
    absorbed = coord.absorb_pending_updates()
    assert absorbed == []
