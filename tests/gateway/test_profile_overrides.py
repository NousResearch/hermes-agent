"""Tests for live per-chat profile switching (/profile set, picker, clear).

The store is ``gateway.profile_overrides`` (a JSON switch table shared across
profiles); ``GatewayRunner._profile_name_for_source`` consults it before
``gateway.profile_routes``; the ``/profile`` handler in
``GatewaySlashCommandsMixin`` writes it, either from an explicit name or from an
interactive picker selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import profile_overrides
from gateway.config import Platform
from gateway.slash_commands import GatewaySlashCommandsMixin


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Point the override store at a temp file (never the real HERMES_HOME)."""
    path = tmp_path / "profile_overrides.json"
    monkeypatch.setattr(profile_overrides, "_overrides_path", lambda: path)
    return path


def _source(platform=Platform.TELEGRAM, chat_id="42", thread_id=None, profile="default"):
    return SimpleNamespace(
        platform=platform, chat_id=chat_id, thread_id=thread_id, profile=profile,
        guild_id=None, parent_chat_id=None, user_id="7",
        _transport_adapter_ref=None,
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_set_resolve_clear_round_trip(store):
    assert profile_overrides.resolve_override("telegram", "42") is None
    profile_overrides.set_override("telegram", "42", "roster")
    assert profile_overrides.resolve_override("telegram", "42") == "roster"
    assert json.loads(store.read_text()) == {"telegram:42": "roster"}
    assert profile_overrides.clear_override("telegram", "42") is True
    assert profile_overrides.resolve_override("telegram", "42") is None
    assert profile_overrides.clear_override("telegram", "42") is False


def test_thread_pin_beats_chat_pin(store):
    profile_overrides.set_override("telegram", "42", "roster")
    profile_overrides.set_override("telegram", "42", "pessoal", thread_id="99")
    assert profile_overrides.resolve_override("telegram", "42", "99") == "pessoal"
    assert profile_overrides.resolve_override("telegram", "42", "100") == "roster"


def test_pins_are_scoped_per_platform_and_chat(store):
    profile_overrides.set_override("telegram", "42", "roster")
    assert profile_overrides.resolve_override("discord", "42") is None
    assert profile_overrides.resolve_override("telegram", "43") is None


def test_corrupt_file_reads_as_empty_instead_of_raising(store):
    store.write_text("{not json")
    assert profile_overrides.list_overrides() == {}


# ---------------------------------------------------------------------------
# Resolution: _profile_name_for_source consults the pin before routes
# ---------------------------------------------------------------------------


def _runner(config):
    from gateway.run import GatewayRunner
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = config
    return runner


def _config(multiplex=True, routes=None):
    return SimpleNamespace(
        multiplex_profiles=multiplex,
        profile_routes=routes or [],
        _runtime_profile_homes=[("default", Path("/h/default")), ("roster", Path("/h/roster"))],
    )


def test_pin_wins_over_configured_routes(store, monkeypatch):
    from gateway import profile_routing

    monkeypatch.setattr(
        profile_routing, "match_profile_route",
        lambda *a, **k: SimpleNamespace(name="route-1", profile="default"),
    )
    profile_overrides.set_override("telegram", "42", "roster")
    runner = _runner(_config(routes=[{"profile": "default", "chat_ids": ["42"]}]))
    assert runner._profile_name_for_source(_source(), adapter_profile="default") == "roster"


def test_pin_works_without_any_configured_routes(store):
    profile_overrides.set_override("telegram", "42", "roster")
    runner = _runner(_config(routes=[]))
    assert runner._profile_name_for_source(_source(), adapter_profile="default") == "roster"


def test_pin_naming_an_unserved_profile_is_ignored(store, caplog):
    profile_overrides.set_override("telegram", "42", "ghost")
    runner = _runner(_config(routes=[]))
    with caplog.at_level("WARNING"):
        assert runner._profile_name_for_source(_source(), adapter_profile="default") is None
    assert "profile is not served" in caplog.text


def test_multiplexing_off_ignores_the_pin_entirely(store):
    profile_overrides.set_override("telegram", "42", "roster")
    runner = _runner(_config(multiplex=False))
    assert runner._profile_name_for_source(_source(), adapter_profile="default") is None


# ---------------------------------------------------------------------------
# Command surface
# ---------------------------------------------------------------------------


class _Harness(GatewaySlashCommandsMixin):
    """Only the command surface under test; the runner internals are stubbed."""

    def __init__(self, multiplex=True, served=("default", "roster"), picker=False):
        self.config = SimpleNamespace(multiplex_profiles=multiplex)
        self._served = list(served)
        self._picker_available = picker
        self.picker_choices = None
        self.picker_callback = None

    def _served_profiles(self):
        return list(self._served)

    def _session_key_for_source(self, source):
        return f"agent:main:{source.platform.value}:{source.chat_id}"

    async def _try_send_choice_picker(self, event, session_key, title, choices, on_choice_selected):
        if not self._picker_available:
            return False
        self.picker_choices = choices
        self.picker_callback = on_choice_selected
        return True


def _event(args="", source=None):
    return SimpleNamespace(
        get_command_args=lambda: args,
        source=source or _source(),
    )


@pytest.mark.asyncio
async def test_set_pins_the_chat(store):
    host = _Harness()
    reply = await host._handle_profile_command(_event("set roster"))
    assert "roster" in reply
    assert profile_overrides.resolve_override("telegram", "42") == "roster"


@pytest.mark.asyncio
async def test_set_with_an_unknown_name_pins_nothing(store):
    host = _Harness()
    reply = await host._handle_profile_command(_event("set ghost"))
    assert "ghost" in reply
    assert profile_overrides.resolve_override("telegram", "42") is None


@pytest.mark.asyncio
async def test_set_without_a_name_opens_the_picker_and_the_choice_pins(store):
    host = _Harness(picker=True, served=("default", "roster"))
    reply = await host._handle_profile_command(_event("set"))
    assert reply is None  # picker owns the response
    assert [c["value"] for c in host.picker_choices] == ["default", "roster"]
    assert host.picker_choices[0]["is_current"] is True

    answer = await host.picker_callback("42", "roster")
    assert "roster" in answer
    assert profile_overrides.resolve_override("telegram", "42") == "roster"


@pytest.mark.asyncio
async def test_picker_without_a_picker_capable_platform_lists_the_names(store):
    host = _Harness(picker=False)
    reply = await host._handle_profile_command(_event("set"))
    assert "default" in reply and "roster" in reply
    assert profile_overrides.resolve_override("telegram", "42") is None


@pytest.mark.asyncio
async def test_clear_drops_the_pin(store):
    profile_overrides.set_override("telegram", "42", "roster")
    host = _Harness()
    reply = await host._handle_profile_command(_event("clear"))
    assert reply
    assert profile_overrides.resolve_override("telegram", "42") is None


@pytest.mark.asyncio
async def test_status_reports_the_active_pin(store, monkeypatch):
    from gateway import slash_commands

    monkeypatch.setattr(
        slash_commands, "_execute",
        lambda name, options=None: SimpleNamespace(data={"profile": "roster", "home": "/h/roster"}),
    )
    profile_overrides.set_override("telegram", "42", "roster")
    host = _Harness()
    reply = await host._handle_profile_command(_event(""))
    assert "roster" in reply
    assert len(reply.splitlines()) == 3  # header + home + pin line


@pytest.mark.asyncio
async def test_switching_requires_multiplexing(store):
    host = _Harness(multiplex=False)
    reply = await host._handle_profile_command(_event("set roster"))
    assert "multiplex_profiles" in reply
    assert profile_overrides.resolve_override("telegram", "42") is None
