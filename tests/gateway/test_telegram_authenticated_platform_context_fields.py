"""W2: the Telegram adapter populates the four optional AuthenticatedPlatformContext fields
(message_id, profile_name, session_incarnation, profile_home) from the event and from the
already-resolved routing identity — never invented, never from model arguments.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore
from gateway.platform_context import (
    AuthenticatedPlatformContext,
    get_authenticated_platform_context,
)
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.profile_routing import parse_profile_routes
from plugins.platforms.telegram.adapter import TelegramAdapter


def _runner(home):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=False)
    runner.config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True, extra={})}
    runner.config.profile_routes = parse_profile_routes([])
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {}
    runner._primary_profile_name = "default"
    return runner


def _adapter(runner):
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.gateway_runner = runner
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra={})
    adapter._bot = SimpleNamespace(id=999, username="test_bot")
    adapter._pending_messages = {}
    adapter._active_sessions = {}
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    return adapter


@pytest.mark.asyncio
async def test_handle_message_populates_optional_identity_fields(tmp_path, monkeypatch):
    home = tmp_path / "hh"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    runner = _runner(home)
    adapter = _adapter(runner)

    source = adapter.build_source(chat_id="123", chat_type="dm", user_id="456")
    event = MessageEvent(text="hi", source=source, message_id="777")

    captured = {}

    async def fake_base_handle_message(self, ev):
        captured["context"] = get_authenticated_platform_context()

    with patch.object(BasePlatformAdapter, "handle_message", fake_base_handle_message):
        await adapter.handle_message(event)

    # The scope must be closed again once handle_message returns.
    assert get_authenticated_platform_context() is None

    context = captured["context"]
    assert context is not None
    assert context.platform == "telegram"
    assert context.message_id == "777"
    assert context.profile_name == "default"
    assert context.profile_home == str(home)
    assert context.session_incarnation == adapter._event_session_key(event)
    assert context.session_incarnation  # non-empty: this is the same value used elsewhere as session_incarnation


@pytest.mark.asyncio
async def test_thread_id_none_stays_none_and_optional_fields_survive_it(tmp_path, monkeypatch):
    """thread_id=None is legitimate (SPEC 2.2) and must not disturb the other optional fields."""
    home = tmp_path / "hh"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    runner = _runner(home)
    adapter = _adapter(runner)

    source = adapter.build_source(chat_id="123", chat_type="dm", user_id="456", thread_id=None)
    event = MessageEvent(text="hi", source=source, message_id="778")

    captured = {}

    async def fake_base_handle_message(self, ev):
        captured["context"] = get_authenticated_platform_context()

    with patch.object(BasePlatformAdapter, "handle_message", fake_base_handle_message):
        await adapter.handle_message(event)

    context = captured["context"]
    assert context.thread_id is None
    assert context.message_id == "778"
    assert context.profile_name == "default"
    assert context.profile_home == str(home)


def test_nine_field_context_equals_five_field_reconstruction():
    """The 4 optional fields are excluded from __eq__/__hash__ (compare=False): only the 5
    identity fields (platform, account_id, user_id, chat_id, thread_id) determine equality.
    This is what lets resolve_authenticated_platform_context/set_authenticated_platform_context
    accept an explicit context built with fewer fields than the ambient one.
    """
    ambient = AuthenticatedPlatformContext(
        platform="telegram",
        account_id="acc",
        user_id="456",
        chat_id="123",
        thread_id="9",
        message_id="777",
        profile_name="default",
        session_incarnation="lane-1",
        profile_home="/srv/hermes-home",
    )
    five_field = AuthenticatedPlatformContext(
        platform="telegram",
        account_id="acc",
        user_id="456",
        chat_id="123",
        thread_id="9",
    )
    assert ambient == five_field
    assert hash(ambient) == hash(five_field)


def test_differing_only_in_optional_field_is_still_equal_to_ambient():
    """A context whose ONLY difference from the ambient one is an optional field must still
    compare equal -- otherwise a caller could smuggle different message_id/profile_name/
    session_incarnation/profile_home data into the ambient context past the forgery guard
    in set_authenticated_platform_context / resolve_authenticated_platform_context.
    """
    ambient = AuthenticatedPlatformContext(
        platform="telegram",
        account_id="acc",
        user_id="456",
        chat_id="123",
        thread_id="9",
        message_id="777",
        profile_name="default",
        session_incarnation="lane-1",
        profile_home="/srv/hermes-home",
    )
    different_optional_fields = AuthenticatedPlatformContext(
        platform="telegram",
        account_id="acc",
        user_id="456",
        chat_id="123",
        thread_id="9",
        message_id="999-not-the-real-message",
        profile_name="someone-elses-profile",
        session_incarnation="forged-lane",
        profile_home="/tmp/forged-home",
    )
    assert ambient == different_optional_fields
