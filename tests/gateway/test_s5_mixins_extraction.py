"""Regression tests for the wave-1 shard-s5 mixin extraction from base.py.

Covers the methods moved out of ``BasePlatformAdapter`` into the new
``gateway.platforms.session_lifecycle_mixin`` and
``gateway.platforms.source_builder_mixin`` modules:

- ``has_pending_interrupt`` / ``get_pending_message`` (cluster c7,
  SessionLifecycleMixin)
- ``build_source`` (cluster c8, SourceBuilderMixin)

The methods are still reachable through the adapter class via MRO; these
tests pin their behavior so the extraction cannot silently change it.
"""

import asyncio
from types import SimpleNamespace

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.platforms.session_lifecycle_mixin import SessionLifecycleMixin
from gateway.platforms.source_builder_mixin import SourceBuilderMixin
from gateway.session import SessionSource


class DummyAdapter(BasePlatformAdapter):
    """Minimal concrete adapter: implements only the abstract methods."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="1")

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _event(message_id: str = "1") -> MessageEvent:
    return MessageEvent(
        text="hello",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm"),
        message_id=message_id,
    )


def test_mixins_are_in_adapter_mro():
    assert SessionLifecycleMixin in BasePlatformAdapter.__mro__
    assert SourceBuilderMixin in BasePlatformAdapter.__mro__


def test_has_pending_interrupt_via_mixin():
    adapter = DummyAdapter()
    key = "session-1"
    assert adapter.has_pending_interrupt(key) is False

    interrupt = asyncio.Event()
    adapter._active_sessions[key] = interrupt
    assert adapter.has_pending_interrupt(key) is False  # not set yet

    interrupt.set()
    assert adapter.has_pending_interrupt(key) is True

    # unrelated sessions are never reported as interrupting
    assert adapter.has_pending_interrupt("session-other") is False
    assert adapter.has_pending_interrupt("") is False


def test_get_pending_message_via_mixin():
    adapter = DummyAdapter()
    key = "session-2"
    assert adapter.get_pending_message(key) is None

    event = _event()
    adapter._pending_messages[key] = event
    # pop semantics: returns the message AND clears it
    assert adapter.get_pending_message(key) is event
    assert adapter.get_pending_message(key) is None


def test_build_source_defaults_and_stringification():
    adapter = DummyAdapter()
    source = adapter.build_source(
        chat_id=123,
        chat_name="Test Chat",
        chat_type="group",
        user_id=42,
        user_name="alice",
        thread_id=7,
        guild_id="g1",
        parent_chat_id="p1",
        message_id="m1",
    )
    assert isinstance(source, SessionSource)
    assert source.platform == Platform.TELEGRAM
    assert source.chat_id == "123"
    assert source.chat_name == "Test Chat"
    assert source.chat_type == "group"
    assert source.user_id == "42"
    assert source.user_name == "alice"
    assert source.thread_id == "7"
    # SessionSource.__post_init__ mirrors the deprecated guild_id alias onto
    # scope_id (pre-existing dataclass behavior, unchanged by the extraction).
    assert source.scope_id == "g1"
    assert source.guild_id == "g1"
    assert source.parent_chat_id == "p1"
    assert source.message_id == "m1"
    assert source.profile is None
    assert source.role_authorized is False


def test_build_source_scope_id_is_canonical():
    adapter = DummyAdapter()
    source = adapter.build_source(chat_id="1", scope_id="s1", guild_id="g1")
    # scope_id wins on conflict (SessionSource.__post_init__ semantics).
    assert source.scope_id == "s1"
    assert source.guild_id == "s1"


def test_build_source_normalizes_blank_topic():
    adapter = DummyAdapter()
    assert adapter.build_source(chat_id="1", chat_topic="   ").chat_topic is None
    assert adapter.build_source(chat_id="1", chat_topic="").chat_topic is None
    assert adapter.build_source(chat_id="1", chat_topic="General").chat_topic == "General"


def test_build_source_stamps_transport_ref():
    adapter = DummyAdapter()
    source = adapter.build_source(chat_id="1")
    ref = getattr(source, "_transport_adapter_ref", None)
    assert ref is not None
    assert ref() is adapter


def test_build_source_resolves_profile_from_runner():
    adapter = DummyAdapter()
    runner = SimpleNamespace(_profile_name_for_source=lambda source: "alice")
    adapter.gateway_runner = runner
    source = adapter.build_source(chat_id="1", guild_id="g1", thread_id="t1")
    assert source.profile == "alice"


def test_build_source_runner_failure_defaults_profile():
    adapter = DummyAdapter()

    class Boom:
        def _profile_name_for_source(self, source):
            raise RuntimeError("boom")

    adapter.gateway_runner = Boom()
    source = adapter.build_source(chat_id="1")
    assert source.profile is None
