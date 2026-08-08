"""Tests for the opt-in inline slash-command promotion in the WhatsApp
owner-reply flow.

Exercises ``coerce_inline_owner_command`` in isolation — no gateway/adapter —
by constructing a ``MessageEvent`` + ``SessionSource`` directly. Uses a real
built-in gateway command (``/status``) so the tests do not depend on any plugin
being loaded. JIDs are fictitious (upstream fixture style).
"""

import pathlib
import re

import pytest

from gateway.platforms.base import (
    coerce_inline_owner_command,
    MessageEvent,
    MessageType,
)
from gateway.session import SessionSource
from gateway.config import Platform


def _owner_dm_event(text, from_owner=True, chat_type="dm", platform=Platform.WHATSAPP):
    src = SessionSource(
        platform=platform,
        chat_id="6281234567890@s.whatsapp.net",
        chat_type=chat_type,
        user_id="6281234567890@s.whatsapp.net",
    )
    md = {"whatsapp_from_owner": True} if from_owner else {}
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=src, metadata=md)


@pytest.fixture(autouse=True)
def _flag_on(monkeypatch):
    monkeypatch.setenv("WHATSAPP_INLINE_COMMANDS", "true")
    yield


# A known command in the middle of an owner DM is promoted to the front.
def test_inline_command_promoted_in_owner_dm():
    event = _owner_dm_event("please run /status now")
    coerce_inline_owner_command(event)
    assert event.text == "/status please run now"
    assert event.get_command() == "status"


# The "[owner reply] " marker is stripped before scanning and absent after rewrite.
def test_inline_respects_owner_reply_prefix():
    event = _owner_dm_event("[owner reply] hey /status thanks")
    coerce_inline_owner_command(event)
    assert event.text == "/status hey thanks"
    assert "[owner reply]" not in event.text
    assert event.get_command() == "status"


# A command already at the start (after the marker) still sheds the marker.
def test_command_at_start_after_prefix_drops_prefix():
    event = _owner_dm_event("[owner reply] /status foo")
    coerce_inline_owner_command(event)
    assert event.text == "/status foo"
    assert event.get_command() == "status"

    # variant without the prefix -> unchanged text, still a command
    event2 = _owner_dm_event("/status foo")
    coerce_inline_owner_command(event2)
    assert event2.text == "/status foo"
    assert event2.get_command() == "status"


# Flag off (the default): byte-identical no-op.
def test_flag_off_is_identity(monkeypatch):
    monkeypatch.delenv("WHATSAPP_INLINE_COMMANDS", raising=False)
    event = _owner_dm_event("please run /status now")
    coerce_inline_owner_command(event)
    assert event.text == "please run /status now"
    assert event.get_command() is None


# A path-like token has an internal "/" and never matches.
def test_path_like_token_not_promoted():
    event = _owner_dm_event("see /home/user/notes for details")
    coerce_inline_owner_command(event)
    assert event.text == "see /home/user/notes for details"
    assert event.get_command() is None


# A command name embedded in a URL is not an isolated token.
def test_url_not_treated_as_command():
    event = _owner_dm_event("check http://example.com/status please")
    coerce_inline_owner_command(event)
    assert event.text == "check http://example.com/status please"
    assert event.get_command() is None


# Span-based matching: a URL occurrence must not shadow the real command.
def test_url_then_real_command_promotes_real():
    event = _owner_dm_event("see http://ex.com/status then /status now")
    coerce_inline_owner_command(event)
    assert event.text == "/status see http://ex.com/status then now"
    assert event.get_command() == "status"


# The first KNOWN command wins; an unknown slash word before it does not block.
def test_unknown_then_known_promotes_known():
    event = _owner_dm_event("/banana please /status now")
    coerce_inline_owner_command(event)
    assert event.text == "/status /banana please now"
    assert event.get_command() == "status"


# A path-like token before a known command does not block promotion.
def test_path_then_known_promotes_known():
    event = _owner_dm_event("/home/user please /status now")
    coerce_inline_owner_command(event)
    assert event.text == "/status /home/user please now"
    assert event.get_command() == "status"


# Unknown slash words alone are never promoted.
def test_unknown_slash_word_not_promoted():
    event = _owner_dm_event("talk about /banana today")
    coerce_inline_owner_command(event)
    assert event.text == "talk about /banana today"
    assert event.get_command() is None


# The "@bot" mention suffix is dropped and kept out of the args.
def test_mention_suffix_stripped():
    event = _owner_dm_event("hey /status@somebot ok")
    coerce_inline_owner_command(event)
    assert event.text == "/status hey ok"
    assert event.get_command() == "status"


# Command at the end: empty remainder, no trailing space.
def test_command_at_end_empty_remainder():
    event = _owner_dm_event("please /status")
    coerce_inline_owner_command(event)
    assert event.text == "/status please"
    assert event.get_command() == "status"

    # command alone -> empty remainder, no trailing space
    event2 = _owner_dm_event("/status")
    coerce_inline_owner_command(event2)
    assert event2.text == "/status"
    assert event2.get_command() == "status"


# Group chats are out of scope.
def test_group_not_affected():
    event = _owner_dm_event("run /status", chat_type="group")
    coerce_inline_owner_command(event)
    assert event.text == "run /status"
    assert event.get_command() is None


# Non-WhatsApp platforms are out of scope.
def test_non_whatsapp_not_affected():
    event = _owner_dm_event("run /status", platform=Platform.TELEGRAM)
    coerce_inline_owner_command(event)
    assert event.text == "run /status"
    assert event.get_command() is None


# Events without the owner marker are untouched.
def test_not_from_owner_not_affected():
    event = _owner_dm_event("run /status", from_owner=False)
    coerce_inline_owner_command(event)
    assert event.text == "run /status"
    assert event.get_command() is None


# Strict boolean gate: a truthy NON-bool marker must NOT qualify as owner.
def test_truthy_non_bool_owner_marker_rejected():
    for marker in ("false", "0", 1, "yes"):
        event = MessageEvent(
            text="run /status",
            message_type=MessageType.TEXT,
            source=SessionSource(
                platform=Platform.WHATSAPP,
                chat_id="6281234567890@s.whatsapp.net",
                chat_type="dm",
                user_id="6281234567890@s.whatsapp.net",
            ),
            metadata={"whatsapp_from_owner": marker},
        )
        coerce_inline_owner_command(event)
        assert event.text == "run /status", f"marker {marker!r} must not qualify"
        assert event.get_command() is None


# iOS en/em-dash auto-correction: args are normalized downstream.
def test_endash_normalized_in_args():
    # iOS auto-corrects "-" to en/em dash. We do NOT normalize before the scan
    # (offsets stay intact); get_command_args() re-applies the dash-fix so args
    # come out with a real ASCII dash.
    event = _owner_dm_event("do /status —v")
    coerce_inline_owner_command(event)
    assert event.get_command() == "status"
    assert "-v" in event.get_command_args()


# Static extraction — does not import the adapter module (no import side effects).
def test_owner_reply_prefix_constant_in_sync():
    from gateway.platforms.base import _OWNER_REPLY_PREFIX as base_prefix

    adapter_src = (
        pathlib.Path(__file__).resolve().parents[2]
        / "plugins" / "platforms" / "whatsapp" / "adapter.py"
    ).read_text(encoding="utf-8")
    m = re.search(r'_OWNER_REPLY_PREFIX\s*=\s*(["\'])(.*?)\1', adapter_src)
    assert m, "could not locate _OWNER_REPLY_PREFIX in adapter.py"
    assert m.group(2) == base_prefix


# YAML config propagation with env-var precedence.
def test_yaml_inline_commands_env_and_precedence(monkeypatch):
    from plugins.platforms.whatsapp.adapter import _apply_yaml_config

    # env clean -> YAML populates the env var (bool -> "true")
    monkeypatch.delenv("WHATSAPP_INLINE_COMMANDS", raising=False)
    _apply_yaml_config({}, {"inline_commands": True})
    assert __import__("os").environ["WHATSAPP_INLINE_COMMANDS"] == "true"

    # env already set -> env wins over YAML
    monkeypatch.setenv("WHATSAPP_INLINE_COMMANDS", "false")
    _apply_yaml_config({}, {"inline_commands": True})
    assert __import__("os").environ["WHATSAPP_INLINE_COMMANDS"] == "false"


# ── Text-batching sender boundary (owner-reply vs contact) ─────────────────
#
# The adapter's rapid-fire text batching is keyed by session (chat-scoped in a
# DM) and the merged event keeps the FIRST chunk's metadata/source. Without a
# sender boundary, an owner chunk and a contact chunk landing in the same
# debounce window would merge, so the combined text would inherit
# ``whatsapp_from_owner`` from whichever arrived first — letting a contact's
# inline command run under the owner marker, or losing the owner's own marker.
# These tests pin the boundary: batches never mix owner-reply and contact text.

import asyncio

from unittest.mock import AsyncMock

from gateway.config import PlatformConfig


def _batch_adapter():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.01
    adapter._text_batch_split_delay_seconds = 0.01
    adapter.handle_message = AsyncMock()
    return adapter


def _dm_text_event(text, from_owner):
    src = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="6281234567890@s.whatsapp.net",
        chat_type="dm",
        user_id=(
            "15551230000@s.whatsapp.net"
            if from_owner
            else "6281234567890@s.whatsapp.net"
        ),
    )
    md = {"whatsapp_from_owner": True} if from_owner else {}
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=src, metadata=md)


# Owner-attribution boundary predicate: same marker batches, mixed never does.
def test_can_batch_text_events_owner_boundary():
    adapter = _batch_adapter()
    owner_a = _dm_text_event("a", from_owner=True)
    owner_b = _dm_text_event("b", from_owner=True)
    contact_a = _dm_text_event("c", from_owner=False)
    contact_b = _dm_text_event("d", from_owner=False)
    assert adapter._can_batch_text_events(owner_a, owner_b) is True
    assert adapter._can_batch_text_events(contact_a, contact_b) is True
    assert adapter._can_batch_text_events(owner_a, contact_a) is False
    assert adapter._can_batch_text_events(contact_a, owner_a) is False


# Contact chunk first, owner chunk second: two separate dispatches; the
# contact's inline command never gains the owner marker.
@pytest.mark.asyncio
async def test_batch_contact_then_owner_not_merged():
    adapter = _batch_adapter()
    adapter._enqueue_text_event(_dm_text_event("hey /status now", from_owner=False))
    adapter._enqueue_text_event(_dm_text_event("[owner reply] ok", from_owner=True))
    await asyncio.sleep(0.05)

    events = [c.args[0] for c in adapter.handle_message.await_args_list]
    assert len(events) == 2, "owner and contact chunks must not merge"
    contact_event = events[0]
    assert (contact_event.metadata or {}).get("whatsapp_from_owner") is not True
    assert "/status" in contact_event.text
    coerce_inline_owner_command(contact_event)
    assert contact_event.get_command() is None


# Owner chunk first, contact chunk second: the contact's text must not ride
# into the owner-attributed event (no privilege leak into promoted args).
@pytest.mark.asyncio
async def test_batch_owner_then_contact_no_privilege_leak():
    adapter = _batch_adapter()
    adapter._enqueue_text_event(
        _dm_text_event("[owner reply] look at this", from_owner=True)
    )
    adapter._enqueue_text_event(_dm_text_event("please run /status", from_owner=False))
    await asyncio.sleep(0.05)

    events = [c.args[0] for c in adapter.handle_message.await_args_list]
    assert len(events) == 2, "owner and contact chunks must not merge"
    owner_event, contact_event = events
    assert (owner_event.metadata or {}).get("whatsapp_from_owner") is True
    assert "/status" not in owner_event.text
    coerce_inline_owner_command(owner_event)
    assert owner_event.get_command() is None
    coerce_inline_owner_command(contact_event)
    assert contact_event.get_command() is None


# Owner-only multi-chunk burst still merges and the inline command works on
# the combined text.
@pytest.mark.asyncio
async def test_batch_owner_multichunk_still_merges_and_promotes():
    adapter = _batch_adapter()
    adapter._enqueue_text_event(_dm_text_event("[owner reply] hang on", from_owner=True))
    adapter._enqueue_text_event(
        _dm_text_event("[owner reply] /status please", from_owner=True)
    )
    await asyncio.sleep(0.05)

    assert adapter.handle_message.await_count == 1, "same-owner chunks must merge"
    merged = adapter.handle_message.await_args_list[0].args[0]
    assert (merged.metadata or {}).get("whatsapp_from_owner") is True
    coerce_inline_owner_command(merged)
    assert merged.get_command() == "status"
