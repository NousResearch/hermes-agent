"""Tests for Telegram ``channel_prompts`` resolution and composite topic keys.

A forum topic can override its group prompt with the composite key
``<chat_id>:<thread_id>``, while the legacy bare ``<thread_id>`` key keeps
working. Resolution precedence is therefore:

    composite topic key  >  legacy topic key  >  group fallback

The resolver is unit-tested directly, and the ``_build_message_event`` call
site is covered so the wiring (not just the lookup) stays honest.
"""

from types import SimpleNamespace

import pytest

import plugins.platforms.telegram.telegram_context as telegram_context
from gateway.platforms.base import resolve_channel_prompt
from gateway.platforms.event import MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter(channel_prompts=None):
    adapter = object.__new__(TelegramAdapter)
    adapter.config = SimpleNamespace(extra={"channel_prompts": channel_prompts or {}})
    return adapter


# ── Common resolver (gateway.platforms.base) ────────────────────────────


@pytest.mark.parametrize(
    "prompts, channel_id, parent_id, fallback_keys, expected",
    [
        ({"k1": "Exact"}, "k1", None, (), "Exact"),
        ({"k1": "Exact"}, "absent", None, (), None),
        ({"k2": "Parent"}, "absent", "k2", (), "Parent"),
        ({"k1": "Exact", "k2": "Parent"}, "k1", "k2", (), "Exact"),
        ({"k3": "Fallback"}, "absent", "absent2", ("k3",), "Fallback"),
        ({"k2": "Seq"}, ["absent", "k2"], None, (), "Seq"),
        ({"blank": "   ", "k2": "V"}, "blank", "k2", (), "V"),
        ({}, "k1", None, (), None),
    ],
)
def test_common_resolver(prompts, channel_id, parent_id, fallback_keys, expected):
    extra = {"channel_prompts": prompts}
    assert resolve_channel_prompt(extra, channel_id, parent_id, *fallback_keys) == expected


@pytest.mark.parametrize("extra", [{}, {"channel_prompts": None}, {"channel_prompts": "invalid"}])
def test_common_resolver_tolerates_invalid_config(extra):
    assert resolve_channel_prompt(extra, "k1") is None


# ── Telegram resolver precedence ────────────────────────────────────────


def test_composite_topic_key_precedes_group_and_legacy():
    adapter = _make_adapter({
        "-1001:42": "Composite topic",
        "42": "Legacy topic",
        "-1001": "Group prompt",
    })
    assert adapter._resolve_channel_prompt("-1001", "42") == "Composite topic"


def test_legacy_topic_key_precedes_group_fallback():
    adapter = _make_adapter({
        "42": "Legacy topic",
        "-1001": "Group prompt",
    })
    assert adapter._resolve_channel_prompt("-1001", "42") == "Legacy topic"


def test_group_fallback_when_topic_prompt_absent():
    adapter = _make_adapter({
        "-1001": "Group prompt",
    })
    assert adapter._resolve_channel_prompt("-1001", "42") == "Group prompt"


def test_absence_of_rule_returns_none():
    adapter = _make_adapter({
        "-1009:99": "Other topic",
    })
    assert adapter._resolve_channel_prompt("-1001", "42") is None
    assert adapter._resolve_channel_prompt("-1001") is None
    assert adapter._resolve_channel_prompt(None) is None


def test_collision_same_thread_id_different_groups():
    adapter = _make_adapter({
        "-1001:42": "Group A Topic 42",
        "-1002:42": "Group B Topic 42",
        "-1001": "Group A General",
        "-1002": "Group B General",
        "42": "Legacy 42",
    })
    assert adapter._resolve_channel_prompt("-1001", "42") == "Group A Topic 42"
    assert adapter._resolve_channel_prompt("-1002", "42") == "Group B Topic 42"


def test_collision_composite_and_group_fallback():
    adapter = _make_adapter({
        "-1001:42": "Group A Topic 42",
        "-1002": "Group B General",
    })
    assert adapter._resolve_channel_prompt("-1001", "42") == "Group A Topic 42"
    assert adapter._resolve_channel_prompt("-1002", "42") == "Group B General"


def test_composite_key_passed_as_single_argument():
    """A pre-joined ``chat_id:thread_id`` resolves through the same precedence."""
    adapter = _make_adapter({
        "-1001:42": "Composite topic",
        "-1001": "Group prompt",
    })
    assert adapter._resolve_channel_prompt("-1001:42") == "Composite topic"


def test_channel_and_parent_keyword_arguments():
    """Legacy call shape: the topic id arrives as channel_id, the group as parent_id."""
    adapter = _make_adapter({
        "-1001:42": "Composite topic",
        "-1001": "Group prompt",
    })
    assert adapter._resolve_channel_prompt(channel_id="42", parent_id="-1001") == "Composite topic"
    assert adapter._resolve_channel_prompt(channel_id="-1001") == "Group prompt"


def test_missing_config_is_safe():
    bare = object.__new__(TelegramAdapter)
    bare.config = None
    assert bare._resolve_channel_prompt("-1001", "42") is None


# ── Wiring: _build_message_event ───────────────────────────────────────


def _patch_event_collaborators(monkeypatch, adapter):
    """Stub everything ``_build_message_event`` needs except prompt resolution."""
    monkeypatch.setattr(adapter, "_chat_type_str", lambda chat: "supergroup")
    monkeypatch.setattr(adapter, "_effective_message_thread_id", lambda message: "42")
    monkeypatch.setattr(
        adapter, "_resolve_topic_binding", lambda message, chat_type, thread_id_str: (None, None)
    )
    monkeypatch.setattr(adapter, "_reply_context", lambda message: (None, None))
    monkeypatch.setattr(adapter, "build_source", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(telegram_context, "group_identity_prompt", lambda adapter, message, prompt: prompt)


def _topic_message():
    return SimpleNamespace(
        text="hello",
        caption=None,
        chat=SimpleNamespace(id=-1001, type="supergroup", is_forum=True, title="Forum"),
        from_user=SimpleNamespace(id=456, full_name="Bob", is_bot=False),
        message_thread_id=42,
        is_topic_message=True,
        reply_to_message=None,
        message_id=101,
        date=None,
    )


def test_build_message_event_uses_composite_topic_prompt(monkeypatch):
    adapter = _make_adapter({"-1001:42": "Topic 42 prompt", "-1001": "Group prompt"})
    _patch_event_collaborators(monkeypatch, adapter)
    event = adapter._build_message_event(_topic_message(), msg_type=MessageType.TEXT)
    assert event.channel_prompt == "Topic 42 prompt"


def test_build_message_event_falls_back_to_group_prompt(monkeypatch):
    adapter = _make_adapter({"-1001": "Group prompt"})
    _patch_event_collaborators(monkeypatch, adapter)
    event = adapter._build_message_event(_topic_message(), msg_type=MessageType.TEXT)
    assert event.channel_prompt == "Group prompt"
