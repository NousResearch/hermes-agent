"""Tests for Feishu session resolver and session cards."""

import os
import pytest

from agent.managed_agents.feishu_session_resolver import (
    ResolutionResult,
    AmbiguityInfo,
    resolve_feishu_session,
    check_ambiguity,
    record_card_session_binding,
    _is_group_chat,
)
from agent.managed_agents.feishu_session_cards import (
    build_ambiguity_card,
    build_card_acknowledgement,
    build_rejection_card,
    _session_label,
    _truncate,
)
from agent.managed_agents.feishu_entry_adapter import FeishuEntryAdapter
from agent.managed_agents.entry_event import EntryEvent
from agent.managed_agents.workspace import DEFAULT_WORKSPACE_ID
from agent.managed_agents.session import DEFAULT_SESSION_ID
from agent.managed_agents.session_binding import put_binding, get_binding_value, _binding_values, _bindings_lock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_bindings():
    """Clear session bindings between tests to avoid test pollution."""
    with _bindings_lock:
        _binding_values.clear()
    yield
    with _bindings_lock:
        _binding_values.clear()


def _p2p_event(**overrides) -> EntryEvent:
    """Create a p2p EntryEvent (direct message)."""
    base = {
        "event_id": "evt-001",
        "entrypoint": "feishu",
        "external_channel_id": "ou_dm999_unique",
        "external_thread_id": None,
        "external_user_id": "ou_user456",
        "workspace_id": "ws-feishu-ou_dm999_unique",
        "session_id": "ses-feishu-ou_dm999_unique",
        "message": "hello",
        "origin_entrypoint": "feishu",
    }
    base.update(overrides)
    return EntryEvent.from_dict(base)


def _group_event(**overrides) -> EntryEvent:
    """Create a group chat EntryEvent."""
    base = {
        "event_id": "evt-002",
        "entrypoint": "feishu",
        "external_channel_id": "oc_group888_unique",
        "external_thread_id": None,
        "external_user_id": "ou_user456",
        "workspace_id": "ws-feishu-oc_group888_unique",
        "session_id": "ses-feishu-oc_group888_unique",
        "message": "hello group",
        "origin_entrypoint": "feishu",
    }
    base.update(overrides)
    return EntryEvent.from_dict(base)


def _thread_event(**overrides) -> EntryEvent:
    """Create a group + thread EntryEvent."""
    base = {
        "event_id": "evt-003",
        "entrypoint": "feishu",
        "external_channel_id": "oc_group888_unique",
        "external_thread_id": "om_thread777_unique",
        "external_user_id": "ou_user456",
        "workspace_id": "ws-feishu-oc_group888_unique",
        "session_id": "ses-feishu-thread-om_thread777_unique",
        "message": "hello thread",
        "origin_entrypoint": "feishu",
    }
    base.update(overrides)
    return EntryEvent.from_dict(base)


# ===========================================================================
# Resolution Chain Tests (5B1)
# ===========================================================================

class TestResolutionChain:
    """Test the priority resolution chain for Feishu sessions."""

    def test_p2p_no_binding_derives_default(self):
        """Scenario 1: Private chat, no prior binding → derive default."""
        event = _p2p_event()
        result = resolve_feishu_session(event, active_sessions=None)
        assert result.workspace_id is not None
        assert result.session_id is not None
        assert result.source == "default"
        assert result.ambiguous is False

    def test_p2p_card_binding_exists(self):
        """Scenario 2: Private chat with card binding → use card binding."""
        put_binding("feishu", "ou_dm999_unique", None, "ws-p2p-card", "ses-p2p-card", source="card")
        event = _p2p_event()
        result = resolve_feishu_session(event, active_sessions=None)
        assert result.session_id == "ses-p2p-card"
        assert result.source == "card"
        assert result.ambiguous is False

    def test_group_no_thread_no_binding_single_session(self):
        """Scenario 3: Group chat, no thread, no binding, 1 session → not ambiguous."""
        event = _group_event()
        result = resolve_feishu_session(event, active_sessions=("ses-1",))
        assert result.ambiguous is False
        assert result.needs_card is False
        assert result.source in ("default", "card")

    def test_group_thread_binding(self):
        """Scenario 4: Group chat with thread → thread-derived session."""
        event = _thread_event()
        result = resolve_feishu_session(event, active_sessions=None)
        assert "thread" in result.session_id or result.source in ("card", "thread")
        assert result.ambiguous is False

    def test_group_card_binding_exists(self):
        """Scenario 5: Group chat, card binding exists → use card binding."""
        put_binding("feishu", "oc_group888_unique", None, "ws-card", "ses-card-group", source="card")
        event = _group_event()
        result = resolve_feishu_session(event, active_sessions=None)
        assert result.session_id == "ses-card-group"
        assert result.source == "card"

    def test_group_thread_card_overrides_thread_derived(self):
        """Card binding for a specific thread overrides thread-derived session."""
        put_binding("feishu", "oc_group888_unique", "om_thread777_unique", "ws-card", "ses-card-thread", source="card")
        event = _thread_event()
        result = resolve_feishu_session(event, active_sessions=None)
        assert result.session_id == "ses-card-thread"
        assert result.source == "card"

    def test_group_ambiguity_flag_disabled_by_default(self):
        """When FEISHU_AMBIGUITY_CARD_ENABLED is False, no ambiguity even with multiple sessions."""
        # Default is disabled
        event = _group_event()
        result = resolve_feishu_session(event, active_sessions=("ses-1", "ses-2", "ses-3"))
        assert result.ambiguous is False
        assert result.needs_card is False

    def test_group_existing_binding_no_ambiguity(self):
        """Group with existing binding (not card) → no ambiguity even with multiple sessions."""
        put_binding("feishu", "oc_group888_unique", None, "ws-default", "ses-default-group", source="default")
        event = _group_event()
        result = resolve_feishu_session(event, active_sessions=("ses-1", "ses-2"))
        assert result.session_id == "ses-default-group"
        assert result.ambiguous is False


# ===========================================================================
# Ambiguity Check Tests
# ===========================================================================

class TestCheckAmbiguity:
    """Test check_ambiguity function."""

    def test_p2p_never_ambiguous(self):
        """P2P messages are never ambiguous."""
        event = _p2p_event()
        result = check_ambiguity(event, active_sessions=("ses-1", "ses-2"))
        assert result is None

    def test_thread_never_ambiguous(self):
        """Thread messages are never ambiguous (they have clear context)."""
        event = _thread_event()
        result = check_ambiguity(event, active_sessions=("ses-1", "ses-2"))
        assert result is None

    def test_group_with_binding_not_ambiguous(self):
        """Group with existing binding is not ambiguous."""
        put_binding("feishu", "oc_group888_unique", None, "ws-default", "ses-bound", source="default")
        event = _group_event()
        result = check_ambiguity(event, active_sessions=("ses-1", "ses-2"))
        assert result is None

    def test_group_default_disabled(self):
        """When feature flag is off, check_ambiguity always returns None."""
        # Default is disabled
        event = _group_event()
        result = check_ambiguity(event, active_sessions=("ses-1", "ses-2"))
        assert result is None


# ===========================================================================
# Card Session Binding Tests
# ===========================================================================

class TestRecordCardSessionBinding:
    """Test record_card_session_binding function."""

    def test_card_binding_creates_binding(self):
        """Card selection writes binding with source='card'."""
        record_card_session_binding("oc_abc_card1", None, "ws-feishu-oc_abc_card1", "ses-selected")
        val = get_binding_value("feishu", "oc_abc_card1", None)
        assert val is not None
        assert val.session_id == "ses-selected"
        assert val.source == "card"

    def test_card_binding_with_thread(self):
        """Card selection with thread context."""
        record_card_session_binding("oc_abc_card2", "om_thread1", "ws-feishu-oc_abc_card2", "ses-thread-selected")
        val = get_binding_value("feishu", "oc_abc_card2", "om_thread1")
        assert val is not None
        assert val.session_id == "ses-thread-selected"
        assert val.source == "card"


# ===========================================================================
# Session Cards Tests
# ===========================================================================

class TestSessionCards:
    """Test Feishu interactive card generation."""

    def test_ambiguity_card_with_sessions(self):
        """Ambiguity card includes buttons for each available session."""
        info = AmbiguityInfo(
            needs_card=True,
            workspace_id="ws-feishu-oc_abc",
            chat_id="oc_abc",
            thread_id=None,
            available_sessions=("ses-1", "ses-2"),
        )
        card = build_ambiguity_card(info, message_preview="hello")
        assert "elements" in card
        assert "header" in card
        # Find the action element
        actions = [e for e in card["elements"] if e.get("tag") == "action"]
        assert len(actions) >= 1
        assert len(actions[0]["actions"]) == 2

    def test_ambiguity_card_empty_message(self):
        """Ambiguity card works with empty message preview."""
        info = AmbiguityInfo(
            needs_card=True,
            workspace_id="ws-feishu-oc_abc",
            chat_id="oc_abc",
            thread_id=None,
            available_sessions=("ses-1",),
        )
        card = build_ambiguity_card(info)
        assert "elements" in card

    def test_no_session_card(self):
        """No available sessions → rejection card."""
        info = AmbiguityInfo(
            needs_card=True,
            workspace_id="ws-feishu-oc_abc",
            chat_id="oc_abc",
            thread_id=None,
            available_sessions=(),
        )
        card = build_ambiguity_card(info)
        # Should return rejection card
        assert card["header"]["title"]["content"] == "无法路由"

    def test_acknowledgement_card(self):
        """Acknowledgement card confirms session selection."""
        card = build_card_acknowledgement("ses-feishu-oc_abc", "ws-feishu")
        text = card["elements"][0]["text"]["content"]
        assert "已绑定到会话" in text or "已选择会话" in text

    def test_rejection_card(self):
        """Rejection card explains no routing available."""
        card = build_rejection_card("oc_abc")
        assert "无法路由" in card["header"]["title"]["content"]

    def test_session_label_thread(self):
        """Thread session ID gets human-readable label."""
        assert "线程" in _session_label("ses-feishu-thread-om_thread999")

    def test_session_label_regular(self):
        """Regular session ID gets human-readable label."""
        assert "会话" in _session_label("ses-feishu-oc_abc")

    def test_truncate_short(self):
        """Short text is not truncated."""
        assert _truncate("hello", 50) == "hello"

    def test_truncate_long(self):
        """Long text is truncated."""
        result = _truncate("a" * 100, 50)
        assert len(result) == 50


# ===========================================================================
# FeishuEntryAdapter resolve_session_with_ambiguity
# ===========================================================================

class TestFeishuAdapterResolveWithAmbiguity:
    """Test FeishuEntryAdapter.resolve_session_with_ambiguity method."""

    def test_p2p_resolves(self):
        """P2P event resolves via adapter method."""
        adapter = FeishuEntryAdapter()
        raw = {"chat_id": "ou_dm_adapter_test", "message_id": "om_1", "open_id": "ou_u1", "content": "hi"}
        event = adapter.normalize_event(raw)
        result = adapter.resolve_session_with_ambiguity(event)
        assert result.workspace_id is not None
        assert result.session_id is not None
        # No prior binding → default source
        assert result.source in ("default", "card")

    def test_group_thread_resolves(self):
        """Group + thread resolves via adapter method."""
        adapter = FeishuEntryAdapter()
        raw = {"chat_id": "oc_grp_adapter_test", "message_id": "om_2", "open_id": "ou_u1", "content": "hi", "thread_id": "om_thread_adapter"}
        event = adapter.normalize_event(raw)
        result = adapter.resolve_session_with_ambiguity(event)
        assert result.session_id is not None
        assert result.source in ("card", "thread", "default")


# ===========================================================================
# _is_group_chat helper
# ===========================================================================

class TestIsGroupChat:
    """Test _is_group_chat heuristic."""

    def test_oc_prefix_is_group(self):
        event = EntryEvent.from_dict({
            "event_id": "e1", "entrypoint": "feishu",
            "external_channel_id": "oc_abc123",
        })
        assert _is_group_chat(event) is True

    def test_ou_prefix_is_not_group(self):
        event = EntryEvent.from_dict({
            "event_id": "e2", "entrypoint": "feishu",
            "external_channel_id": "ou_dm456",
        })
        assert _is_group_chat(event) is False

    def test_session_with_group_marker(self):
        event = EntryEvent.from_dict({
            "event_id": "e3", "entrypoint": "feishu",
            "external_channel_id": "custom_channel",
            "session_id": "ses-feishu-group-abc",
        })
        assert _is_group_chat(event) is True
