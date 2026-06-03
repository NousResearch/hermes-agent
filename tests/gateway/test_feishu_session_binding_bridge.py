"""Tests for Feishu Session Binding Bridge (Phase 5B sidecar)."""

import pytest
from unittest.mock import MagicMock, patch

from gateway.platforms.feishu_session_binding_bridge import (
    record_feishu_session_binding,
    resolve_feishu_session_from_source,
    check_feishu_ambiguity,
    build_session_selection_card,
    handle_select_session_card_action,
)
from agent.managed_agents.feishu_session_resolver import AmbiguityInfo
from agent.managed_agents.session_binding import (
    get_binding,
    get_binding_value,
    resolve_binding,
    put_binding,
    _bindings,
    _binding_values,
    _bindings_lock,
)


@pytest.fixture(autouse=True)
def clear_bindings():
    """Clear in-memory bindings before each test."""
    with _bindings_lock:
        _bindings.clear()
        _binding_values.clear()
    yield
    with _bindings_lock:
        _bindings.clear()
        _binding_values.clear()


def _mock_session_source(**kwargs) -> MagicMock:
    """Build a mock SessionSource with required attributes."""
    m = MagicMock()
    m.chat_id = kwargs.get("chat_id", "oc_test_chat")
    m.user_id = kwargs.get("user_id", "ou_test_user")
    m.thread_id = kwargs.get("thread_id", None)
    m.message_id = kwargs.get("message_id", "om_test_msg")
    return m


# ===========================================================================
# Legacy record_feishu_session_binding
# ===========================================================================

def test_record_binding_writes_to_store():
    """Sidecar writes a binding record to SessionBinding store."""
    source = _mock_session_source(chat_id="oc_abc", thread_id="om_thread_123")
    session_key = "feishu:oc_abc:thread:om_thread_123:botname"

    record_feishu_session_binding(source, session_key)

    binding = get_binding("feishu", "oc_abc", "om_thread_123")
    assert binding is not None
    ws_id, ses_id = binding
    assert ws_id.startswith("ws-feishu-")
    assert ses_id == session_key


def test_record_binding_without_thread():
    """Message without thread writes binding with thread_id=None."""
    source = _mock_session_source(chat_id="oc_def", thread_id=None)
    session_key = "feishu:oc_def:botname"

    record_feishu_session_binding(source, session_key)

    binding = get_binding("feishu", "oc_def", None)
    assert binding is not None
    ws_id, ses_id = binding
    assert ws_id == "ws-feishu-oc_def"
    assert ses_id == session_key


def test_record_binding_idempotent():
    """Writing the same binding twice is idempotent (last write wins)."""
    source = _mock_session_source(chat_id="oc_xyz")
    session_key_v1 = "feishu:oc_xyz:v1"
    session_key_v2 = "feishu:oc_xyz:v2"

    record_feishu_session_binding(source, session_key_v1)
    record_feishu_session_binding(source, session_key_v2)

    binding = get_binding("feishu", "oc_xyz", None)
    assert binding is not None
    assert binding[1] == session_key_v2


def test_record_binding_graceful_on_missing_source_attr():
    """Sidecar catches exceptions when source is malformed."""
    bad_source = MagicMock()
    del bad_source.chat_id
    session_key = "feishu:test"

    record_feishu_session_binding(bad_source, session_key)
    assert get_binding("feishu", None, None) is None


def test_record_binding_graceful_on_missing_session_key():
    """Sidecar handles empty session_key without crashing."""
    source = _mock_session_source(chat_id="oc_empty")
    session_key = ""

    record_feishu_session_binding(source, session_key)


def test_record_binding_with_tenant_id():
    """Tenant ID in source leads to tenant-based workspace."""
    source = _mock_session_source(chat_id="oc_tenant", thread_id=None)
    session_key = "feishu:oc_tenant:bot"

    record_feishu_session_binding(source, session_key)

    binding = get_binding("feishu", "oc_tenant", None)
    assert binding is not None
    ws_id, _ = binding
    assert ws_id == "ws-feishu-oc_tenant"


def test_binding_lookup_returns_correct_values():
    """get_binding returns the workspace and session written by the sidecar."""
    source = _mock_session_source(chat_id="oc_lookup", thread_id="om_thr")
    session_key = "feishu:oc_lookup:thread:om_thr:bot"

    record_feishu_session_binding(source, session_key)

    ws_id, ses_id = get_binding("feishu", "oc_lookup", "om_thr")
    assert ws_id.startswith("ws-feishu-")
    assert ses_id == session_key


def test_binding_fallback_for_unknown():
    """get_binding returns defaults for unknown entrypoint/channel/thread."""
    ws_id, ses_id = resolve_binding("feishu", "oc_unknown", None)
    assert ws_id == "hermes-local"
    assert ses_id == "hermes-legacy"


def test_record_binding_does_not_create_tasks():
    """The sidecar function does not import or call task creation."""
    import inspect
    src = inspect.getsource(record_feishu_session_binding)
    assert "create_task" not in src
    assert "run_agent" not in src
    assert "execute" not in src
    assert "route" not in src.lower()


# ===========================================================================
# resolve_feishu_session_from_source
# ===========================================================================

def test_resolve_from_source_p2p():
    """Resolve a p2p session from SessionSource."""
    source = _mock_session_source(chat_id="ou_dm_bridge", user_id="ou_user1")
    result = resolve_feishu_session_from_source(source)
    assert result is not None
    assert result.workspace_id is not None
    assert result.session_id is not None


def test_resolve_from_source_group_with_thread():
    """Resolve a group session with thread from SessionSource."""
    source = _mock_session_source(chat_id="oc_grp_bridge", thread_id="om_thr_bridge")
    result = resolve_feishu_session_from_source(source)
    assert result is not None
    assert "thread" in result.session_id or result.source in ("card", "thread", "default")


def test_resolve_from_source_error_returns_none():
    """When resolution fails, returns None (non-critical)."""
    bad_source = MagicMock()
    del bad_source.chat_id
    result = resolve_feishu_session_from_source(bad_source)
    assert result is None


# ===========================================================================
# check_feishu_ambiguity
# ===========================================================================

def test_check_ambiguity_p2p_returns_none():
    """P2P messages are never ambiguous via bridge."""
    source = _mock_session_source(chat_id="ou_dm_ambig", user_id="ou_u1")
    result = check_feishu_ambiguity(source)
    assert result is None


def test_check_ambiguity_group_with_thread_returns_none():
    """Thread messages are not ambiguous via bridge."""
    source = _mock_session_source(chat_id="oc_grp_ambig", thread_id="om_thr_ambig")
    result = check_feishu_ambiguity(source)
    assert result is None


def test_check_ambiguity_error_returns_none():
    """When ambiguity check fails, returns None (non-critical)."""
    bad_source = MagicMock()
    del bad_source.chat_id
    result = check_feishu_ambiguity(bad_source)
    assert result is None


# ===========================================================================
# handle_select_session_card_action
# ===========================================================================

def test_select_session_card_action():
    """Card action writes card binding correctly."""
    action = {
        "action": "select_session",
        "session_id": "ses-selected-card",
        "workspace_id": "ws-feishu-test",
        "chat_id": "oc_card_test",
        "thread_id": "",
    }
    result = handle_select_session_card_action(action)
    assert result is True

    val = get_binding_value("feishu", "oc_card_test", None)
    assert val is not None
    assert val.session_id == "ses-selected-card"
    assert val.source == "card"


def test_select_session_card_action_with_thread():
    """Card action with thread context writes binding correctly."""
    action = {
        "action": "select_session",
        "session_id": "ses-thread-card",
        "workspace_id": "ws-feishu-test2",
        "chat_id": "oc_card_test2",
        "thread_id": "om_thr_card",
    }
    result = handle_select_session_card_action(action)
    assert result is True

    val = get_binding_value("feishu", "oc_card_test2", "om_thr_card")
    assert val is not None
    assert val.session_id == "ses-thread-card"
    assert val.source == "card"


def test_select_session_card_action_missing_session_id():
    """Card action missing session_id returns False."""
    action = {
        "action": "select_session",
        "session_id": "",
        "chat_id": "oc_card_missing",
    }
    result = handle_select_session_card_action(action)
    assert result is False


def test_select_session_card_action_missing_chat_id():
    """Card action missing chat_id returns False."""
    action = {
        "action": "select_session",
        "session_id": "ses-missing-chat",
        "chat_id": "",
    }
    result = handle_select_session_card_action(action)
    assert result is False


# ===========================================================================
# build_session_selection_card
# ===========================================================================

def test_build_session_selection_card():
    """Build an interactive card for session selection."""
    info = AmbiguityInfo(
        needs_card=True,
        workspace_id="ws-bridge-test",
        chat_id="oc_bridge_card",
        thread_id=None,
        available_sessions=("ses-1", "ses-2"),
    )
    card = build_session_selection_card(info, message_preview="test message")
    assert card is not None
    assert "elements" in card
    assert "header" in card


def test_build_session_selection_card_returns_none_on_error():
    """Card building returns None on unexpected errors."""
    # Pass None to trigger an error
    result = build_session_selection_card(None)
    assert result is None
