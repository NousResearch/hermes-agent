"""Tests for Feishu Session Binding Bridge (Phase 5B sidecar)."""

import pytest
from unittest.mock import MagicMock

from gateway.platforms.feishu_session_binding_bridge import record_feishu_session_binding
from agent.managed_agents.session_binding import get_binding, resolve_binding, put_binding, _bindings


@pytest.fixture(autouse=True)
def clear_bindings():
    """Clear in-memory bindings before each test."""
    _bindings.clear()
    yield
    _bindings.clear()


def _mock_session_source(**kwargs) -> MagicMock:
    """Build a mock SessionSource with required attributes."""
    m = MagicMock()
    m.chat_id = kwargs.get("chat_id", "oc_test_chat")
    m.user_id = kwargs.get("user_id", "ou_test_user")
    m.thread_id = kwargs.get("thread_id", None)
    m.message_id = kwargs.get("message_id", "om_test_msg")
    return m


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Graceful failure
# ---------------------------------------------------------------------------

def test_record_binding_graceful_on_missing_source_attr():
    """Sidecar catches exceptions when source is malformed."""
    bad_source = MagicMock()
    # missing chat_id attribute -> ValueError in normalize_event
    del bad_source.chat_id
    session_key = "feishu:test"

    # Must not raise
    record_feishu_session_binding(bad_source, session_key)

    # Binding should not exist (no binding written because sidecar failed)
    assert get_binding("feishu", None, None) is None


def test_record_binding_graceful_on_missing_session_key():
    """Sidecar handles empty session_key without crashing."""
    source = _mock_session_source(chat_id="oc_empty")
    session_key = ""

    # Must not raise
    record_feishu_session_binding(source, session_key)


# ---------------------------------------------------------------------------
# Workspace derivation
# ---------------------------------------------------------------------------

def test_record_binding_with_tenant_id():
    """Tenant ID in source leads to tenant-based workspace."""
    source = _mock_session_source(chat_id="oc_tenant", thread_id=None)
    # Simulate a payload that includes tenant_id by overriding workspace derivation
    # The sidecar uses FeishuEntryAdapter which derives workspace from chat_id when
    # tenant_id is absent in the raw payload.  Since our mock source has no tenant_id,
    # workspace falls back to chat_id.
    session_key = "feishu:oc_tenant:bot"

    record_feishu_session_binding(source, session_key)

    binding = get_binding("feishu", "oc_tenant", None)
    assert binding is not None
    ws_id, _ = binding
    assert ws_id == "ws-feishu-oc_tenant"


# ---------------------------------------------------------------------------
# Lookup round-trip
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# No side effects on existing flow
# ---------------------------------------------------------------------------

def test_record_binding_does_not_create_tasks():
    """The sidecar function does not import or call task creation."""
    import inspect
    src = inspect.getsource(record_feishu_session_binding)
    assert "create_task" not in src
    assert "run_agent" not in src
    assert "execute" not in src
    assert "route" not in src.lower()
