"""Tests for conversation-provenance metadata on Mem0 writes."""

import json

from plugins.memory.mem0 import (
    Mem0MemoryProvider,
    _PROVENANCE_KWARGS,
    _PROVENANCE_VALUE_MAX_CHARS,
)


class CaptureBackend:
    """Backend stub that records add() calls."""

    def __init__(self):
        self.adds = []

    def add(self, messages, *, user_id, agent_id, infer, metadata):
        self.adds.append({
            "messages": messages, "user_id": user_id,
            "agent_id": agent_id, "infer": infer, "metadata": metadata,
        })
        return {"status": "PENDING", "event_id": "evt-1"}

    def close(self):
        pass


GATEWAY_KWARGS = {
    "platform": "slack",
    "user_id": "U0AAAA1",
    "user_id_alt": "alt-42",
    "user_name": "Ada Lovelace",
    "chat_id": "C0BBBB2",
    "chat_name": "customer-care",
    "chat_type": "channel",
    "thread_id": "1784193622.582189",
    "session_title": "Login issue triage",
    # Host-local init kwargs that must never reach remote metadata:
    "hermes_home": "/home/user/.hermes",
    "agent_identity": "default",
    "agent_workspace": "hermes",
    "gateway_session_key": "agent:main:slack:channel:C0BBBB2:1784193622.582189",
}


def _provider(init_kwargs=None, session_id="sess-1"):
    provider = Mem0MemoryProvider()
    provider.initialize(session_id, **(init_kwargs or {}))
    provider._backend = CaptureBackend()
    return provider


class TestWriteProvenanceMetadata:

    def test_tool_add_carries_conversation_provenance(self):
        provider = _provider(GATEWAY_KWARGS)
        result = json.loads(provider.handle_tool_call("mem0_add", {"content": "fact"}))
        assert "error" not in result
        metadata = provider._backend.adds[0]["metadata"]
        assert metadata["gateway_user_id"] == "U0AAAA1"
        assert metadata["gateway_user_id_alt"] == "alt-42"
        assert metadata["user_name"] == "Ada Lovelace"
        assert metadata["chat_id"] == "C0BBBB2"
        assert metadata["chat_name"] == "customer-care"
        assert metadata["chat_type"] == "channel"
        assert metadata["thread_id"] == "1784193622.582189"
        assert metadata["session_title"] == "Login issue triage"
        assert metadata["channel"] == "slack"
        assert metadata["session_id"] == "sess-1"

    def test_host_local_kwargs_never_leak(self):
        provider = _provider(GATEWAY_KWARGS)
        provider.handle_tool_call("mem0_add", {"content": "fact"})
        metadata = provider._backend.adds[0]["metadata"]
        allowed = {meta_key for _, meta_key in _PROVENANCE_KWARGS} | {"channel", "session_id"}
        assert set(metadata) <= allowed
        assert "hermes_home" not in metadata
        assert "gateway_session_key" not in metadata

    def test_cli_defaults_stay_minimal(self):
        provider = _provider({}, session_id="sess-cli")
        provider.handle_tool_call("mem0_add", {"content": "fact"})
        metadata = provider._backend.adds[0]["metadata"]
        assert metadata == {"channel": "cli", "session_id": "sess-cli"}

    def test_values_are_stringified_and_truncated(self):
        kwargs = dict(GATEWAY_KWARGS, user_id=123456789, session_title="x" * 1000)
        provider = _provider(kwargs)
        provider.handle_tool_call("mem0_add", {"content": "fact"})
        metadata = provider._backend.adds[0]["metadata"]
        assert metadata["gateway_user_id"] == "123456789"
        assert len(metadata["session_title"]) == _PROVENANCE_VALUE_MAX_CHARS

    def test_sync_turn_prefers_per_call_session_id(self):
        provider = _provider(GATEWAY_KWARGS, session_id="sess-old")
        provider.sync_turn("hello", "hi there", session_id="sess-new")
        provider._sync_thread.join(timeout=5.0)
        metadata = provider._backend.adds[0]["metadata"]
        assert metadata["session_id"] == "sess-new"
        assert metadata["thread_id"] == "1784193622.582189"

    def test_sync_turn_falls_back_to_initialize_session_id(self):
        provider = _provider(GATEWAY_KWARGS, session_id="sess-init")
        provider.sync_turn("hello", "hi there")
        provider._sync_thread.join(timeout=5.0)
        assert provider._backend.adds[0]["metadata"]["session_id"] == "sess-init"

    def test_reinitialize_replaces_stale_provenance(self):
        provider = _provider(GATEWAY_KWARGS)
        provider.initialize("sess-2", platform="telegram", user_id="tg-7")
        provider._backend = CaptureBackend()
        provider.handle_tool_call("mem0_add", {"content": "fact"})
        metadata = provider._backend.adds[0]["metadata"]
        assert metadata["gateway_user_id"] == "tg-7"
        assert metadata["channel"] == "telegram"
        assert "chat_id" not in metadata
