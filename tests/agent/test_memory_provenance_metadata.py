"""Completed-turn provenance metadata reaches opted-in memory providers only."""
import threading
from unittest.mock import MagicMock

from tests.agent.test_memory_provider import FakeMemoryProvider
from agent.memory_manager import MemoryManager


class TurnMetadataMemoryProvider(FakeMemoryProvider):
    def __init__(self, name="external"):
        super().__init__(name=name)
        self.synced_messages = []

    def sync_turn(self, user_content, assistant_content, *, session_id="",
                  messages=None, metadata=None):
        self.synced_turns.append((user_content, assistant_content, session_id, metadata))
        self.synced_messages.append(messages)


def test_sync_all_forwards_metadata_to_opted_in_provider():
    mgr = MemoryManager()
    provider = TurnMetadataMemoryProvider()
    mgr.add_provider(provider)
    metadata = {"platform": "telegram", "chat_id": "chat-42"}

    mgr.sync_all("user", "assistant", metadata=metadata)
    assert mgr.flush_pending(timeout=2)

    assert provider.synced_turns == [("user", "assistant", "", metadata)]


def test_legacy_provider_does_not_receive_metadata():
    mgr = MemoryManager()
    legacy = FakeMemoryProvider("legacy")
    mgr.add_provider(legacy)

    mgr.sync_all("user", "assistant", metadata={"platform": "telegram"})
    assert mgr.flush_pending(timeout=2)

    assert legacy.synced_turns == [("user", "assistant")]


def test_sync_all_deeply_snapshots_optional_context_before_background_execution():
    mgr = MemoryManager()
    provider = TurnMetadataMemoryProvider()
    mgr.add_provider(provider)
    blocker = threading.Event()
    mgr._submit_background(lambda: blocker.wait(timeout=2))
    messages = [{"role": "user", "content": [{"type": "text", "text": "original"}]}]
    metadata = {"platform": "telegram", "scope": {"chat_id": "original"}}

    mgr.sync_all("user", "assistant", messages=messages, metadata=metadata)
    messages[0]["content"][0]["text"] = "mutated"
    metadata["scope"]["chat_id"] = "mutated"
    blocker.set()
    assert mgr.flush_pending(timeout=2)

    assert provider.synced_turns == [
        ("user", "assistant", "", {"platform": "telegram", "scope": {"chat_id": "original"}}),
    ]
    assert provider.synced_messages == [
        [{"role": "user", "content": [{"type": "text", "text": "original"}]}],
    ]


def test_completed_turn_forwards_runtime_provenance_metadata():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    agent.session_id = "test_session_001"
    agent.platform = "telegram"
    agent._chat_id = "chat-42"
    agent._thread_id = "thread-7"
    agent._gateway_session_key = "agent:main:telegram:profile:group:chat-42:thread-7"
    agent._parent_session_id = "parent-session"

    agent._sync_external_memory_for_turn(
        original_user_message="원문을 찾아줘",
        final_response="원문을 찾았어",
        interrupted=False,
    )

    _, kwargs = agent._memory_manager.sync_all.call_args
    assert kwargs["metadata"] == {
        "session_id": "test_session_001",
        "parent_session_id": "parent-session",
        "lineage_key": "agent:main:telegram:profile:group:chat-42:thread-7",
        "platform": "telegram",
        "chat_id": "chat-42",
        "thread_id": "thread-7",
        "origin_kind": "direct",
    }
