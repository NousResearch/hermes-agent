from copy import deepcopy
import json

from agent.codex_responses_adapter import _summarize_user_message_for_log
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider, PRE_COMPRESS_CHECKPOINT_API_VERSION


_DATA_URI = "data:image/png;base64,aGVsbG8="


class _RecordingProvider(MemoryProvider):
    pre_compress_checkpoint_api_version = PRE_COMPRESS_CHECKPOINT_API_VERSION

    def __init__(self) -> None:
        self.sync_user = None
        self.sync_assistant = None
        self.sync_messages = None
        self.sync_turn_author = None
        self.session_end_messages = None
        self.pre_compress_messages = None

    @property
    def name(self) -> str:
        return "recording"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self):
        return []

    def sync_turn(
        self, user_content, assistant_content, *, session_id="", messages=None, turn_author=None,
    ) -> None:
        self.sync_user = user_content
        self.sync_assistant = assistant_content
        self.sync_messages = messages
        self.sync_turn_author = turn_author

    def on_session_end(self, messages) -> None:
        self.session_end_messages = messages

    def on_pre_compress(self, messages, *, require_checkpoint=False) -> str:
        self.pre_compress_messages = messages
        return ""


def _messages_with_data_uri():
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "keep this text"},
                {"type": "image_url", "image_url": {"url": _DATA_URI}},
            ],
        },
        {
            "role": "assistant",
            "content": f"answer {_DATA_URI}",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "inspect", "arguments": json.dumps({"image": _DATA_URI})},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": [{"type": "text", "text": f"tool output {_DATA_URI}"}],
        },
    ]


def _assert_sanitized(value) -> None:
    serialized = json.dumps(value, ensure_ascii=False)
    assert "data:image/png;base64" not in serialized
    assert "[embedded data]" in serialized


def test_log_summary_strips_data_uri_from_scalar_and_text_parts():
    scalar = _summarize_user_message_for_log(f"before {_DATA_URI} after")
    assert scalar == "before [embedded data] after"

    parts = _summarize_user_message_for_log([
        {"type": "text", "text": f"caption {_DATA_URI}"},
        {"type": "image_url", "image_url": {"url": _DATA_URI}},
    ])
    assert parts == "[1 image] caption [embedded data]"


def test_memory_manager_sanitizes_nested_messages_without_mutating_transcript():
    manager = MemoryManager()
    provider = _RecordingProvider()
    manager.add_provider(provider)
    messages = _messages_with_data_uri()
    original = deepcopy(messages)
    turn_author = {"id": "bot:lynn", "name": "Lynn", "is_bot": True}

    manager.sync_all(
        f"user {_DATA_URI}",
        f"assistant {_DATA_URI}",
        session_id="session-1",
        messages=messages,
        turn_author=turn_author,
    )
    assert manager.flush_pending(timeout=5.0)

    assert provider.sync_user == "user [embedded data]"
    assert provider.sync_assistant == "assistant [embedded data]"
    _assert_sanitized(provider.sync_messages)
    assert provider.sync_messages is not messages
    assert provider.sync_turn_author == turn_author
    assert messages == original

    manager.on_session_end(messages)
    _assert_sanitized(provider.session_end_messages)
    assert provider.session_end_messages is not messages
    assert messages == original

    manager.on_pre_compress(messages)
    _assert_sanitized(provider.pre_compress_messages)
    assert provider.pre_compress_messages is not messages
    assert messages == original


def test_memory_manager_sanitizes_checkpoint_evidence_without_mutating_it():
    manager = MemoryManager()
    provider = _RecordingProvider()
    manager.add_provider(provider)
    messages = [{"role": "user", "content": "plain raw transcript"}]
    evidence_messages = [{"role": "assistant", "content": {"image_url": {"url": _DATA_URI}}}]
    original_evidence = deepcopy(evidence_messages)

    manager.on_pre_compress(messages, evidence_messages=evidence_messages)

    _assert_sanitized(provider.pre_compress_messages)
    assert provider.pre_compress_messages is not evidence_messages
    assert evidence_messages == original_evidence
