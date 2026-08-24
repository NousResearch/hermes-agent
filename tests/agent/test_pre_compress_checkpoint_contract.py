"""Host-side contract tests for the opt-in pre-compress checkpoint API (v1).

The contract has three parts:
- providers opt in by advertising ``pre_compress_checkpoint_api_version``;
- ``MemoryManager`` exposes capability probing and a ``require_checkpoint``
  mode whose failure must propagate instead of being swallowed;
- the compression host normalizes messages to direct user/assistant evidence
  before handing them to providers.
"""

import pytest

from agent.conversation_compression import (
    CompressionCheckpointUnavailable,
    _checkpoint_blocked,
    _direct_messages_for_pre_compress_memory,
)
from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY
from agent.memory_manager import MemoryManager
from agent.memory_provider import (
    PRE_COMPRESS_CHECKPOINT_API_VERSION,
    MemoryProvider,
)


class _BaseStubProvider(MemoryProvider):
    def __init__(self, name="stub"):
        self._name = name
        self.pre_compress_calls = []

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        return None

    def get_tool_schemas(self):
        return []

    def on_pre_compress(self, messages):
        self.pre_compress_calls.append(messages)
        return f"{self._name} context"


class _CheckpointProvider(_BaseStubProvider):
    pre_compress_checkpoint_api_version = PRE_COMPRESS_CHECKPOINT_API_VERSION


class _FailingCheckpointProvider(_CheckpointProvider):
    def on_pre_compress(self, messages):
        raise RuntimeError("durable store unreachable")


class _FailingLegacyProvider(_BaseStubProvider):
    def on_pre_compress(self, messages):
        raise RuntimeError("legacy best-effort failure")


def test_provider_base_class_defaults_to_best_effort_api_version_zero():
    assert MemoryProvider.pre_compress_checkpoint_api_version == 0


def test_direct_messages_filter_keeps_only_direct_source_evidence():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "durable user decision"},
        {"role": "assistant", "content": "direct assistant answer"},
        {"role": "assistant", "content": "call", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "content": "tool output", "tool_call_id": "t1"},
        {
            "role": "assistant",
            "content": "previous compaction summary",
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        },
        "not-a-dict",
    ]

    direct = _direct_messages_for_pre_compress_memory(messages)

    assert [m["content"] for m in direct] == [
        "durable user decision",
        "direct assistant answer",
    ]


def test_manager_advertises_checkpoint_capability_only_with_capable_provider():
    # The host allows one external provider per manager, so capability is
    # probed on two separate managers.
    legacy_manager = MemoryManager()
    legacy_manager.add_provider(_BaseStubProvider("legacy"))
    assert legacy_manager.supports_pre_compress_checkpoint(
        PRE_COMPRESS_CHECKPOINT_API_VERSION
    ) is False

    durable_manager = MemoryManager()
    durable_manager.add_provider(_CheckpointProvider("durable"))
    assert durable_manager.supports_pre_compress_checkpoint(
        PRE_COMPRESS_CHECKPOINT_API_VERSION
    ) is True


def test_manager_require_checkpoint_raises_without_capable_provider():
    manager = MemoryManager()
    manager.add_provider(_BaseStubProvider("legacy"))

    with pytest.raises(RuntimeError, match="pre-compress checkpoint"):
        manager.on_pre_compress(
            [{"role": "user", "content": "evidence"}],
            require_checkpoint=True,
            checkpoint_api_version=PRE_COMPRESS_CHECKPOINT_API_VERSION,
        )


def test_manager_require_checkpoint_propagates_checkpoint_provider_failure():
    manager = MemoryManager()
    manager.add_provider(_FailingCheckpointProvider("durable"))

    with pytest.raises(RuntimeError, match="durable store unreachable"):
        manager.on_pre_compress(
            [{"role": "user", "content": "evidence"}],
            require_checkpoint=True,
            checkpoint_api_version=PRE_COMPRESS_CHECKPOINT_API_VERSION,
        )


def test_manager_require_checkpoint_succeeds_and_returns_provider_context():
    manager = MemoryManager()
    durable = _CheckpointProvider("durable")
    manager.add_provider(durable)

    combined = manager.on_pre_compress(
        [{"role": "user", "content": "evidence"}],
        require_checkpoint=True,
        checkpoint_api_version=PRE_COMPRESS_CHECKPOINT_API_VERSION,
    )

    assert "durable context" in combined
    assert durable.pre_compress_calls


def test_manager_best_effort_mode_keeps_historical_swallow_semantics():
    manager = MemoryManager()
    manager.add_provider(_FailingLegacyProvider("legacy"))

    combined = manager.on_pre_compress([{"role": "user", "content": "evidence"}])

    assert combined == ""


def test_checkpoint_blocked_error_is_prefixed_and_typed():
    error = _checkpoint_blocked("no active provider")
    assert isinstance(error, CompressionCheckpointUnavailable)
    assert str(error).startswith("BLOCKED_MISSING_PREREQUISITE:")
    assert "no active provider" in str(error)


def test_compressed_summary_marker_survives_restart_via_resume_history(tmp_path):
    """The persistent marker reaches the resumed model history — and only it.

    ``get_messages_as_conversation`` keeps its existing marker-free contract;
    the resume path carries ``_compressed_summary`` so checkpoint providers
    keep excluding derivative summaries after a process restart.
    """
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="cli")
    db.append_message("s1", "user", "durable user evidence")
    db.append_message(
        "s1", "assistant", "derivative summary", _compressed_summary=True
    )

    reopened = SessionDB(tmp_path / "state.db")
    model_history, _display = reopened.get_resume_conversations("s1")
    by_content = {m.get("content"): m for m in model_history}
    assert by_content["derivative summary"].get("_compressed_summary") is True
    assert "_compressed_summary" not in by_content["durable user evidence"]

    plain = reopened.get_messages_as_conversation("s1")
    assert all("_compressed_summary" not in m for m in plain)
