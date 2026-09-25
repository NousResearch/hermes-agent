"""Checkpoint providers may read raw evidence without changing the v2 positional contract."""

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import PRE_COMPRESS_CHECKPOINT_API_VERSION


class _CheckpointProvider:
    name = "checkpoint-probe"
    pre_compress_checkpoint_api_version = PRE_COMPRESS_CHECKPOINT_API_VERSION

    def __init__(self):
        self.received = None

    def on_pre_compress(self, messages, *, require_checkpoint=False, raw_messages=None):
        self.received = (messages, require_checkpoint, raw_messages)
        return ""


def test_checkpoint_receives_normalized_and_raw_evidence() -> None:
    manager = MemoryManager()
    provider = _CheckpointProvider()
    manager._providers = [provider]
    raw = [{"role": "tool", "content": "the complete tool output"}]
    normalized = [{"role": "user", "content": "the normalized user statement"}]
    assert manager.on_pre_compress(raw, evidence_messages=normalized, require_checkpoint=True) == ""
    assert provider.received == (normalized, True, raw)


def test_checkpoint_without_raw_keyword_keeps_existing_v2_contract() -> None:
    class ExistingProvider:
        name = "existing-checkpoint"
        pre_compress_checkpoint_api_version = PRE_COMPRESS_CHECKPOINT_API_VERSION

        def __init__(self):
            self.received = None

        def on_pre_compress(self, messages, *, require_checkpoint=False):
            self.received = (messages, require_checkpoint)
            return ""

    manager = MemoryManager()
    provider = ExistingProvider()
    manager._providers = [provider]
    raw = [{"role": "tool", "content": "tool detail"}]
    normalized = [{"role": "user", "content": "summary"}]
    manager.on_pre_compress(raw, evidence_messages=normalized, require_checkpoint=True)
    assert provider.received == (normalized, True)


def test_legacy_provider_keeps_raw_positional_contract() -> None:
    class LegacyProvider:
        name = "legacy"
        pre_compress_checkpoint_api_version = 1

        def __init__(self):
            self.received = None

        def on_pre_compress(self, messages):
            self.received = messages
            return ""

    manager = MemoryManager()
    provider = LegacyProvider()
    manager._providers = [provider]
    raw = [{"role": "tool", "content": "tool detail"}]
    manager.on_pre_compress(raw, evidence_messages=[])
    assert provider.received is raw


def test_required_checkpoint_failure_aborts_memory_handoff() -> None:
    manager = MemoryManager()
    provider = _CheckpointProvider()
    manager._providers = [provider]
    def fail(*args, **kwargs):
        raise RuntimeError("ingestion job was not accepted")
    provider.on_pre_compress = fail
    with pytest.raises(RuntimeError, match="not accepted"):
        manager.on_pre_compress([], evidence_messages=[], require_checkpoint=True)
