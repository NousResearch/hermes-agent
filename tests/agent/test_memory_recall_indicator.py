"""MemoryManager.describe_recall — the deterministic recall indicator.

When auto-recall injects memory, Hermes surfaces a model-independent
"🧠 <provider> — recalled N memories" status line so the user SEES memory
working regardless of whether the model chooses to mention it. These tests
lock the formatting (singular/plural/generic) and the aggregation across
providers, all deterministically (no LLM, no network).
"""
from typing import Optional

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider, RecallStatus


class _FakeProvider(MemoryProvider):
    """Provider with a settable recall_status for indicator tests."""

    def __init__(self, name: str, status: Optional[RecallStatus], *, raises: bool = False):
        self._name = name
        self._status = status
        self._raises = raises

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str = "", **kwargs) -> None:
        pass

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs) -> str:
        return ""

    def recall_status(self) -> Optional[RecallStatus]:
        if self._raises:
            raise RuntimeError("boom")
        return self._status


def test_no_status_returns_empty_string():
    mgr = MemoryManager()
    mgr.add_provider(_FakeProvider("hindsight", None))
    assert mgr.describe_recall() == ""


def test_no_providers_returns_empty_string():
    assert MemoryManager().describe_recall() == ""


def test_single_memory_is_singular():
    mgr = MemoryManager()
    mgr.add_provider(_FakeProvider("hindsight", RecallStatus("Hindsight", 1)))
    assert mgr.describe_recall() == "🧠 Hindsight — recalled 1 memory"


def test_multiple_memories_are_plural():
    mgr = MemoryManager()
    mgr.add_provider(_FakeProvider("hindsight", RecallStatus("Hindsight", 3)))
    assert mgr.describe_recall() == "🧠 Hindsight — recalled 3 memories"


def test_zero_count_renders_generic():
    # count 0 = content injected but no discrete count (e.g. reflect synthesis).
    mgr = MemoryManager()
    mgr.add_provider(_FakeProvider("hindsight", RecallStatus("Hindsight", 0)))
    assert mgr.describe_recall() == "🧠 Hindsight — recalled relevant memory"


def test_aggregates_multiple_providers():
    # builtin is always accepted first; a second external is rejected, so use
    # builtin + one external to exercise the join path.
    mgr = MemoryManager()
    mgr.add_provider(_FakeProvider("builtin", RecallStatus("Notes", 2)))
    mgr.add_provider(_FakeProvider("hindsight", RecallStatus("Hindsight", 5)))
    result = mgr.describe_recall()
    assert "🧠 Notes — recalled 2 memories" in result
    assert "🧠 Hindsight — recalled 5 memories" in result


def test_failing_provider_is_skipped_not_fatal():
    mgr = MemoryManager()
    mgr.add_provider(_FakeProvider("builtin", None, raises=True))
    mgr.add_provider(_FakeProvider("hindsight", RecallStatus("Hindsight", 1)))
    # The raising provider is swallowed; the healthy one still surfaces.
    assert mgr.describe_recall() == "🧠 Hindsight — recalled 1 memory"
