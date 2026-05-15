from pathlib import Path

from agent.memory_manager import MemoryManager
from plugins.memory.memory_fragmentation import MemoryFragmentationProvider


def _bare_agent(memory_manager):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = memory_manager
    agent.session_id = "session-fragmentation-hook"
    return agent


def test_after_turn_sync_writes_memory_fragment(tmp_path):
    provider = MemoryFragmentationProvider()
    provider.initialize(
        "session-fragmentation-hook",
        hermes_home=str(tmp_path),
        platform="cli",
        user_id="user-1",
    )
    manager = MemoryManager()
    manager.add_provider(provider)
    agent = _bare_agent(manager)

    agent._sync_external_memory_for_turn(
        original_user_message="Develop a quant strategy with moving average and volatility filters.",
        final_response="Completed the quant strategy and wrote reports/performance.md with CAGR, Sortino, and max drawdown analysis.",
        interrupted=False,
    )

    records_path = tmp_path / "memory_fragmentation" / "fragments.jsonl"
    assert records_path.exists()
    records_text = records_path.read_text(encoding="utf-8")
    assert "quant" in records_text.lower()
    assert "reports/performance.md" in records_text


def test_after_turn_interrupt_does_not_write_memory_fragment(tmp_path):
    provider = MemoryFragmentationProvider()
    provider.initialize("session-fragmentation-hook", hermes_home=str(tmp_path), platform="cli")
    manager = MemoryManager()
    manager.add_provider(provider)
    agent = _bare_agent(manager)

    agent._sync_external_memory_for_turn(
        original_user_message="Develop a quant strategy with moving average filters.",
        final_response="Partial result that the user never saw completed.",
        interrupted=True,
    )

    records_path = tmp_path / "memory_fragmentation" / "fragments.jsonl"
    assert not records_path.exists()
