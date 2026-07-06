from __future__ import annotations

from run_agent import AIAgent


def _agent(model: str = "dflash", base_url: str = "http://10.10.20.211:8080/v1") -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.model = model
    agent.provider = "custom"
    agent.base_url = base_url
    agent._base_url_lower = base_url.lower()
    return agent


def test_dflash_compacts_large_text_tool_results_by_default(monkeypatch) -> None:
    monkeypatch.delenv("HERMES_COMPACT_TEXT_TOOL_RESULTS", raising=False)
    monkeypatch.delenv("HERMES_TEXT_TOOL_RESULT_COMPACT_THRESHOLD", raising=False)
    monkeypatch.delenv("HERMES_TEXT_TOOL_RESULT_COMPACT_CHARS", raising=False)
    agent = _agent()

    result = "HEAD\n" + ("A" * 3_000) + "CENTER_MARKER" + ("B" * 3_000) + "\nTAIL\n"

    compacted = agent._tool_result_content_for_active_model("terminal", result)

    assert len(compacted) < len(result)
    assert "HEAD" in compacted
    assert "TAIL" in compacted
    assert "CENTER_MARKER" not in compacted
    assert "Tool output compacted for dflash" in compacted


def test_non_dflash_keeps_large_text_tool_results_by_default(monkeypatch) -> None:
    monkeypatch.delenv("HERMES_COMPACT_TEXT_TOOL_RESULTS", raising=False)
    agent = _agent(model="qwen3.6-27b-256k")
    result = "x" * 20_000

    assert agent._tool_result_content_for_active_model("terminal", result) == result


def test_text_tool_result_compaction_can_be_forced(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_COMPACT_TEXT_TOOL_RESULTS", "1")
    monkeypatch.setenv("HERMES_TEXT_TOOL_RESULT_COMPACT_THRESHOLD", "1000")
    monkeypatch.setenv("HERMES_TEXT_TOOL_RESULT_COMPACT_CHARS", "800")
    agent = _agent(model="qwen3.6-27b-256k")
    result = "A" * 2_000 + "B" * 2_000

    compacted = agent._tool_result_content_for_active_model("skill_view", result)

    assert len(compacted) < 1_100
    assert compacted.startswith("A")
    assert compacted.endswith("B")


def test_text_tool_result_compaction_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_COMPACT_TEXT_TOOL_RESULTS", "0")
    agent = _agent()
    result = "x" * 20_000

    assert agent._tool_result_content_for_active_model("terminal", result) == result
