"""Promoted answers are delivered once while their replay carriers remain intact."""
from types import SimpleNamespace

import pytest

from tests.agent.test_promoted_reasoning_length import _response, _run, loop_agent as loop_agent


def _render_reasoning(result, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("display:\n  show_reasoning: true\n", encoding="utf-8")
    from gateway.config import Platform
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._show_reasoning = True
    source = SimpleNamespace(platform=Platform.TELEGRAM)
    return runner._hmwa_prepend_reasoning(result, result["final_response"], source, False)


@pytest.mark.parametrize("content", [None, ""])
@pytest.mark.parametrize("source", ["sdk", "stream"])
def test_promoted_reply_has_one_delivery_and_preserves_replay(loop_agent, tmp_path, monkeypatch, content, source):
    answer = "The answer is 42."
    result = _run(loop_agent, [_response(content=content, reasoning=answer, source=source)])
    assert result["api_calls"] == 1
    assert result["final_response"] == answer
    row = result["messages"][-1]
    assert not row.get("content")
    assert row["reasoning"] == row["api_content"] == answer
    rendered = _render_reasoning(result, tmp_path, monkeypatch)
    assert rendered == answer
    assert result["last_reasoning"] is None


@pytest.mark.parametrize("source", ["sdk", "stream"])
def test_visible_answer_retains_distinct_reasoning_display(loop_agent, tmp_path, monkeypatch, source):
    answer = "The answer is 42."
    reasoning = "I checked the calculation carefully."
    result = _run(loop_agent, [_response(content=answer, reasoning=reasoning, source=source)])
    assert result["api_calls"] == 1
    assert result["final_response"] == answer
    assert result["last_reasoning"] == reasoning
    assert result["messages"][-1]["content"] == answer
    rendered = _render_reasoning(result, tmp_path, monkeypatch)
    assert rendered.count(answer) == 1
    assert rendered.count(reasoning) == 1
