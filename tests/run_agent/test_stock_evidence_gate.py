from __future__ import annotations

from types import SimpleNamespace

from run_agent import AIAgent
from tools.evotraders_evidence import check_min_evidence


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.model = "test-model"
    agent.session_id = "test-session"
    agent.valid_tool_names = {"evotraders_route_and_call"}
    agent._evotraders_evidence_recent = []
    agent._emit_status = lambda *_args, **_kwargs: None
    agent._save_session_log = lambda *_args, **_kwargs: None
    agent._has_content_after_think_block = lambda text: bool((text or "").strip())
    agent._build_assistant_message = (
        lambda _assistant_message, _finish_reason: {"role": "assistant", "content": "interim"}
    )
    return agent


def test_get_stock_evidence_missing_detects_gap():
    agent = _bare_agent()
    missing = AIAgent._get_stock_evidence_missing(agent, "请分析 600519 后市")
    assert isinstance(missing, list)
    assert missing


def test_maybe_enqueue_stock_evidence_autofetch_appends_followup_messages():
    agent = _bare_agent()
    messages = []
    triggered = AIAgent._maybe_enqueue_stock_evidence_autofetch(
        agent,
        original_user_message="请分析 600519 后市",
        final_response="先给你一个初步结论",
        messages=messages,
        assistant_message=SimpleNamespace(),
        finish_reason="stop",
        already_attempted=False,
    )
    assert triggered is True
    assert len(messages) == 2
    assert messages[0]["role"] == "assistant"
    assert messages[1]["role"] == "user"
    assert "evotraders_route_and_call" in messages[1]["content"]


def test_apply_stock_evidence_gate_replaces_last_assistant_content():
    agent = _bare_agent()
    messages = [{"role": "assistant", "content": "原始结论"}]
    gated = AIAgent._apply_stock_evidence_gate_if_needed(
        agent,
        original_user_message="分析 000001",
        final_response="买入建议",
        interrupted=False,
        messages=messages,
    )
    assert gated is not None
    assert "最小证据集不足" in gated
    assert messages[-1]["content"] == gated


def test_stock_evidence_missing_includes_code_gap():
    # 有快照/结构/时点，但证据是别的股票代码时，应命中 code 缺口。
    ev = {
        "has_snapshot": True,
        "has_structure": True,
        "has_timepoint": True,
        "codes": ["000001"],
    }
    out = check_min_evidence(ev, intent="stock_analysis", required_codes=["600519"])
    assert out["ok"] is False
    assert any(str(x).startswith("code:") for x in out["missing"])

