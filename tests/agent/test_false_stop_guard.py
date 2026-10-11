"""Behavior-contract tests for the false-stop guard.

Does not start a conversation loop and does not open a session database.
The 183-character gold string is inlined verbatim.
"""

from __future__ import annotations

from agent.conversation_compression import _SYNTHETIC_USER_FLAGS
from agent.false_stop_guard import (
    FALSE_STOP_NUDGE,
    FALSE_STOP_SIGNALS,
    ROUTING_BLOCK_KEYS,
    classify_false_stop_budget,
    false_stop_decision,
)
from agent.session_persistence import _EPHEMERAL_SCAFFOLDING_FLAGS
from agent.turn_finalizer import _VERIFICATION_CONTINUATION_FLAGS
from agent.turn_stop_gates import apply_stop_gates

GOLD_TEXT = "\n".join([
    "路由",
    "- 分类：动态",
    "- 流程族：标准编码（standard_coding）",
    "- 路径：本机薄补丁拆上游拉取请求（PR）",
    "- 技能：hermes-local-to-upstream-pr",
    "- 验证：去重、干净工作树套用、远程拉取请求链接",
    "- 计划强度：常明（中）",
    "- 执行强度：无影灯（高质量）",
    "",
    "先读上游补丁技能，并核实暂存与两个临时工作树里有没有还没落地的东西。",
])

_ROUTE = "路由\n- 分类：动态\n- 路径：本地\n\n"


def _route(body: str) -> str:
    return _ROUTE + body


def _decide(text: str, *, finish_reason="stop", tools=0, cont=0, budget="remaining") -> str:
    return false_stop_decision(text, finish_reason, tools, cont, budget)


def test_gold_183_nudges():
    assert len(GOLD_TEXT) == 183
    assert _decide(GOLD_TEXT) == "nudge"


def test_bare_heading_positive_nudges():
    text = "\n".join([
        "路由",
        "- 分类：动态",
        "- 技能：hermes-local-to-upstream-pr",
        "",
        "我先读文件",
        "## 怎么做",
    ])
    assert _decide(text) == "nudge"


def test_extended_frozen_key_counts_as_routing_block():
    text = "\n".join([
        "路由",
        "- 对抗审查类型：v3-plan-grill",
        "",
        "我先读文件",
    ])
    assert _decide(text) == "nudge"


def test_english_commitments_nudge():
    assert _decide(_route("Let me read the file.")) == "nudge"
    assert _decide(_route("let me read the file.")) == "nudge"
    assert _decide(_route("I'll check the log.")) == "nudge"
    assert _decide(_route("Next I will open the notes.")) == "nudge"


def test_user_directed_releases():
    assert _decide(_route("你可以先读文件。")) == "release"
    assert _decide(_route("建议你先看日志。")) == "release"
    assert _decide(_route("你可以打开那份说明。")) == "release"


def test_question_ending_releases():
    assert _decide(_route("我先读文件？")) == "release"
    assert _decide(_route("Let me read this?")) == "release"


def test_routing_block_without_signal_releases():
    assert _decide("路由\n- 分类：动态\n- 路径：本地\n") == "release"
    assert _decide(_route("本轮没有额外动作。")) == "release"


def test_refusal_releases():
    assert _decide(_route("我无法完成该请求。该请求被拒绝。")) == "release"
    assert _decide(_route("拒绝执行此操作。")) == "release"


def test_already_done_releases():
    assert _decide(_route("任务已完成。")) == "release"
    assert _decide(_route("文件已写入。")) == "release"
    assert _decide(_route("补丁已提交。")) == "release"


def test_long_analysis_without_routing_block_releases():
    text = (
        "我将从三方面说明这个问题的边界与证据。"
        "第一，近窗样本只支持观察，不支持改写生产路径，因此这里只复述已见事实。"
        "第二，对照比例在采样窗口内保持稳定，没有出现必须立刻动手的新漂移。"
        "第三，所以本段是终答，不是下一步动作清单，不再追加读取或检索。"
        "补充一句以便越过两百字：以上三段已经把观察、稳定性和终答边界说完，读者可以据此自行核对，不必再等下一轮动作。"
        "这段只说明已经看见的边界，不构成新的承诺，也不要求继续检索或改写。"
    )
    assert "路由" not in text.splitlines()[0]
    assert len(text) > 200
    assert text.count("。") >= 2
    assert _decide(text) == "release"


def test_mixed_normal_final_answer_releases():
    body = (
        "结论已经写完，本轮不再追加动作。"
        "The review closed on the evidence already in hand, and this mixed note stays descriptive. "
        "I will leave the remaining choice with you. "
        "Letter me a summary only if you ask; let's not open another pass. "
        "第二段只复述已落地的结果，不承诺新的读取或检索。"
        "开始总结到此为止，表外祈使不算续跑信号。"
    )
    assert len(body) > 200
    assert _decide(_route(body)) == "release"


def test_user_procedure_close_releases_but_self_promise_still_nudges():
    assert _decide(_route("接下来打开目录即可。")) == "release"
    assert _decide(_route("我先打开目录即可。")) == "nudge"


def test_process_tail_releases():
    assert _decide(_route("先确认再执行。")) == "release"
    assert _decide(_route("先读日志再执行。")) == "release"
    assert _decide(_route("接下来打开目录再处理。")) == "release"
    assert _decide(_route("开始检索索引再决定。")) == "release"


def test_table_external_imperative_releases():
    assert _decide(_route("开始总结范围。")) == "release"
    assert _decide(_route("先确认范围。")) == "release"


def test_substantial_delivery_suppresses_signal():
    sentence = "我将从三方面说明已经完成的结论。"
    body = sentence * 14
    assert len(body) > 200
    assert body.count("。") >= 2
    assert _decide(_route(body)) == "release"


def test_budget_three_branches():
    assert _decide(GOLD_TEXT, budget="exhausted") == "release"
    assert _decide(GOLD_TEXT, budget="unknown") == "nudge"
    assert _decide(GOLD_TEXT, budget="remaining") == "nudge"


def test_second_continuation_releases():
    assert _decide(GOLD_TEXT, cont=1, budget="remaining") == "release"


def test_stop_and_zero_tools_required():
    assert _decide(GOLD_TEXT, finish_reason="length") == "release"
    assert _decide(GOLD_TEXT, tools=1) == "release"


def test_word_boundary_does_not_match_letter_me():
    assert _decide(_route("Letter me a note about the result.")) == "release"


def test_whitelist_tuples_include_flag():
    assert "_false_stop_synthetic" in _VERIFICATION_CONTINUATION_FLAGS
    assert "_false_stop_synthetic" in _SYNTHETIC_USER_FLAGS
    assert "_false_stop_synthetic" in _EPHEMERAL_SCAFFOLDING_FLAGS


def test_frozen_set_is_extended_23():
    assert len(ROUTING_BLOCK_KEYS) == 23
    for key in ("对抗审查类型", "对抗审查角色", "用户烤问", "计划场景", "场景来源"):
        assert key in ROUTING_BLOCK_KEYS
    assert FALSE_STOP_SIGNALS["imperative_verbs"] == (
        "读", "看", "核实", "加载", "检索", "打开",
    )


def test_nudge_is_one_sentence_and_not_the_gold():
    assert FALSE_STOP_NUDGE.endswith("。")
    assert FALSE_STOP_NUDGE.count("。") == 1
    assert "先读上游补丁技能" not in FALSE_STOP_NUDGE


class _Budget:
    def __init__(self, remaining):
        self.remaining = remaining


class _Unreadable:
    @property
    def remaining(self):
        raise RuntimeError("unreadable")


def test_budget_classifier_missing_unknown_and_exhausted():
    assert classify_false_stop_budget(None, 10, 1) == "unknown"
    assert classify_false_stop_budget(_Budget(1), None, 1) == "unknown"
    assert classify_false_stop_budget(_Unreadable(), 10, 1) == "unknown"
    assert classify_false_stop_budget(_Budget(2), 10, 1) == "remaining"
    assert classify_false_stop_budget(_Budget(0), 10, 1) == "exhausted"
    assert classify_false_stop_budget(_Budget(3), 4, 4) == "exhausted"


class _GateAgent:
    def __init__(self):
        self._false_stop_continuations = 0
        self.flushed = 0

    def _emit_interim_assistant_message(self, msg):
        self.emitted = msg

    def _flush_messages_to_session_db(self, messages, history):
        self.flushed += 1

    def _interim_content_was_streamed(self, text):
        return False


def _silence_earlier_gates(monkeypatch):
    monkeypatch.setattr("agent.turn_stop_gates._verify_on_stop_nudge", lambda agent: None)
    monkeypatch.setattr("agent.turn_stop_gates._pre_verify_nudge", lambda *args, **kwargs: None)
    monkeypatch.setattr("agent.turn_stop_gates._kanban_stop_nudge", lambda *args, **kwargs: None)


def _run_gate(agent, text, *, tools=None):
    final_msg = {"role": "assistant", "content": text, "finish_reason": "stop"}
    if tools:
        final_msg["tool_calls"] = tools
    messages = []
    verdict = apply_stop_gates(
        agent,
        final_msg,
        final_response=text,
        messages=messages,
        conversation_history=None,
        pending_verification_response=None,
        pending_verification_response_previewed=None,
    )
    return verdict, final_msg, messages


def test_stop_gate_nudges_once_and_leaves_assistant_unflagged(monkeypatch):
    _silence_earlier_gates(monkeypatch)
    agent = _GateAgent()
    verdict, final_msg, messages = _run_gate(agent, GOLD_TEXT)
    assert verdict.continue_turn is True
    assert verdict.final_response is None
    assert final_msg["finish_reason"] == "false_stop_continue"
    assert "_false_stop_synthetic" not in final_msg
    assert messages[-1]["role"] == "user"
    assert messages[-1]["_false_stop_synthetic"] is True
    assert agent._false_stop_continuations == 1
    assert agent.flushed == 1

    again, _, messages2 = _run_gate(agent, GOLD_TEXT)
    assert again.continue_turn is False
    assert messages2 == []


def test_stop_gate_releases_when_tools_present(monkeypatch):
    _silence_earlier_gates(monkeypatch)
    agent = _GateAgent()
    verdict, final_msg, messages = _run_gate(agent, GOLD_TEXT, tools=[{"id": "call_1"}])
    assert verdict.continue_turn is False
    assert final_msg["finish_reason"] == "stop"
    assert messages == []

