from agent import advisor


class _Agent:
    def __init__(self, cfg):
        self._advisor_config = cfg
        self.session_id = "sess-advisor"
        self.model = "stub-model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.api_key = ""
        self.api_mode = "chat_completions"


def _messages(final="done"):
    return [
        {"role": "user", "content": "edit the file and verify"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "terminal", "arguments": "pytest tests/foo.py"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "1 passed"},
        {"role": "assistant", "content": final},
    ]


def test_disabled_advisor_does_not_call_llm(monkeypatch):
    def boom(**kwargs):
        raise AssertionError("advisor LLM should not be called")

    monkeypatch.setattr(advisor, "_call_aux_llm", boom)
    result = advisor.run_final_advisor_gate(
        _Agent({"enabled": False}),
        messages=_messages(),
        final_response="done",
        api_call_count=2,
        original_user_message="edit the file",
        turn_exit_reason="text_response(finish_reason=stop)",
    )
    assert result.final_response == "done"
    assert result.receipt is None
    assert result.response_changed is False



def test_actionful_only_runs_for_current_tool_turn_without_final_assistant(monkeypatch):
    captured = {}

    def fake_call(**kwargs):
        captured["payload"] = kwargs["messages"][1]["content"]
        return '{"verdict":"PASS","findings":[],"summary":"ok"}'

    monkeypatch.setattr(advisor, "_call_aux_llm", fake_call)
    monkeypatch.setattr(advisor, "_persist_receipt", lambda agent, receipt: None)
    msgs = [
        {"role": "user", "content": "previous work"},
        {"role": "assistant", "content": "previous done"},
        {"role": "user", "content": "run a check"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "terminal", "arguments": "pytest"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "1 passed"},
    ]
    result = advisor.run_final_advisor_gate(
        _Agent({"enabled": True, "mode": "observe"}),
        messages=msgs,
        final_response="check passed",
        api_call_count=1,
        original_user_message="run a check",
        turn_exit_reason="text_response(finish_reason=stop)",
    )
    assert result.receipt is not None
    assert "previous done" not in captured["payload"]
    assert "run a check" in captured["payload"]
    assert "1 passed" in captured["payload"]


def test_actionful_only_ignores_tool_calls_from_prior_turns(monkeypatch):
    def boom(**kwargs):
        raise AssertionError("advisor LLM should not run for current simple turn")

    monkeypatch.setattr(advisor, "_call_aux_llm", boom)
    msgs = [
        {"role": "user", "content": "previous work"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "old", "function": {"name": "terminal", "arguments": "ls"}}]},
        {"role": "tool", "tool_call_id": "old", "content": "ok"},
        {"role": "assistant", "content": "previous done"},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "you are welcome"},
    ]
    result = advisor.run_final_advisor_gate(
        _Agent({"enabled": True, "mode": "final_gate"}),
        messages=msgs,
        final_response="you are welcome",
        api_call_count=1,
        original_user_message="thanks",
        turn_exit_reason="text_response(finish_reason=stop)",
    )
    assert result.receipt is None
    assert result.final_response == "you are welcome"


def test_observe_mode_records_receipt_but_does_not_change_response(monkeypatch):
    receipts = []
    monkeypatch.setattr(
        advisor,
        "_call_aux_llm",
        lambda **kwargs: '{"verdict":"CHANGES_REQUIRED","findings":[{"message":"missing test evidence"}],"summary":"needs caveat"}',
    )
    monkeypatch.setattr(advisor, "_persist_receipt", lambda agent, receipt: receipts.append(receipt))
    msgs = _messages("all done")
    result = advisor.run_final_advisor_gate(
        _Agent({"enabled": True, "mode": "observe", "phases": {"final": "always"}}),
        messages=msgs,
        final_response="all done",
        api_call_count=1,
        original_user_message="edit the file",
        turn_exit_reason="text_response(finish_reason=stop)",
    )
    assert result.final_response == "all done"
    assert result.response_changed is False
    assert result.receipt is not None
    assert result.receipt["verdict"] == "CHANGES_REQUIRED"
    assert receipts and receipts[0]["enforced"] is False
    assert msgs[-1]["content"] == "all done"



def test_payload_redacts_secret_like_values_before_advisor_call(monkeypatch):
    captured = {}

    def fake_call(**kwargs):
        captured["payload"] = kwargs["messages"][1]["content"]
        return '{"verdict":"PASS","findings":[],"summary":"ok"}'

    monkeypatch.setattr(advisor, "_call_aux_llm", fake_call)
    monkeypatch.setattr(advisor, "_persist_receipt", lambda agent, receipt: None)
    secret = "sk-" + "a" * 32
    msgs = _messages(f"done with api_key={secret}")
    msgs[1]["tool_calls"][0]["function"]["arguments"] = f"curl -H 'Authorization: Bearer {secret}'"
    msgs[2]["content"] = f"logged token={secret}"
    result = advisor.run_final_advisor_gate(
        _Agent({"enabled": True, "mode": "observe", "phases": {"final": "always"}}),
        messages=msgs,
        final_response=f"done with api_key={secret}",
        api_call_count=1,
        original_user_message=f"use token={secret}",
        turn_exit_reason="text_response(finish_reason=stop)",
    )
    assert result.receipt is not None
    assert secret not in captured["payload"]
    assert "[redacted]" in captured["payload"]


def test_final_gate_repairs_changes_required_and_updates_last_assistant(monkeypatch):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs["messages"][0]["content"])
        if len(calls) == 1:
            return '{"verdict":"CHANGES_REQUIRED","findings":[{"message":"do not claim deploy"}],"summary":"repair"}'
        return "Corrected: tests passed locally; deploy not verified."

    monkeypatch.setattr(advisor, "_call_aux_llm", fake_call)
    monkeypatch.setattr(advisor, "_persist_receipt", lambda agent, receipt: None)
    msgs = _messages("Done; deployed and verified.")
    result = advisor.run_final_advisor_gate(
        _Agent({
            "enabled": True,
            "mode": "final_gate",
            "max_repair_iterations": 1,
            "phases": {"final": "always"},
        }),
        messages=msgs,
        final_response="Done; deployed and verified.",
        api_call_count=1,
        original_user_message="edit the file",
        turn_exit_reason="text_response(finish_reason=stop)",
    )
    assert result.repaired is True
    assert result.response_changed is True
    assert result.turn_exit_reason == "advisor_repaired"
    assert result.final_response == "Corrected: tests passed locally; deploy not verified."
    assert msgs[-1]["content"] == result.final_response
    assert len(calls) == 2



def test_rewrite_appends_final_assistant_if_history_only_has_tool_call_assistant(monkeypatch):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs["messages"][0]["content"])
        if len(calls) == 1:
            return '{"verdict":"CHANGES_REQUIRED","findings":[{"message":"add caveat"}],"summary":"repair"}'
        return "Repaired final response"

    monkeypatch.setattr(advisor, "_call_aux_llm", fake_call)
    monkeypatch.setattr(advisor, "_persist_receipt", lambda agent, receipt: None)
    msgs = [
        {"role": "user", "content": "run a check"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "terminal", "arguments": "pytest"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "1 passed"},
    ]
    result = advisor.run_final_advisor_gate(
        _Agent({
            "enabled": True,
            "mode": "final_gate",
            "max_calls_per_turn": 2,
            "max_repair_iterations": 1,
            "phases": {"final": "always"},
        }),
        messages=msgs,
        final_response="Draft final response",
        api_call_count=2,
        original_user_message="run a check",
        turn_exit_reason="max_iterations_reached(2/2)",
    )
    assert result.final_response == "Repaired final response"
    assert msgs[-1] == {"role": "assistant", "content": "Repaired final response"}
    assert msgs[1]["tool_calls"]


def test_final_gate_blocks_critical_verdict(monkeypatch):
    monkeypatch.setattr(
        advisor,
        "_call_aux_llm",
        lambda **kwargs: '{"verdict":"BLOCK","findings":[{"message":"secret exposure"}],"summary":"blocked"}',
    )
    monkeypatch.setattr(advisor, "_persist_receipt", lambda agent, receipt: None)
    msgs = _messages("token is sk-abc123")
    result = advisor.run_final_advisor_gate(
        _Agent({"enabled": True, "mode": "final_gate", "phases": {"final": "always"}}),
        messages=msgs,
        final_response="token is sk-abc123",
        api_call_count=1,
        original_user_message="show token",
        turn_exit_reason="text_response(finish_reason=stop)",
    )
    assert result.blocked is True
    assert result.turn_exit_reason == "advisor_blocked"
    assert "Advisor block reason" in result.final_response
    assert msgs[-1]["content"] == result.final_response


def test_final_gate_fail_closed_blocks_when_advisor_unavailable(monkeypatch):
    def broken_call(**kwargs):
        raise RuntimeError("aux down")

    monkeypatch.setattr(advisor, "_call_aux_llm", broken_call)
    monkeypatch.setattr(advisor, "_persist_receipt", lambda agent, receipt: None)
    msgs = _messages("draft")
    result = advisor.run_final_advisor_gate(
        _Agent({
            "enabled": True,
            "mode": "final_gate",
            "failure_policy": "fail_closed",
            "phases": {"final": "always"},
        }),
        messages=msgs,
        final_response="draft",
        api_call_count=1,
        original_user_message="do risky work",
        turn_exit_reason="text_response(finish_reason=stop)",
    )
    assert result.unavailable is True
    assert result.blocked is True
    assert result.turn_exit_reason == "advisor_unavailable_fail_closed"
    assert msgs[-1]["content"] == result.final_response


def test_extract_json_object_accepts_markdown_wrapped_json():
    data = advisor._extract_json_object('```json\n{"verdict":"PASS","findings":[]}\n```')
    assert data["verdict"] == "PASS"
    assert data["findings"] == []
