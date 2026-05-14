from agent.trajectory import build_agent_run_trace


def _assistant_tool(call_id, name, args):
    import json
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def _tool(call_id, name, content):
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


def _trace(messages, **overrides):
    kwargs = {
        "messages": messages,
        "run_id": "run-1",
        "started_at": "2026-05-14T12:00:00Z",
        "ended_at": "2026-05-14T12:00:03Z",
        "origin": "discord",
        "task_summary": "Implement trace support",
        "completed": True,
        "interrupted": False,
        "turn_exit_reason": "final_response",
    }
    kwargs.update(overrides)
    return build_agent_run_trace(**kwargs)


def test_run_trace_aggregates_tools_side_effects_files_and_verifier():
    messages = [
        {"role": "user", "content": "patch and test the code"},
        _assistant_tool("c1", "write_file", {"path": "agent/foo.py", "content": "x"}),
        _tool("c1", "write_file", "ok"),
        _assistant_tool("c2", "terminal", {"command": "python -m pytest tests/agent/test_foo.py"}),
        _tool("c2", "terminal", "2 passed exit_code: 0"),
    ]

    trace = _trace(messages)

    assert trace["schema_version"] == "agent_run_trace.v1"
    assert trace["origin"] == "discord"
    assert trace["intent_type"] == "code_patch"
    assert trace["risk_level"] == "medium"
    assert {t["name"]: t for t in trace["tools_used"]}["write_file"]["side_effect"] is True
    assert "terminal" in trace["side_effect_tools"]
    assert "agent/foo.py" in trace["files_touched"]
    assert "tests/agent/test_foo.py" in trace["files_touched"]
    assert trace["verifier_detected"] is True
    assert trace["verifier"] == {
        "type": "test",
        "command_or_endpoint": "python -m pytest tests/agent/test_foo.py",
        "result": "pass",
    }
    assert trace["outcome"] == "success"


def test_run_trace_marks_external_actions_high_risk():
    messages = [
        _assistant_tool("c1", "terminal", {"command": "git push origin main"}),
        _tool("c1", "terminal", "pushed exit_code: 0"),
    ]

    trace = _trace(messages, task_summary="deploy change")

    assert trace["risk_level"] == "high"
    assert trace["external_actions"] == ["git_push"]
    assert trace["promotion_hint"]["skill"] is True


def test_run_trace_detects_recovered_tool_errors_and_eval_promotion():
    messages = [
        _assistant_tool("c1", "terminal", {"command": "python -m pytest tests/foo.py"}),
        _tool("c1", "terminal", "Traceback: boom\nexit_code: 1"),
        {"role": "assistant", "content": "Recovered and fixed it."},
    ]

    trace = _trace(messages, completed=True)

    assert trace["failure_mode"] == "tool_error_recovered"
    assert trace["verifier"]["result"] == "fail"
    assert trace["promotion_hint"]["eval_case"] is True
    assert trace["outcome"] == "partial"


def test_run_trace_privacy_and_skill_labels():
    messages = [
        {"role": "user", "content": "Use memory. email me@example.com token [REDACTED]"},
        _assistant_tool("c1", "skill_view", {"name": "systematic-debugging"}),
        _tool("c1", "skill_view", "loaded"),
        _assistant_tool("c2", "memory", {"action": "add", "target": "memory", "content": "fact"}),
        _tool("c2", "memory", "saved"),
    ]

    trace = _trace(messages)

    assert trace["skills_loaded"] == ["systematic-debugging"]
    assert trace["privacy_gate"]["secret_redaction_applied"] is True
    assert trace["privacy_gate"]["durable_memory_write_detected"] is True
    assert trace["privacy_gate"]["pii_risk"] == "possible"


def test_run_trace_marks_max_iterations_failure_mode():
    trace = _trace([], completed=False, turn_exit_reason="max_iterations_reached(90/90)")

    assert trace["failure_mode"] == "max_iterations"
    assert trace["outcome"] == "failed"
    assert trace["promotion_hint"]["eval_case"] is True
