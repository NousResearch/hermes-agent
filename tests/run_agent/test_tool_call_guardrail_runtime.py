"""Runtime tests for tool-call loop guardrails."""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(
    *tool_names: str,
    max_iterations: int = 10,
    config: dict | None = None,
    platform: str | None = None,
) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("hermes_cli.config.load_config_readonly", return_value=config or {}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform=platform or "cli",
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _seed_exact_failures(agent: AIAgent, tool_name: str, args: dict, count: int = 2) -> None:
    for _ in range(count):
        agent._tool_guardrails.after_call(
            tool_name,
            args,
            json.dumps({"error": "boom"}),
            failed=True,
        )


def _hard_stop_config(**overrides) -> dict:
    cfg = {
        "tool_loop_guardrails": {
            "warnings_enabled": True,
            "hard_stop_enabled": True,
            "hard_stop_after": {
                "exact_failure": 2,
                "same_tool_failure": 8,
                "idempotent_no_progress": 5,
            },
        }
    }
    cfg["tool_loop_guardrails"].update(overrides)
    return cfg


def test_gateway_platform_uses_hard_stop_default_without_cli_opt_in():
    agent = _make_agent("web_search", platform="telegram")
    args = {"query": "same"}

    _seed_exact_failures(agent, "web_search", args, count=5)

    decision = getattr(agent, "_tool_guardrails").before_call("web_search", args)
    assert decision.action == "block"
    assert decision.code == "repeated_exact_failure_block"


@pytest.mark.parametrize("platform", ["desktop", "acp"])
def test_interactive_platforms_keep_warning_only_default(platform):
    agent = _make_agent("web_search", platform=platform)
    args = {"query": "same"}

    _seed_exact_failures(agent, "web_search", args, count=5)

    decision = getattr(agent, "_tool_guardrails").before_call("web_search", args)
    assert decision.action == "allow"
    assert decision.code == "allow"


def test_default_sequential_path_warns_repeated_exact_failure_without_blocking_execution():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-soft")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_called_once()
    assert len(starts) == 1
    assert any(event[0][0] == "tool.completed" for event in progress)
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-soft"
    assert "repeated_exact_failure_warning" in messages[0]["content"]
    assert "repeated_exact_failure_block" not in messages[0]["content"]
    assert agent._tool_guardrail_halt_decision is None


def test_config_enabled_hard_stop_blocks_repeated_exact_failure_before_execution():
    agent = _make_agent("web_search", config=_hard_stop_config())
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-block")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert starts == []
    assert progress == []
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-block"
    assert "repeated_exact_failure_block" in messages[0]["content"]


def test_sequential_after_call_appends_guidance_to_tool_result_without_extra_messages():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args, count=1)
    tc = _mock_tool_call("web_search", json.dumps(args), "c-warn")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    assert [m["role"] for m in messages] == ["tool"]
    assert messages[0]["tool_call_id"] == "c-warn"
    assert "Tool loop warning" in messages[0]["content"]
    assert "repeated_exact_failure_warning" in messages[0]["content"]


def test_same_tool_failure_warning_tells_model_to_recover_with_tools():
    agent = _make_agent("terminal")
    guardrails = getattr(agent, "_tool_guardrails")
    guardrails.after_call(
        "terminal",
        {"command": "bad-1"},
        json.dumps({"exit_code": 1}),
        failed=True,
    )
    guardrails.after_call(
        "terminal",
        {"command": "bad-2"},
        json.dumps({"exit_code": 1}),
        failed=True,
    )
    tc = _mock_tool_call("terminal", json.dumps({"command": "bad-3"}), "c-recover")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"exit_code": 1})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    content = messages[0]["content"]
    assert "same_tool_failure_warning" in content
    assert "Do not switch to text-only replies" in content
    assert "keep using tools" in content
    assert "pwd && ls -la" in content
    assert "absolute path" in content
    assert "different tool" in content


def test_config_enabled_hard_stop_concurrent_path_does_not_submit_blocked_calls_and_preserves_result_order():
    agent = _make_agent("web_search", config=_hard_stop_config())
    blocked_args = {"query": "blocked"}
    allowed_args = {"query": "allowed"}
    _seed_exact_failures(agent, "web_search", blocked_args)
    starts = []
    progress_events = []
    agent.tool_start_callback = lambda tool_call_id, name, args: starts.append((tool_call_id, name, args))
    agent.tool_progress_callback = lambda event, name, preview, args, **kw: progress_events.append((event, name, args, kw))
    calls = [
        _mock_tool_call("web_search", json.dumps(blocked_args), "c-block"),
        _mock_tool_call("web_search", json.dumps(allowed_args), "c-allow"),
    ]
    msg = SimpleNamespace(content="", tool_calls=calls)
    messages = []
    executed = []

    def fake_handle(name, args, task_id, **kwargs):
        executed.append((name, args, kwargs["tool_call_id"]))
        return json.dumps({"ok": args["query"]})

    with patch("run_agent.handle_function_call", side_effect=fake_handle):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    assert executed == [("web_search", allowed_args, "c-allow")]
    assert [m["tool_call_id"] for m in messages] == ["c-block", "c-allow"]
    assert "repeated_exact_failure_block" in messages[0]["content"]
    assert json.loads(messages[1]["content"]) == {"ok": "allowed"}
    assert starts == [("c-allow", "web_search", allowed_args)]
    started_events = [event for event in progress_events if event[0] == "tool.started"]
    completed_events = [event for event in progress_events if event[0] == "tool.completed"]
    assert started_events == [("tool.started", "web_search", allowed_args, {})]
    assert len(completed_events) == 1
    assert completed_events[0][1] == "web_search"


def test_relay_rewrite_precedes_sequential_policy_approval_checkpoint_and_dispatch():
    agent = _make_agent("write_file")
    original_args = {"path": "/original/path", "content": "old"}
    final_args = {"path": "/approved/path", "content": "new"}
    tc = _mock_tool_call("write_file", json.dumps(original_args), "c-rewrite")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    observed = {
        "plugin": [],
        "guardrail": [],
        "approval": [],
        "checkpoint": [],
        "start": [],
        "dispatch": [],
    }

    original_before_call = agent._tool_guardrails.before_call

    def observe_guardrail(name, args):
        observed["guardrail"].append((name, dict(args)))
        return original_before_call(name, args)

    def relay_execute(name, args, callback, **kwargs):
        del name, args, kwargs
        return callback(dict(final_args)), dict(final_args)

    def observe_plugin(name, args, **kwargs):
        del kwargs
        observed["plugin"].append((name, dict(args)))
        return (None, None)

    def observe_approval(name, args):
        observed["approval"].append((name, dict(args)))
        return None

    def dispatch(name, args, task_id, **kwargs):
        del task_id, kwargs
        observed["dispatch"].append((name, dict(args)))
        return json.dumps({"ok": True})

    agent._checkpoint_mgr = SimpleNamespace(
        enabled=True,
        get_working_dir_for_path=lambda path: path,
        ensure_checkpoint=lambda path, reason: observed["checkpoint"].append(
            (path, reason)
        ),
    )
    agent.tool_start_callback = lambda _call_id, name, args: observed["start"].append(
        (name, dict(args))
    )

    with (
        patch("agent.relay_tools.execute", side_effect=relay_execute),
        patch(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            side_effect=observe_plugin,
        ),
        patch.object(agent._tool_guardrails, "before_call", side_effect=observe_guardrail),
        patch(
            "acp_adapter.edit_approval.maybe_require_edit_approval",
            side_effect=observe_approval,
        ),
        patch("model_tools.registry.dispatch", side_effect=dispatch),
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    expected = [("write_file", final_args)]
    assert observed["plugin"] == expected
    assert observed["guardrail"] == expected
    assert observed["approval"] == expected
    assert observed["start"] == expected
    assert observed["dispatch"] == expected
    assert observed["checkpoint"] == [
        ("/approved/path", "before write_file")
    ]


def test_relay_rewrite_is_guarded_before_dispatch_in_concurrent_path():
    agent = _make_agent("web_search", config=_hard_stop_config())
    original_args = {"query": "original"}
    blocked_args = {"query": "blocked"}
    _seed_exact_failures(agent, "web_search", blocked_args)
    tc = _mock_tool_call("web_search", json.dumps(original_args), "c-rewrite-block")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    starts = []

    def relay_execute(name, args, callback, **kwargs):
        del name, args, kwargs
        return callback(dict(blocked_args)), dict(blocked_args)

    agent.tool_start_callback = lambda *args: starts.append(args)
    with (
        patch("agent.relay_tools.execute", side_effect=relay_execute),
        patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as dispatch,
    ):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    dispatch.assert_not_called()
    assert starts == []
    assert "repeated_exact_failure_block" in messages[0]["content"]


def test_plugin_pre_tool_block_wins_without_counting_as_toolguard_block():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    tc = _mock_tool_call("web_search", json.dumps(args), "c-plugin")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with (
        patch(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            return_value=("plugin policy", None),
        ),
        patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc,
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert "plugin policy" in messages[0]["content"]
    assert agent._tool_guardrails.before_call("web_search", args).action == "allow"


def test_default_run_conversation_warns_without_guardrail_halt():
    agent = _make_agent("web_search", max_iterations=10)
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 4)
    ]
    responses.append(_mock_response(content="done", finish_reason="stop", tool_calls=None))
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert mock_hfc.call_count == 3
    assert result["turn_exit_reason"].startswith("text_response")
    assert "guardrail" not in result
    assert result["final_response"] == "done"
    tool_contents = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert any("repeated_exact_failure_warning" in content for content in tool_contents)




def test_guardrail_halt_emits_final_response_through_stream_delta_callback():
    """Regression for #30770: when the guardrail halts the loop, the
    synthesized halt message must be pushed through ``stream_delta_callback``
    so SSE/TUI clients see why the agent stopped instead of a silent stream
    close.  Without this the chat-completions SSE writer drains an empty
    queue and emits a finish chunk with zero content (indistinguishable
    from a crash for Open WebUI and similar clients).
    """
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 10)
    ]
    agent.client.chat.completions.create.side_effect = responses

    deltas: list = []
    agent.stream_delta_callback = lambda d: deltas.append(d)
    # The mocked client returns SimpleNamespace responses which aren't
    # iterable as streaming chunks; force the non-streaming code path so
    # the guardrail-halt branch is reached without engaging the real
    # streaming machinery.
    agent._disable_streaming = True

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert result["turn_exit_reason"] == "guardrail_halt"
    halt_text = result["final_response"]
    assert "stopped retrying" in halt_text

    # The halt message must have been pushed through the callback at least
    # once.  Empty-queue SSE writers were the bug — clients saw no content
    # delta before the finish chunk.
    text_deltas = [d for d in deltas if isinstance(d, str)]
    assert halt_text in text_deltas, (
        f"halt message was never streamed; callback only saw {deltas!r}"
    )



# ── guardrail-halt wrap-up ────────────────────────────────────────────────
#
# A halt used to end the turn on a fixed "I stopped retrying X" string, which
# discards everything the model actually observed. The wrap-up gives it one
# tool-free call to write the real answer. Two invariants matter as much as
# the behavior: the toolset narrows for exactly ONE call (and never by
# mutating agent.tools), and no synthetic user message is injected.


def _model_that_loops_until_tools_are_withdrawn(summary: str):
    """Stand-in for a real model: keeps re-issuing the same failing call while
    tools are offered, and can only answer in prose once they are withdrawn.

    Keying off the ``tools`` kwarg (rather than a fixed response list) is the
    point — it ties the wrap-up's observable contract, "no tools offered", to
    the behavior it is supposed to buy, "the model writes the answer".
    """
    counter = {"n": 0}

    def _respond(*_args, **kwargs):
        if not kwargs.get("tools"):
            return _mock_response(content=summary, finish_reason="stop")
        counter["n"] += 1
        return _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call("web_search", json.dumps({"query": "same"}), f"c{counter['n']}")
            ],
        )

    return _respond


def _run_halting_turn(agent, summary="Here is what I found before I was stopped."):
    agent.client.chat.completions.create.side_effect = (
        _model_that_loops_until_tools_are_withdrawn(summary)
    )
    agent._disable_streaming = True
    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")
    return result


def test_wrapup_suppresses_tools_for_exactly_one_call():
    """The halt call offers no tools; nothing else in the turn is affected,
    and agent.tools itself is never mutated (a changed toolset would follow
    the conversation into later turns and break the prompt cache)."""
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())
    tools_before = list(agent.tools)

    result = _run_halting_turn(agent)

    sent_tools = [
        kw.get("tools") for _, kw in agent.client.chat.completions.create.call_args_list
    ]
    toolless = [i for i, t in enumerate(sent_tools) if not t]
    assert len(toolless) == 1, f"expected exactly one tool-free call, got {toolless}"
    # ...and it is the last call of the turn.
    assert toolless[0] == len(sent_tools) - 1
    assert agent.tools == tools_before
    assert result["final_response"]


def test_wrapup_injects_no_synthetic_user_message():
    """AGENTS.md message hygiene: no user message may be fabricated mid-loop,
    and roles must never repeat back to back."""
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())

    _run_halting_turn(agent)

    _, last_kwargs = agent.client.chat.completions.create.call_args_list[-1]
    roles = [m["role"] for m in last_kwargs["messages"]]
    # Exactly one user message: the real prompt.
    assert roles.count("user") == 1, roles
    # The wrap-up call is entered straight off tool results.
    assert roles[-1] == "tool", roles[-3:]
    for a, b in zip(roles, roles[1:]):
        assert not (a == b == "user"), f"consecutive user messages: {roles}"


def test_wrapup_final_response_is_the_models_summary_not_the_canned_string():
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())

    result = _run_halting_turn(agent, summary="I hit a 404 wall; here are partial results.")

    assert result["final_response"] == "I hit a 404 wall; here are partial results."
    assert "stopped retrying" not in result["final_response"]


def test_wrapup_preserves_guardrail_metadata_on_the_turn_result():
    """The wrap-up clears the live halt attribute so the loop can make the
    final call; the turn result must still report the guardrail."""
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())

    result = _run_halting_turn(agent)

    assert result.get("guardrail"), "guardrail metadata lost through the wrap-up"


def _always_calls_tools():
    """A model that never stops calling tools, tool-free wrap-up included.

    Unbounded on purpose: a fixed response list would let the turn end by
    running out of seeded responses, which looks identical to the branch the
    test means to pin.
    """
    counter = {"n": 0}

    def _respond(*_args, **_kwargs):
        counter["n"] += 1
        return _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call("web_search", json.dumps({"query": "same"}), f"c{counter['n']}")
            ],
        )

    return _respond


def test_second_halt_after_wrapup_falls_back_to_canned_response():
    """Only one wrap-up per turn — a model that keeps calling tools through
    the wrap-up gets the bounded canned halt, so the loop still terminates."""
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())
    agent.client.chat.completions.create.side_effect = _always_calls_tools()
    agent._disable_streaming = True
    statuses: list[str] = []
    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_emit_status", side_effect=lambda m, *a, **k: statuses.append(str(m))),
    ):
        result = agent.run_conversation("search repeatedly")

    assert result["turn_exit_reason"] == "guardrail_halt"
    assert "stopped retrying" in result["final_response"]
    # Termination came from the second-halt branch, not from exhausting the
    # iteration budget: the halt fired well inside it...
    calls = len(agent.client.chat.completions.create.call_args_list)
    assert calls < agent.max_iterations, calls
    # ...and exactly two halts were recorded — the wrap-up, then the fallback.
    halts = [s for s in statuses if "Tool guardrail halted" in s]
    assert len(halts) == 2, halts
    assert "requesting a final summary" in halts[0]
    assert "requesting a final summary" not in halts[1]


# ── the halt landing on the last budgeted call ────────────────────────────
#
# The wrap-up ``continue``s to the top of the loop, so if the halt lands on
# the last call the loop condition is already false. Without the grace flag
# the turn would fall out of the loop having made neither the wrap-up call
# nor the canned fallback under it, ending with no ``guardrail_halt`` at all.
# (Only the iteration edge is exercised here: build_turn_context rebuilds
# ``iteration_budget`` from ``max_iterations`` every turn, so the budget can
# only run out at or after that same point.)


def _assert_wrapped_up_at_the_edge(agent, result, budget_summary):
    """The summary came from the wrap-up, not from the post-loop
    max-iterations fallback.

    Both end in a tool-free call, so the discriminator has to be which code
    path produced it: ``_handle_max_iterations`` must never run, and its
    signature -- a synthetic user message asking for a summary -- must be
    absent from the request actually sent.
    """
    calls = agent.client.chat.completions.create.call_args_list
    sent_tools = [kw.get("tools") for _, kw in calls]
    assert not sent_tools[-1], "the wrap-up call never happened"
    budget_summary.assert_not_called()
    roles = [m["role"] for m in calls[-1][1]["messages"]]
    assert roles.count("user") == 1, roles
    assert roles[-1] == "tool", roles[-3:]
    assert result["final_response"] == "Partial results before the stop."
    assert result.get("guardrail"), "guardrail metadata lost at the budget edge"


def test_wrapup_still_runs_when_halt_lands_on_the_last_iteration():
    """max_iterations is reached by the halting call itself."""
    # 3 tool-calling iterations is exactly what this config needs to halt.
    # 3 tool-calling iterations is exactly what this config needs to halt.
    agent = _make_agent("web_search", max_iterations=3, config=_hard_stop_config())

    with patch.object(agent, "_handle_max_iterations") as budget_summary:
        result = _run_halting_turn(agent, summary="Partial results before the stop.")

    _assert_wrapped_up_at_the_edge(agent, result, budget_summary)


def test_canned_fallback_still_reached_when_halt_lands_on_the_last_iteration():
    """And when the wrap-up call itself halts again at the budget edge, the
    turn still ends on the canned halt rather than an empty response."""
    agent = _make_agent("web_search", max_iterations=3, config=_hard_stop_config())
    agent.client.chat.completions.create.side_effect = _always_calls_tools()
    agent._disable_streaming = True
    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert result["turn_exit_reason"] == "guardrail_halt"
    assert "stopped retrying" in result["final_response"]
