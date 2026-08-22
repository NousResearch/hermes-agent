"""Runtime tests for tool-call loop guardrails."""

import json
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    _MAX_PRUNED_SKILL_MARKERS,
    _skill_pruned_marker,
    _summarize_tool_result,
)
from agent.tool_executor import refresh_pending_skill_reloads
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


def _make_agent(*tool_names: str, max_iterations: int = 10, config: dict | None = None) -> AIAgent:
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


def test_pruned_skill_blocks_other_tools_before_dispatch():
    agent = _make_agent("skill_view", "web_search")
    refresh_pending_skill_reloads(
        agent,
        [
            {
                "role": "tool",
                "content": _summarize_tool_result(
                    "skill_view", '{"name":"pdf"}', "x" * 6000
                ),
            }
        ],
    )
    tc = _mock_tool_call("web_search", json.dumps({"query": "blocked"}), "c-pruned")
    messages = []

    with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as dispatch:
        agent._execute_tool_calls_sequential(
            SimpleNamespace(content="", tool_calls=[tc]), messages, "task-1"
        )

    dispatch.assert_not_called()
    assert "skill_view(name='pdf')" in messages[0]["content"]
    assert agent._pending_skill_reloads == ["pdf"]


def test_reload_success_does_not_release_later_tool_in_same_sequential_batch():
    agent = _make_agent("skill_view", "web_search")
    refresh_pending_skill_reloads(
        agent, [{"role": "user", "content": _skill_pruned_marker("pdf")}]
    )
    calls = [
        _mock_tool_call("skill_view", json.dumps({"name": "pdf"}), "c-reload"),
        _mock_tool_call(
            "web_search", json.dumps({"query": "must wait"}), "c-after-reload"
        ),
    ]
    messages = []
    executed = []

    def dispatch(name, args, task_id, **kwargs):
        del task_id, kwargs
        executed.append((name, args))
        if name == "skill_view":
            return json.dumps({"success": True, "name": "pdf", "content": "rules"})
        return json.dumps({"ok": True})

    with patch("run_agent.handle_function_call", side_effect=dispatch):
        agent._execute_tool_calls_sequential(
            SimpleNamespace(content="", tool_calls=calls), messages, "task-1"
        )

    assert executed == [("skill_view", {"name": "pdf"})]
    assert agent._pending_skill_reloads == []
    assert "skill_view(name='pdf')" in messages[1]["content"]


def test_reload_success_does_not_race_later_tool_in_same_concurrent_batch():
    agent = _make_agent("skill_view", "web_search")
    refresh_pending_skill_reloads(
        agent, [{"role": "user", "content": _skill_pruned_marker("pdf")}]
    )
    calls = [
        _mock_tool_call("skill_view", json.dumps({"name": "pdf"}), "c-reload"),
        _mock_tool_call(
            "web_search", json.dumps({"query": "must wait"}), "c-after-reload"
        ),
    ]
    messages = []
    executed = []
    reload_finished = threading.Event()

    def relay_execute(name, args, callback, **kwargs):
        del kwargs
        if name == "web_search":
            assert reload_finished.wait(timeout=5)
        result = callback(dict(args))
        if name == "skill_view":
            reload_finished.set()
        return result, dict(args)

    def dispatch(name, args, task_id, **kwargs):
        del task_id, kwargs
        executed.append((name, args))
        if name == "skill_view":
            return json.dumps({"success": True, "name": "pdf", "content": "rules"})
        return json.dumps({"ok": True})

    with (
        patch("agent.relay_tools.execute", side_effect=relay_execute),
        patch("run_agent.handle_function_call", side_effect=dispatch),
    ):
        agent._execute_tool_calls_concurrent(
            SimpleNamespace(content="", tool_calls=calls), messages, "task-1"
        )

    assert executed == [("skill_view", {"name": "pdf"})]
    assert agent._pending_skill_reloads == []
    assert "skill_view(name='pdf')" in messages[1]["content"]


def test_reload_boundary_survives_mixed_batch_segmentation():
    agent = _make_agent("skill_view", "write_file")
    refresh_pending_skill_reloads(
        agent, [{"role": "user", "content": _skill_pruned_marker("pdf")}]
    )
    calls = [
        _mock_tool_call("skill_view", json.dumps({"name": "pdf"}), "c-reload"),
        _mock_tool_call(
            "write_file",
            json.dumps({"path": "/tmp/must-wait", "content": "blocked"}),
            "c-after-reload",
        ),
    ]
    messages = []
    executed = []

    def dispatch(name, args, task_id, **kwargs):
        del task_id, kwargs
        executed.append((name, args))
        if name == "skill_view":
            return json.dumps({"success": True, "name": "pdf", "content": "rules"})
        return json.dumps({"success": True})

    with patch("run_agent.handle_function_call", side_effect=dispatch):
        agent._execute_tool_calls(
            SimpleNamespace(content="", tool_calls=calls), messages, "task-1"
        )

    assert executed == [("skill_view", {"name": "pdf"})]
    assert agent._pending_skill_reloads == []
    assert "skill_view(name='pdf')" in messages[1]["content"]


def test_successful_skill_view_roundtrip_clears_pending_reload_but_failure_does_not():
    agent = _make_agent("skill_view", "web_search")
    markers = [
        {"role": "user", "content": _skill_pruned_marker(f"skill-{i}")}
        for i in range(_MAX_PRUNED_SKILL_MARKERS + 3)
    ]
    refresh_pending_skill_reloads(agent, markers)
    assert len(agent._pending_skill_reloads) == _MAX_PRUNED_SKILL_MARKERS
    assert agent._pending_skill_reloads[0] == "skill-0"

    failed = _mock_tool_call(
        "skill_view", json.dumps({"name": "skill-0"}), "c-reload-failed"
    )
    with patch(
        "run_agent.handle_function_call",
        return_value=json.dumps({"success": False, "error": "missing"}),
    ):
        agent._execute_tool_calls_sequential(
            SimpleNamespace(content="", tool_calls=[failed]), [], "task-1"
        )
    assert agent._pending_skill_reloads[0] == "skill-0"

    loaded = _mock_tool_call(
        "skill_view", json.dumps({"name": "skill-0"}), "c-reload-ok"
    )
    with patch(
        "run_agent.handle_function_call",
        return_value=json.dumps({"success": True, "name": "skill-0", "content": "rules"}),
    ):
        agent._execute_tool_calls_sequential(
            SimpleNamespace(content="", tool_calls=[loaded]), [], "task-1"
        )

    assert "skill-0" not in agent._pending_skill_reloads
    assert agent._pending_skill_reloads[0] == "skill-1"

    # Rebuilding state on the next turn treats the retained marker as
    # historical because the later persisted skill_view result succeeded.
    history = [
        {"role": "user", "content": _skill_pruned_marker("skill-0")},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "c-persisted-reload",
                "type": "function",
                "function": {
                    "name": "skill_view",
                    "arguments": json.dumps({"name": "skill-0"}),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "c-persisted-reload",
            "content": json.dumps(
                {"success": True, "name": "skill-0", "content": "rules"}
            ),
        },
    ]
    assert refresh_pending_skill_reloads(agent, history) == []

    # A later compaction marker for the same skill re-arms the guard.
    history.append({"role": "user", "content": _skill_pruned_marker("skill-0")})
    assert refresh_pending_skill_reloads(agent, history) == ["skill-0"]


def test_pending_reload_refresh_scans_only_appended_tail_and_replaced_history():
    agent = _make_agent("skill_view")
    history = [{"role": "user", "content": _skill_pruned_marker("pdf")}]

    with patch("agent.tool_executor._message_text", wraps=lambda msg: msg["content"]) as text:
        assert refresh_pending_skill_reloads(agent, history) == ["pdf"]
        assert text.call_count == 1

        text.reset_mock()
        assert refresh_pending_skill_reloads(agent, history) == ["pdf"]
        text.assert_not_called()

        history.append({"role": "user", "content": "appended tail"})
        text.reset_mock()
        assert refresh_pending_skill_reloads(agent, history) == ["pdf"]
        assert text.call_count == 1

        replacement = [dict(message) for message in history]
        replacement[0]["content"] = _skill_pruned_marker("xlsx")
        text.reset_mock()
        assert refresh_pending_skill_reloads(agent, replacement) == ["xlsx"]
        assert text.call_count == len(replacement)


def test_pending_reload_refresh_force_full_rescans_after_compression():
    agent = _make_agent("skill_view")
    history = [{"role": "user", "content": _skill_pruned_marker("pdf")}]
    assert refresh_pending_skill_reloads(agent, history) == ["pdf"]

    # Compression may rewrite a retained row in place, preserving both list
    # length and object identity. Its caller must explicitly invalidate the
    # append-only watermark so the rewritten marker is observed.
    history[0]["content"] = _skill_pruned_marker("xlsx")
    assert refresh_pending_skill_reloads(agent, history) == ["pdf"]
    assert refresh_pending_skill_reloads(agent, history, force_full=True) == ["xlsx"]

    history[0]["content"] = _skill_pruned_marker("docx")
    agent.reset_session_state()
    assert refresh_pending_skill_reloads(agent, history) == ["docx"]


def test_pending_reload_tail_scan_matches_skill_call_to_later_result():
    agent = _make_agent("skill_view")
    history = [
        {"role": "user", "content": _skill_pruned_marker("pdf")},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c-tail-reload",
                    "type": "function",
                    "function": {
                        "name": "skill_view",
                        "arguments": json.dumps({"name": "pdf"}),
                    },
                }
            ],
        },
    ]
    assert refresh_pending_skill_reloads(agent, history) == ["pdf"]

    history.append(
        {
            "role": "tool",
            "tool_call_id": "c-tail-reload",
            "content": json.dumps(
                {"success": True, "name": "pdf", "content": "rules"}
            ),
        }
    )
    assert refresh_pending_skill_reloads(agent, history) == []


def test_compression_exception_refreshes_pending_reloads_from_mutated_transcript():
    agent = _make_agent("skill_view")
    messages = [{"role": "user", "content": "before compression"}]
    assert refresh_pending_skill_reloads(agent, messages) == []

    def fail_after_mutating_transcript(*args, **kwargs):
        del args, kwargs
        messages[0]["content"] = _skill_pruned_marker("pdf")
        raise RuntimeError("synthetic compression failure")

    with (
        patch(
            "agent.conversation_compression.compress_context",
            side_effect=fail_after_mutating_transcript,
        ),
        patch(
            "agent.conversation_compression.resolve_context_compression_timeouts",
            return_value=(0, 0),
        ),
    ):
        try:
            agent._compress_context(messages, "system prompt", force=True)
        except RuntimeError as exc:
            assert str(exc) == "synthetic compression failure"
        else:
            raise AssertionError("compression failure was not propagated")

    assert agent._pending_skill_reloads == ["pdf"]


def test_malformed_successful_skill_view_body_keeps_reload_guard_armed():
    agent = _make_agent("skill_view")
    refresh_pending_skill_reloads(
        agent, [{"role": "user", "content": _skill_pruned_marker("pdf")}]
    )
    loaded = _mock_tool_call(
        "skill_view", json.dumps({"name": "pdf"}), "c-malformed-reload"
    )
    messages = []

    with patch(
        "run_agent.handle_function_call",
        return_value=json.dumps(
            {"success": True, "name": "pdf", "content": ["not", "a", "string"]}
        ),
    ):
        agent._execute_tool_calls_sequential(
            SimpleNamespace(content="", tool_calls=[loaded]), messages, "task-1"
        )

    assert agent._pending_skill_reloads == ["pdf"]


def test_support_file_view_does_not_clear_live_or_rehydrated_main_reload(
    tmp_path: Path, monkeypatch
):
    skill_name = "reload-support-boundary"
    hermes_home = tmp_path / "hermes-home"
    skill_dir = hermes_home / "skills" / skill_name
    references_dir = skill_dir / "references"
    references_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "# Main policy\n\nMAIN-POLICY-SENTINEL\n", encoding="utf-8"
    )
    (references_dir / "note.md").write_text(
        "SUPPORT-FILE-SENTINEL\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    agent = _make_agent("skill_view", "web_search")
    refresh_pending_skill_reloads(
        agent, [{"role": "user", "content": _skill_pruned_marker(skill_name)}]
    )
    support_args = {"name": skill_name, "file_path": "references/note.md"}
    support_call = _mock_tool_call(
        "skill_view", json.dumps(support_args), "c-support-only"
    )
    messages = []

    agent._execute_tool_calls_sequential(
        SimpleNamespace(content="", tool_calls=[support_call]),
        messages,
        "task-1",
    )

    assert "SUPPORT-FILE-SENTINEL" in messages[0]["content"]
    assert "MAIN-POLICY-SENTINEL" not in messages[0]["content"]
    assert agent._pending_skill_reloads == [skill_name]

    history = [
        {"role": "user", "content": _skill_pruned_marker(skill_name)},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c-support-only",
                    "type": "function",
                    "function": {
                        "name": "skill_view",
                        "arguments": json.dumps(support_args),
                    },
                }
            ],
        },
        messages[0],
    ]
    assert refresh_pending_skill_reloads(agent, history) == [skill_name]


def test_pending_main_reload_stays_complete_through_dedup_and_result_budgets(
    tmp_path: Path, monkeypatch
):
    skill_name = "reload-full-delivery"
    tail_sentinel = "TAIL-POLICY-SENTINEL"
    hermes_home = tmp_path / "hermes-home"
    skill_dir = hermes_home / "skills" / skill_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "# Full policy\n\n" + ("instruction line\n" * 2_500) + tail_sentinel,
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    agent = _make_agent("skill_view", "web_search")
    call_args = {"name": skill_name}

    # Seed the identical-result tracker with the same successful call. The
    # recovery call below would otherwise be replaced by a result-reference
    # stub before the model could receive the restored policy.
    getattr(agent, "context_compressor").context_length = 200_000
    first_call = _mock_tool_call(
        "skill_view", json.dumps(call_args), "c-before-prune"
    )
    agent._execute_tool_calls_sequential(
        SimpleNamespace(content="", tool_calls=[first_call]), [], "task-1"
    )

    refresh_pending_skill_reloads(
        agent, [{"role": "user", "content": _skill_pruned_marker(skill_name)}]
    )
    # A 16K-token context produces a 9,600-char per-result cap and a
    # 19,200-char aggregate cap, both below this real skill_view response.
    getattr(agent, "context_compressor").context_length = 16_000
    reload_call = _mock_tool_call(
        "skill_view", json.dumps(call_args), "c-reload-full"
    )
    messages = []

    agent._execute_tool_calls_sequential(
        SimpleNamespace(content="", tool_calls=[reload_call]),
        messages,
        "task-1",
    )

    delivered = messages[0]["content"]
    payload = json.loads(delivered)
    assert payload["success"] is True
    assert tail_sentinel in payload["content"]
    assert "<persisted-output>" not in delivered
    assert "result-reference" not in delivered
    assert agent._pending_skill_reloads == []


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
