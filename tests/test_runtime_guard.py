import json
from types import SimpleNamespace

import model_tools
from agent.runtime_guard import reset_current_runtime_guard, set_current_runtime_guard
from agent.runtime_guard_types import GuardDecision
from agent.tool_executor import execute_tool_calls_sequential
from agent.tool_guardrails import ToolCallGuardrailController, toolguard_synthetic_result


def _write_runtime_guard_config(hermes_home, *, enabled, fail_closed=True):
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(
        "\n".join(
            [
                "runtime_guard:",
                f"  enabled: {str(enabled).lower()}",
                f"  fail_closed: {str(fail_closed).lower()}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _patch_registry_dispatch(monkeypatch, result='{"ok": true}'):
    from tools.registry import registry

    calls = []

    def _dispatch(name, args, **kwargs):
        calls.append((name, args, kwargs))
        return result

    monkeypatch.setattr(registry, "dispatch", _dispatch)
    monkeypatch.setattr(model_tools, "_READ_SEARCH_TOOLS", frozenset())
    return calls


def test_runtime_guard_disabled_config_is_noop(tmp_path, monkeypatch):
    _write_runtime_guard_config(tmp_path / "hermes", enabled=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    calls = _patch_registry_dispatch(monkeypatch, result='{"output": "ran"}')

    out = model_tools.handle_function_call(
        "dummy_tool",
        {"value": 1},
        task_id="task-1",
        session_id="session-1",
        tool_call_id="call-1",
        skip_pre_tool_call_hook=True,
    )

    assert out == '{"output": "ran"}'
    assert len(calls) == 1


def test_enabled_runtime_guard_block_returns_structured_metadata(tmp_path, monkeypatch):
    _write_runtime_guard_config(tmp_path / "hermes", enabled=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    calls = _patch_registry_dispatch(monkeypatch)

    class BlockingGuard:
        def guard_tool_action(self, context, **kwargs):
            return GuardDecision.block(
                "blocked by test guard",
                message="blocked message",
                code="policy_block",
                context=context,
                metadata={"rule": "test"},
            )

    token = set_current_runtime_guard(BlockingGuard())
    try:
        out = model_tools.handle_function_call(
            "dummy_tool",
            {"value": 1},
            task_id="task-1",
            session_id="session-1",
            tool_call_id="call-1",
            skip_pre_tool_call_hook=True,
        )
    finally:
        reset_current_runtime_guard(token)

    data = json.loads(out)
    assert calls == []
    assert data["status"] == "blocked"
    assert data["blocked_by"] == "runtime_guard"
    assert data["runtime_guard_block"]["allowed"] is False
    assert data["runtime_guard_block"]["code"] == "policy_block"
    assert data["runtime_guard_block"]["context"]["tool_name"] == "dummy_tool"
    assert data["runtime_guard_block"]["metadata"] == {"rule": "test"}


def test_enabled_runtime_guard_resolution_failure_fails_closed(tmp_path, monkeypatch):
    _write_runtime_guard_config(tmp_path / "hermes", enabled=True, fail_closed=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    calls = _patch_registry_dispatch(monkeypatch)

    out = model_tools.handle_function_call(
        "dummy_tool",
        {},
        task_id="task-1",
        session_id="session-1",
        tool_call_id="call-1",
        skip_pre_tool_call_hook=True,
    )

    data = json.loads(out)
    assert calls == []
    assert data["status"] == "blocked"
    assert data["blocked_by"] == "runtime_guard"
    assert data["code"] == "runtime_guard_error"
    assert data["runtime_guard_block"]["context"]["guard_name"] == "tool_action"


def test_enabled_runtime_guard_exception_fails_closed(tmp_path, monkeypatch):
    _write_runtime_guard_config(tmp_path / "hermes", enabled=True, fail_closed=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    calls = _patch_registry_dispatch(monkeypatch)

    class RaisingGuard:
        def guard_tool_action(self, context, **kwargs):
            raise RuntimeError("guard failed")

    token = set_current_runtime_guard(RaisingGuard())
    try:
        out = model_tools.handle_function_call(
            "dummy_tool",
            {},
            task_id="task-1",
            session_id="session-1",
            tool_call_id="call-1",
            skip_pre_tool_call_hook=True,
        )
    finally:
        reset_current_runtime_guard(token)

    data = json.loads(out)
    assert calls == []
    assert data["status"] == "blocked"
    assert data["blocked_by"] == "runtime_guard"
    assert data["code"] == "runtime_guard_error"


def test_enabled_runtime_guard_exception_can_fail_open_when_configured(tmp_path, monkeypatch):
    _write_runtime_guard_config(tmp_path / "hermes", enabled=True, fail_closed=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    calls = _patch_registry_dispatch(monkeypatch, result='{"output": "ran"}')

    class RaisingGuard:
        def guard_tool_action(self, context, **kwargs):
            raise RuntimeError("guard failed")

    token = set_current_runtime_guard(RaisingGuard())
    try:
        out = model_tools.handle_function_call(
            "dummy_tool",
            {},
            task_id="task-1",
            session_id="session-1",
            tool_call_id="call-1",
            skip_pre_tool_call_hook=True,
        )
    finally:
        reset_current_runtime_guard(token)

    assert out == '{"output": "ran"}'
    assert len(calls) == 1


def test_final_output_guard_replaces_response_when_enabled(tmp_path, monkeypatch):
    from agent.runtime_guard import guarded_final_output_for_agent

    _write_runtime_guard_config(tmp_path / "hermes", enabled=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    class BlockingGuard:
        def guard_final_output(self, context, **kwargs):
            return GuardDecision.block(
                "blocked final output",
                replacement_text="safe replacement",
                context=context,
            )

    token = set_current_runtime_guard(BlockingGuard())
    try:
        out = guarded_final_output_for_agent(SimpleNamespace(session_id="s1", platform="cli"), "unsafe")
    finally:
        reset_current_runtime_guard(token)

    assert out == "safe replacement"


def test_final_output_guard_exception_fails_closed_when_enabled(tmp_path, monkeypatch):
    from agent.runtime_guard import guarded_final_output_for_agent

    _write_runtime_guard_config(tmp_path / "hermes", enabled=True, fail_closed=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    class RaisingGuard:
        def guard_final_output(self, context, **kwargs):
            raise RuntimeError("final guard failed")

    token = set_current_runtime_guard(RaisingGuard())
    try:
        out = guarded_final_output_for_agent(SimpleNamespace(session_id="s1", platform="cli"), "unsafe")
    finally:
        reset_current_runtime_guard(token)

    assert "Response blocked by runtime_guard" in out
    assert "final guard failed" in out


class _SubdirectoryHints:
    def check_tool_call(self, name, args):
        return ""


class _FakeAgent:
    def __init__(self):
        self._interrupt_requested = False
        self.quiet_mode = True
        self.verbose_logging = False
        self.tool_progress_mode = "off"
        self.log_prefix_chars = 80
        self.log_prefix = ""
        self.tool_progress_callback = None
        self.tool_start_callback = None
        self.tool_complete_callback = None
        self.tool_delay = 0
        self.session_id = "session-1"
        self.platform = "cli"
        self.valid_tool_names = set()
        self._current_turn_id = "turn-1"
        self._current_api_request_id = "request-1"
        self._checkpoint_mgr = SimpleNamespace(enabled=False)
        self._tool_guardrails = ToolCallGuardrailController()
        self._subdirectory_hints = _SubdirectoryHints()
        self._context_engine_tool_names = set()
        self._memory_manager = None

    def _vprint(self, *args, **kwargs):
        return None

    def _touch_activity(self, *args, **kwargs):
        return None

    def _should_emit_quiet_tool_messages(self):
        return False

    def _should_start_quiet_spinner(self):
        return False

    def _guardrail_block_result(self, decision):
        return toolguard_synthetic_result(decision)

    def _append_guardrail_observation(self, tool_name, function_args, function_result, *, failed):
        return function_result

    def _record_file_mutation_result(self, function_name, function_args, function_result, is_error):
        return None

    def _tool_result_content_for_active_model(self, tool_name, result):
        return result

    def _apply_pending_steer_to_tool_results(self, messages, num_tool_msgs):
        return None


def test_structured_block_metadata_preserved_in_sequential_tool_result(tmp_path, monkeypatch):
    _write_runtime_guard_config(tmp_path / "hermes", enabled=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    calls = _patch_registry_dispatch(monkeypatch)

    class BlockingGuard:
        def guard_tool_action(self, context, **kwargs):
            return GuardDecision.block(
                "blocked before execution",
                code="runtime_guard_block",
                context=context,
                metadata={"source": "unit-test"},
            )

    token = set_current_runtime_guard(BlockingGuard())
    try:
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="terminal",
                arguments=json.dumps({"command": "echo should-not-run"}),
            ),
        )
        messages = []
        execute_tool_calls_sequential(
            _FakeAgent(),
            SimpleNamespace(tool_calls=[tool_call]),
            messages,
            "task-1",
        )
    finally:
        reset_current_runtime_guard(token)

    assert calls == []
    assert len(messages) == 1
    data = json.loads(messages[0]["content"])
    assert data["blocked_by"] == "runtime_guard"
    assert data["runtime_guard_block"]["context"]["tool_name"] == "terminal"
    assert data["runtime_guard_block"]["metadata"] == {"source": "unit-test"}
