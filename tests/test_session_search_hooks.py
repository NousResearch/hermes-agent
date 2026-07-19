"""Tests for session_search transform_tool_result hooks in agent-runtime paths.

Verifies that post_tool_call fires BEFORE transform_tool_result in the
concurrent (agent_runtime_helpers.invoke_tool) path, and that the shared
helper correctly passes lifecycle metadata to plugin hooks.
"""

import pytest

_HOOK_LOG = []


def _fake_invoke_hook(name, **kwargs):
    global _HOOK_LOG
    _HOOK_LOG.append(name)
    if name == "transform_tool_result":
        return ["TRANSFORMED"]
    return []


@pytest.fixture(autouse=True)
def reset_log():
    global _HOOK_LOG
    _HOOK_LOG = []


class FakeAgent:
    session_id = "s1"
    quiet_mode = True
    valid_tool_names = set()
    enabled_toolsets = []
    disabled_toolsets = []
    tool_progress_callback = None
    tool_complete_callback = None
    log_prefix_chars = 200
    log_prefix = ""
    _current_tool = None
    _subdirectory_hints = None
    _context_engine_tool_names = frozenset()
    _memory_manager = None
    _delegate_spinner = None
    _todo_store = {}
    _memory_store = {}
    _pending_steer = None
    _interrupt_requested = False
    _file_guard_state = None
    _interrupt_lock = None
    verbose_logging = False
    tool_delay = 0
    tool_progress_mode = "off"

    def _get_session_db_for_recall(self):
        import sqlite3
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT)")
        return db

    def _should_emit_quiet_tool_messages(self):
        return False

    def _should_start_quiet_spinner(self):
        return False

    def _get_session_db(self):
        import sqlite3, tempfile
        return sqlite3.connect(":memory:")

    def _append_guardrail_observation(self, name, args, result, failed=False):
        return result

    def _record_file_mutation_result(self, *a, **kw):
        pass

    def _apply_pending_steer_to_tool_results(self, *a, **kw):
        pass

    def _tool_result_content_for_active_model(self, name, result):
        return result

    def _touch_activity(self, *a, **kw):
        pass

    def _print_fn(self, *a, **kw):
        pass

    def _vprint(self, *a, **kw):
        pass

    def _wrap_verbose(self, *a, **kw):
        return ""

    def _dispatch_delegate_task(self, args):
        return "delegated"

    def _flush_session_db_after_tool_progress(self, *a, **kw):
        pass


def test_concurrent_path_hook_ordering(monkeypatch):
    """post_tool_call fires BEFORE transform_tool_result in concurrent path."""
    from agent.agent_runtime_helpers import invoke_tool

    agent = FakeAgent()

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_invoke_hook)
    monkeypatch.setattr("model_tools._emit_post_tool_call_hook",
                        lambda **kw: _HOOK_LOG.append("post_tool_call"))

    result = invoke_tool(agent, "session_search", {"query": "test", "limit": 1},
                         effective_task_id="t1", tool_call_id="tc1")

    assert "post_tool_call" in _HOOK_LOG
    assert "transform_tool_result" in _HOOK_LOG
    post_idx = _HOOK_LOG.index("post_tool_call")
    transform_idx = _HOOK_LOG.index("transform_tool_result")
    assert post_idx < transform_idx
    assert result == "TRANSFORMED"


def test_transform_hook_passes_lifecycle_metadata(monkeypatch):
    """Helper passes task_id, tool_call_id, turn_id, api_request_id to hook."""
    from agent.agent_runtime_helpers import _apply_transform_tool_result_hook

    received_kwargs = {}

    def capture_invoke(name, **kwargs):
        received_kwargs.update(kwargs)
        return []

    class FA:
        session_id = "s1"
        _current_turn_id = "turn-1"
        _current_api_request_id = "req-1"

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", capture_invoke)

    _apply_transform_tool_result_hook(
        "memory", {"key": "val"}, "RESULT", FA(),
        task_id="t42", tool_call_id="tc42",
        turn_id="turn-99", api_request_id="req-99",
        duration_ms=1234,
    )

    assert received_kwargs["task_id"] == "t42"
    assert received_kwargs["tool_call_id"] == "tc42"
    assert received_kwargs["turn_id"] == "turn-99"
    assert received_kwargs["api_request_id"] == "req-99"
    assert received_kwargs["duration_ms"] == 1234
    assert received_kwargs["tool_name"] == "memory"
    assert received_kwargs["result"] == "RESULT"


def test_transform_hook_no_op_when_no_plugin(monkeypatch):
    """Result unchanged when no transform hook is registered."""
    from agent.agent_runtime_helpers import _apply_transform_tool_result_hook

    class FA:
        session_id = "s1"

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)
    result = _apply_transform_tool_result_hook("session_search", {}, "ORIGINAL", FA())
    assert result == "ORIGINAL"
