"""Tests for model_tools.py — function call dispatch, agent-loop interception, legacy toolsets."""

import json
from unittest.mock import patch


from model_tools import (
    handle_function_call,
    get_all_tool_names,
    get_toolset_for_tool,
    _AGENT_LOOP_TOOLS,
    _LEGACY_TOOLSET_MAP,
)


# =========================================================================
# handle_function_call
# =========================================================================

class TestHandleFunctionCall:
    def test_agent_loop_tool_returns_error(self):
        for tool_name in _AGENT_LOOP_TOOLS:
            result = json.loads(handle_function_call(tool_name, {}))
            assert "error" in result
            assert "agent loop" in result["error"].lower()

    def test_unknown_tool_returns_error(self):
        result = json.loads(handle_function_call("totally_fake_tool_xyz", {}))
        assert "error" in result
        assert "totally_fake_tool_xyz" in result["error"]



    def test_post_tool_call_receives_non_negative_integer_duration_ms(self):
        """Regression: post_tool_call and transform_tool_result hooks must
        receive a non-negative integer ``duration_ms`` kwarg measuring
        dispatch latency.  Inspired by Claude Code 2.1.119, which added
        ``duration_ms`` to its PostToolUse hook inputs.
        """
        with (
            patch("model_tools.registry.dispatch", return_value='{"ok":true}'),
            patch("hermes_cli.plugins.has_hook", return_value=True),
            patch("hermes_cli.plugins.invoke_hook") as mock_invoke_hook,
        ):
            handle_function_call("web_search", {"q": "test"}, task_id="t1")

        kwargs_by_hook = {
            c.args[0]: c.kwargs for c in mock_invoke_hook.call_args_list
        }
        assert "duration_ms" in kwargs_by_hook["post_tool_call"]
        assert "duration_ms" in kwargs_by_hook["transform_tool_result"]

        post_duration = kwargs_by_hook["post_tool_call"]["duration_ms"]
        transform_duration = kwargs_by_hook["transform_tool_result"]["duration_ms"]
        assert isinstance(post_duration, int)
        assert post_duration >= 0
        # Both hooks should observe the same measured duration.
        assert post_duration == transform_duration
        # pre_tool_call does NOT get duration_ms (nothing has run yet).
        assert "duration_ms" not in kwargs_by_hook["pre_tool_call"]

    def test_terminal_nonzero_exit_is_reported_as_error(self):
        result = json.dumps({"output": "", "exit_code": 1, "error": None})
        with (
            patch("model_tools.registry.dispatch", return_value=result),
            patch("hermes_cli.plugins.has_hook", return_value=True),
            patch("hermes_cli.plugins.invoke_hook") as mock_invoke_hook,
        ):
            assert handle_function_call("terminal", {"command": "false"}) == result

        kwargs_by_hook = {
            hook.args[0]: hook.kwargs for hook in mock_invoke_hook.call_args_list
        }
        for hook_name in ("post_tool_call", "transform_tool_result"):
            assert kwargs_by_hook[hook_name]["status"] == "error"
            assert kwargs_by_hook[hook_name]["error_type"] == "tool_error"
            assert kwargs_by_hook[hook_name]["error_message"] == "exit 1"

    def test_no_listener_skips_post_and_transform_emit(self):
        """When no plugin is registered for post_tool_call /
        transform_tool_result, the emit path must short-circuit on
        ``has_hook`` and never build/dispatch a payload — so the
        no-listener hot path stays cheap.  ``pre_tool_call`` is always
        polled (block-check), so it may still fire; the observer/transform
        emits must not.
        """
        with (
            patch("model_tools.registry.dispatch", return_value='{"ok":true}'),
            patch("hermes_cli.plugins.has_hook", return_value=False),
            patch("hermes_cli.plugins.invoke_hook") as mock_invoke_hook,
        ):
            result = handle_function_call("web_search", {"q": "test"}, task_id="t1")

        assert result == '{"ok":true}'
        fired = {c.args[0] for c in mock_invoke_hook.call_args_list}
        assert "post_tool_call" not in fired
        assert "transform_tool_result" not in fired

    def test_tool_request_and_execution_middleware_wrap_registry_dispatch(self, monkeypatch):
        seen = {}

        def fake_invoke_middleware(kind, **kwargs):
            if kind == "tool_request":
                return [{
                    "args": {**kwargs["args"], "rewritten": True},
                    "source": "test-middleware",
                    "reason": "rewrite",
                }]
            return []

        def execution_middleware(**kwargs):
            seen["execution_args"] = kwargs["args"]
            return kwargs["next_call"]({**kwargs["args"], "wrapped": True})

        def fake_dispatch(tool_name, args, **kwargs):
            seen["dispatch"] = (tool_name, args, kwargs)
            return json.dumps({"ok": True, "args": args})

        manager = type(
            "Manager",
            (),
            {"_middleware": {"tool_request": [fake_invoke_middleware], "tool_execution": [execution_middleware]}},
        )()
        monkeypatch.setattr("hermes_cli.plugins.invoke_middleware", fake_invoke_middleware)
        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
        hook_calls = []
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
        monkeypatch.setattr("model_tools.registry.dispatch", fake_dispatch)

        result = json.loads(
            handle_function_call(
                "web_search",
                {"q": "test"},
                task_id="task-1",
                tool_call_id="tool-1",
                session_id="session-1",
            )
        )

        assert seen["execution_args"] == {"q": "test", "rewritten": True}
        assert seen["dispatch"][1] == {"q": "test", "rewritten": True, "wrapped": True}
        assert result["args"] == {"q": "test", "rewritten": True, "wrapped": True}
        expected_trace = [{"source": "test-middleware", "reason": "rewrite"}]
        pre_call = next(call for call in hook_calls if call[0] == "pre_tool_call")
        post_call = next(call for call in hook_calls if call[0] == "post_tool_call")
        assert pre_call[1]["middleware_trace"] == expected_trace
        assert post_call[1]["middleware_trace"] == expected_trace

    def test_registry_exception_emits_terminal_tool_hook(self, monkeypatch):
        from hermes_cli import lifecycle

        hook_calls = []
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "post_tool_call")
        monkeypatch.setattr(
            lifecycle,
            "invoke_hook",
            lambda name, **kwargs: hook_calls.append((name, kwargs)) or [],
        )
        monkeypatch.setattr(
            "model_tools.registry.dispatch",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        result = json.loads(
            handle_function_call(
                "web_search",
                {"q": "test"},
                task_id="task-1",
                session_id="session-1",
                tool_call_id="tool-1",
            )
        )

        assert "error" in result
        [post_call] = [call for call in hook_calls if call[0] == "post_tool_call"]
        assert post_call[1]["status"] == "error"
        assert post_call[1]["error_type"] == "RuntimeError"
        assert post_call[1]["duration_ms"] >= 0

    def test_acp_edit_denial_emits_blocked_terminal_tool_hook(self, monkeypatch):
        from hermes_cli import lifecycle

        hook_calls = []
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "post_tool_call")
        monkeypatch.setattr(
            lifecycle,
            "invoke_hook",
            lambda name, **kwargs: hook_calls.append((name, kwargs)) or [],
        )
        monkeypatch.setattr(
            "acp_adapter.edit_approval.maybe_require_edit_approval",
            lambda *_args, **_kwargs: json.dumps({"error": "Edit approval denied"}),
        )
        monkeypatch.setattr(
            "model_tools.registry.dispatch",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("denied edit must not dispatch")
            ),
        )

        result = json.loads(
            handle_function_call(
                "write_file",
                {"path": "private.txt", "content": "private"},
                task_id="task-1",
                session_id="session-1",
                tool_call_id="tool-1",
            )
        )

        assert result == {"error": "Edit approval denied"}
        [post_call] = [call for call in hook_calls if call[0] == "post_tool_call"]
        assert post_call[1]["status"] == "blocked"
        assert post_call[1]["error_type"] == "edit_approval_denied"


# =========================================================================
# Agent loop tools
# =========================================================================

class TestAgentLoopTools:
    def test_expected_tools_in_set(self):
        assert "todo_list" in _AGENT_LOOP_TOOLS
        assert "memory" in _AGENT_LOOP_TOOLS
        assert "session_search" in _AGENT_LOOP_TOOLS
        assert "delegate_task" in _AGENT_LOOP_TOOLS

    def test_no_regular_tools_in_set(self):
        assert "web_search" not in _AGENT_LOOP_TOOLS
        assert "terminal" not in _AGENT_LOOP_TOOLS


# =========================================================================
# Pre-tool-call blocking via plugin hooks
# =========================================================================

class TestPreToolCallBlocking:
    """Verify that pre_tool_call hooks can block tool execution."""

    def test_blocked_tool_returns_error_and_skips_dispatch(self, monkeypatch):
        hook_calls = []

        def fake_invoke_hook(hook_name, **kwargs):
            hook_calls.append((hook_name, kwargs))
            if hook_name == "pre_tool_call":
                return [{"action": "block", "message": "Blocked by policy"}]
            return []

        dispatch_called = False
        _orig_dispatch = None

        def fake_dispatch(*args, **kwargs):
            nonlocal dispatch_called
            dispatch_called = True
            raise AssertionError("dispatch should not run when blocked")

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
        monkeypatch.setattr("model_tools.registry.dispatch", fake_dispatch)

        result = json.loads(handle_function_call("read_file", {"path": "test.txt"}, task_id="t1"))
        assert result == {"error": "Blocked by policy"}
        assert not dispatch_called
        post_call = next(call for call in hook_calls if call[0] == "post_tool_call")
        assert post_call[1]["status"] == "blocked"
        assert post_call[1]["error_type"] == "plugin_block"
        assert post_call[1]["error_message"] == "Blocked by policy"
        assert post_call[1]["duration_ms"] == 0

    def test_blocked_tool_skips_read_loop_notification(self, monkeypatch):
        notifications = []

        def fake_invoke_hook(hook_name, **kwargs):
            if hook_name == "pre_tool_call":
                return [{"action": "block", "message": "Blocked"}]
            return []

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)
        monkeypatch.setattr("model_tools.registry.dispatch",
                            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not run")))
        monkeypatch.setattr("tools.file_tools_read_tracking.notify_other_tool_call",
                            lambda task_id: notifications.append(task_id))

        result = json.loads(handle_function_call("web_search", {"q": "test"}, task_id="t1"))
        assert result == {"error": "Blocked"}
        assert notifications == []

    def test_invalid_hook_returns_do_not_block(self, monkeypatch):
        """Malformed hook returns should be ignored — tool executes normally."""
        def fake_invoke_hook(hook_name, **kwargs):
            if hook_name == "pre_tool_call":
                return [
                    "block",
                    {"action": "block"},           # missing message
                    {"action": "deny", "message": "nope"},
                ]
            return []

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)
        monkeypatch.setattr("model_tools.registry.dispatch",
                            lambda *a, **kw: json.dumps({"ok": True}))

        result = json.loads(handle_function_call("read_file", {"path": "test.txt"}, task_id="t1"))
        assert result == {"ok": True}


    def test_relay_rewrite_is_visible_to_pre_tool_authorization(self, monkeypatch):
        observed = {}

        def rewrite(**kwargs):
            assert kwargs["tool_name"] == "read_file"
            return {**kwargs["args"], "path": "approved.txt"}

        def fake_invoke_hook(hook_name, **kwargs):
            if hook_name == "pre_tool_call":
                observed["pre_tool_args"] = kwargs["args"]
            return []

        def dispatch(_name, args, **_kwargs):
            observed["dispatch_args"] = args
            return json.dumps({"ok": True})

        monkeypatch.setattr(
            "agent.relay_runtime.apply_tool_request_intercepts",
            rewrite,
        )
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
        monkeypatch.setattr("model_tools.registry.dispatch", dispatch)

        handle_function_call(
            "read_file",
            {"path": "original.txt"},
            task_id="t1",
            session_id="s1",
        )

        assert observed["pre_tool_args"]["path"] == "approved.txt"
        assert observed["dispatch_args"]["path"] == "approved.txt"



# =========================================================================
# Legacy toolset map
# =========================================================================

class TestLegacyToolsetMap:
    def test_expected_legacy_names(self):
        expected = [
            "web_tools", "terminal_tools", "vision_tools",
            "image_tools", "skills_tools", "browser_tools", "cronjob_tools",
            "file_tools", "tts_tools",
        ]
        for name in expected:
            assert name in _LEGACY_TOOLSET_MAP, f"Missing legacy toolset: {name}"



# =========================================================================
# Backward-compat wrappers
# =========================================================================

class TestBackwardCompat:
    def test_get_all_tool_names_returns_list(self):
        names = get_all_tool_names()
        assert isinstance(names, list)
        assert len(names) > 0
        # Should contain well-known tools
        assert "web_search" in names
        assert "terminal" in names

    def test_get_toolset_for_tool(self):
        result = get_toolset_for_tool("web_search")
        assert result is not None
        assert isinstance(result, str)




# =========================================================================
# _coerce_number — inf / nan must fall through to the original string
# (regression: fix: eliminate duplicate checkpoint entries and JSON-unsafe coercion)
# =========================================================================

class TestCoerceNumberInfNan:
    """_coerce_number must honor its documented contract ("Returns original
    string on failure") for inf/nan inputs, because float('inf') and
    float('nan') are not JSON-compliant under strict serialization."""

    def test_inf_returns_original_string(self):
        from tools.arg_coercion import _coerce_number
        assert _coerce_number("inf") == "inf"


    def test_nan_returns_original_string(self):
        from tools.arg_coercion import _coerce_number
        assert _coerce_number("nan") == "nan"



    def test_normal_numbers_still_coerce(self):
        """Guard against over-correction — real numbers still coerce."""
        from tools.arg_coercion import _coerce_number
        assert _coerce_number("42") == 42
        assert _coerce_number("3.14") == 3.14
        assert _coerce_number("1e3") == 1000

class TestDisabledToolsetsPlatformBundle:
    """Regression test for #33924: disabling a platform bundle (hermes-*)
    must not remove core tools from other enabled toolsets."""

    def test_disabling_platform_bundle_preserves_core_tools(self):
        """Disabling hermes-yuanbao should not strip core tools from hermes-telegram."""
        from model_tools import get_tool_definitions

        tools_telegram = get_tool_definitions(
            enabled_toolsets=["hermes-telegram"],
            quiet_mode=True,
        )
        tools_telegram_no_yuanbao = get_tool_definitions(
            enabled_toolsets=["hermes-telegram"],
            disabled_toolsets=["hermes-yuanbao"],
            quiet_mode=True,
        )
        names_telegram = {t["function"]["name"] for t in tools_telegram}
        names_no_yuanbao = {t["function"]["name"] for t in tools_telegram_no_yuanbao}

        # Disabling a *different* platform bundle must not remove any tools
        assert names_telegram == names_no_yuanbao, (
            f"Tools lost after disabling hermes-yuanbao: "
            f"{names_telegram - names_no_yuanbao}"
        )

    def test_disabling_platform_bundle_removes_own_tools(self):
        """Disabling hermes-discord should remove discord-specific tools."""
        from model_tools import get_tool_definitions

        tools = get_tool_definitions(
            enabled_toolsets=["hermes-discord"],
            disabled_toolsets=["hermes-discord"],
            quiet_mode=True,
        )
        names = {t["function"]["name"] for t in tools}
        assert "discord" not in names




    def test_bundle_non_core_tools_unknown_falls_back(self):
        """An unknown/garbage bundle name falls back to full resolution (best effort)."""
        from toolsets import bundle_non_core_tools
        # A non-existent bundle resolves to an empty set (no tools), not a crash.
        assert bundle_non_core_tools("hermes-does-not-exist") == set()


class TestDisabledToolsetsPostureToolset:
    """Regression test for #57315: disabling a posture toolset (`coding`,
    posture: True) must preserve the shared core tools it re-lists but does
    not own -- same non-core-delta subtraction as hermes-* bundles (#33924) --
    while atomic toolsets stay fully removable."""

    def test_disabling_coding_preserves_core_but_atomic_disables_still_remove(self):
        from model_tools import get_tool_definitions

        # web_search is check_fn-gated (needs an API key); probe only the core
        # tools actually present in baseline so gating cannot mask the fix.
        core_probe = {"terminal", "read_file", "write_file", "web_search", "execute_code"}

        baseline = {
            t["function"]["name"]
            for t in get_tool_definitions(quiet_mode=True)
        }
        present_core = core_probe & baseline
        # Sanity: at least some probed core tools are available in this env.
        assert present_core, "no probed core tools present in baseline"

        no_coding = {
            t["function"]["name"]
            for t in get_tool_definitions(
                disabled_toolsets=["coding"], quiet_mode=True
            )
        }
        # Previously the full resolve_toolset("coding") subtraction stripped
        # these shared core tools, collapsing the schema to a handful (#57315).
        assert present_core <= no_coding, (
            f"Core tools stripped by disabling 'coding': {present_core - no_coding}"
        )

        # Atomic (non-posture) toolsets must still be fully removable.
        no_terminal = {
            t["function"]["name"]
            for t in get_tool_definitions(
                disabled_toolsets=["terminal"], quiet_mode=True
            )
        }
        assert "terminal" not in no_terminal

        no_file = {
            t["function"]["name"]
            for t in get_tool_definitions(
                disabled_toolsets=["file"], quiet_mode=True
            )
        }
        assert "write_file" not in no_file


# =========================================================================
# Tool Search bridge dispatch
# =========================================================================

class TestBridgeDispatch:
    """handle_function_call routes tool_search/tool_describe inline, unwraps tool_call,
    and refuses tool_call targets outside the session-scoped deferrable catalog."""

    def test_tool_search_and_describe_return_json_strings(self):
        with patch("model_tools.get_tool_definitions", return_value=[]):
            out = handle_function_call("tool_search", {"queries": ["anything"]})
            assert isinstance(out, str) and json.loads(out) is not None
            out = handle_function_call("tool_describe", {"names": ["nope"]})
            assert isinstance(out, str) and json.loads(out) is not None

    def test_tool_call_bad_args_error(self):
        with patch("model_tools.get_tool_definitions", return_value=[]):
            result = json.loads(handle_function_call("tool_call", {}))
        assert "requires a 'name'" in result["error"]

    def test_tool_call_rejects_out_of_scope_and_unwraps_in_scope(self):
        import tools.tool_search as ts
        with patch("model_tools.get_tool_definitions", return_value=[]), \
             patch.object(ts, "resolve_underlying_call", return_value=("mcp_x", {"a": 1}, None)), \
             patch.object(ts, "scoped_deferrable_names", return_value=frozenset()):
            result = json.loads(handle_function_call("tool_call", {"name": "mcp_x"}))
        assert "not available in this session" in result["error"]

        with patch("model_tools.get_tool_definitions", return_value=[]), \
             patch.object(ts, "resolve_underlying_call", return_value=("mcp_x", {"a": 1}, None)), \
             patch.object(ts, "scoped_deferrable_names", return_value=frozenset({"mcp_x"})), \
             patch.object(ts, "validate_deferred_call_args", return_value=None), \
             patch("model_tools.registry.dispatch", return_value='{"ok": true}') as disp:
            out = handle_function_call("tool_call", {"name": "mcp_x"}, task_id="t")
        assert json.loads(out) == {"ok": True}
        assert disp.call_args.args[0] == "mcp_x" and disp.call_args.args[1] == {"a": 1}
