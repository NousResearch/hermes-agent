"""Tests for model_tools.py — function call dispatch, agent-loop interception, legacy toolsets."""

import json
from unittest.mock import ANY, call, patch


from model_tools import (
    handle_function_call,
    get_all_tool_names,
    get_toolset_for_tool,
    _AGENT_LOOP_TOOLS,
    _LEGACY_TOOLSET_MAP,
    TOOL_TO_TOOLSET_MAP,
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

    def test_exception_returns_json_error(self):
        # Even if something goes wrong, should return valid JSON
        result = handle_function_call("web_search", None)  # None args may cause issues
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "error" in parsed
        assert len(parsed["error"]) > 0
        assert "error" in parsed["error"].lower() or "failed" in parsed["error"].lower()

    def test_tool_hooks_receive_session_and_tool_call_ids(self):
        with (
            patch("model_tools.registry.dispatch", return_value='{"ok":true}'),
            patch("hermes_cli.plugins.has_hook", return_value=True),
            patch("hermes_cli.plugins.invoke_hook") as mock_invoke_hook,
        ):
            result = handle_function_call(
                "web_search",
                {"q": "test"},
                task_id="task-1",
                tool_call_id="call-1",
                session_id="session-1",
            )

        assert result == '{"ok":true}'
        assert mock_invoke_hook.call_args_list == [
            call(
                "pre_tool_call",
                tool_name="web_search",
                args={"q": "test"},
                task_id="task-1",
                session_id="session-1",
                tool_call_id="call-1",
                turn_id="",
                api_request_id="",
                middleware_trace=[],
            ),
            call(
                "post_tool_call",
                tool_name="web_search",
                args={"q": "test"},
                result='{"ok":true}',
                task_id="task-1",
                session_id="session-1",
                tool_call_id="call-1",
                turn_id="",
                api_request_id="",
                duration_ms=ANY,
                status="ok",
                error_type=None,
                error_message=None,
                middleware_trace=[],
            ),
            call(
                "transform_tool_result",
                tool_name="web_search",
                args={"q": "test"},
                result='{"ok":true}',
                task_id="task-1",
                session_id="session-1",
                tool_call_id="call-1",
                turn_id="",
                api_request_id="",
                duration_ms=ANY,
                status="ok",
                error_type=None,
                error_message=None,
            ),
        ]

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


# =========================================================================
# Agent loop tools
# =========================================================================

class TestAgentLoopTools:
    def test_expected_tools_in_set(self):
        assert "todo" in _AGENT_LOOP_TOOLS
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
        monkeypatch.setattr("tools.file_tools.notify_other_tool_call",
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

    def test_skip_flag_prevents_double_fire(self, monkeypatch):
        """When skip_pre_tool_call_hook=True, the hook does not fire again.

        The caller (e.g. run_agent._invoke_tool) has already called
        get_pre_tool_call_block_message(), which fires the hook once.
        handle_function_call must NOT fire it a second time — that was
        the classic double-fire bug where observer hooks logged every
        tool call twice.
        """
        hook_calls = []

        def fake_invoke_hook(hook_name, **kwargs):
            hook_calls.append(hook_name)
            return []

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
        monkeypatch.setattr("model_tools.registry.dispatch",
                            lambda *a, **kw: json.dumps({"ok": True}))

        handle_function_call("web_search", {"q": "test"}, task_id="t1",
                             skip_pre_tool_call_hook=True)

        # Single-fire contract: when skip=True the caller already fired
        # pre_tool_call, so handle_function_call must not fire it again.
        assert hook_calls.count("pre_tool_call") == 0, (
            f"pre_tool_call fired {hook_calls.count('pre_tool_call')} times "
            f"with skip_pre_tool_call_hook=True; expected 0 "
            f"(caller already fired it). hook_calls={hook_calls}"
        )
        # post_tool_call and transform_tool_result still fire — only the
        # pre-call block-check path is suppressed by the skip flag.
        assert "post_tool_call" in hook_calls
        assert "transform_tool_result" in hook_calls

    def test_run_agent_pattern_fires_pre_tool_call_exactly_once(self, monkeypatch):
        """End-to-end regression for the double-fire bug.

        Mirrors run_agent._invoke_tool: first calls
        get_pre_tool_call_block_message() (which fires the hook as part of
        its block-directive poll), then calls
        handle_function_call(skip_pre_tool_call_hook=True).  The plugin
        hook MUST fire exactly once across both calls — not twice as it
        did before the fix (observer plugins were seeing every tool
        execution logged twice).
        """
        from hermes_cli.plugins import get_pre_tool_call_block_message

        hook_calls = []

        def fake_invoke_hook(hook_name, **kwargs):
            hook_calls.append(hook_name)
            return []

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)
        monkeypatch.setattr("model_tools.registry.dispatch",
                            lambda *a, **kw: json.dumps({"ok": True}))

        # Step 1: caller checks for a block directive (this fires pre_tool_call once).
        block = get_pre_tool_call_block_message(
            "web_search", {"q": "test"}, task_id="t1",
        )
        assert block is None

        # Step 2: caller dispatches with skip=True so the hook isn't re-fired.
        handle_function_call(
            "web_search", {"q": "test"}, task_id="t1",
            skip_pre_tool_call_hook=True,
        )

        assert hook_calls.count("pre_tool_call") == 1, (
            f"pre_tool_call fired {hook_calls.count('pre_tool_call')} times "
            f"across the run_agent (block-check + dispatch) path; "
            f"expected exactly 1. hook_calls={hook_calls}"
        )


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

    def test_values_are_lists_of_strings(self):
        for name, tools in _LEGACY_TOOLSET_MAP.items():
            assert isinstance(tools, list), f"{name} is not a list"
            for tool in tools:
                assert isinstance(tool, str), f"{name} contains non-string: {tool}"


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

    def test_get_toolset_for_unknown_tool(self):
        result = get_toolset_for_tool("totally_nonexistent_tool")
        assert result is None

    def test_tool_to_toolset_map(self):
        assert isinstance(TOOL_TO_TOOLSET_MAP, dict)
        assert len(TOOL_TO_TOOLSET_MAP) > 0


# =========================================================================
# _coerce_number — inf / nan must fall through to the original string
# (regression: fix: eliminate duplicate checkpoint entries and JSON-unsafe coercion)
# =========================================================================

class TestCoerceNumberInfNan:
    """_coerce_number must honor its documented contract ("Returns original
    string on failure") for inf/nan inputs, because float('inf') and
    float('nan') are not JSON-compliant under strict serialization."""

    def test_inf_returns_original_string(self):
        from model_tools import _coerce_number
        assert _coerce_number("inf") == "inf"

    def test_negative_inf_returns_original_string(self):
        from model_tools import _coerce_number
        assert _coerce_number("-inf") == "-inf"

    def test_nan_returns_original_string(self):
        from model_tools import _coerce_number
        assert _coerce_number("nan") == "nan"

    def test_infinity_spelling_returns_original_string(self):
        from model_tools import _coerce_number
        # Python's float() parses "Infinity" too — still not JSON-safe.
        assert _coerce_number("Infinity") == "Infinity"

    def test_coerced_result_is_strict_json_safe(self):
        """Whatever _coerce_number returns for inf/nan must round-trip
        through strict (allow_nan=False) json.dumps without raising."""
        from model_tools import _coerce_number
        for s in ("inf", "-inf", "nan", "Infinity"):
            result = _coerce_number(s)
            json.dumps({"x": result}, allow_nan=False)  # must not raise

    def test_normal_numbers_still_coerce(self):
        """Guard against over-correction — real numbers still coerce."""
        from model_tools import _coerce_number
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

    def test_disabling_non_platform_toolset_still_works(self):
        """Disabling a regular (non-hermes-) toolset still subtracts all tools."""
        from model_tools import get_tool_definitions

        tools_normal = get_tool_definitions(
            enabled_toolsets=["hermes-telegram"],
            quiet_mode=True,
        )
        tools_no_web = get_tool_definitions(
            enabled_toolsets=["hermes-telegram"],
            disabled_toolsets=["web"],
            quiet_mode=True,
        )
        names_normal = {t["function"]["name"] for t in tools_normal}
        names_no_web = {t["function"]["name"] for t in tools_no_web}

        web_tools = {"web_search", "web_extract"}
        removed = names_normal - names_no_web
        # web tools should be removed (if they were present)
        present_web = web_tools & names_normal
        assert present_web <= removed, (
            f"Web tools not removed: {present_web - removed}"
        )


    def test_disabling_bundle_removes_platform_tools_but_keeps_core(self):
        """Disabling hermes-discord (when enabled) removes discord/discord_admin
        from the resolved delta but keeps core tools — via bundle_non_core_tools."""
        from toolsets import bundle_non_core_tools, _HERMES_CORE_TOOLS

        delta = bundle_non_core_tools("hermes-yuanbao")
        # The delta is the bundle's platform-specific tools, NOT core.
        assert "yb_send_dm" in delta
        assert not (delta & set(_HERMES_CORE_TOOLS)), "core tools must not be in the removal delta"

    def test_bundle_non_core_tools_unknown_falls_back(self):
        """An unknown/garbage bundle name falls back to full resolution (best effort)."""
        from toolsets import bundle_non_core_tools
        # A non-existent bundle resolves to an empty set (no tools), not a crash.
        assert bundle_non_core_tools("hermes-does-not-exist") == set()


class TestDisabledToolsetsConflictVisibility:
    """Regression coverage for the Telegram-DM terminal-tool RCA: disabled_toolsets
    always wins over enabled_toolsets (model_tools.py `_compute_tool_definitions`,
    issue #17309), but that override used to be completely silent. These tests
    verify the new INFO/WARNING logging that makes the silent override visible."""

    def test_disabled_toolset_removal_logs_info(self, caplog):
        """Any toolset actually stripped by disabled_toolsets logs its removed
        tool names at INFO, regardless of quiet_mode."""
        import logging
        from model_tools import _compute_tool_definitions

        with caplog.at_level(logging.INFO, logger="model_tools"):
            _compute_tool_definitions(
                enabled_toolsets=["hermes-telegram"],
                disabled_toolsets=["web"],
                quiet_mode=True,
            )

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any(
            "disabled_toolsets removed toolset 'web'" in r.getMessage()
            for r in info_records
        ), f"Expected an INFO log about 'web' removal, got: {[r.getMessage() for r in info_records]}"

    def test_toolset_enabled_and_disabled_simultaneously_logs_warning(self, caplog):
        """The exact silent-override mechanism from the RCA: a toolset present
        in BOTH enabled_toolsets and disabled_toolsets must emit a WARNING
        naming the toolset, since disabled_toolsets wins without any other
        signal to the caller."""
        import logging
        import model_tools
        from model_tools import _compute_tool_definitions

        # Clear dedup state so the warning isn't suppressed by an earlier test
        # or an earlier call using the same toolset name.
        model_tools._WARNED_TOOLSET_CONFLICTS.discard(("terminal",))

        with caplog.at_level(logging.WARNING, logger="model_tools"):
            tools = _compute_tool_definitions(
                enabled_toolsets=["terminal"],
                disabled_toolsets=["terminal"],
                quiet_mode=True,
            )

        names = {t["function"]["name"] for t in tools}
        assert "terminal" not in names, "disabled_toolsets must still win over enabled_toolsets"

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "'terminal' is both explicitly enabled" in r.getMessage()
            for r in warning_records
        ), f"Expected a conflict WARNING for 'terminal', got: {[r.getMessage() for r in warning_records]}"

    def test_conflict_warning_fires_once_per_toolset(self, caplog):
        """Dedup guard: repeated resolution with the same conflicting toolset
        must not spam the log on every call."""
        import logging
        import model_tools
        from model_tools import _compute_tool_definitions

        model_tools._WARNED_TOOLSET_CONFLICTS.discard(("file",))

        with caplog.at_level(logging.WARNING, logger="model_tools"):
            for _ in range(3):
                _compute_tool_definitions(
                    enabled_toolsets=["file"],
                    disabled_toolsets=["file"],
                    quiet_mode=True,
                )

        matches = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "'file' is both explicitly enabled" in r.getMessage()
        ]
        assert len(matches) == 1, f"Expected exactly one conflict warning, got {len(matches)}"

    def test_no_conflict_when_toolset_only_disabled(self, caplog):
        """A toolset that is disabled but was never explicitly enabled is a
        normal disablement, not a conflict — no WARNING should fire."""
        import logging
        from model_tools import _compute_tool_definitions

        with caplog.at_level(logging.WARNING, logger="model_tools"):
            _compute_tool_definitions(
                enabled_toolsets=["hermes-telegram"],
                disabled_toolsets=["memory"],
                quiet_mode=True,
            )

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not any(
            "'memory' is both explicitly enabled" in r.getMessage()
            for r in warning_records
        )

    def test_docstring_documents_unconditional_override(self):
        """The docstring must state the real precedence (disabled_toolsets
        always wins), not the old inaccurate 'only if enabled_toolsets is
        None' wording that motivated this RCA."""
        from model_tools import get_tool_definitions

        doc = get_tool_definitions.__doc__ or ""
        assert "always" in doc.lower()
        assert "disabled_toolsets" in doc
