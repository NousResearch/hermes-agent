"""Tests for model_tools.py — function call dispatch, agent-loop interception, legacy toolsets."""

import json
from unittest.mock import call, patch

import pytest

from model_tools import (
    handle_function_call,
    get_all_tool_names,
    get_tool_definitions,
    get_toolset_for_tool,
    _AGENT_LOOP_TOOLS,
    _LEGACY_TOOLSET_MAP,
    TOOL_TO_TOOLSET_MAP,
)
from toolsets import resolve_toolset


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
            ),
            call(
                "post_tool_call",
                tool_name="web_search",
                args={"q": "test"},
                result='{"ok":true}',
                task_id="task-1",
                session_id="session-1",
                tool_call_id="call-1",
            ),
        ]


# =========================================================================
# Agent loop tools
# =========================================================================

class TestAgentLoopTools:
    def test_expected_tools_in_set(self):
        assert "todo" in _AGENT_LOOP_TOOLS
        assert "memory" in _AGENT_LOOP_TOOLS
        assert "session_search" in _AGENT_LOOP_TOOLS
        assert "delegate_task" in _AGENT_LOOP_TOOLS
        assert "session_model" in _AGENT_LOOP_TOOLS

    def test_no_regular_tools_in_set(self):
        assert "web_search" not in _AGENT_LOOP_TOOLS
        assert "terminal" not in _AGENT_LOOP_TOOLS


# =========================================================================
# Legacy toolset map
# =========================================================================

class TestLegacyToolsetMap:
    def test_expected_legacy_names(self):
        expected = [
            "web_tools", "terminal_tools", "vision_tools", "moa_tools",
            "image_tools", "skills_tools", "browser_tools", "cronjob_tools",
            "rl_tools", "file_tools", "tts_tools",
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
        assert "messaging_control" in names
        assert "qq_control" in names
        assert "employee_route_control" in names

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


class TestMessagingUnifiedControlVisibility:
    def test_unified_qq_toolset_resolves_to_messaging_control_only(self):
        assert set(resolve_toolset("qq")) == {"messaging_control"}

    def test_unified_weixin_toolset_resolves_to_messaging_control_only(self):
        assert set(resolve_toolset("weixin")) == {"messaging_control"}

    def test_get_tool_definitions_hides_platform_specific_tools_when_messaging_control_is_available(self):
        defs = [
            {
                "type": "function",
                "function": {"name": "messaging_control", "description": "", "parameters": {"type": "object"}},
            },
            {
                "type": "function",
                "function": {"name": "employee_route_control", "description": "", "parameters": {"type": "object"}},
            },
            {
                "type": "function",
                "function": {"name": "qq_control", "description": "", "parameters": {"type": "object"}},
            },
            {
                "type": "function",
                "function": {"name": "qq_social_control", "description": "", "parameters": {"type": "object"}},
            },
            {
                "type": "function",
                "function": {"name": "qq_intel_control", "description": "", "parameters": {"type": "object"}},
            },
            {
                "type": "function",
                "function": {"name": "qq_group_policy", "description": "", "parameters": {"type": "object"}},
            },
            {
                "type": "function",
                "function": {"name": "qq_group_archive", "description": "", "parameters": {"type": "object"}},
            },
            {
                "type": "function",
                "function": {"name": "qq_group_moderation", "description": "", "parameters": {"type": "object"}},
            },
            {
                "type": "function",
                "function": {"name": "qq_group_file", "description": "", "parameters": {"type": "object"}},
            },
            {
                "type": "function",
                "function": {"name": "weixin_control", "description": "", "parameters": {"type": "object"}},
            },
            {
                "type": "function",
                "function": {"name": "weixin_group_policy", "description": "", "parameters": {"type": "object"}},
            },
            {
                "type": "function",
                "function": {"name": "weixin_group_archive", "description": "", "parameters": {"type": "object"}},
            },
        ]

        with patch("model_tools.registry.get_definitions", return_value=defs):
            result = get_tool_definitions(quiet_mode=True)

        names = [item["function"]["name"] for item in result]
        assert "messaging_control" in names
        assert "employee_route_control" not in names
        assert "qq_control" not in names
        assert "qq_social_control" not in names
        assert "qq_intel_control" not in names
        assert "qq_group_policy" not in names
        assert "qq_group_archive" not in names
        assert "qq_group_moderation" not in names
        assert "qq_group_file" not in names
        assert "weixin_control" not in names
        assert "weixin_group_policy" not in names
        assert "weixin_group_archive" not in names

    def test_unified_toolset_keeps_only_messaging_control_and_strengthens_description(self):
        defs_by_name = {
            "messaging_control": {
                "type": "function",
                "function": {
                    "name": "messaging_control",
                    "description": "Unified messaging control.",
                    "parameters": {"type": "object"},
                },
            }
        }

        with patch(
            "model_tools.registry.get_definitions",
            side_effect=lambda tool_names, quiet=False: [
                defs_by_name[name] for name in sorted(tool_names) if name in defs_by_name
            ],
        ):
            result = get_tool_definitions(enabled_toolsets=["qq"], quiet_mode=True)

        assert [item["function"]["name"] for item in result] == ["messaging_control"]
        description = result[0]["function"]["description"].lower()
        assert "terminal" in description
        assert "execute_code" in description
        assert "approval" in description
        assert "platform" in description


class TestToolDefinitionOrdering:
    def test_get_tool_definitions_preserves_declared_toolset_order(self):
        defs_by_name = {
            name: {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in ("patch", "read_file", "search_files", "write_file")
        }

        with patch(
            "model_tools.registry.get_definitions",
            side_effect=lambda tool_names, quiet=False: [
                defs_by_name[name] for name in sorted(tool_names) if name in defs_by_name
            ],
        ):
            result = get_tool_definitions(enabled_toolsets=["file"], quiet_mode=True)

        assert [item["function"]["name"] for item in result] == [
            "read_file",
            "write_file",
            "patch",
            "search_files",
        ]
