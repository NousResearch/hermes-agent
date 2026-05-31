"""Tests for tool execution safety guardrails — timeout, truncation, validation."""

import json
import os
import time
from unittest.mock import ANY, patch

import pytest

from model_tools import (
    _execute_tool_with_timeout,
    _load_tool_safety_config,
    _truncate_tool_result,
    _validate_tool_args_against_schema,
    handle_function_call,
)


# =========================================================================
# Tool Safety Config Loading
# =========================================================================

class TestToolSafetyLoadConfig:
    """Tests for _load_tool_safety_config() — lazy loading + env overrides."""

    def test_defaults_when_no_config(self):
        with (
            patch("model_tools._TOOL_SAFETY_CONFIG", None),
            patch("model_tools._TOOL_SAFETY_CONFIG_LOCK"),
            patch("hermes_cli.config.load_config", side_effect=Exception("no config")),
            patch.dict(os.environ, {}, clear=True),
        ):
            cfg = _load_tool_safety_config()
            assert cfg["tool_timeout_seconds"] == 0.0
            assert cfg["tool_max_output_chars"] == 0
            assert cfg["tool_validate_input"] is False

    def test_from_config_yaml(self):
        fake_user_config = {
            "tools": {
                "safety": {
                    "timeout_seconds": 30,
                    "max_output_chars": 5000,
                    "validate_input": True,
                }
            }
        }
        with (
            patch("model_tools._TOOL_SAFETY_CONFIG", None),
            patch("model_tools._TOOL_SAFETY_CONFIG_LOCK"),
            patch("hermes_cli.config.load_config", return_value=fake_user_config),
            patch.dict(os.environ, {}, clear=True),
        ):
            cfg = _load_tool_safety_config()
            assert cfg["tool_timeout_seconds"] == 30.0
            assert cfg["tool_max_output_chars"] == 5000
            assert cfg["tool_validate_input"] is True

    def test_env_var_overrides_config(self):
        with (
            patch("model_tools._TOOL_SAFETY_CONFIG", None),
            patch("model_tools._TOOL_SAFETY_CONFIG_LOCK"),
            patch("hermes_cli.config.load_config", return_value={"tools": {"safety": {"timeout_seconds": 10}}}),
            patch.dict(os.environ, {"HERMES_TOOL_TIMEOUT": "60"}, clear=True),
        ):
            cfg = _load_tool_safety_config()
            assert cfg["tool_timeout_seconds"] == 60.0

    def test_env_var_validate_input_triggers(self):
        with (
            patch("model_tools._TOOL_SAFETY_CONFIG", None),
            patch("model_tools._TOOL_SAFETY_CONFIG_LOCK"),
            patch("hermes_cli.config.load_config", return_value={}),
            patch.dict(os.environ, {"HERMES_TOOL_VALIDATE_INPUT": "true"}, clear=True),
        ):
            cfg = _load_tool_safety_config()
            assert cfg["tool_validate_input"] is True

    def test_caching_returns_same_object(self):
        with (
            patch("model_tools._TOOL_SAFETY_CONFIG", None),
            patch("model_tools._TOOL_SAFETY_CONFIG_LOCK"),
            patch("hermes_cli.config.load_config", return_value={}),
            patch.dict(os.environ, {}, clear=True),
        ):
            cfg1 = _load_tool_safety_config()
            cfg2 = _load_tool_safety_config()
            assert cfg1 is cfg2


# =========================================================================
# Tool Input Validation
# =========================================================================

class TestToolValidation:
    """Tests for _validate_tool_args_against_schema()."""

    SCHEMA = {
        "name": "test_tool",
        "description": "A test tool",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The name"},
                "count": {"type": "integer", "description": "Item count"},
                "tags": {"type": "array", "description": "Item tags"},
            },
            "required": ["name", "count"],
        },
    }

    def test_valid_args_passes(self):
        with patch("model_tools.registry.get_schema", return_value=self.SCHEMA):
            error = _validate_tool_args_against_schema("test_tool", {"name": "foo", "count": 3})
            assert error is None

    def test_missing_required_field_returns_error(self):
        with patch("model_tools.registry.get_schema", return_value=self.SCHEMA):
            error = _validate_tool_args_against_schema("test_tool", {"name": "foo"})
            assert error is not None
            assert "count" in error
            assert "missing required field" in error

    def test_type_mismatch_string_returns_error(self):
        with patch("model_tools.registry.get_schema", return_value=self.SCHEMA):
            error = _validate_tool_args_against_schema("test_tool", {"name": True, "count": 3})
            assert error is not None
            assert "expected string" in error

    def test_type_mismatch_integer_returns_error(self):
        with patch("model_tools.registry.get_schema", return_value=self.SCHEMA):
            error = _validate_tool_args_against_schema("test_tool", {"name": "foo", "count": "bar"})
            assert error is not None
            assert "expected integer" in error

    def test_array_field_passes(self):
        with patch("model_tools.registry.get_schema", return_value=self.SCHEMA):
            error = _validate_tool_args_against_schema("test_tool", {"name": "foo", "count": 3, "tags": ["a", "b"]})
            assert error is None

    def test_type_mismatch_array_returns_error(self):
        with patch("model_tools.registry.get_schema", return_value=self.SCHEMA):
            error = _validate_tool_args_against_schema("test_tool", {"name": "foo", "count": 3, "tags": "not_an_array"})
            assert error is not None
            assert "expected array" in error

    def test_empty_string_required_field_returns_error(self):
        with patch("model_tools.registry.get_schema", return_value=self.SCHEMA):
            error = _validate_tool_args_against_schema("test_tool", {"name": "", "count": 3})
            assert error is not None
            assert "missing required field" in error

    def test_no_schema_returns_none(self):
        with patch("model_tools.registry.get_schema", return_value=None):
            error = _validate_tool_args_against_schema("unknown_tool", {"name": "foo"})
            assert error is None

    def test_none_required_field_detected(self):
        with patch("model_tools.registry.get_schema", return_value=self.SCHEMA):
            error = _validate_tool_args_against_schema("test_tool", {"name": "foo", "count": None})
            assert error is not None
            assert "missing required field" in error


# =========================================================================
# Tool Output Truncation
# =========================================================================

class TestToolTruncation:
    """Tests for _truncate_tool_result()."""

    def test_no_truncation_when_disabled(self):
        assert _truncate_tool_result("foo", "hello world", 0) == "hello world"
        assert _truncate_tool_result("foo", "hello world", -1) == "hello world"

    def test_no_truncation_when_within_limit(self):
        result = _truncate_tool_result("foo", "short", 100)
        assert result == "short"

    def test_truncation_when_exceeds_limit(self):
        long_text = "a" * 200
        result = _truncate_tool_result("foo", long_text, 100)
        assert len(result) <= 130  # 100 + marker (~26 chars)
        assert "truncated" in result
        assert "... (truncated" in result
        assert "100 chars" in result

    def test_exact_limit_no_truncation(self):
        text = "a" * 50
        result = _truncate_tool_result("foo", text, 50)
        assert result == text

    def test_empty_string_returns_empty(self):
        assert _truncate_tool_result("foo", "", 100) == ""


# =========================================================================
# Tool Timeout
# =========================================================================

class TestToolTimeout:
    """Tests for _execute_tool_with_timeout()."""

    def test_dispatches_synchronously_when_timeout_is_zero(self):
        with patch("model_tools.registry.dispatch", return_value='{"ok":true}') as mock_dispatch:
            result = _execute_tool_with_timeout(
                "web_search", {"q": "test"}, task_id="t1", user_task="ut1",
                timeout_seconds=0.0,
            )
        assert result == '{"ok":true}'
        mock_dispatch.assert_called_once_with(
            "web_search", {"q": "test"},
            task_id="t1", user_task="ut1",
        )

    def test_execute_code_passes_enabled_tools(self):
        with patch("model_tools.registry.dispatch", return_value='{"ok":true}') as mock_dispatch:
            result = _execute_tool_with_timeout(
                "execute_code", {"code": "print(1)"}, task_id="t1",
                enabled_tools=["bash", "python"],
                timeout_seconds=0.0,
            )
        assert result == '{"ok":true}'
        mock_dispatch.assert_called_once_with(
            "execute_code", {"code": "print(1)"},
            task_id="t1",
            enabled_tools=["bash", "python"],
        )

    def test_timeout_returns_error_json(self):
        with patch("model_tools.registry.dispatch") as mock_dispatch:
            def _slow(_name, _args, **_kw):
                time.sleep(10)
                return "done"

            mock_dispatch.side_effect = _slow
            result = _execute_tool_with_timeout(
                "slow_tool", {}, task_id="t1",
                timeout_seconds=0.1,
            )
            parsed = json.loads(result)
            assert "error" in parsed
            assert "did not complete within" in parsed["error"]

    def test_fast_tool_with_timeout_completes(self):
        with patch("model_tools.registry.dispatch", return_value='{"ok":true}') as mock_dispatch:
            result = _execute_tool_with_timeout(
                "fast_tool", {}, task_id="t1",
                timeout_seconds=5.0,
            )
        assert result == '{"ok":true}'


# =========================================================================
# Integration: handle_function_call respects safety config
# =========================================================================

class TestHandleFunctionCallSafety:
    """Tests that the safety wrappers integrate correctly in handle_function_call."""

    def test_validation_blocked_by_config(self):
        fake_safety_cfg = {
            "tool_timeout_seconds": 0.0,
            "tool_max_output_chars": 0,
            "tool_validate_input": True,
        }
        with (
            patch("model_tools._load_tool_safety_config", return_value=fake_safety_cfg),
            patch("model_tools.registry.get_schema", return_value=None),
            patch("model_tools.registry.dispatch", return_value='{"ok":true}') as mock_dispatch,
        ):
            result = handle_function_call("web_search", {"q": "test"})
        assert result == '{"ok":true}'
        mock_dispatch.assert_called_once()

    def test_validation_rejects_missing_required(self):
        schema = {
            "name": "test_tool",
            "parameters": {
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
            },
        }
        fake_safety_cfg = {
            "tool_timeout_seconds": 0.0,
            "tool_max_output_chars": 0,
            "tool_validate_input": True,
        }
        with (
            patch("model_tools._load_tool_safety_config", return_value=fake_safety_cfg),
            patch("model_tools.registry.get_schema", return_value=schema),
        ):
            result = handle_function_call("test_tool", {"not_name": "bar"})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "missing required field" in parsed["error"]

    def test_truncation_happens_when_configured(self):
        long_result = "x" * 5000
        fake_safety_cfg = {
            "tool_timeout_seconds": 0.0,
            "tool_max_output_chars": 100,
            "tool_validate_input": False,
        }
        with (
            patch("model_tools._load_tool_safety_config", return_value=fake_safety_cfg),
            patch("model_tools.registry.dispatch", return_value=long_result),
        ):
            result = handle_function_call("web_search", {"q": "test"})
        assert len(result) < 5000
        assert "... (truncated" in result

    def test_timeout_fires_when_configured(self):
        fake_safety_cfg = {
            "tool_timeout_seconds": 0.05,
            "tool_max_output_chars": 0,
            "tool_validate_input": False,
        }
        with (
            patch("model_tools._load_tool_safety_config", return_value=fake_safety_cfg),
            patch("model_tools.registry.dispatch") as mock_dispatch,
        ):
            def _slow(_name, _args, **_kw):
                time.sleep(5)
                return "done"

            mock_dispatch.side_effect = _slow
            result = handle_function_call("web_search", {"q": "slow"})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "did not complete within" in parsed["error"]

    def test_all_disabled_no_overhead(self):
        fake_safety_cfg = {
            "tool_timeout_seconds": 0.0,
            "tool_max_output_chars": 0,
            "tool_validate_input": False,
        }
        with (
            patch("model_tools._load_tool_safety_config", return_value=fake_safety_cfg),
            patch("model_tools.registry.dispatch", return_value='{"ok":true}') as mock_dispatch,
            patch("hermes_cli.plugins.invoke_hook", return_value=[]),
        ):
            result = handle_function_call("web_search", {"q": "test"}, task_id="t1", tool_call_id="c1", session_id="s1")
        assert result == '{"ok":true}'
        mock_dispatch.assert_called_once()
