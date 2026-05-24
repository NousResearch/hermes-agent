"""Tests for RuntimeGuard — blocks forbidden patterns at tool/gateway/message level."""

import json
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _clear_guard_cache():
    """Clear the RuntimeGuard pattern cache before each test."""
    from agent.runtime_guard import _invalidate_cache, _user_patterns
    _invalidate_cache()
    _user_patterns.clear()
    yield
    _invalidate_cache()
    _user_patterns.clear()


@pytest.fixture
def tmp_hermes_home(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for tests."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


# ── Unit tests ───────────────────────────────────────────────────────

class TestRuntimeGuardPatterns:
    """Test default forbidden patterns compile and match correctly."""

    def test_default_patterns_compile(self):
        """All default patterns should compile without error."""
        from agent.runtime_guard import _load_patterns
        patterns = _load_patterns()
        assert len(patterns) >= 10  # At least the defaults
        for gp, compiled in patterns:
            assert isinstance(compiled, re.Pattern)
            assert gp.pattern  # pattern string is non-empty

    def test_blocks_hermes_labs_variants(self):
        """Should block various forms of 'Hermes Labs'."""
        from agent.runtime_guard import RuntimeGuard
        texts = [
            "Check Hermes Labs documentation",
            "Connect to hermes_labs server",
            "hermes-labs config",
            "HERMES LABS settings",
        ]
        for text in texts:
            result = RuntimeGuard.check_message(text)
            assert not result.allowed, f"Should block: {text}"
            assert "Hermes Labs" in result.reason or "legacy" in result.reason.lower()

    def test_blocks_hermesnous_variants(self):
        """Should block various forms of 'HermesNous'."""
        from agent.runtime_guard import RuntimeGuard
        texts = [
            "Use HermesNous API",
            "Connect to hermes_nous",
            "hermes-nous endpoint",
            "HERMESNOUS config",
        ]
        for text in texts:
            result = RuntimeGuard.check_message(text)
            assert not result.allowed, f"Should block: {text}"
            assert "HermesNous" in result.reason or "legacy" in result.reason.lower()

    def test_blocks_forbidden_ports(self):
        """Should block forbidden runtime ports 7421/7422."""
        from agent.runtime_guard import RuntimeGuard
        texts = [
            "Connect to 127.0.0.1:7421",
            "Use localhost:7422",
            "http://127.0.0.1:7421/api",
            "http://localhost:7422/api",
        ]
        for text in texts:
            result = RuntimeGuard.check_message(text)
            assert not result.allowed, f"Should block: {text}"

    def test_blocks_hermes_mcp(self):
        """Should block hermes-mcp references."""
        from agent.runtime_guard import RuntimeGuard
        texts = [
            "Use hermes-mcp server",
            "Connect to hermes_mcp",
        ]
        for text in texts:
            result = RuntimeGuard.check_message(text)
            assert not result.allowed, f"Should block: {text}"

    def test_false_positives(self):
        """Common words should NOT be blocked (no false positives)."""
        from agent.runtime_guard import RuntimeGuard
        safe_texts = [
            "Go to the laboratory",
            "I love labs",
            "The lab results are ready",
            "Nous avons besoin de plus de temps",  # French "we need"
            "The MCP protocol is standard",
            "Check the server at port 8080",
            "Connect to localhost:3000",
            "Using 127.0.0.1:5000 for dev",
        ]
        for text in safe_texts:
            result = RuntimeGuard.check_message(text)
            assert result.allowed, f"Should NOT block: {text}"

    def test_empty_text_allowed(self):
        """Empty text should be allowed."""
        from agent.runtime_guard import RuntimeGuard
        result = RuntimeGuard.check_message("")
        assert result.allowed
        result = RuntimeGuard.check_message(None)
        assert result.allowed

    def test_tool_call_check_blocks_in_args(self):
        """Should block forbidden patterns in tool call arguments."""
        from agent.runtime_guard import RuntimeGuard
        result = RuntimeGuard.check_tool_call(
            "web_search",
            {"query": "HermesNous documentation"},
        )
        assert not result.allowed

    def test_tool_call_allowed(self):
        """Safe tool calls should pass."""
        from agent.runtime_guard import RuntimeGuard
        result = RuntimeGuard.check_tool_call(
            "web_search",
            {"query": "Python best practices"},
        )
        assert result.allowed

    def test_terminal_command_blocked(self):
        """Should block forbidden patterns in terminal commands."""
        from agent.runtime_guard import RuntimeGuard
        result = RuntimeGuard.check_terminal_command("curl http://127.0.0.1:7421")
        assert not result.allowed

    def test_terminal_command_allowed(self):
        """Safe commands should pass."""
        from agent.runtime_guard import RuntimeGuard
        result = RuntimeGuard.check_terminal_command("ls -la /tmp")
        assert result.allowed


class TestRuntimeGuardDynamicPatterns:
    """Test adding/removing user-configured patterns."""

    def test_add_pattern(self):
        """Adding a pattern should block matching text."""
        from agent.runtime_guard import RuntimeGuard, _invalidate_cache
        _invalidate_cache()
        pid = RuntimeGuard.add_pattern(r"(?i)\btest-forbidden\b", "Test forbidden")
        assert pid is not None
        result = RuntimeGuard.check_message("Use test-forbidden tool")
        assert not result.allowed
        assert "Test forbidden" in result.reason

    def test_remove_pattern(self):
        """Removing a pattern should allow previously blocked text."""
        from agent.runtime_guard import RuntimeGuard, _invalidate_cache
        _invalidate_cache()
        pid = RuntimeGuard.add_pattern(r"(?i)\btemp-block\b", "Temp block")
        result = RuntimeGuard.check_message("Use temp-block here")
        assert not result.allowed
        removed = RuntimeGuard.remove_pattern(pid)
        assert removed
        result = RuntimeGuard.check_message("Use temp-block here")
        assert result.allowed

    def test_invalid_pattern_returns_none(self):
        """Invalid regex should return None."""
        from agent.runtime_guard import RuntimeGuard, _invalidate_cache
        _invalidate_cache()
        pid = RuntimeGuard.add_pattern("[invalid(regex", "Bad pattern")
        assert pid is None


class TestRuntimeGuardListPatterns:
    """Test listing patterns."""

    def test_list_patterns_returns_dicts(self):
        """list_patterns should return a list of pattern dicts."""
        from agent.runtime_guard import RuntimeGuard
        patterns = RuntimeGuard.list_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) >= 10  # At least defaults
        for p in patterns:
            assert "id" in p
            assert "pattern" in p
            assert "description" in p
            assert "source" in p


class TestRuntimeGuardIntegration:
    """Integration tests: guard wired into model_tools, terminal_tool, gateway."""

    def test_model_tools_handle_function_call_blocked(self, tmp_hermes_home):
        """handle_function_call should block a forbidden tool call."""
        from model_tools import handle_function_call
        result = handle_function_call(
            "web_search",
            {"query": "HermesNous API docs"},
        )
        data = json.loads(result)
        assert "error" in data
        assert "runtime guard" in data["error"].lower() or "blocked" in data["error"].lower()

    def test_model_tools_handle_function_call_allowed(self, tmp_hermes_home):
        """handle_function_call should allow safe tool calls."""
        from model_tools import handle_function_call
        # This should pass the guard (but may fail for other reasons like
        # missing API key — we just check it's not a guard block)
        result = handle_function_call(
            "web_search",
            {"query": "Python tutorials"},
        )
        data = json.loads(result)
        # Should NOT be a runtime guard error
        if "error" in data:
            assert "runtime guard" not in data["error"].lower()

    def test_terminal_tool_blocked(self):
        """Terminal tool should block forbidden commands."""
        from tools.terminal_tool import terminal_tool
        # Use force=False to allow filter checks
        with patch.dict(os.environ, {"HERMES_HOME": tempfile.mkdtemp()}):
            result = terminal_tool("curl http://127.0.0.1:7421", force=False)
            data = json.loads(result)
            assert data.get("status") == "blocked" or data.get("exit_code") == -1

    def test_terminal_tool_allowed(self):
        """Terminal tool should allow safe commands."""
        from tools.terminal_tool import terminal_tool
        with patch.dict(os.environ, {"HERMES_HOME": tempfile.mkdtemp()}):
            result = terminal_tool("echo hello", force=False)
            data = json.loads(result)
            # Should NOT be blocked by guard
            assert data.get("status") != "blocked" or data.get("exit_code") == 0
