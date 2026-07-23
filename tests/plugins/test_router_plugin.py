"""Regression tests for the router plugin (plugins/router/).

Covers the pure functions that don't require LLM mocking — pattern matching,
noise filtering, prompt building, binary resolution, cache behaviour, and the
classification pipeline up to (but not including) the LLM call itself.

The actual LLM inference path and subprocess spawning are integration-tested
via the pre_agent_dispatch hook dispatch tests, not here.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from unittest import mock

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME so router config lookups don't touch real config."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _load_module():
    """Import the router plugin module directly."""
    repo_root = Path(__file__).resolve().parents[2]
    sys_path = str(repo_root)
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    try:
        return importlib.import_module("plugins.router")
    finally:
        if sys_path in sys.path:
            # Don't leave the import path polluted
            pass


ROUTER = None  # cached module


def _mod():
    global ROUTER
    if ROUTER is None:
        ROUTER = _load_module()
    return ROUTER


# ── Pattern matching ──────────────────────────────────────────────────

class TestMatchPatterns:
    def test_empty_patterns(self):
        """No patterns → no match regardless of message."""
        mod = _mod()
        assert mod._match_patterns("anything", []) is False
        assert mod._match_patterns("", []) is False

    def test_simple_literal_match(self):
        mod = _mod()
        assert mod._match_patterns("deploy to production", ["deploy"]) is True

    def test_no_match(self):
        mod = _mod()
        assert mod._match_patterns("hello world", ["deploy", "restart"]) is False

    def test_case_insensitive_match(self):
        mod = _mod()
        assert mod._match_patterns("DEPLOY NOW", ["deploy"]) is True

    def test_regex_pattern(self):
        mod = _mod()
        assert mod._match_patterns("fix bug #1234", [r"#\d+"]) is True

    def test_regex_word_boundary(self):
        mod = _mod()
        # "test" should not match "testing" when using word boundary
        assert mod._match_patterns("testing", [r"\btest\b"]) is False
        assert mod._match_patterns("run test now", [r"\btest\b"]) is True

    def test_invalid_regex_warns_but_continues(self, caplog):
        """Invalid regex patterns are logged and treated as no-match."""
        mod = _mod()
        result = mod._match_patterns("anything", [r"[invalid("])
        assert result is False
        # Should have logged a warning
        assert any("Invalid router pattern" in r.message for r in caplog.records)

    def test_multiple_patterns_first_match_wins(self):
        mod = _mod()
        # Second pattern would match but first already did
        assert mod._match_patterns("hello", ["world", "hello"]) is True


# ── Noise line detection ──────────────────────────────────────────────

class TestIsNoiseLine:
    def test_empty_line_is_noise(self):
        mod = _mod()
        assert mod._is_noise_line("") is True
        assert mod._is_noise_line("   ") is True

    def test_tokens_line_is_noise(self):
        mod = _mod()
        assert mod._is_noise_line("Tokens: 1.2k/200k") is True

    def test_cost_line_is_noise(self):
        mod = _mod()
        assert mod._is_noise_line("Cost: $0.003") is True

    def test_model_line_is_noise(self):
        mod = _mod()
        assert mod._is_noise_line("Model: deepseek-v4-pro") is True

    def test_session_line_is_noise(self):
        mod = _mod()
        assert mod._is_noise_line("Session: abc123") is True

    def test_box_drawing_characters_are_noise(self):
        mod = _mod()
        for line in ["──────", "╭─ Results", "╰─ done", "├─ step", "│ data"]:
            assert mod._is_noise_line(line), f"should be noise: {line!r}"

    def test_circle_symbols_are_noise(self):
        mod = _mod()
        assert mod._is_noise_line("⏺ Starting") is True
        assert mod._is_noise_line("● Running") is True
        assert mod._is_noise_line("○ Idle") is True

    def test_real_content_is_not_noise(self):
        mod = _mod()
        assert mod._is_noise_line("Hello, this is a real response.") is False
        assert mod._is_noise_line("   def foo(): pass") is False


# ── Hermes response extraction ────────────────────────────────────────

class TestExtractHermesResponse:
    def test_removes_noise_lines(self):
        mod = _mod()
        output = "\n".join([
            "Tokens: 100",
            "Model: test",
            "",
            "The actual response content",
            "",
            "Cost: $0.001",
        ])
        result = mod._extract_hermes_response(output)
        assert result == "The actual response content"

    def test_fallback_to_raw_on_all_noise(self):
        """When all lines are noise, returns the raw output."""
        mod = _mod()
        output = "Tokens: 100\nModel: test\nCost: $0.001"
        result = mod._extract_hermes_response(output)
        assert result == output

    def test_empty_output(self):
        mod = _mod()
        assert mod._extract_hermes_response("") == ""

    def test_preserves_content_structure(self):
        mod = _mod()
        output = "Line 1\nLine 2\nLine 3"
        result = mod._extract_hermes_response(output)
        assert result == "Line 1\nLine 2\nLine 3"


# ── Orchestrator prompt building ──────────────────────────────────────

class TestBuildOrchestratorPrompt:
    def test_simple_message_no_history_no_context(self):
        mod = _mod()
        result = mod._build_orchestrator_prompt("do something")
        assert "=== Current user request ===" in result
        assert "do something" in result
        assert "=== Previous conversation ===" not in result

    def test_with_history(self):
        mod = _mod()
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = mod._build_orchestrator_prompt("new task", history=history)
        assert "=== Previous conversation ===" in result
        assert "[user]: hello" in result
        assert "[assistant]: hi there" in result
        assert "=== Current user request ===" in result
        assert "new task" in result

    def test_history_truncated_to_last_6_messages(self):
        mod = _mod()
        history = [
            {"role": "user", "content": f"msg{i}"} for i in range(20)
        ]
        result = mod._build_orchestrator_prompt("final", history=history)
        # Only last 6 (3 exchanges) should appear
        assert "msg0" not in result
        assert "msg14" in result  # index 14 is in last 6
        assert "msg19" in result

    def test_history_content_truncated_at_2000_chars(self):
        mod = _mod()
        long_text = "x" * 3000
        history = [{"role": "user", "content": long_text}]
        result = mod._build_orchestrator_prompt("task", history=history)
        # Content should be truncated
        assert "x" * 2500 not in result

    def test_history_with_multimodal_content(self):
        """List-type content (from multimodal messages) is flattened."""
        mod = _mod()
        history = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "image description"},
                {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
            ],
        }]
        result = mod._build_orchestrator_prompt("task", history=history)
        assert "image description" in result

    def test_with_session_context_cwd(self):
        mod = _mod()
        result = mod._build_orchestrator_prompt(
            "task",
            session_context={"cwd": "/home/user/project"},
        )
        assert "Working directory: /home/user/project" in result

    def test_empty_session_context_no_cwd_line(self):
        mod = _mod()
        result = mod._build_orchestrator_prompt(
            "task",
            session_context={"cwd": ""},
        )
        assert "Working directory:" not in result


# ── Hermes binary resolution ─────────────────────────────────────────

class TestResolveHermesBin:
    def test_env_var_takes_priority(self, monkeypatch):
        mod = _mod()
        monkeypatch.setenv("HERMES_BIN", "/custom/path/hermes")
        assert mod._resolve_hermes_bin() == "/custom/path/hermes"

    def test_env_var_even_if_file_missing(self, monkeypatch):
        """Env var is returned regardless of file existence."""
        mod = _mod()
        monkeypatch.setenv("HERMES_BIN", "/nonexistent/path/hermes")
        assert mod._resolve_hermes_bin() == "/nonexistent/path/hermes"

    def test_fallback_to_hermes_when_no_candidates_found(self, monkeypatch):
        """When no env var and no candidates exist, returns 'hermes'."""
        mod = _mod()
        monkeypatch.delenv("HERMES_BIN", raising=False)
        with mock.patch.object(mod.os.path, "isfile", return_value=False):
            assert mod._resolve_hermes_bin() == "hermes"


# ── Classification cache ──────────────────────────────────────────────

class TestClassificationCache:
    def test_get_miss_when_empty(self):
        mod = _mod()
        # Clear cache so we start fresh
        mod._classification_cache.clear()
        assert mod._get_cached_classification("session-1") is None

    def test_cache_hit_within_ttl(self):
        mod = _mod()
        mod._classification_cache.clear()
        mod._cache_classification("session-1", {"classification": "complex"})
        result = mod._get_cached_classification("session-1")
        assert result == {"classification": "complex"}

    def test_cache_expired_after_ttl(self, monkeypatch):
        mod = _mod()
        mod._classification_cache.clear()
        # Cache with a timestamp far in the past
        mod._classification_cache["session-old"] = (
            0,  # epoch 0
            {"classification": "complex"},
        )
        result = mod._get_cached_classification("session-old")
        assert result is None
        # Expired entry should be cleaned up
        assert "session-old" not in mod._classification_cache

    def test_cache_different_sessions_independent(self):
        mod = _mod()
        mod._classification_cache.clear()
        mod._cache_classification("session-a", {"classification": "simple"})
        mod._cache_classification("session-b", {"classification": "complex"})
        assert mod._get_cached_classification("session-a") == \
            {"classification": "simple"}
        assert mod._get_cached_classification("session-b") == \
            {"classification": "complex"}


# ── Classify (without LLM) ───────────────────────────────────────────

class TestClassify:
    def test_disabled_returns_simple(self):
        """When router.enabled is false, classify returns simple."""
        mod = _mod()
        cfg: dict = {"enabled": False}
        result = mod.classify("complex task requiring deep analysis", cfg)
        assert result["classification"] == "simple"

    def test_disabled_without_config_key(self):
        """Missing 'enabled' key → same as disabled."""
        mod = _mod()
        result = mod.classify("anything", {})
        assert result["classification"] == "simple"

    def test_always_simple_pattern_wins(self):
        """always_simple patterns bypass everything → simple."""
        mod = _mod()
        cfg: dict = {
            "enabled": True,
            "rules": {
                "always_simple": ["hello", "how are you"],
                "always_complex": ["deploy"],
            },
        }
        result = mod.classify("hello world", cfg)
        assert result["classification"] == "simple"
        assert result["confidence"] == 1.0
        assert "always_simple" in result["reason"]

    def test_always_complex_pattern_wins(self):
        """always_complex patterns → complex regardless of LLM."""
        mod = _mod()
        cfg: dict = {
            "enabled": True,
            "rules": {
                "always_simple": ["hello"],
                "always_complex": ["deploy", "restart"],
            },
        }
        result = mod.classify("please deploy now", cfg)
        assert result["classification"] == "complex"
        assert result["confidence"] == 1.0
        assert "always_complex" in result["reason"]

    def test_always_simple_beats_always_complex(self):
        """Simple patterns are checked first — they beat complex patterns."""
        mod = _mod()
        cfg: dict = {
            "enabled": True,
            "rules": {
                "always_simple": ["deploy safely"],
                "always_complex": ["deploy"],
            },
        }
        # "deploy safely" matches BOTH patterns
        result = mod.classify("deploy safely", cfg)
        assert result["classification"] == "simple"

    def test_no_pattern_match_falls_through_to_llm(self):
        """When neither pattern matches, LLM classifier is called.
        We mock to avoid real inference."""
        mod = _mod()
        cfg: dict = {
            "enabled": True,
            "rules": {"always_simple": [], "always_complex": []},
        }
        with mock.patch.object(
            mod, "_classify_with_llm",
            return_value={"classification": "complex", "confidence": 0.9,
                          "reason": "mock"},
        ):
            result = mod.classify("complex analysis request", cfg)
        assert result["classification"] == "complex"
        assert result["confidence"] == 0.9


# ── Hook callback (_on_pre_agent_dispatch) ────────────────────────────

class TestOnPreAgentDispatch:
    def test_returns_none_when_disabled(self):
        """When router is disabled, hook returns None (no-op)."""
        mod = _mod()
        with mock.patch.object(mod, "_load_router_config",
                               return_value={"enabled": False}):
            result = mod._on_pre_agent_dispatch(message="test")
        assert result is None

    def test_returns_none_for_empty_message(self):
        """Empty messages are ignored."""
        mod = _mod()
        with mock.patch.object(mod, "_load_router_config",
                               return_value={"enabled": True}):
            result = mod._on_pre_agent_dispatch(message="")
        assert result is None

    def test_returns_none_for_non_string_message(self):
        """Non-string messages are ignored."""
        mod = _mod()
        with mock.patch.object(mod, "_load_router_config",
                               return_value={"enabled": True}):
            result = mod._on_pre_agent_dispatch(message=123)
        assert result is None

    def test_cached_simple_returns_none(self):
        """Cached 'simple' classification → no routing, returns None."""
        mod = _mod()
        mod._classification_cache.clear()
        mod._cache_classification("sess", {"classification": "simple"})
        with mock.patch.object(mod, "_load_router_config",
                               return_value={"enabled": True}):
            result = mod._on_pre_agent_dispatch(
                message="hello",
                session_key="sess",
            )
        assert result is None

    def test_cached_complex_routes_to_orchestrator(self):
        """Cached 'complex' classification → routes."""
        mod = _mod()
        mod._classification_cache.clear()
        mod._cache_classification("sess", {"classification": "complex"})
        with mock.patch.object(mod, "_load_router_config",
                               return_value={
                                   "enabled": True,
                                   "orchestrator": {"profile": "test-profile"},
                               }):
            with mock.patch.object(
                mod, "route_to_orchestrator",
                return_value="orchestrator response",
            ):
                result = mod._on_pre_agent_dispatch(
                    message="complex task",
                    session_key="sess",
                )
        assert result is not None
        assert result["action"] == "route"
        assert result["result"] == "orchestrator response"

    def test_classification_simple_returns_none(self):
        """Fresh classification → simple → no routing."""
        mod = _mod()
        mod._classification_cache.clear()
        with mock.patch.object(mod, "_load_router_config",
                               return_value={"enabled": True}):
            with mock.patch.object(
                mod, "classify",
                return_value={"classification": "simple", "confidence": 0.8,
                              "reason": "looks easy"},
            ):
                result = mod._on_pre_agent_dispatch(
                    message="what time is it",
                    session_key="sess",
                )
        assert result is None

    def test_streamed_flag_set_when_callback_provided(self):
        """When stream_callback is provided, 'streamed' is True."""
        mod = _mod()
        mod._classification_cache.clear()
        mod._cache_classification("sess", {"classification": "complex"})

        def _noop(text: str) -> None:
            pass

        with mock.patch.object(mod, "_load_router_config",
                               return_value={
                                   "enabled": True,
                                   "orchestrator": {"profile": "test-profile"},
                               }):
            with mock.patch.object(
                mod, "route_to_orchestrator",
                return_value="response",
            ):
                result = mod._on_pre_agent_dispatch(
                    message="complex task",
                    session_key="sess",
                    stream_callback=_noop,
                )
        assert result["streamed"] is True

    def test_streamed_false_when_no_callback(self):
        """When no stream_callback, 'streamed' is False."""
        mod = _mod()
        mod._classification_cache.clear()
        mod._cache_classification("sess", {"classification": "complex"})
        with mock.patch.object(mod, "_load_router_config",
                               return_value={
                                   "enabled": True,
                                   "orchestrator": {"profile": "test-profile"},
                               }):
            with mock.patch.object(
                mod, "route_to_orchestrator",
                return_value="response",
            ):
                result = mod._on_pre_agent_dispatch(
                    message="complex task",
                    session_key="sess",
                )
        assert result["streamed"] is False


# ── Plugin registration ───────────────────────────────────────────────

class TestPluginRegistration:
    def test_register_calls_register_hook(self):
        """register(ctx) registers the pre_agent_dispatch hook."""
        mod = _mod()

        class FakeCtx:
            def __init__(self):
                self.hooks: list = []

            def register_hook(self, name, callback):
                self.hooks.append((name, callback))

        ctx = FakeCtx()
        mod.register(ctx)
        assert len(ctx.hooks) == 1
        assert ctx.hooks[0][0] == "pre_agent_dispatch"
        assert callable(ctx.hooks[0][1])
