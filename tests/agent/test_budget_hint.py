"""Tests for agent.budget_hint — context-window budget hint injection.

Adapted from openai/codex's TokenBudgetRemainingContext: when the estimated
request approaches the model's context window, the model is told how much
room remains so it can keep responses focused and avoid forced truncation or
lossy compression.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest

from agent.budget_hint import DEFAULT_BUDGET_HINT_THRESHOLD, build_budget_hint


class TestBuildBudgetHint:
    def test_below_threshold_returns_none(self):
        # 10K / 100K = 10% < 70% threshold → no hint, prompt prefix stays stable.
        assert build_budget_hint(10_000, 100_000, 0.70) is None

    def test_at_threshold_returns_hint(self):
        hint = build_budget_hint(70_000, 100_000, 0.70)
        assert hint is not None
        assert "70%" in hint
        assert "30,000" in hint  # remaining tokens formatted with commas

    def test_above_threshold_returns_hint(self):
        hint = build_budget_hint(85_000, 100_000, 0.70)
        assert hint is not None
        assert "85%" in hint
        assert "15,000" in hint

    def test_full_window_reports_zero_remaining(self):
        hint = build_budget_hint(100_000, 100_000, 0.50)
        assert hint is not None
        assert "100%" in hint
        assert "0" in hint

    def test_disabled_threshold_never_injects(self):
        # threshold <= 0 disables the hint entirely, even at 100% usage.
        assert build_budget_hint(100_000, 100_000, 0.0) is None
        assert build_budget_hint(100_000, 100_000, -1.0) is None

    def test_degenerate_inputs_return_none(self):
        assert build_budget_hint(-1, 100_000, 0.70) is None  # negative usage
        assert build_budget_hint(50_000, 0, 0.70) is None  # unknown window
        assert build_budget_hint(50_000, -100, 0.70) is None  # negative window

    def test_hint_mentions_context_budget_explicitly(self):
        hint = build_budget_hint(80_000, 100_000, 0.50)
        assert hint is not None
        assert "Context budget" in hint
        assert "tokens" in hint

    def test_default_threshold_constant_matches_documented_default(self):
        # The centralized default must mirror cli-config.yaml.example
        # (compression.budget_hint_threshold: 0.70) — turn-context builds
        # that never pass through init_agent fall back to it, and a 0.0
        # fallback would silently disable the hint for them.
        assert DEFAULT_BUDGET_HINT_THRESHOLD == 0.70


# ---------------------------------------------------------------------------
# build_turn_context wiring — the getattr-default seam (review #91974)
# ---------------------------------------------------------------------------


class _FakeTodoStore:
    def has_items(self):
        return True


class _FakeGuardrails:
    def reset_for_turn(self):
        pass


class _FakeAgent:
    """Minimal stand-in covering only what build_turn_context touches.

    Deliberately does NOT set ``_budget_hint_threshold`` — the point of the
    regression test is that the getattr default (DEFAULT_BUDGET_HINT_THRESHOLD,
    0.70) kicks in instead of a 0.0 fallback that would silently disable the
    hint for agents built outside init_agent (older serialized agents,
    alternate constructors).
    """

    def __init__(self):
        self.session_id = "sess-1"
        self.model = "test/model"
        self.provider = "openrouter"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = "sk-x"
        self.api_mode = "chat_completions"
        self.platform = "cli"
        self.quiet_mode = True
        self.max_iterations = 90
        self.tools = []
        self.valid_tool_names = set()
        self._skip_mcp_refresh = True
        self.compression_enabled = False
        self.context_compressor = types.SimpleNamespace(
            protect_first_n=2, protect_last_n=2, context_length=100_000
        )
        self._cached_system_prompt = "SYSTEM"
        self._memory_store = None
        self._memory_manager = None
        self._memory_nudge_interval = 0
        self._turns_since_memory = 0
        self._user_turn_count = 0
        self._todo_store = _FakeTodoStore()
        self._tool_guardrails = _FakeGuardrails()
        self._compression_warning = None
        self._interrupt_requested = False
        self._memory_write_origin = "assistant_tool"
        self._stream_context_scrubber = None
        self._stream_think_scrubber = None

    def _ensure_db_session(self):
        pass

    def _restore_primary_runtime(self):
        pass

    def _cleanup_dead_connections(self):
        return False

    def _emit_status(self, _msg):
        pass

    def _replay_compression_warning(self):
        pass

    def _hydrate_todo_store(self, *_a, **_k):
        pass

    def _safe_print(self, *_a, **_k):
        pass

    def _persist_session(self, messages, _history=None):
        pass


def _build(agent, **overrides):
    from agent.turn_context import build_turn_context

    kwargs = dict(
        agent=agent,
        user_message="hello",
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        restore_or_build_system_prompt=lambda *a, **k: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=lambda s: s,
        set_session_context=lambda _sid: None,
        set_current_write_origin=lambda _o: None,
        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
    )
    kwargs.update(overrides)
    return build_turn_context(**kwargs)


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


class TestBuildTurnContextBudgetHintWiring:
    """The hint must reach TurnContext.budget_hint through build_turn_context.

    Regression for the Enough1122 review on #91974: the getattr default for
    ``_budget_hint_threshold`` used to be 0.0 (disabled), contradicting the
    documented 0.70 default. Any code path that builds a turn context without
    passing through init_agent's config wiring silently lost the feature.
    """

    def test_default_threshold_applies_without_attribute(self):
        agent = _FakeAgent()
        assert not hasattr(agent, "_budget_hint_threshold")
        with patch(
            "agent.turn_context.estimate_messages_tokens_rough",
            return_value=90_000,  # 90% > 0.70 default → hint fires
        ):
            ctx = _build(agent)
        assert ctx.budget_hint, "hint must fire via the 0.70 default threshold"
        assert "90%" in ctx.budget_hint

    def test_explicit_attribute_overrides_default(self):
        agent = _FakeAgent()
        agent._budget_hint_threshold = 0.50  # explicit wiring wins
        with patch(
            "agent.turn_context.estimate_messages_tokens_rough",
            return_value=60_000,  # 60% > 0.50 override, but < 0.70 default
        ):
            ctx = _build(agent)
        assert ctx.budget_hint, "explicit threshold must win over the default"
        assert "60%" in ctx.budget_hint

    def test_zero_threshold_disables_hint(self):
        agent = _FakeAgent()
        agent._budget_hint_threshold = 0.0  # explicit opt-out
        with patch(
            "agent.turn_context.estimate_messages_tokens_rough",
            return_value=90_000,
        ):
            ctx = _build(agent)
        assert ctx.budget_hint == ""


class TestComposeUserApiContentBudgetHint:
    """compose_user_api_content forwards budget_hint as a third injection."""

    def _compose(self):
        from agent.turn_context import compose_user_api_content

        return compose_user_api_content

    def test_budget_hint_appended_after_other_injections(self):
        compose = self._compose()
        out = compose("hello", "MEM", "PLUGIN", "HINT")
        assert out is not None
        assert out.startswith("hello")
        assert "HINT" in out
        # Hint comes last; memory block comes before plugin context.
        assert out.index("HINT") > out.index("PLUGIN")
        assert out.index("PLUGIN") > out.index("MEM")

    def test_budget_hint_only_injection(self):
        compose = self._compose()
        out = compose("hello", "", "", "HINT")
        assert out is not None
        assert out == "hello\n\nHINT"

    def test_empty_budget_hint_keeps_old_behavior(self):
        # Backward compatibility: the 4th arg defaults to "" and changes nothing.
        compose = self._compose()
        assert compose("hello", "", "") is None  # no injections → None (unchanged)
        out = compose("hello", "MEM", "PLUGIN")
        assert out is not None
        assert "HINT" not in out
        assert "MEM" in out and "PLUGIN" in out

    def test_non_string_content_returns_none_even_with_hint(self):
        compose = self._compose()
        # Multimodal content defeats injection; the hint must not force one.
        assert compose(["part1", "part2"], "", "", "HINT") is None
