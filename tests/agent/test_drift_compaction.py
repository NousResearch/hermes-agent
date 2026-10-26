"""Tests for the opt-in JEV drift-triggered compaction policy.

Covers ``agent.turn_context._should_drift_compact`` — the pure predicate that decides
whether a topic change may compact the old context early — and the lexical dependence
heuristic behind it. Both are intentionally side-effect-free so the policy can be
verified without constructing a live agent or DB. ``TestDriftCompactionPass`` then
drives the real turn pipeline (mocked provider, real ``run_conversation``).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.turn_context import (
    _DRIFT_DEPENDENT_MIN,
    _DRIFT_INDEPENDENT_MAX,
    _should_drift_compact,
    drift_dependence_score,
    drift_score_ambiguous,
)
from run_agent import AIAgent


def _decide(**overrides):
    """Call the predicate with sensible defaults (new topic + large context => fire)."""
    kwargs = dict(
        enabled=True,
        tokens=60_000,
        window_tokens=65_536,
        min_ratio=0.5,
        dependence_score=0.0,
        confirm_independent=None,
        cooldown_active=False,
        in_task=False,
        floor_tokens=20_000,
        last_compaction_tokens=0,
    )
    kwargs.update(overrides)
    return _should_drift_compact(**kwargs)


class TestShouldDriftCompact:

    def test_independent_new_direction_fires(self):
        assert _decide() is True

    def test_disabled(self):
        assert _decide(enabled=False) is False

    def test_never_during_task_continuation(self):
        # The #1 false-drift failure mode: a follow-up that looks like a new topic but
        # depends on in-flight work.
        assert _decide(in_task=True) is False

    def test_never_below_min_ratio(self):
        # 30k < 0.5 * 65,536 — drift compaction is opportunistic, not an overflow valve.
        assert _decide(tokens=30_000) is False

    def test_never_below_floor(self):
        assert _decide(tokens=25_000, min_ratio=0.0, floor_tokens=40_000) is False

    def test_last_compaction_raises_floor(self):
        # Must gain a floor's worth of NEW content after a compaction (#97239 contract).
        assert _decide(
            tokens=45_000, min_ratio=0.0, floor_tokens=20_000, last_compaction_tokens=40_000
        ) is False

    def test_dependent_continuation_never_fires(self):
        assert _decide(dependence_score=0.9) is False

    def test_ambiguous_band_requires_model_ok(self):
        ambiguous = (_DRIFT_INDEPENDENT_MAX + _DRIFT_DEPENDENT_MIN) / 2
        assert drift_score_ambiguous(ambiguous) is True
        assert _decide(dependence_score=ambiguous, confirm_independent=None) is False
        assert _decide(dependence_score=ambiguous, confirm_independent=False) is False
        assert _decide(dependence_score=ambiguous, confirm_independent=True) is True

    def test_no_signal_requires_model_ok(self):
        assert _decide(dependence_score=None, confirm_independent=None) is False
        assert _decide(dependence_score=None, confirm_independent=True) is True

    def test_cooldown_active_blocks(self):
        assert _decide(cooldown_active=True) is False

    def test_zero_window_never_fires(self):
        assert _decide(window_tokens=0) is False


class TestDependenceScore:

    def test_disjoint_texts_score_low(self):
        score = drift_dependence_score(
            "redeploy the kubernetes cluster in the frankfurt datacenter",
            ["the quarterly budget spreadsheet needs revision", "invoice totals look wrong"],
        )
        assert score is not None
        assert score <= _DRIFT_INDEPENDENT_MAX

    def test_short_generic_followup_has_no_signal(self):
        # "fix the second bug too" is dependent on history despite zero lexical overlap;
        # too little signal must route to the model check, never the clear-independent band.
        assert drift_dependence_score(
            "fix the second bug too",
            ["the gateway crash loop kept respawning the telegram adapter"],
        ) is None

    def test_shared_vocabulary_scores_high(self):
        score = drift_dependence_score(
            "why did the gateway crash loop keep respawning",
            ["the gateway crash loop kept respawning the telegram adapter"],
        )
        assert score is not None
        assert score >= _DRIFT_DEPENDENT_MIN

    def test_too_little_signal_is_none(self):
        assert drift_dependence_score("ok", ["nothing useful here"]) is None
        assert drift_dependence_score("a real message with words", []) is None


class TestDriftModelCheck:
    """The ambiguous-band judge's parse contract (client is a mocked aux model)."""

    @staticmethod
    def _fake_client(content):
        msg = SimpleNamespace(content=content)
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        return client

    def test_parses_yes_no(self):
        from agent.turn_context_compaction import _drift_model_check

        agent = SimpleNamespace(provider="x", model="m")
        with patch("agent.turn_context_compaction._drift_confirm_client",
                   return_value=(self._fake_client("NO"), "m")):
            assert _drift_model_check(agent, "question", "context") is False
        with patch("agent.turn_context_compaction._drift_confirm_client",
                   return_value=(self._fake_client("YES"), "m")):
            assert _drift_model_check(agent, "question", "context") is True

    def test_unavailable_client_stays_none(self):
        from agent.turn_context_compaction import _drift_model_check

        agent = SimpleNamespace(provider="x", model="m")
        with patch("agent.turn_context_compaction._drift_confirm_client", return_value=(None, None)):
            assert _drift_model_check(agent, "question", "context") is None

    def test_ambiguous_band_detection(self):
        assert drift_score_ambiguous(None) is True
        assert drift_score_ambiguous(0.0) is False
        assert drift_score_ambiguous(1.0) is False


# ── Integration: the drift pass inside the real turn pipeline ──
# Mirrors tests/agent/test_413_compression.py's harness: a real AIAgent whose
# provider client is a MagicMock, driven through the genuine run_conversation()
# turn-start path (run_turn_start_compaction → _drift_compaction).


def _mock_response(content="ok"):
    msg = SimpleNamespace(content=content, tool_calls=None,
                          reasoning_content=None, reasoning=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        model="test/model", usage=None,
    )


def _new_agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.compression_enabled = True
    a.save_trajectories = False
    # JEV drift-compaction knobs under test (normally set from the compression config).
    a.compression_drift_compact = True
    a.compression_drift_min_ratio = 0.5
    a.compression_drift_confirm = True
    a.compression_drift_cooldown_seconds = 600
    comp = a.context_compressor
    comp.context_length = 65_536
    comp.threshold_tokens = 130_000          # drift floor = 130k * 0.2 = 26k
    comp.summary_target_ratio = 0.2
    comp.protect_first_n = 3
    comp.protect_last_n = 2
    comp.awaiting_real_usage_after_compression = False
    comp.last_compression_rough_tokens = 0
    return a


_TOPIC_A_HISTORY = [
    {"role": "user", "content": "kubernetes cluster migration: rollout the replica set across the scheduler namespace"},
    {"role": "assistant", "content": "replica set rollout scheduled across the kubernetes namespace, cluster nodes drained"},
    {"role": "user", "content": "check the deployment pods after the scheduler migration rollout"},
    {"role": "assistant", "content": "deployment pods healthy, replica scheduler migration complete"},
    {"role": "user", "content": "document the kubernetes namespace rollout for the cluster migration"},
    {"role": "assistant", "content": "namespace rollout documented, scheduler replica migration recorded"},
]


def _run_turn(agent, user_text, **kwargs):
    """One real turn: mocked provider, fresh-estimate pins, recording compressor."""
    agent.client.chat.completions.create.side_effect = [_mock_response()]
    with (
        patch("agent.turn_context._preflight_request_tokens", return_value=60_000),
        patch.object(agent, "_compress_context",
                     side_effect=lambda messages, system_message, **kw: (messages, "compressed"))
        as mock_compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(user_text, **kwargs)
    triggers = [c.kwargs.get("trigger") for c in mock_compress.call_args_list]
    return result, triggers


class TestDriftCompactionPass:

    def test_fires_on_topic_switch(self):
        agent = _new_agent()
        result, triggers = _run_turn(
            agent,
            "sourdough starter hydration: what levain ratio keeps the oven crust open",
            conversation_history=[dict(m) for m in _TOPIC_A_HISTORY],
        )
        assert "drift" in triggers, f"expected drift compaction, saw triggers={triggers}"
        assert result.get("completed") is True

    def test_same_topic_continuation_never_fires(self):
        agent = _new_agent()
        _, triggers = _run_turn(
            agent,
            "continue the kubernetes rollout and check the replica scheduler namespace again",
            conversation_history=[dict(m) for m in _TOPIC_A_HISTORY],
        )
        assert "drift" not in triggers, f"continuation must not compact, saw triggers={triggers}"

    def test_task_continuation_never_fires(self):
        agent = _new_agent()
        _, triggers = _run_turn(
            agent,
            "sourdough starter hydration: what levain ratio keeps the oven crust open",
            conversation_history=[dict(m) for m in _TOPIC_A_HISTORY],
            task_id="task-123",
        )
        assert "drift" not in triggers, f"in-flight task must not compact, saw triggers={triggers}"

    def test_nothing_outside_protected_region_never_fires(self):
        agent = _new_agent()
        _, triggers = _run_turn(
            agent,
            "sourdough starter hydration levain crust oven flour",
            conversation_history=[{"role": "assistant", "content": "kubernetes rollout"}],
        )
        assert "drift" not in triggers, f"nothing compressible, saw triggers={triggers}"

    def test_ambiguous_band_uses_model_check(self):
        # Partial lexical overlap lands in the gray band; only the model may authorize.
        ambiguous_msg = (
            "kubernetes rollout questions plus sourdough starter hydration and levain crust"
        )
        for confirm, expect_fire in ((False, False), (True, True)):
            agent = _new_agent()
            with patch("agent.turn_context_compaction._drift_model_check",
                       return_value=confirm) as mock_check:
                _, triggers = _run_turn(
                    agent, ambiguous_msg,
                    conversation_history=[dict(m) for m in _TOPIC_A_HISTORY],
                )
            assert mock_check.called, "ambiguous band must consult the model check"
            assert ("drift" in triggers) is expect_fire, (
                f"confirm={confirm} should {'fire' if expect_fire else 'not fire'}, "
                f"saw triggers={triggers}")

    def test_cooldown_blocks_repeat_firing(self):
        agent = _new_agent()
        _, first = _run_turn(
            agent,
            "sourdough starter hydration levain crust oven flour baking",
            conversation_history=[dict(m) for m in _TOPIC_A_HISTORY],
        )
        assert "drift" in first
        _, second = _run_turn(
            agent,
            "violin luthier varnish recipe for maple scroll carving",
            conversation_history=[dict(m) for m in _TOPIC_A_HISTORY],
        )
        assert "drift" not in second, f"10-min cooldown must suppress, saw triggers={second}"
