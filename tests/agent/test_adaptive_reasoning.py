"""Tests for opt-in adaptive reasoning escalation (agent/adaptive_reasoning.py).

Covers classification boundaries, config parsing, per-turn apply/restore,
manual-override precedence, notification emission + dedup, and the
continuation/reset behavior for follow-up turns.
"""

import unittest
from types import SimpleNamespace

from agent.adaptive_reasoning import (
    NOTICE_KEY,
    adaptive_reasoning_turn,
    begin_adaptive_reasoning_turn,
    classify_reasoning_effort,
    end_adaptive_reasoning_turn,
    extract_message_text,
    parse_adaptive_reasoning_config,
)


def _make_agent(
    *,
    baseline="medium",
    enabled=True,
    max_effort="xhigh",
    min_effort=None,
    user_override=False,
    reasoning_config="__from_baseline__",
    platform="cli",
):
    if reasoning_config == "__from_baseline__":
        reasoning_config = {"enabled": True, "effort": baseline}
    adaptive = {"enabled": True, "max_effort": max_effort} if enabled else None
    if adaptive is not None and min_effort is not None:
        adaptive["min_effort"] = min_effort
    notices = []
    return SimpleNamespace(
        reasoning_config=reasoning_config,
        adaptive_reasoning=adaptive,
        reasoning_user_override=user_override,
        _adaptive_prev_effort=None,
        _adaptive_last_notified_effort=None,
        notice_callback=notices.append,
        _notices=notices,
        platform=platform,
    )


DEBUG_MSG = (
    "Why does the gateway keep failing after I restart it? "
    "error: connection refused"
)
XHIGH_MSG = (
    "Design the cross-component architecture for migrating our session "
    "store to a multi-process model"
)


# ---------------------------------------------------------------------------
# Classification boundaries
# ---------------------------------------------------------------------------

class TestClassifyBoundaries(unittest.TestCase):
    def test_casual_and_trivial_classify_low(self):
        for msg in ("hi", "thanks!", "ok", "sounds good", "ping", "lol"):
            level, reason = classify_reasoning_effort(msg)
            self.assertEqual(level, "low", msg)
            self.assertTrue(reason, msg)

    def test_short_factual_question_classifies_low(self):
        level, reason = classify_reasoning_effort("What is the capital of France?")
        self.assertEqual(level, "low")
        self.assertTrue(reason)

    def test_mechanical_single_step_classifies_low(self):
        for msg in (
            "show me the readme",
            "list the files in /tmp",
            "print the current config",
        ):
            self.assertEqual(classify_reasoning_effort(msg)[0], "low", msg)

    def test_signal_bearing_action_stays_medium_not_low(self):
        # nginx is an infrastructure signal: a complexity signal always
        # overrides low, even for a short imperative.
        self.assertEqual(
            classify_reasoning_effort("restart the nginx service")[0], "medium"
        )

    def test_realistic_near_misses_stay_medium(self):
        for msg in (
            "go ahead",  # continuation without a prior escalated turn
            "how do I center a div?",  # a real (small) task, not a fact lookup
            "check whether the fix works",  # verification, not mechanical retrieval
            "summarize this file",  # unbounded content, unknown weight
            "list the open PRs and summarize the risky ones",  # compound request
        ):
            self.assertEqual(classify_reasoning_effort(msg)[0], "medium", msg)

    def test_complexity_signals_override_low(self):
        for msg in (
            "fix the crash",  # debugging signal in a terse message
            "show me the segfault",  # mechanical verb + error signal
            "why error: ENOENT?",  # error evidence
        ):
            self.assertEqual(classify_reasoning_effort(msg)[0], "medium", msg)
        # A factual-shaped question carrying a real signal keeps its
        # escalation classification — signals override the simple shape.
        self.assertEqual(
            classify_reasoning_effort("What is a race condition?")[0], "high"
        )

    def test_multiple_questions_stay_medium(self):
        self.assertEqual(
            classify_reasoning_effort(
                "What is the gateway port? And where is the config file?"
            )[0],
            "medium",
        )

    def test_code_snippet_stays_medium(self):
        self.assertEqual(
            classify_reasoning_effort("what is this?\n```\nx = 1\n```")[0],
            "medium",
        )

    def test_empty_and_whitespace_stay_medium(self):
        # No positive simplicity evidence (e.g. an image-only message whose
        # text extraction is empty) — never downshift on absence of signal.
        self.assertEqual(classify_reasoning_effort("")[0], "medium")
        self.assertEqual(classify_reasoning_effort("   \n ")[0], "medium")

    def test_debugging_with_error_evidence_is_high(self):
        level, reason = classify_reasoning_effort(DEBUG_MSG)
        self.assertEqual(level, "high")
        self.assertTrue(reason)

    def test_implementation_request_is_high(self):
        level, _ = classify_reasoning_effort(
            "Implement a new session-title feature with tests"
        )
        self.assertEqual(level, "high")

    def test_consequential_infra_is_high_but_lone_keyword_is_not(self):
        self.assertEqual(
            classify_reasoning_effort("deploy the new version to production")[0],
            "high",
        )
        # Infrastructure keyword without corroboration stays at baseline.
        self.assertEqual(
            classify_reasoning_effort("show me the kubernetes pod list")[0],
            "medium",
        )

    def test_architecture_cross_component_is_xhigh(self):
        self.assertEqual(classify_reasoning_effort(XHIGH_MSG)[0], "xhigh")

    def test_high_stakes_production_diagnosis_is_xhigh(self):
        level, _ = classify_reasoning_effort(
            "We are seeing data corruption in production after the last "
            "deploy - diagnose and fix it"
        )
        self.assertEqual(level, "xhigh")

    def test_lone_xhigh_keyword_without_corroboration_is_not_xhigh(self):
        # Security matters, but a bare mention with no supporting signals
        # must not jump straight to xhigh.
        level, _ = classify_reasoning_effort(
            "Do a security audit of the gateway auth flow"
        )
        self.assertEqual(level, "high")

    def test_continuation_inherits_prior_effort(self):
        level, reason = classify_reasoning_effort("go ahead", prior_effort="high")
        self.assertEqual(level, "high")
        self.assertIn("continuing", reason)

    def test_continuation_without_prior_effort_is_medium(self):
        self.assertEqual(classify_reasoning_effort("go ahead")[0], "medium")

    def test_unrelated_trivial_turn_after_escalation_drops(self):
        # A non-continuation message re-classifies from scratch even with a
        # prior escalated turn — nothing stays stuck at xhigh.
        level, _ = classify_reasoning_effort(
            "what time is it in Tokyo?", prior_effort="xhigh"
        )
        self.assertEqual(level, "low")

    def test_deterministic(self):
        for msg in (DEBUG_MSG, "What is the capital of France?", "thanks!"):
            self.assertEqual(
                classify_reasoning_effort(msg),
                classify_reasoning_effort(msg),
            )


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

class TestParseAdaptiveConfig(unittest.TestCase):
    def test_absent_or_disabled_returns_none(self):
        self.assertIsNone(parse_adaptive_reasoning_config(None))
        self.assertIsNone(parse_adaptive_reasoning_config({}))
        self.assertIsNone(parse_adaptive_reasoning_config({"enabled": False}))
        self.assertIsNone(parse_adaptive_reasoning_config("yes"))

    def test_enabled_defaults_max_effort(self):
        cfg = parse_adaptive_reasoning_config({"enabled": True})
        self.assertEqual(cfg, {"enabled": True, "max_effort": "xhigh"})

    def test_invalid_max_effort_falls_back_to_xhigh(self):
        cfg = parse_adaptive_reasoning_config(
            {"enabled": True, "max_effort": "turbo"}
        )
        self.assertEqual(cfg["max_effort"], "xhigh")

    def test_valid_max_effort_kept(self):
        cfg = parse_adaptive_reasoning_config(
            {"enabled": True, "max_effort": "high"}
        )
        self.assertEqual(cfg["max_effort"], "high")

    def test_min_effort_low_kept(self):
        cfg = parse_adaptive_reasoning_config(
            {"enabled": True, "min_effort": "low"}
        )
        self.assertEqual(
            cfg, {"enabled": True, "max_effort": "xhigh", "min_effort": "low"}
        )

    def test_min_effort_absent_or_empty_stays_escalation_only(self):
        self.assertNotIn(
            "min_effort", parse_adaptive_reasoning_config({"enabled": True})
        )
        self.assertNotIn(
            "min_effort",
            parse_adaptive_reasoning_config({"enabled": True, "min_effort": ""}),
        )

    def test_invalid_min_effort_dropped(self):
        # "none" must never become a selectable adaptive level — thinking
        # presence stays stable for the life of a conversation.
        for bad in ("none", "turbo", False, "off"):
            cfg = parse_adaptive_reasoning_config(
                {"enabled": True, "min_effort": bad}
            )
            self.assertNotIn("min_effort", cfg, repr(bad))
            self.assertEqual(cfg["max_effort"], "xhigh", repr(bad))

    def test_min_effort_above_max_effort_dropped(self):
        cfg = parse_adaptive_reasoning_config(
            {"enabled": True, "min_effort": "xhigh", "max_effort": "high"}
        )
        self.assertNotIn("min_effort", cfg)
        self.assertEqual(cfg["max_effort"], "high")

    def test_min_effort_equal_to_max_effort_kept(self):
        cfg = parse_adaptive_reasoning_config(
            {"enabled": True, "min_effort": "high", "max_effort": "high"}
        )
        self.assertEqual(cfg.get("min_effort"), "high")

    def test_parsed_config_round_trips(self):
        # Delegation propagates the parent's already-parsed dict into the
        # child's AIAgent constructor, which parses it again — the parse must
        # be a fixed point for both shapes.
        for raw in (
            {"enabled": True},
            {"enabled": True, "min_effort": "low", "max_effort": "high"},
        ):
            once = parse_adaptive_reasoning_config(raw)
            self.assertEqual(parse_adaptive_reasoning_config(once), once)


# ---------------------------------------------------------------------------
# extract_message_text
# ---------------------------------------------------------------------------

class TestExtractMessageText(unittest.TestCase):
    def test_string_passthrough(self):
        self.assertEqual(extract_message_text("hello"), "hello")

    def test_multimodal_blocks(self):
        blocks = [
            {"type": "text", "text": "debug this crash"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]
        self.assertEqual(extract_message_text(blocks), "debug this crash")

    def test_unknown_shape_is_empty(self):
        self.assertEqual(extract_message_text(42), "")


# ---------------------------------------------------------------------------
# Per-turn apply / restore
# ---------------------------------------------------------------------------

class TestBeginEndTurn(unittest.TestCase):
    def test_escalates_for_turn_and_restores_baseline(self):
        agent = _make_agent()
        saved = agent.reasoning_config
        token = begin_adaptive_reasoning_turn(agent, DEBUG_MSG)
        self.assertIsNotNone(token)
        self.assertEqual(agent.reasoning_config, {"enabled": True, "effort": "high"})
        end_adaptive_reasoning_turn(agent, token)
        self.assertIs(agent.reasoning_config, saved)

    def test_no_escalation_for_trivial_turn(self):
        agent = _make_agent()
        token = begin_adaptive_reasoning_turn(agent, "thanks!")
        self.assertIsNone(token)
        self.assertEqual(agent.reasoning_config["effort"], "medium")
        self.assertEqual(agent._notices, [])

    def test_disabled_feature_is_inert(self):
        agent = _make_agent(enabled=False)
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, DEBUG_MSG))

    def test_user_override_suppresses_escalation(self):
        agent = _make_agent(user_override=True)
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, DEBUG_MSG))
        self.assertEqual(agent._notices, [])

    def test_thinking_disabled_baseline_is_respected(self):
        agent = _make_agent(reasoning_config={"enabled": False})
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, DEBUG_MSG))
        self.assertEqual(agent.reasoning_config, {"enabled": False})

    def test_unknown_baseline_effort_skips(self):
        agent = _make_agent(
            reasoning_config={"enabled": True, "effort": "turbo"}
        )
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, DEBUG_MSG))

    def test_absent_baseline_is_inert(self):
        # No explicit reasoning baseline (reasoning_config is None) means the
        # provider decides whether thinking exists at all. Toggling thinking
        # presence mid-conversation fragments the Anthropic prompt-cache
        # namespace, so adaptive must be inert — in both directions.
        agent = _make_agent(reasoning_config=None)
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, DEBUG_MSG))
        self.assertIsNone(agent.reasoning_config)
        self.assertEqual(agent._notices, [])

        agent = _make_agent(reasoning_config=None, min_effort="low")
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, "thanks!"))
        self.assertIsNone(agent.reasoning_config)

    def test_baseline_at_or_above_classification_never_changes(self):
        agent = _make_agent(baseline="high")
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, DEBUG_MSG))
        agent = _make_agent(baseline="xhigh")
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, XHIGH_MSG))

    def test_max_effort_ceiling_clamps_xhigh_to_high(self):
        agent = _make_agent(max_effort="high")
        token = begin_adaptive_reasoning_turn(agent, XHIGH_MSG)
        self.assertIsNotNone(token)
        self.assertEqual(agent.reasoning_config["effort"], "high")

    def test_moa_turn_is_skipped(self):
        agent = _make_agent()
        self.assertIsNone(
            begin_adaptive_reasoning_turn(agent, DEBUG_MSG, moa_config={"preset": "x"})
        )

    def test_restore_is_identity_guarded(self):
        # A mid-turn rewrite (fallback model re-resolving effort) must win
        # over the end-of-turn restore.
        agent = _make_agent()
        token = begin_adaptive_reasoning_turn(agent, DEBUG_MSG)
        replacement = {"enabled": True, "effort": "low"}
        agent.reasoning_config = replacement
        end_adaptive_reasoning_turn(agent, token)
        self.assertIs(agent.reasoning_config, replacement)

    def test_end_with_none_token_is_noop(self):
        agent = _make_agent()
        end_adaptive_reasoning_turn(agent, None)
        self.assertEqual(agent.reasoning_config["effort"], "medium")

    def test_never_raises_on_broken_agent(self):
        broken = SimpleNamespace(adaptive_reasoning={"enabled": True})
        self.assertIsNone(begin_adaptive_reasoning_turn(broken, object()))


# ---------------------------------------------------------------------------
# Conservative downshift (min_effort)
# ---------------------------------------------------------------------------

class TestDownshift(unittest.TestCase):
    def test_min_omitted_keeps_escalation_only(self):
        # Backward compatibility: without min_effort the configured baseline
        # is the floor, so a low-classified turn changes nothing.
        agent = _make_agent()
        self.assertIsNone(
            begin_adaptive_reasoning_turn(agent, "What is the capital of France?")
        )
        self.assertEqual(agent.reasoning_config["effort"], "medium")
        self.assertEqual(agent._notices, [])

    def test_min_low_applies_low_for_turn_and_restores(self):
        agent = _make_agent(min_effort="low")
        saved = agent.reasoning_config
        token = begin_adaptive_reasoning_turn(
            agent, "What is the capital of France?"
        )
        self.assertIsNotNone(token)
        self.assertEqual(agent.reasoning_config, {"enabled": True, "effort": "low"})
        end_adaptive_reasoning_turn(agent, token)
        self.assertIs(agent.reasoning_config, saved)

    def test_min_low_casual_turn_downshifts(self):
        agent = _make_agent(min_effort="low")
        token = begin_adaptive_reasoning_turn(agent, "thanks!")
        self.assertIsNotNone(token)
        self.assertEqual(agent.reasoning_config["effort"], "low")

    def test_downshift_clamps_to_floor_not_below(self):
        # baseline high with min_effort medium: a casual turn drops to the
        # configured floor (medium), never to low.
        agent = _make_agent(baseline="high", min_effort="medium")
        token = begin_adaptive_reasoning_turn(agent, "thanks!")
        self.assertIsNotNone(token)
        self.assertEqual(agent.reasoning_config["effort"], "medium")

    def test_min_at_or_above_baseline_never_changes_a_simple_turn(self):
        # min == baseline: floor is the baseline — nothing to lower.
        agent = _make_agent(min_effort="medium")
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, "thanks!"))
        # min above baseline never raises a simple turn either: the baseline
        # is already the floor of the adaptive range.
        agent = _make_agent(min_effort="high")
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, "thanks!"))
        self.assertEqual(agent.reasoning_config["effort"], "medium")

    def test_complex_turn_still_escalates_with_min_low(self):
        agent = _make_agent(min_effort="low")
        token = begin_adaptive_reasoning_turn(agent, XHIGH_MSG)
        self.assertIsNotNone(token)
        self.assertEqual(agent.reasoning_config["effort"], "xhigh")

    def test_user_override_suppresses_downshift(self):
        agent = _make_agent(min_effort="low", user_override=True)
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, "thanks!"))
        self.assertEqual(agent._notices, [])

    def test_thinking_disabled_baseline_never_downshifts(self):
        # reasoning_effort: none stays none — never re-enable thinking.
        agent = _make_agent(
            min_effort="low", reasoning_config={"enabled": False}
        )
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, "thanks!"))
        self.assertEqual(agent.reasoning_config, {"enabled": False})

    def test_downshift_is_not_inherited_by_continuation(self):
        # "go ahead" after a lowered turn refers to previously discussed
        # work — it must run at the baseline, not inherit the downshift.
        agent = _make_agent(min_effort="low")
        token = begin_adaptive_reasoning_turn(agent, "thanks!")
        self.assertEqual(agent.reasoning_config["effort"], "low")
        end_adaptive_reasoning_turn(agent, token)
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, "go ahead"))
        self.assertEqual(agent.reasoning_config["effort"], "medium")

    def test_low_notice_text_and_dedup(self):
        agent = _make_agent(min_effort="low")
        token = begin_adaptive_reasoning_turn(agent, "thanks!")
        end_adaptive_reasoning_turn(agent, token)
        self.assertEqual(len(agent._notices), 1)
        notice = agent._notices[0]
        self.assertIn("Reasoning lowered to Low", notice.text)
        self.assertIn("—", notice.text)
        self.assertEqual(notice.kind, "ttl")
        self.assertEqual(notice.key, NOTICE_KEY)
        # Consecutive lowered turns notify once.
        token = begin_adaptive_reasoning_turn(
            agent, "What is the capital of France?"
        )
        end_adaptive_reasoning_turn(agent, token)
        self.assertEqual(len(agent._notices), 1)

    def test_low_notice_resets_after_baseline_turn(self):
        agent = _make_agent(min_effort="low")
        token = begin_adaptive_reasoning_turn(agent, "thanks!")
        end_adaptive_reasoning_turn(agent, token)
        self.assertIsNone(
            begin_adaptive_reasoning_turn(agent, "how do I center a div?")
        )
        token = begin_adaptive_reasoning_turn(agent, "thanks!")
        end_adaptive_reasoning_turn(agent, token)
        self.assertEqual(len(agent._notices), 2)

    def test_low_then_high_notifies_each_change(self):
        agent = _make_agent(min_effort="low")
        token = begin_adaptive_reasoning_turn(agent, "thanks!")
        end_adaptive_reasoning_turn(agent, token)
        token = begin_adaptive_reasoning_turn(agent, DEBUG_MSG)
        end_adaptive_reasoning_turn(agent, token)
        self.assertEqual(len(agent._notices), 2)
        self.assertIn("lowered to Low", agent._notices[0].text)
        self.assertIn("raised to High", agent._notices[1].text)

    def test_subagent_platform_emits_no_notice(self):
        # Delegate children adapt silently — a child toast on the parent's
        # rail (or a forced console line) would be noise.
        vprints = []
        agent = _make_agent(min_effort="low", platform="subagent")
        agent.notice_callback = None
        agent._vprint = lambda *a, **k: vprints.append(a)
        token = begin_adaptive_reasoning_turn(agent, DEBUG_MSG)
        self.assertIsNotNone(token)
        self.assertEqual(agent.reasoning_config["effort"], "high")
        self.assertEqual(vprints, [])

    def test_session_state_is_per_agent_object(self):
        # Two live agents (e.g. simultaneous Discord threads) must not share
        # continuation/dedup state or effective efforts.
        a = _make_agent(min_effort="low")
        b = _make_agent(min_effort="low")
        token_a = begin_adaptive_reasoning_turn(a, DEBUG_MSG)
        token_b = begin_adaptive_reasoning_turn(b, "thanks!")
        self.assertEqual(a.reasoning_config["effort"], "high")
        self.assertEqual(b.reasoning_config["effort"], "low")
        self.assertEqual(a._adaptive_prev_effort, "high")
        self.assertIsNone(b._adaptive_prev_effort)
        self.assertEqual(len(a._notices), 1)
        self.assertEqual(len(b._notices), 1)
        end_adaptive_reasoning_turn(a, token_a)
        end_adaptive_reasoning_turn(b, token_b)
        self.assertEqual(a.reasoning_config["effort"], "medium")
        self.assertEqual(b.reasoning_config["effort"], "medium")


# ---------------------------------------------------------------------------
# Notification emission + dedup
# ---------------------------------------------------------------------------

class TestEscalationNotice(unittest.TestCase):
    def test_notice_fired_with_level_and_reason(self):
        agent = _make_agent()
        begin_adaptive_reasoning_turn(agent, DEBUG_MSG)
        self.assertEqual(len(agent._notices), 1)
        notice = agent._notices[0]
        self.assertIn("Reasoning raised to High", notice.text)
        self.assertIn("—", notice.text)
        self.assertEqual(notice.kind, "ttl")
        self.assertEqual(notice.key, NOTICE_KEY)
        self.assertEqual(notice.level, "info")

    def test_xhigh_notice_labels_level(self):
        agent = _make_agent()
        begin_adaptive_reasoning_turn(agent, XHIGH_MSG)
        self.assertIn("Reasoning raised to XHigh", agent._notices[0].text)

    def test_consecutive_same_level_turns_notify_once(self):
        agent = _make_agent()
        token = begin_adaptive_reasoning_turn(agent, DEBUG_MSG)
        end_adaptive_reasoning_turn(agent, token)
        token = begin_adaptive_reasoning_turn(agent, DEBUG_MSG)
        end_adaptive_reasoning_turn(agent, token)
        self.assertEqual(len(agent._notices), 1)

    def test_continuation_turn_does_not_renotify(self):
        agent = _make_agent()
        token = begin_adaptive_reasoning_turn(agent, DEBUG_MSG)
        end_adaptive_reasoning_turn(agent, token)
        token = begin_adaptive_reasoning_turn(agent, "go ahead")
        self.assertIsNotNone(token)
        self.assertEqual(agent.reasoning_config["effort"], "high")
        end_adaptive_reasoning_turn(agent, token)
        self.assertEqual(len(agent._notices), 1)

    def test_renotifies_after_a_baseline_turn(self):
        agent = _make_agent()
        token = begin_adaptive_reasoning_turn(agent, DEBUG_MSG)
        end_adaptive_reasoning_turn(agent, token)
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, "what time is it in Tokyo?"))
        token = begin_adaptive_reasoning_turn(agent, DEBUG_MSG)
        end_adaptive_reasoning_turn(agent, token)
        self.assertEqual(len(agent._notices), 2)

    def test_level_change_notifies_again(self):
        agent = _make_agent()
        token = begin_adaptive_reasoning_turn(agent, DEBUG_MSG)
        end_adaptive_reasoning_turn(agent, token)
        token = begin_adaptive_reasoning_turn(agent, XHIGH_MSG)
        end_adaptive_reasoning_turn(agent, token)
        self.assertEqual(len(agent._notices), 2)
        self.assertIn("XHigh", agent._notices[1].text)

    def test_notice_failure_does_not_break_escalation(self):
        agent = _make_agent()

        def _boom(_notice):
            raise RuntimeError("driver died")

        agent.notice_callback = _boom
        token = begin_adaptive_reasoning_turn(agent, DEBUG_MSG)
        self.assertIsNotNone(token)
        self.assertEqual(agent.reasoning_config["effort"], "high")


# ---------------------------------------------------------------------------
# Wiring through AIAgent.run_conversation (real forwarder, stubbed loop)
# ---------------------------------------------------------------------------

class TestRunConversationWiring(unittest.TestCase):
    """The forwarder must apply the escalated effort for the whole turn and
    restore the baseline on both clean and exceptional exits."""

    def _make_real_agent(self):
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent.session_id = "adaptive-test-session"
        agent.platform = "cli"
        agent.model = "test-model"
        agent.reasoning_config = {"enabled": True, "effort": "medium"}
        agent.adaptive_reasoning = {"enabled": True, "max_effort": "xhigh"}
        agent.reasoning_user_override = False
        agent._adaptive_prev_effort = None
        agent._adaptive_last_notified_effort = None
        agent.notice_callback = None
        agent._vprint = lambda *a, **k: None
        agent._session_db = None
        agent._reset_activity_labels_after_turn = lambda: None
        return agent

    def test_escalated_during_turn_restored_after(self):
        from unittest.mock import patch

        agent = self._make_real_agent()
        baseline = agent.reasoning_config
        seen = {}

        def _fake_loop(agent_arg, *args, **kwargs):
            seen["effort"] = dict(agent_arg.reasoning_config)
            return {"final_response": "ok", "messages": []}

        with patch("agent.conversation_loop.run_conversation", _fake_loop):
            result = agent.run_conversation(DEBUG_MSG)

        self.assertEqual(result["final_response"], "ok")
        self.assertEqual(seen["effort"], {"enabled": True, "effort": "high"})
        self.assertIs(agent.reasoning_config, baseline)

    def test_baseline_restored_when_turn_raises(self):
        from unittest.mock import patch

        agent = self._make_real_agent()
        baseline = agent.reasoning_config

        def _boom(agent_arg, *args, **kwargs):
            raise KeyboardInterrupt()

        with patch("agent.conversation_loop.run_conversation", _boom):
            with self.assertRaises(KeyboardInterrupt):
                agent.run_conversation(DEBUG_MSG)

        self.assertIs(agent.reasoning_config, baseline)

    def test_low_applied_during_turn_and_restored_after(self):
        from unittest.mock import patch

        agent = self._make_real_agent()
        agent.adaptive_reasoning["min_effort"] = "low"
        baseline = agent.reasoning_config
        seen = {}

        def _fake_loop(agent_arg, *args, **kwargs):
            seen["effort"] = dict(agent_arg.reasoning_config)
            return {"final_response": "Paris", "messages": []}

        with patch("agent.conversation_loop.run_conversation", _fake_loop):
            result = agent.run_conversation("What is the capital of France?")

        self.assertEqual(result["final_response"], "Paris")
        self.assertEqual(seen["effort"], {"enabled": True, "effort": "low"})
        self.assertIs(agent.reasoning_config, baseline)

    def test_no_adaptive_config_means_untouched_turn(self):
        from unittest.mock import patch

        agent = self._make_real_agent()
        agent.adaptive_reasoning = None
        seen = {}

        def _fake_loop(agent_arg, *args, **kwargs):
            seen["effort"] = dict(agent_arg.reasoning_config)
            return {"final_response": "ok", "messages": []}

        with patch("agent.conversation_loop.run_conversation", _fake_loop):
            agent.run_conversation(DEBUG_MSG)

        self.assertEqual(seen["effort"], {"enabled": True, "effort": "medium"})

    def test_forwarder_is_decorated(self):
        """The seam is the decorator on AIAgent.run_conversation itself — a
        rebase that drops the decorator line must fail here, not in prod."""
        import inspect

        from run_agent import AIAgent

        self.assertTrue(hasattr(AIAgent.run_conversation, "__wrapped__"))
        inner = inspect.unwrap(AIAgent.run_conversation)
        self.assertEqual(inner.__name__, "run_conversation")
        # functools.wraps keeps the public signature intact for callers and
        # introspection (moa_config is still a declared keyword).
        self.assertIn(
            "moa_config", inspect.signature(AIAgent.run_conversation).parameters
        )


# ---------------------------------------------------------------------------
# adaptive_reasoning_turn decorator
# ---------------------------------------------------------------------------

class TestAdaptiveReasoningTurnDecorator(unittest.TestCase):
    def _agent(self, **overrides):
        agent = _make_agent(**overrides)
        return agent

    def test_wraps_body_and_restores_on_return(self):
        agent = self._agent()
        baseline = agent.reasoning_config
        seen = {}

        @adaptive_reasoning_turn
        def turn(a, user_message, extra=None, moa_config=None):
            seen["effort"] = dict(a.reasoning_config)
            return "ok"

        self.assertEqual(turn(agent, DEBUG_MSG), "ok")
        self.assertEqual(seen["effort"], {"enabled": True, "effort": "high"})
        self.assertIs(agent.reasoning_config, baseline)

    def test_restores_when_body_raises(self):
        agent = self._agent()
        baseline = agent.reasoning_config

        @adaptive_reasoning_turn
        def turn(a, user_message, moa_config=None):
            self.assertEqual(a.reasoning_config["effort"], "high")
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            turn(agent, DEBUG_MSG)
        self.assertIs(agent.reasoning_config, baseline)

    def test_positional_moa_config_is_seen(self):
        """moa_config is the last positional in the forwarder; a positional
        caller must still hit the MoA opt-out."""
        agent = self._agent()
        baseline = agent.reasoning_config
        seen = {}

        @adaptive_reasoning_turn
        def turn(a, user_message, extra=None, moa_config=None):
            seen["effort"] = a.reasoning_config["effort"]

        turn(agent, DEBUG_MSG, None, {"preset": "x"})
        self.assertEqual(seen["effort"], "medium")
        self.assertIs(agent.reasoning_config, baseline)
        turn(agent, DEBUG_MSG, None, moa_config={"preset": "x"})
        self.assertEqual(seen["effort"], "medium")

    def test_preserves_wrapped_metadata(self):
        import inspect

        def turn(a, user_message, moa_config=None):
            """doc"""

        wrapped = adaptive_reasoning_turn(turn)
        self.assertEqual(wrapped.__name__, "turn")
        self.assertEqual(wrapped.__doc__, "doc")
        self.assertIs(wrapped.__wrapped__, turn)
        self.assertEqual(
            list(inspect.signature(wrapped).parameters), ["a", "user_message", "moa_config"]
        )


if __name__ == "__main__":
    unittest.main()
