"""Tests for opt-in adaptive reasoning escalation (agent/adaptive_reasoning.py).

Covers classification boundaries, config parsing, per-turn apply/restore,
manual-override precedence, notification emission + dedup, and the
continuation/reset behavior for follow-up turns.
"""

import unittest
from types import SimpleNamespace

import pytest

from hermes_constants import VALID_REASONING_EFFORTS, parse_reasoning_effort

from agent.adaptive_reasoning import (
    CACHE_NOTICE_KEY,
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


# A lone code-class signal (debugging) with nothing corroborating: high.
DEBUG_MSG = "Why does the gateway keep failing after I restart it?"
# The same code work corroborated by error evidence: xhigh.
DEBUG_ERROR_MSG = (
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

    def test_compound_factual_requests_never_downshift(self):
        for text in (
            "Which tests failed and fix them?",
            "Which tests failed — fix them?",
            "Which tests failed: fix them?",
            "Which tests failed - fix them?",
            "Which tests failed (fix them)?",
            "Which tests failed fix them?",
            "What failed so we can repair it?",
            "What is the capital of France: book a flight?",
            "What time is it in Tokyo — notify me?",
            "Show me the readme: fix it",
            "Read README.md — fix it",
            "List the files in /tmp remove them",
            "Print the current config to decide what to change",
            "What is required to fix this?",
            "Who knows how to fix this?",
            "Show me how to fix this",
            "What failed then repair it?",
            "Where is the issue, fix it?",
            "Who owns this plus notify them?",
            "Which option works or should we build another?",
            "What changed & revert it?",
            "Where is the file / delete it?",
            "What failed before fixing it?",
            "Which tests failed after updating dependencies?",
            "Which approach should we use?",
            "What would you change?",
            "What is the best option?",
            "Which tests run if I update this?",
        ):
            for baseline in ("high", "xhigh"):
                with self.subTest(text=text, baseline=baseline):
                    agent = _make_agent(baseline=baseline, min_effort="low")
                    original = agent.reasoning_config
                    with adaptive_reasoning_turn(agent, text):
                        self.assertEqual(agent.reasoning_config["effort"], baseline)
                        self.assertEqual(agent._notices, [])
                    self.assertIs(agent.reasoning_config, original)

    def test_complete_simple_requests_downshift_high_baselines(self):
        # Closed lookup/retrieval shapes still save effort; denying every
        # factual/mechanical request would hide the compound-classifier bug.
        for text in (
            "What is the capital of France?",
            "what time is it in Tokyo?",
            "What is the current date?",
            "Which tests failed?",
            "Where is the config file?",
            "show me the readme",
            "list the files in /tmp",
            "print the current config",
            "thanks!",
        ):
            for baseline in ("high", "xhigh"):
                with self.subTest(text=text, baseline=baseline):
                    agent = _make_agent(baseline=baseline, min_effort="low")
                    original = agent.reasoning_config
                    with adaptive_reasoning_turn(agent, text):
                        self.assertEqual(agent.reasoning_config["effort"], "low")
                        self.assertEqual(len(agent._notices), 1)
                        self.assertIn("lowered to Low", agent._notices[0].text)
                    self.assertIs(agent.reasoning_config, original)

    def test_signal_bearing_action_keeps_baseline(self):
        # nginx is an infrastructure signal: a complexity signal always
        # overrides low, even for a short imperative.
        self.assertEqual(
            classify_reasoning_effort("restart the nginx service")[0], None
        )

    def test_realistic_near_misses_keep_baseline(self):
        for msg in (
            "go ahead",  # continuation without a prior escalated turn
            "how do I center a div?",  # a real (small) task, not a fact lookup
            "check whether the fix works",  # verification, not mechanical retrieval
            "summarize this file",  # unbounded content, unknown weight
            "list the open PRs and summarize the risky ones",  # compound request
        ):
            self.assertEqual(classify_reasoning_effort(msg)[0], None, msg)

    def test_complexity_signals_override_low(self):
        for msg in (
            "fix the crash",  # debugging signal in a terse message
            "show me the segfault",  # mechanical verb + error signal
            "why error: ENOENT?",  # error evidence
        ):
            self.assertEqual(classify_reasoning_effort(msg)[0], None, msg)
        # A factual-shaped question carrying a real signal keeps its
        # escalation classification — signals override the simple shape.
        self.assertEqual(
            classify_reasoning_effort("What is a race condition?")[0], "high"
        )

    def test_multiple_questions_keep_baseline(self):
        self.assertEqual(
            classify_reasoning_effort(
                "What is the gateway port? And where is the config file?"
            )[0],
            None,
        )

    def test_code_snippet_keeps_baseline(self):
        self.assertEqual(
            classify_reasoning_effort("what is this?\n```\nx = 1\n```")[0],
            None,
        )

    def test_empty_and_whitespace_keep_baseline(self):
        # No positive simplicity evidence (e.g. an image-only message whose
        # text extraction is empty) — never downshift on absence of signal.
        self.assertEqual(classify_reasoning_effort("")[0], None)
        self.assertEqual(classify_reasoning_effort("   \n ")[0], None)

    def test_lone_debugging_signal_is_high(self):
        level, reason = classify_reasoning_effort(DEBUG_MSG)
        self.assertEqual(level, "high")
        self.assertTrue(reason)

    def test_debugging_with_error_evidence_is_xhigh(self):
        # Code work corroborated by an error trace leans to xhigh.
        level, reason = classify_reasoning_effort(DEBUG_ERROR_MSG)
        self.assertEqual(level, "xhigh")
        self.assertIn("an error trace", reason)

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
            None,
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

    def test_continuation_without_prior_effort_keeps_baseline(self):
        self.assertEqual(classify_reasoning_effort("go ahead")[0], None)

    def test_unrelated_trivial_turn_after_escalation_drops(self):
        # A non-continuation message re-classifies from scratch even with a
        # prior escalated turn — nothing stays stuck at xhigh.
        level, _ = classify_reasoning_effort(
            "what time is it in Tokyo?", prior_effort="xhigh"
        )
        self.assertEqual(level, "low")

    def test_lone_code_keyword_stays_high(self):
        for msg in (
            DEBUG_MSG,
            "Implement a new session-title feature with tests",
            "implement a parser for this file format",
        ):
            with self.subTest(msg=msg):
                self.assertEqual(classify_reasoning_effort(msg)[0], "high")

    def test_code_work_with_corroboration_is_xhigh(self):
        for msg, corroboration in (
            (DEBUG_ERROR_MSG, "an error trace"),
            ("implement the redis integration for prod", "production impact"),
            ("refactor the parser to fix the crash in the tokenizer", "debugging/diagnosis"),
            ("Implement the following:\n1. parse\n2. validate\n3. store", "a multi-step request"),
            ("debug why this function returns None\n```\nx = f()\n```", "included code"),
            ("integrate the scheduler with the deploy pipeline", "consequential infrastructure work"),
            ("implement the title feature. " + "context " * 70, "a detailed request"),
        ):
            with self.subTest(msg=msg[:40]):
                level, reason = classify_reasoning_effort(msg)
                self.assertEqual(level, "xhigh")
                self.assertIn(corroboration, reason)

    def test_non_code_categories_keep_the_score_rule(self):
        # Research/analysis are not code-class: they never take the code path.
        for msg in (
            "research the vector database landscape for our use case",
            "compare and research these two approaches",
        ):
            with self.subTest(msg=msg):
                self.assertEqual(classify_reasoning_effort(msg)[0], "high")
        # Security/architecture keep needing a total score of 6.
        self.assertEqual(
            classify_reasoning_effort("Do a security audit of the gateway auth flow")[0], "high")
        self.assertEqual(classify_reasoning_effort(XHIGH_MSG)[0], "xhigh")
        # Terse code turns still never escalate on their own.
        self.assertIsNone(classify_reasoning_effort("fix the crash")[0])

    def test_short_followup_inherits_a_high_or_xhigh_prior(self):
        for prior in ("high", "xhigh"):
            for msg in ("fix it", "fix it please", "try again", "no, do it properly",
                        "why?", "still broken", "redo it", "it still doesn't work",
                        "lets fix it", "same error"):
                with self.subTest(prior=prior, msg=msg):
                    level, reason = classify_reasoning_effort(msg, prior_effort=prior)
                    self.assertEqual(level, prior)
                    self.assertIn("continuing", reason)

    def test_short_followup_without_escalated_prior_is_unchanged(self):
        for prior in (None, "low", "medium"):
            for msg in ("fix it", "try again", "why?", "still broken", "redo it"):
                with self.subTest(prior=prior, msg=msg):
                    self.assertIsNone(classify_reasoning_effort(msg, prior_effort=prior)[0])

    def test_unrelated_casual_turn_after_xhigh_reclassifies(self):
        for msg in ("thanks", "hi", "lol", "hi how are you"):
            with self.subTest(msg=msg):
                self.assertEqual(classify_reasoning_effort(msg, prior_effort="xhigh")[0], "low")
        # A follow-up with more than the closed phrase re-classifies too.
        self.assertIsNone(
            classify_reasoning_effort("fix it and also update the docs", prior_effort="xhigh")[0])

    def test_greeting_with_how_are_you_is_casual(self):
        for msg in ("hi how are you", "hey how are you?", "hello, how are you",
                    "how are you doing?", "hi there how's it going"):
            with self.subTest(msg=msg):
                level = classify_reasoning_effort(msg)[0]
                if "," in msg:
                    self.assertIsNone(level)  # the comma veto still applies
                else:
                    self.assertEqual(level, "low")
        # Closed set: anything appended falls back to the baseline.
        for msg in ("hi how are you, can you fix the build", "how are you handling retries?"):
            with self.subTest(msg=msg):
                self.assertNotEqual(classify_reasoning_effort(msg)[0], "low")

    def test_second_question_mark_and_trailing_newline_do_not_veto_low(self):
        for msg in ("What is the capital of France??", "What is the capital of France?\n",
                    "show me the readme\n", "  thanks!  \n", "hi how are you??"):
            with self.subTest(msg=msg):
                self.assertEqual(classify_reasoning_effort(msg)[0], "low")
        # Real compounds and inner structure still veto.
        for msg in ("What is the gateway port? And where is the config file?",
                    "What is the capital of France?\nWhich tests failed?",
                    "What changed & revert it?", "thanks; now fix the build",
                    "what is this?\n```\nx = 1\n```"):
            with self.subTest(msg=msg):
                self.assertIsNone(classify_reasoning_effort(msg)[0])

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
# Human-authored text and synthetic system turns
# ---------------------------------------------------------------------------

# Skill body words that escalate on their own (implement/debug/architecture/security).
_SKILL_BODY = (
    "# Release engineering\n"
    + "When asked, implement the change, debug any failing test, then evaluate the "
    "architecture trade-offs and run a security review before you deploy to production.\n" * 30
)


def _auto_loaded_payload(human):
    """The gateway's model-facing shape: auto-loaded skill dump + speaker prefix + text."""
    from agent.skill_commands import _build_skill_message

    note = '[IMPORTANT: The "release" skill is auto-loaded. Follow its instructions for this session.]'
    dump = _build_skill_message({"name": "release", "content": _SKILL_BODY}, None, note)
    return f"{dump}\n\n[Example] {human}"


def _skill_invocation(instruction=""):
    from agent.skill_commands import _build_skill_message

    note = ('[IMPORTANT: The user has invoked the "release" skill, indicating they want '
            "you to follow its instructions. The full skill content is loaded below.]")
    return _build_skill_message(
        {"name": "release", "content": _SKILL_BODY}, None, note, user_instruction=instruction,
    )


def _bundle_invocation(instruction=""):
    from agent.skill_commands import _build_skill_message, _scaffold_header

    header = _scaffold_header('"frfr" skill bundle', ["release"], lead_lines=["Bundle: frfr"],
                              user_instruction=instruction)
    block = _build_skill_message({"name": "release", "content": _SKILL_BODY}, None,
                                 '[Loaded as part of the "frfr" skill bundle.]')
    return f"{header}\n\n{block}"


def _synthetic_turns():
    """Every synthetic completion shape, from the real producers where they are callable."""
    import time
    from tools import process_registry_notifications as notes

    now = time.time()
    failing = {"status": "failed", "summary": "Traceback: exit code 1, error: connection refused"}
    return {
        "process completion": notes.format_process_notification({
            "type": "completion", "session_id": "proc_1", "command": "pytest -x tests/",
            "exit_code": 1, "output": "Traceback (most recent call last):\nError: failed",
            "completed_at": now,
        }),
        "process watch match": notes.format_process_notification({
            "type": "watch_match", "session_id": "proc_2", "command": "tail -f app.log",
            "pattern": "ERROR", "output": "ERROR crash in auth", "completed_at": now,
        }),
        "async delegation": notes._format_async_delegation({
            "delegation_id": "deleg_1", "goal": "debug the gateway crash",
            "results": [failing], "completed_at": now,
        }),
        "async delegation batch": notes._format_batch_delegation(
            {"results": [failing, failing], "completed_at": now}, "deleg_2", now,
        ),
        "batched process header": "[IMPORTANT: 3 background processes completed. Results below.]\n"
                                  + "exit code 1, error: refused\n" * 40,
        "legacy subagent line": "A background subagent you dispatched earlier has finished. "
                                "Error: exit code 2 after refactor and migration.",
        "legacy fan-out line": "A background fan-out of 3 subagent(s) you dispatched earlier has "
                               "finished. Traceback: exit code 1.",
    }


class TestHumanAuthoredClassification(unittest.TestCase):
    """Classify what the human wrote, never the enrichment or a synthetic notice."""

    def test_skill_dump_with_casual_human_text_does_not_escalate(self):
        payload = _auto_loaded_payload("thanks!")
        agent = _make_agent()
        baseline = agent.reasoning_config
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, payload, human_text="thanks!"))
        self.assertIs(agent.reasoning_config, baseline)
        self.assertEqual(agent._notices, [])

    def test_skill_dump_with_complex_human_text_still_escalates(self):
        payload = _auto_loaded_payload(DEBUG_MSG)
        agent = _make_agent()
        token = begin_adaptive_reasoning_turn(agent, payload, human_text=DEBUG_MSG)
        self.assertIsNotNone(token)
        self.assertEqual(agent.reasoning_config["effort"], "high")
        self.assertEqual(len(agent._notices), 1)
        end_adaptive_reasoning_turn(agent, token)

    def test_no_human_text_falls_back_to_the_payload(self):
        # Surfaces without a pre-injection text (CLI, subagent goals) classify
        # the payload exactly as before, skill dump included.
        payload = _auto_loaded_payload("thanks!")
        effort, _ = classify_reasoning_effort(payload)
        self.assertIn(effort, ("high", "xhigh"))
        agent = _make_agent()
        token = begin_adaptive_reasoning_turn(agent, payload)
        self.assertEqual(agent.reasoning_config["effort"], effort)
        end_adaptive_reasoning_turn(agent, token)

    def test_skill_invocation_classifies_only_the_typed_instruction(self):
        for build in (_skill_invocation, _bundle_invocation):
            with self.subTest(build=build.__name__):
                self.assertIn(classify_reasoning_effort(build("thanks"))[0], ("high", "xhigh"))
                for message in (build("thanks, looks good"), build("")):
                    # The gateway's title_user_message is the rewritten scaffold too.
                    for human in (None, message):
                        agent = _make_agent()
                        self.assertIsNone(
                            begin_adaptive_reasoning_turn(agent, message, human_text=human))
                        self.assertEqual(agent._notices, [])
                agent = _make_agent()
                token = begin_adaptive_reasoning_turn(agent, build(DEBUG_MSG))
                self.assertEqual(agent.reasoning_config["effort"], "high")
                end_adaptive_reasoning_turn(agent, token)

    def test_bare_skill_invocation_never_downshifts(self):
        agent = _make_agent(min_effort="low")
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, _skill_invocation("")))
        self.assertEqual(agent.reasoning_config, {"enabled": True, "effort": "medium"})

    def test_empty_human_text_keeps_the_no_adjustment_path(self):
        # Image-only turn: the payload may carry enrichment text, the human wrote nothing.
        blocks = [{"type": "text", "text": DEBUG_MSG}, {"type": "image_url", "image_url": {"url": "x"}}]
        for min_effort in (None, "low"):
            agent = _make_agent(min_effort=min_effort)
            self.assertIsNone(begin_adaptive_reasoning_turn(agent, blocks, human_text=""))
            self.assertEqual(agent._notices, [])

    def test_plain_messages_classify_exactly_as_before(self):
        from agent.adaptive_reasoning import classification_text

        for message in ("thanks!", "What is the capital of France?", DEBUG_MSG, XHIGH_MSG,
                        "go ahead", "rename notes.txt to todo.txt", "[draft] fix the typo",
                        "[IMPORTANT: read this] why does the build fail? error: exit code 2"):
            with self.subTest(message=message):
                self.assertEqual(classification_text(message), message)
                self.assertEqual(classification_text(message, message), message)
                for min_effort in (None, "low"):
                    plain, same = _make_agent(min_effort=min_effort), _make_agent(min_effort=min_effort)
                    a = begin_adaptive_reasoning_turn(plain, message)
                    b = begin_adaptive_reasoning_turn(same, message, human_text=message)
                    self.assertEqual(plain.reasoning_config, same.reasoning_config)
                    self.assertEqual(a is None, b is None)

    def test_synthetic_turns_get_no_adjustment_and_no_notice(self):
        turns = _synthetic_turns()
        for name, text in turns.items():
            with self.subTest(shape=name):
                self.assertIn(classify_reasoning_effort(text)[0], ("high", "xhigh", None))
                for human in (None, text):
                    for min_effort in (None, "low"):
                        agent = _make_agent(min_effort=min_effort)
                        agent.api_mode = "anthropic_messages"
                        baseline = agent.reasoning_config
                        self.assertIsNone(
                            begin_adaptive_reasoning_turn(agent, text, human_text=human))
                        self.assertIs(agent.reasoning_config, baseline)
                        self.assertEqual(agent._notices, [])
        # The shapes that used to escalate are the reason this exists.
        for name in ("async delegation batch", "process watch match", "batched process header"):
            self.assertIn(classify_reasoning_effort(turns[name])[0], ("high", "xhigh"), name)

    def test_synthetic_turn_is_transparent_to_continuation_and_dedup(self):
        agent = _make_agent()
        token = begin_adaptive_reasoning_turn(agent, DEBUG_MSG)
        end_adaptive_reasoning_turn(agent, token)
        self.assertEqual(len(agent._notices), 1)
        self.assertIsNone(
            begin_adaptive_reasoning_turn(agent, _synthetic_turns()["async delegation batch"]))
        self.assertEqual(agent._adaptive_prev_effort, "high")
        token = begin_adaptive_reasoning_turn(agent, "go ahead")
        self.assertEqual(agent.reasoning_config["effort"], "high")
        self.assertEqual(len(agent._notices), 1)  # same level: still deduplicated
        end_adaptive_reasoning_turn(agent, token)

    def test_synthetic_detector_covers_the_timeline_shapes(self):
        # Local copy of hermes_state_timeline._SYNTHETIC_PROMPT: never narrower.
        from agent.adaptive_reasoning import _SYNTHETIC_TURN_RE
        from hermes_state_timeline import _SYNTHETIC_PROMPT

        samples = [*_synthetic_turns().values(), "[ASYNC COMPLETE x]", "[async delegation complete — d]",
                   "  [IMPORTANT: Background process p exited", "thanks!", DEBUG_MSG,
                   _skill_invocation("thanks")]
        for text in samples:
            with self.subTest(text=text[:40]):
                if _SYNTHETIC_PROMPT.match(text):
                    self.assertTrue(_SYNTHETIC_TURN_RE.match(text))
        self.assertIsNone(_SYNTHETIC_TURN_RE.match(_skill_invocation("thanks")))
        self.assertIsNone(_SYNTHETIC_TURN_RE.match(DEBUG_MSG))

    def test_rebase_rerun_keeps_classifying_the_human_text(self):
        from agent.adaptive_reasoning import rebase_adaptive_turn

        agent = _make_agent()
        with adaptive_reasoning_turn(agent, _auto_loaded_payload("thanks!"), human_text="thanks!"):
            self.assertEqual(agent.reasoning_config, {"enabled": True, "effort": "medium"})
            agent.reasoning_config = {"enabled": True, "effort": "medium"}  # restored primary
            rebase_adaptive_turn(agent)
            self.assertEqual(agent.reasoning_config["effort"], "medium")
        self.assertEqual(agent._notices, [])


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

    def test_effort_only_thinking_off_is_never_adjusted(self):
        # Desktop Thinking Off can arrive as {effort: "none"} with ``enabled`` missing or still True,
        # and providers now honor it as disabled (#107238). It is not a low baseline to adjust from.
        for off in ({"effort": "none"}, {"enabled": True, "effort": "none"},
                    {"enabled": True, "effort": "disabled"}, {"effort": "false"}):
            for message, min_effort in ((DEBUG_MSG, None), ("thanks!", "low")):
                agent = _make_agent(reasoning_config=off, min_effort=min_effort)
                with adaptive_reasoning_turn(agent, message):
                    self.assertIs(agent.reasoning_config, off)
                self.assertIs(agent.reasoning_config, off)
                self.assertEqual(agent._notices, [])
        # Control: the same turn does escalate from a real baseline.
        agent = _make_agent()
        with adaptive_reasoning_turn(agent, DEBUG_MSG):
            self.assertEqual(agent.reasoning_config, {"enabled": True, "effort": "high"})

    def test_unknown_baseline_effort_skips(self):
        agent = _make_agent(
            reasoning_config={"enabled": True, "effort": "turbo"}
        )
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, DEBUG_MSG))

    def test_absent_baseline_is_inert(self):
        # No explicit reasoning baseline (reasoning_config is None) means the
        # provider decides whether thinking exists at all. Toggling thinking
        # presence mid-conversation invalidates more of the Anthropic prompt
        # cache than a level change, so adaptive must be inert — in both directions.
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


@pytest.mark.parametrize("baseline", ("none", *VALID_REASONING_EFFORTS))
@pytest.mark.parametrize("pinned", (False, True))
@pytest.mark.parametrize("policy,simple_target,complex_target", [
    ({}, None, "xhigh"),
    ({"min_effort": "minimal"}, "low", "xhigh"),
    ({"min_effort": "low"}, "low", "xhigh"),
    ({"min_effort": "medium"}, "medium", "xhigh"),
    ({"min_effort": "high", "max_effort": "high"}, "high", "high"),
    ({"min_effort": "low", "max_effort": "minimal"}, None, "minimal"),
    ({"min_effort": "none", "max_effort": "medium"}, None, "medium"),
    ({"min_effort": "minimal", "max_effort": "minimal"}, "minimal", "minimal"),
])
def test_adjustment_respects_baseline_direction(baseline, pinned, policy, simple_target, complex_target):
    """Bounds constrain adjustments, not the baseline; simple never raises it."""
    original = parse_reasoning_effort(baseline)
    agent = _make_agent(reasoning_config=original, user_override=pinned)
    agent.adaptive_reasoning = parse_adaptive_reasoning_config({"enabled": True, **policy})
    for message, direction, target in (
        ("summarize this file", "keep", None),
        ("go ahead", "keep", None),
        ("", "keep", None),
        ("Which tests failed fix them?", "keep", None),
        ("Which tests failed — fix them?", "keep", None),
        ("restart the nginx service", "keep", None),
        ("hello", "lower", simple_target),
        ("What is the capital of France?", "lower", simple_target),
        ("show me the readme", "lower", simple_target),
        (DEBUG_MSG, "raise", min("high", complex_target, key=VALID_REASONING_EFFORTS.index)),
        (XHIGH_MSG, "raise", complex_target),
    ):
        expected = original
        if not pinned and baseline != "none" and target is not None:
            levels = (baseline, target)
            selected = (min if direction == "lower" else max)(
                levels, key=VALID_REASONING_EFFORTS.index
            )
            expected = parse_reasoning_effort(selected)
        agent._notices.clear()
        # Each row starts a new task, not a continuation of the previous row.
        agent._adaptive_prev_effort = None
        agent._adaptive_last_notified_effort = None
        with adaptive_reasoning_turn(agent, message):
            assert agent.reasoning_config == expected, message
            assert bool(agent._notices) == (expected != original), message
            if expected == original:
                assert agent.reasoning_config is original
                assert agent._adaptive_prev_effort is None
                assert agent._adaptive_last_notified_effort is None
        assert agent.reasoning_config is original


# ---------------------------------------------------------------------------
# Calibration: xhigh lean, short-turn context, reachable low (agent level)
# ---------------------------------------------------------------------------

class TestCalibrationOnAgent(unittest.TestCase):
    def _turn(self, agent, text):
        token = begin_adaptive_reasoning_turn(agent, text)
        effort = agent.reasoning_config["effort"]
        if token is not None:
            end_adaptive_reasoning_turn(agent, token)
        return effort

    def test_fix_it_after_an_escalated_turn_inherits_the_level(self):
        for first, level in ((DEBUG_MSG, "high"), (DEBUG_ERROR_MSG, "xhigh")):
            with self.subTest(level=level):
                agent = _make_agent(min_effort="low")
                self.assertEqual(self._turn(agent, first), level)
                self.assertEqual(self._turn(agent, "fix it"), level)
                self.assertEqual(self._turn(agent, "still broken"), level)
                self.assertEqual(len(agent._notices), 1)  # same level: no re-notice

    def test_fix_it_without_an_escalated_prior_stays_at_baseline(self):
        for min_effort in (None, "low"):
            agent = _make_agent(min_effort=min_effort)
            self.assertIsNone(begin_adaptive_reasoning_turn(agent, "fix it"))
            self.assertEqual(agent.reasoning_config["effort"], "medium")
            self.assertEqual(agent._notices, [])

    def test_casual_turn_after_xhigh_drops_back(self):
        agent = _make_agent(min_effort="low")
        self.assertEqual(self._turn(agent, DEBUG_ERROR_MSG), "xhigh")
        self.assertEqual(self._turn(agent, "thanks"), "low")
        # The drop clears the continuation state: "fix it" now runs at baseline.
        self.assertEqual(self._turn(agent, "fix it"), "medium")
        # Escalation-only config: the casual turn returns to the baseline.
        agent = _make_agent()
        self.assertEqual(self._turn(agent, DEBUG_ERROR_MSG), "xhigh")
        self.assertEqual(self._turn(agent, "thanks"), "medium")
        self.assertEqual(self._turn(agent, "fix it"), "medium")

    def test_hi_how_are_you_lowers_only_when_min_effort_permits(self):
        agent = _make_agent(min_effort="low")
        self.assertEqual(self._turn(agent, "hi how are you"), "low")
        for min_effort in (None, "medium", "high"):
            with self.subTest(min_effort=min_effort):
                agent = _make_agent(min_effort=min_effort)
                self.assertIsNone(begin_adaptive_reasoning_turn(agent, "hi how are you"))
                self.assertEqual(agent.reasoning_config["effort"], "medium")

    def test_two_question_marks_reach_the_floor_never_below(self):
        agent = _make_agent(baseline="high", min_effort="medium")
        self.assertEqual(self._turn(agent, "What is the capital of France??"), "medium")
        agent = _make_agent(baseline="low", min_effort="low")
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, "What is the capital of France??"))

    def test_max_effort_caps_code_xhigh_and_inherited_followups(self):
        agent = _make_agent(max_effort="high")
        self.assertEqual(self._turn(agent, DEBUG_ERROR_MSG), "high")
        self.assertEqual(self._turn(agent, "fix it"), "high")
        agent = _make_agent(max_effort="medium")
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, DEBUG_ERROR_MSG))
        self.assertIsNone(begin_adaptive_reasoning_turn(agent, "fix it"))


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

    def test_only_positive_simple_classification_can_lower_a_high_baseline(self):
        for baseline in ("high", "xhigh"):
            for message in ("how do I center a div?", DEBUG_MSG, "go ahead"):
                agent = _make_agent(baseline=baseline, min_effort="low")
                saved = agent.reasoning_config
                self.assertIsNone(begin_adaptive_reasoning_turn(agent, message))
                self.assertIs(agent.reasoning_config, saved)

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


class TestAnthropicCacheNotice(unittest.TestCase):
    """Once-per-session cache warning on the anthropic-adapter route."""

    @staticmethod
    def _cache_notices(agent):
        return [n for n in agent._notices if n.key == CACHE_NOTICE_KEY]

    @staticmethod
    def _level_notices(agent):
        return [n for n in agent._notices if n.key == NOTICE_KEY]

    def _run_turns(self, agent, *messages):
        for message in messages:
            with adaptive_reasoning_turn(agent, message):
                pass

    def test_anthropic_route_warns_exactly_once_per_session(self):
        agent = _make_agent()
        agent.api_mode = "anthropic_messages"
        # Turn 1 carries the level-change notice (priority); turn 2 the warning.
        self._run_turns(agent, DEBUG_MSG)
        self.assertEqual(len(self._level_notices(agent)), 1)
        self.assertEqual(self._cache_notices(agent), [])
        self._run_turns(agent, DEBUG_MSG, XHIGH_MSG, DEBUG_MSG, DEBUG_MSG)
        cache = self._cache_notices(agent)
        self.assertEqual(len(cache), 1)
        self.assertIs(agent._notices[1], cache[0])
        self.assertIn("Anthropic's prompt cache", cache[0].text)
        self.assertIn("cost", cache[0].text)
        self.assertEqual(cache[0].kind, "ttl")
        self.assertEqual(cache[0].level, "warn")

    def test_session_reset_rearms_warning(self):
        from agent.adaptive_reasoning import init_adaptive_reasoning_state

        agent = _make_agent()
        agent.api_mode = "anthropic_messages"
        self._run_turns(agent, DEBUG_MSG, DEBUG_MSG)
        init_adaptive_reasoning_state(agent, {"enabled": True, "max_effort": "xhigh"})
        self._run_turns(agent, DEBUG_MSG, DEBUG_MSG)
        self.assertEqual(len(self._cache_notices(agent)), 2)

    def test_non_anthropic_route_never_warns(self):
        for api_mode in ("chat_completions", "codex_responses", "bedrock_converse", None):
            agent = _make_agent()
            agent.api_mode = api_mode
            self._run_turns(agent, DEBUG_MSG, DEBUG_MSG, XHIGH_MSG, DEBUG_MSG)
            self.assertEqual(self._cache_notices(agent), [], api_mode)
            self.assertEqual(agent.reasoning_config["effort"], "medium")

    def test_disabled_or_inert_adaptive_never_warns(self):
        cases = (
            _make_agent(enabled=False),
            _make_agent(reasoning_config=None),
            _make_agent(reasoning_config={"enabled": False}),
            _make_agent(user_override=True),
        )
        for agent in cases:
            agent.api_mode = "anthropic_messages"
            self._run_turns(agent, DEBUG_MSG, DEBUG_MSG, XHIGH_MSG)
            self.assertEqual(agent._notices, [])

    def test_unadjusted_turns_never_warn(self):
        agent = _make_agent()
        agent.api_mode = "anthropic_messages"
        self._run_turns(agent, "what time is it in Tokyo?", "thanks!", "what time is it in Tokyo?")
        self.assertEqual(agent._notices, [])

    def test_no_rail_and_subagent_never_warn(self):
        vprints = []
        agent = _make_agent()
        agent.api_mode = "anthropic_messages"
        agent.notice_callback = None
        agent._vprint = lambda *a, **k: vprints.append(a)
        self._run_turns(agent, DEBUG_MSG, DEBUG_MSG)
        self.assertEqual(len(vprints), 1)  # the level change only; no console warning
        self.assertNotIn("cache", vprints[0][0])

        child = _make_agent(platform="subagent")
        child.api_mode = "anthropic_messages"
        self._run_turns(child, DEBUG_MSG, DEBUG_MSG)
        self.assertEqual(child._notices, [])

    def test_level_change_notice_unaffected_on_anthropic_route(self):
        agent = _make_agent()
        agent.api_mode = "anthropic_messages"
        self._run_turns(agent, DEBUG_MSG, DEBUG_MSG, XHIGH_MSG)
        levels = self._level_notices(agent)
        self.assertEqual(len(levels), 2)
        self.assertIn("Reasoning raised to High", levels[0].text)
        self.assertIn("Reasoning raised to XHigh", levels[1].text)
        self.assertEqual([n.key for n in agent._notices], [NOTICE_KEY, CACHE_NOTICE_KEY, NOTICE_KEY])

    def test_rebase_rerun_does_not_double_notice_in_one_turn(self):
        from agent.adaptive_reasoning import rebase_adaptive_turn

        agent = _make_agent()
        agent.api_mode = "anthropic_messages"
        with adaptive_reasoning_turn(agent, DEBUG_MSG):
            agent.reasoning_config = {"enabled": True, "effort": "medium"}  # primary restore
            rebase_adaptive_turn(agent)
            self.assertEqual(agent.reasoning_config["effort"], "high")
        self.assertEqual([n.key for n in agent._notices], [NOTICE_KEY])
        self._run_turns(agent, DEBUG_MSG)
        self.assertEqual([n.key for n in agent._notices], [NOTICE_KEY, CACHE_NOTICE_KEY])


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

    def test_title_user_message_is_the_classified_text(self):
        """The gateway passes its pre-injection text as title_user_message; the facade hands it
        to adaptive, and a caller without one (CLI, delegate child) still classifies the payload."""
        from unittest.mock import patch

        payload = _auto_loaded_payload("thanks!")
        for title, expected in (("thanks!", "medium"), (DEBUG_MSG, "high"), (None, "xhigh")):
            with self.subTest(title=title):
                agent = self._make_real_agent()
                seen = {}

                def _fake_loop(agent_arg, *args, **kwargs):
                    seen["effort"] = agent_arg.reasoning_config["effort"]
                    return {"final_response": "ok", "messages": []}

                with patch("agent.conversation_loop.run_conversation", _fake_loop):
                    agent.run_conversation(payload, title_user_message=title)
                self.assertEqual(seen["effort"], expected)
                self.assertEqual(agent.reasoning_config, {"enabled": True, "effort": "medium"})

    def test_host_cancellation_remains_active_during_adaptive_turn(self):
        from unittest.mock import Mock, patch
        from agent.interrupt_scope import InterruptScope, bind_interrupt_scope

        for raises in (False, True):
            with self.subTest(raises=raises):
                agent = self._make_real_agent()
                agent.hard_interrupt = Mock()
                baseline = agent.reasoning_config
                scope = InterruptScope()

                def loop(current, *args, **kwargs):
                    self.assertEqual(current.reasoning_config["effort"], "high")
                    scope.cancel("cancel adaptive turn")
                    current.hard_interrupt.assert_called_once_with(
                        "cancel adaptive turn", tool_reason="host cancelled the command",
                    )
                    if raises:
                        raise KeyboardInterrupt()
                    return {"final_response": "cancelled", "messages": []}

                with bind_interrupt_scope(scope), patch(
                    "agent.conversation_loop.run_conversation", loop,
                ):
                    if raises:
                        with self.assertRaises(KeyboardInterrupt):
                            agent.run_conversation(DEBUG_MSG)
                    else:
                        agent.run_conversation(DEBUG_MSG)
                self.assertIs(agent.reasoning_config, baseline)
                agent.hard_interrupt.reset_mock()
                scope.cancel("after turn")
                agent.hard_interrupt.assert_not_called()

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

    def test_rejected_admission_never_adapts_or_notifies(self):
        from unittest.mock import patch

        agent = self._make_real_agent()
        baseline = agent.reasoning_config
        notices = []
        agent.notice_callback = notices.append
        rejected = {"interrupted": True, "final_response": "cancelled", "messages": []}

        def reject(*args, **kwargs):
            self.assertIs(agent.reasoning_config, baseline)
            return SimpleNamespace(early_result=rejected)

        with patch("agent.turn_facade_lease.admit_durable_turn_lease", reject):
            self.assertIs(agent.run_conversation(DEBUG_MSG), rejected)
        self.assertEqual(notices, [])
        self.assertIsNone(agent._adaptive_prev_effort)
        self.assertIs(agent.reasoning_config, baseline)
        self.assertEqual(rejected["messages"][0]["content"], DEBUG_MSG)
        from agent.session_persistence import _PERSIST_AFTER_ADMISSION_INTERRUPT
        self.assertTrue(rejected["messages"][0][_PERSIST_AFTER_ADMISSION_INTERRUPT])

    def test_review_fence_sees_baseline(self):
        from unittest.mock import patch

        agent = self._make_real_agent()
        baseline = agent.reasoning_config

        def fence(a):
            self.assertIs(a.reasoning_config, baseline)

        with (
            patch("agent.background_review.cancel_background_review_for_live_turn", fence),
            patch("agent.conversation_loop.run_conversation", return_value={}),
        ):
            agent.run_conversation(DEBUG_MSG)
        self.assertIs(agent.reasoning_config, baseline)

    def test_positional_moa_config_keeps_baseline(self):
        import inspect
        from unittest.mock import patch

        agent = self._make_real_agent()
        baseline = agent.reasoning_config
        signature = inspect.signature(agent.run_conversation)
        arguments = signature.bind(DEBUG_MSG, moa_config={"preset": "x"})
        arguments.apply_defaults()

        def loop(a, *args, **kwargs):
            self.assertIs(a.reasoning_config, baseline)
            return {}

        with patch("agent.conversation_loop.run_conversation", loop):
            agent.run_conversation(*arguments.args, **arguments.kwargs)
        self.assertIs(agent.reasoning_config, baseline)


class TestAdaptiveReasoningTurnScope(unittest.TestCase):
    def test_restores_on_interrupt(self):
        agent = _make_agent()
        baseline = agent.reasoning_config
        with self.assertRaises(KeyboardInterrupt):
            with adaptive_reasoning_turn(agent, DEBUG_MSG):
                self.assertEqual(agent.reasoning_config["effort"], "high")
                raise KeyboardInterrupt()
        self.assertIs(agent.reasoning_config, baseline)


@pytest.mark.parametrize("baseline", ("none", *VALID_REASONING_EFFORTS))
@pytest.mark.parametrize("ceiling", ("low", "medium", "xhigh"))
@pytest.mark.parametrize("pinned", (False, True))
def test_real_child_construction_preserves_adaptive_and_upstream_policies(
    tmp_path, monkeypatch, baseline, ceiling, pinned,
):
    """Real profile loader -> child resolver -> AIAgent -> admitted turn; no model request."""
    from unittest.mock import patch

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_yaml import safe_dump
    from run_agent import AIAgent
    from tools.delegate_tool import _build_child_agent

    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text(safe_dump({
        "delegation": {
            "compression_threshold_tokens": 32000,
            "fallback_providers": [{"provider": "custom", "model": "worker-fallback"}],
            **({"reasoning_effort": baseline} if pinned else {}),
        },
    }))
    scope = set_hermes_home_override(tmp_path)
    parent = child = None
    try:
        parent = AIAgent(
            model="test-model", provider="custom", api_key="test-only",
            base_url="http://127.0.0.1:1/v1", api_mode="chat_completions",
            enabled_toolsets=[], quiet_mode=True, skip_memory=True, skip_context_files=True,
            reasoning_config=parse_reasoning_effort(baseline),
            adaptive_reasoning={"enabled": True, "min_effort": "low", "max_effort": ceiling},
        )
        parent._cache_ttl = "1h"
        child = _build_child_agent(
            task_index=0, goal=XHIGH_MSG, context=None, toolsets=None, model="child-model",
            max_iterations=10, task_count=1, parent_agent=parent,
            routing_cfg={"fallback_providers": [{"provider": "custom", "model": "review-fallback"}]},
        )
        assert child.adaptive_reasoning == parent.adaptive_reasoning
        assert child.adaptive_reasoning is not parent.adaptive_reasoning
        assert child.reasoning_user_override is pinned
        assert child._cache_ttl == "5m"
        assert child.context_compressor.threshold_tokens_cap == 32000
        assert child._fallback_chain[0]["model"] == "review-fallback"
        saved = child.reasoning_config
        assert saved == parent.reasoning_config == parse_reasoning_effort(baseline)
        escalated = simple = saved
        if baseline != "none" and not pinned:
            escalated = parse_reasoning_effort(max(baseline, ceiling, key=VALID_REASONING_EFFORTS.index))
            simple = parse_reasoning_effort(min(baseline, "low", key=VALID_REASONING_EFFORTS.index))

        # Exercise the real delegate resolver and admitted-turn forwarder.
        # A low/medium ceiling still permits a concrete continuation from minimal.
        for message, expected in (
            (XHIGH_MSG, escalated), ("go ahead", escalated), ("  YES\n please!", escalated),
            ("summarize this file", saved), ("go ahead", saved),
            ("hello", simple), ("go ahead", saved),
        ):
            def loop(agent, *args, **kwargs):
                assert agent.reasoning_config == expected, message
                assert parent.reasoning_config == saved
                return {"final_response": "ok"}

            with patch("agent.conversation_loop.run_conversation", loop):
                assert child.run_conversation(message)["final_response"] == "ok"
            assert child.reasoning_config is saved
    finally:
        if child is not None:
            child.close()
        if parent is not None:
            parent.close()
        reset_hermes_home_override(scope)



class TestRealTurnStartRuntimeRestore(unittest.TestCase):
    """Real AIAgent.run_conversation -> conversation_loop -> build_turn_context ->
    restore_primary_runtime (no loop mock): the turn after a fallback restores the primary runtime
    INSIDE the adaptive turn. The adjusted effort must reach the wire request, and the restored
    primary baseline (not the fallback's) must be what remains after the turn."""

    DEBUG = DEBUG_MSG

    def _agent(self):
        from unittest.mock import MagicMock, patch
        from run_agent import AIAgent

        with (
            patch("model_tools.get_tool_definitions", return_value=[]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
            patch("agent.context_compressor.get_model_context_length", return_value=200_000),
        ):
            agent = AIAgent(
                api_key="test-key-12345678", base_url="https://my-llm.example.com/v1", provider="custom",
                model="primary-model", quiet_mode=True, skip_context_files=True, skip_memory=True,
                reasoning_config={"enabled": True, "effort": "medium"},
                adaptive_reasoning={"enabled": True, "max_effort": "xhigh", "min_effort": "low"},
            )
        agent._cached_system_prompt = "You are helpful."
        agent.compression_enabled = False
        agent.save_trajectories = False
        return agent

    def _run(self, agent, message):
        from types import SimpleNamespace as NS
        from unittest.mock import MagicMock, patch

        seen = []

        def create(**kwargs):
            seen.append((agent.model, kwargs.get("extra_body", {}).get("reasoning"),
                         kwargs.get("reasoning_effort"), dict(agent.reasoning_config or {})))
            msg = NS(content="ok", tool_calls=None, reasoning=None, reasoning_content=None,
                     reasoning_details=None)
            return NS(choices=[NS(message=msg, finish_reason="stop")], model="m", usage=None)

        client = MagicMock()
        client.chat.completions.create.side_effect = create
        with (
            patch("agent.process_bootstrap.OpenAI", return_value=client),
            patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.client = client
            result = agent.run_conversation(message)
        return result, seen

    def _simulate_prior_fallback(self, agent, fallback_effort):
        # What try_activate_fallback leaves behind for the NEXT turn: fallback route + its own
        # re-resolved reasoning_config, with the primary snapshot pending restore at turn start.
        agent.model = "fallback-model"
        agent._fallback_activated = True
        agent.reasoning_config = {"enabled": True, "effort": fallback_effort}

    def test_init_snapshot_restore_keeps_non_adaptive_parity(self):
        """The init-time snapshot carries no reasoning_config (upstream: "older snapshots ... keep the
        current value"), so the fallback's effort stays the baseline with or without adaptive. Adaptive
        escalates for the turn on the restored primary route and leaves exactly that value behind."""
        agent = self._agent()
        self.assertNotIn("reasoning_config", agent._primary_runtime)
        self._simulate_prior_fallback(agent, "low")
        result, seen = self._run(agent, self.DEBUG)
        self.assertEqual(result["final_response"], "ok")
        self.assertEqual(agent.model, "primary-model", "turn start restored the primary runtime")
        self.assertEqual((seen[0][0], seen[0][3]["effort"]), ("primary-model", "high"))
        self.assertEqual(agent.reasoning_config, {"enabled": True, "effort": "low"})

    def test_escalation_survives_restore_of_switch_snapshot_reasoning(self):
        """A switch_model snapshot DOES carry reasoning_config. Restoring it at turn start must not
        silently discard this turn's escalation: classify against the restored primary baseline."""
        agent = self._agent()
        agent._primary_runtime["reasoning_config"] = {"enabled": True, "effort": "medium"}
        self._simulate_prior_fallback(agent, "low")
        _, seen = self._run(agent, self.DEBUG)
        self.assertEqual((seen[0][0], seen[0][3]["effort"]), ("primary-model", "high"))
        self.assertEqual(agent.reasoning_config, {"enabled": True, "effort": "medium"})
        self.assertFalse(agent._fallback_activated)

    def test_restore_to_primary_baseline_meeting_the_level_is_no_adjustment(self):
        agent = self._agent()
        agent._primary_runtime["reasoning_config"] = {"enabled": True, "effort": "high"}
        self._simulate_prior_fallback(agent, "low")
        _, seen = self._run(agent, self.DEBUG)
        self.assertEqual(seen[0][3]["effort"], "high")
        self.assertEqual(agent.reasoning_config, {"enabled": True, "effort": "high"})
        self.assertIsNone(agent._adaptive_prev_effort)

    def test_restore_to_disabled_primary_never_reenables_thinking(self):
        agent = self._agent()
        agent._primary_runtime["reasoning_config"] = {"enabled": False}
        self._simulate_prior_fallback(agent, "medium")
        _, seen = self._run(agent, self.DEBUG)
        self.assertEqual(seen[0][3], {"enabled": False})
        self.assertEqual(agent.reasoning_config, {"enabled": False})

    def test_no_fallback_turn_restores_own_baseline(self):
        agent = self._agent()
        baseline = agent.reasoning_config
        _, seen = self._run(agent, self.DEBUG)
        self.assertEqual(seen[0][3]["effort"], "high")
        self.assertIs(agent.reasoning_config, baseline)


if __name__ == "__main__":
    unittest.main()
