"""Compaction must preserve policy, not only the imperative (#84718).

The issue's core finding: at a compaction boundary the todo list is re-injected
verbatim while the skill instructions that constrained it are deleted, the
originating request survives only by positional luck, and the session row
reports nothing about any of it. These tests cover the three halves fixed on
the compression side — skill policy digests, the semantic origin anchor, and
the observability counters.
"""

import json
import os
import tempfile
from unittest.mock import patch

from agent.context_compressor import (
    ContextCompressor,
    _ORIGIN_ANCHOR_HEADING,
    _SKILL_DIGEST_HEADING_PREFIX,
    _SKILL_DIGEST_MAX_CHARS,
    _SKILL_DIGEST_REINJECT_BUDGET_CHARS,
    _collect_ghosted_skill_digests,
    _extract_origin_anchor,
    _extract_skill_digest_block,
    _reinject_origin_anchor,
    _reinject_pruned_skill_markers,
    _skill_policy_digest,
    _skill_pruned_marker,
    _summarize_tool_result,
)

SKILL_BODY = """---
name: hodle-ui-validation
description: UI validation rules
---

# Hodle UI Validation

Some prose about branding that is not policy.

## Workflow

1. Capture the home desktop surface.
2. Capture the home mobile surface.
3. Attach the evidence to the PR.

## Rules

You MUST capture only the surface that actually changed.
NEVER delete a route you did not create.
Ordinary prose mentioning must in lowercase is not a rule.
"""


class TestSkillPolicyDigest:
    def test_digest_keeps_workflow_and_rules(self):
        digest = _skill_policy_digest(SKILL_BODY, "hodle-ui-validation")
        assert digest.startswith(f"{_SKILL_DIGEST_HEADING_PREFIX} hodle-ui-validation]")
        assert "## Workflow" in digest
        assert "1. Capture the home desktop surface." in digest
        assert "You MUST capture only the surface that actually changed." in digest
        assert "NEVER delete a route you did not create." in digest

    def test_digest_drops_prose_and_unrelated_sections(self):
        digest = _skill_policy_digest(SKILL_BODY, "hodle-ui-validation")
        assert "Some prose about branding" not in digest
        # Lowercase "must" is prose, not a rule.
        assert "Ordinary prose mentioning must in lowercase" not in digest

    def test_digest_is_hard_capped(self):
        body = "## Workflow\n" + "\n".join(f"- MUST do step {i}" for i in range(5000))
        digest = _skill_policy_digest(body, "huge")
        assert len(digest) <= _SKILL_DIGEST_MAX_CHARS + 64
        assert "[digest truncated]" in digest

    def test_digest_empty_when_no_policy_lines(self):
        assert _skill_policy_digest("# Title\n\nJust prose.\n", "plain") == ""
        assert _skill_policy_digest("", "plain") == ""

    def test_round_trips_through_extraction(self):
        digest = _skill_policy_digest(SKILL_BODY, "hodle-ui-validation")
        text = "some summary\n\n" + digest + "\n"
        assert _extract_skill_digest_block(text, "hodle-ui-validation") == digest.rstrip()
        assert _extract_skill_digest_block(text, "other") == ""


class TestPrunedSkillRowCarriesPolicy:
    def test_pruned_skill_view_result_keeps_marker_and_digest(self):
        big = SKILL_BODY + ("x" * 6000)
        summary = _summarize_tool_result(
            "skill_view", json.dumps({"name": "hodle-ui-validation"}), big,
        )
        assert _skill_pruned_marker("hodle-ui-validation") in summary
        assert "NEVER delete a route you did not create." in summary

    def test_small_skill_is_untouched(self):
        summary = _summarize_tool_result(
            "skill_view", json.dumps({"name": "tiny"}), "MUST do a thing",
        )
        assert _SKILL_DIGEST_HEADING_PREFIX not in summary
        assert "SKILL_PRUNED" not in summary

    def test_pruned_row_is_not_re_summarized_away(self):
        """A digest-bearing row exceeds the 400-char "already summarized" cap.

        Without the marker-aware guard, a later prune pass would summarize it
        again and delete both the digest AND the ghost-skill marker.
        """
        big = SKILL_BODY + ("x" * 6000)
        pruned_row = _summarize_tool_result(
            "skill_view", json.dumps({"name": "hodle-ui-validation"}), big,
        )
        assert len(pruned_row) > 400  # the guard's length ceiling
        c = ContextCompressor(model="test", quiet_mode=True)
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "c1",
                    "function": {
                        "name": "skill_view",
                        "arguments": json.dumps({"name": "hodle-ui-validation"}),
                    },
                }],
            },
            {"role": "tool", "tool_call_id": "c1", "content": pruned_row},
        ] + [{"role": "user", "content": f"turn {i}"} for i in range(6)]
        out, _pruned = c._prune_old_tool_results(messages, protect_tail_count=1)
        row = next(m for m in out if m.get("tool_call_id") == "c1")
        assert _skill_pruned_marker("hodle-ui-validation") in row["content"]
        assert "NEVER delete a route you did not create." in row["content"]


class TestDigestReinjection:
    def test_digests_ride_the_summary_boundary(self):
        digest = _skill_policy_digest(SKILL_BODY, "alpha")
        out = _reinject_pruned_skill_markers(
            "summary text", ["alpha"], {"alpha": digest},
        )
        assert _skill_pruned_marker("alpha") in out
        assert "NEVER delete a route you did not create." in out

    def test_digest_budget_is_bounded(self):
        big_digest = f"{_SKILL_DIGEST_HEADING_PREFIX} s]\n" + ("MUST y\n" * 400)
        names = [f"s{i}" for i in range(20)]
        digests = {n: big_digest for n in names}
        out = _reinject_pruned_skill_markers("summary", names, digests)
        assert len(out) < _SKILL_DIGEST_REINJECT_BUDGET_CHARS + 8000

    def test_no_digests_matches_previous_behavior(self):
        out = _reinject_pruned_skill_markers("summary", ["alpha"], None)
        assert _skill_pruned_marker("alpha") in out
        assert _SKILL_DIGEST_HEADING_PREFIX not in out

    def test_collect_digests_from_raw_and_pruned_rows(self):
        big = SKILL_BODY + ("x" * 6000)
        pruned_row = _summarize_tool_result(
            "skill_view", json.dumps({"name": "beta"}), big,
        )
        turns = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "c1",
                    "function": {
                        "name": "skill_view",
                        "arguments": json.dumps({"name": "alpha"}),
                    },
                }],
            },
            {"role": "tool", "tool_call_id": "c1", "content": big},
            {"role": "tool", "tool_call_id": "c2", "content": pruned_row},
        ]
        digests = _collect_ghosted_skill_digests(turns)
        assert "NEVER delete a route" in digests["alpha"]
        assert "NEVER delete a route" in digests["beta"]


class TestOriginAnchor:
    def test_anchor_appended_and_extractable(self):
        out = _reinject_origin_anchor("summary", "fix the Hodle logo")
        assert _ORIGIN_ANCHOR_HEADING in out
        assert _extract_origin_anchor(out) == "fix the Hodle logo"

    def test_anchor_is_not_duplicated(self):
        once = _reinject_origin_anchor("summary", "fix the logo")
        twice = _reinject_origin_anchor(once, "fix the logo")
        assert twice.count(_ORIGIN_ANCHOR_HEADING) == 1

    def test_empty_anchor_is_a_noop(self):
        assert _reinject_origin_anchor("summary", "") == "summary"

    def test_capture_prefers_earliest_real_user_turn(self):
        c = ContextCompressor(model="test", quiet_mode=True)
        c._capture_origin_anchor([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "  "},
            {"role": "user", "content": "the Hodle logo looks wrong"},
            {"role": "user", "content": "remove that"},
        ])
        assert c._origin_anchor_text == "the Hodle logo looks wrong"

    def test_capture_is_sticky(self):
        c = ContextCompressor(model="test", quiet_mode=True)
        c._capture_origin_anchor([{"role": "user", "content": "first ask"}])
        c._capture_origin_anchor([{"role": "user", "content": "later ask"}])
        assert c._origin_anchor_text == "first ask"

    def test_capture_carries_forward_from_previous_summary(self):
        c = ContextCompressor(model="test", quiet_mode=True)
        c._previous_summary = _reinject_origin_anchor("old summary", "original ask")
        c._capture_origin_anchor([{"role": "user", "content": "a much later ask"}])
        assert c._origin_anchor_text == "original ask"

    def test_request_carrying_a_secret_is_not_pinned(self):
        """A pinned anchor rides every future summary; a masked credential must not."""
        c = ContextCompressor(model="test", quiet_mode=True)
        c._capture_origin_anchor([
            {"role": "user", "content": "deploy with ghp_" + ("a" * 36)},
        ])
        assert c._origin_anchor_text == ""

    def test_anchor_survives_the_deterministic_fallback(self):
        c = ContextCompressor(
            model="test", quiet_mode=True, protect_first_n=1, protect_last_n=1,
        )
        msgs = _compressible_messages("the Hodle logo looks wrong, use h-logo")
        with patch("agent.context_compressor.call_llm", side_effect=Exception("timeout")):
            result = c.compress(msgs)
        merged = "\n".join(
            m.get("content", "") for m in result if isinstance(m.get("content"), str)
        )
        assert _ORIGIN_ANCHOR_HEADING in merged
        assert "the Hodle logo looks wrong, use h-logo" in merged


class _StatsDB:
    def __init__(self):
        self.calls = []

    def record_compaction_stats(self, session_id, **kwargs):
        self.calls.append((session_id, kwargs))


def _compressible_messages(first_user="do the thing"):
    """A transcript long enough to have a real middle window to compact."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": first_user},
    ]
    for i in range(6):
        msgs.append({"role": "assistant", "content": f"working on step {i} " * 20})
        msgs.append({"role": "user", "content": f"and then? {i}"})
    msgs.append({"role": "assistant", "content": "done"})
    msgs.append({"role": "user", "content": "status?"})
    return msgs


class TestCompactionObservability:
    def test_completed_compaction_records_counters(self):
        c = ContextCompressor(
            model="test", quiet_mode=True, protect_first_n=1, protect_last_n=1,
        )
        db = _StatsDB()
        c._session_db = db
        c._session_id = "sess-1"
        with patch("agent.context_compressor.call_llm", side_effect=Exception("timeout")):
            c.compress(_compressible_messages())
        assert db.calls, "a completed compaction must record its counters"
        session_id, kwargs = db.calls[-1]
        assert session_id == "sess-1"
        assert set(kwargs) == {
            "pruned_skills", "pruned_tool_outputs", "tokens_reclaimed",
        }
        assert kwargs["tokens_reclaimed"] >= 0

    def test_recording_failure_never_breaks_compaction(self):
        class _Boom:
            def record_compaction_stats(self, *a, **k):
                raise RuntimeError("db down")

        c = ContextCompressor(
            model="test", quiet_mode=True, protect_first_n=1, protect_last_n=1,
        )
        c._session_db = _Boom()
        c._session_id = "sess-2"
        with patch("agent.context_compressor.call_llm", side_effect=Exception("timeout")):
            out = c.compress(_compressible_messages())
        assert out  # compaction still returned a transcript

    def test_state_db_accumulates_and_reads_back(self, monkeypatch):
        home = tempfile.mkdtemp()
        monkeypatch.setenv("HERMES_HOME", home)
        os.makedirs(home, exist_ok=True)
        from hermes_state import SessionDB

        db = SessionDB()
        db.create_session("s-obs", source="cli")
        db.record_compaction_stats(
            "s-obs", pruned_skills=3, pruned_tool_outputs=41, tokens_reclaimed=118_432,
        )
        db.record_compaction_stats(
            "s-obs", pruned_skills=1, pruned_tool_outputs=2, tokens_reclaimed=-5,
        )
        stats = db.get_compaction_stats("s-obs")
        assert stats == {
            "compaction_count": 2,
            "pruned_skills": 4,
            "pruned_tool_outputs": 43,
            "tokens_reclaimed": 118_432,  # negative reclaim clamped to 0
        }
        assert db.get_compaction_stats("missing")["compaction_count"] == 0
        bulk = db.get_compaction_stats_map(["s-obs", "missing"])
        assert set(bulk) == {"s-obs"}
