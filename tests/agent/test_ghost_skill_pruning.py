"""Ghost-skill defense tests (#32106, salvage of PR #44166).

When compaction reduces an old ``skill_view`` result to a metadata-only
summary, the model still believes the skill is loaded even though its
instructions are gone. The defense has three layers:

- P0/P1: the pruned tool-result summary carries a canonical
  ``[SKILL_PRUNED: ...]`` marker with the exact reload call, and the
  system prompt (SKILLS_GUIDANCE) tells the model how to react to it.
- Phase-1 protection: a skill loaded just before compaction (or actively
  referenced in the protected tail) keeps its full body through the
  ordinary prune passes.
- P2: markers entering the summarizer are extracted BEFORE the aux LLM
  call and deterministically re-injected if the model paraphrased them
  away — including on the static fallback path.

Test patterns for the marker emit checks adapted from PR #32375
(@LeonSGP43) with credit.
"""

from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    SKILL_PRUNED_MARKER_PREFIX,
    SUMMARY_PREFIX,
    ContextCompressor,
    _collect_protected_skill_names,
    _extract_pruned_skill_names,
    _MAX_PRUNED_SKILL_MARKERS,
    _reinject_pruned_skill_markers,
    _skill_pruned_marker,
    _summarize_tool_result,
)


def _make_compressor(**overrides):
    kwargs = dict(
        model="test/model",
        quiet_mode=True,
        protect_first_n=1,
        protect_last_n=2,
    )
    kwargs.update(overrides)
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100000
    ):
        return ContextCompressor(**kwargs)


def _skill_view_pair(call_id, skill_name, size=6000):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "skill_view",
                    "arguments": f'{{"name":"{skill_name}"}}',
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"# {skill_name} instructions\n" + "x" * size,
        },
    ]


class TestSkillPrunedMarkerEmit:
    """Marker emit — patterns adapted from PR #32375 (@LeonSGP43)."""

    def test_skill_view_summary_marks_pruned_content(self):
        summary = _summarize_tool_result(
            "skill_view", '{"name":"docker-management"}', "x" * 6000
        )
        assert summary.startswith("[skill_view] name=docker-management (6,000 chars)")
        assert _skill_pruned_marker("docker-management") in summary
        assert "reload with skill_view(name='docker-management')" in summary

    def test_small_skill_view_summary_not_marked(self):
        summary = _summarize_tool_result(
            "skill_view", '{"name":"docker-management"}', "x" * 1234
        )
        assert summary == "[skill_view] name=docker-management (1,234 chars)"
        assert SKILL_PRUNED_MARKER_PREFIX not in summary

    def test_other_skill_tool_summaries_remain_metadata_only(self):
        summary = _summarize_tool_result(
            "skills_list", '{"name":"docker-management"}', "x" * 6000
        )
        assert summary == "[skills_list] name=docker-management (6,000 chars)"
        assert SKILL_PRUNED_MARKER_PREFIX not in summary

    def test_marker_extractor_round_trips_the_emitted_marker(self):
        """Emit and check sides share one canonical string.

        The original PR #44166 emitted ``[SKILL_PRUNED:`` but presence-
        checked ``[SKILL_PRUNED]`` — re-injection fired even when the
        marker had survived. Pin the round trip.
        """
        summary = _summarize_tool_result("skill_view", '{"name":"pdf"}', "x" * 6000)
        assert _extract_pruned_skill_names(summary) == ["pdf"]
        assert _skill_pruned_marker("pdf") in summary


class TestReinjectPrunedSkillMarkers:
    def test_no_duplicate_when_canonical_marker_survived(self):
        marker = _skill_pruned_marker("pdf")
        out = _reinject_pruned_skill_markers("summary body\n" + marker, ["pdf"])
        assert out.count(SKILL_PRUNED_MARKER_PREFIX) == 1

    def test_reinjects_when_marker_paraphrased_away(self):
        out = _reinject_pruned_skill_markers(
            "The pdf skill was loaded earlier but its content was summarized.",
            ["pdf"],
        )
        assert _skill_pruned_marker("pdf") in out
        assert "## Pruned Skills" in out

    def test_partial_survival_reinjects_only_missing(self):
        marker_a = _skill_pruned_marker("alpha")
        out = _reinject_pruned_skill_markers("body\n" + marker_a, ["alpha", "beta"])
        assert out.count(_skill_pruned_marker("alpha")) == 1
        assert out.count(_skill_pruned_marker("beta")) == 1

    def test_empty_name_list_is_a_no_op(self):
        assert _reinject_pruned_skill_markers("body", []) == "body"

    def test_marker_block_is_not_classified_as_handoff_content(self):
        """Markers must never turn a row into summary/handoff content."""
        marker = _skill_pruned_marker("pdf")
        assert ContextCompressor.classify_summary_content(marker) is None
        row = "[skill_view] name=pdf (6,000 chars) " + marker
        assert ContextCompressor.classify_summary_content(row) is None
        # And the strip helper leaves marker-bearing rows untouched.
        c = ContextCompressor.__new__(ContextCompressor)
        msg = {"role": "user", "content": row}
        assert c._strip_context_summary_handoff_message(msg) == msg

    def test_reinjected_summary_still_classifies_standalone(self):
        body = _reinject_pruned_skill_markers("## Goal\nwork\n", ["pdf"])
        full = SUMMARY_PREFIX + "\n\n" + body
        assert ContextCompressor.classify_summary_content(full) == "standalone"


class TestProtectedSkillPrune:
    """Phase-1 prune must not demote a just-loaded skill (#32106)."""

    def _filler(self, n, start=0):
        out = []
        for i in range(n):
            role = "user" if (start + i) % 2 == 0 else "assistant"
            out.append({"role": role, "content": f"filler {start + i} " + "y" * 400})
        return out

    def test_old_single_use_skill_is_pruned_with_marker(self):
        c = _make_compressor()
        msgs = _skill_view_pair("call_s", "old-skill") + self._filler(14)
        result, pruned = c._prune_old_tool_results(msgs, protect_tail_count=4)
        assert pruned >= 1
        skill_row = result[1]
        assert _skill_pruned_marker("old-skill") in skill_row["content"]

    def test_recently_loaded_skill_survives_prune(self):
        c = _make_compressor()
        # skill loaded within the last 10 messages, but OUTSIDE the
        # protected tail count — without the guard it would be demoted.
        msgs = (
            self._filler(10)
            + _skill_view_pair("call_s", "fresh-skill")
            + self._filler(6, start=10)
        )
        result, _ = c._prune_old_tool_results(msgs, protect_tail_count=4)
        skill_row = result[11]
        assert skill_row["content"].startswith("# fresh-skill instructions")
        assert SKILL_PRUNED_MARKER_PREFIX not in skill_row["content"]

    def test_skill_named_in_tail_user_message_survives_prune(self):
        c = _make_compressor()
        msgs = (
            _skill_view_pair("call_s", "steered-skill")
            + self._filler(14)
            + [{"role": "user", "content": "keep following the steered-skill steps"}]
        )
        result, _ = c._prune_old_tool_results(msgs, protect_tail_count=4)
        skill_row = result[1]
        assert skill_row["content"].startswith("# steered-skill instructions")

    def test_pressure_demotion_overrides_skill_protection(self):
        """Pass-4 must still demote protected skill bodies (#61932 guard)."""
        c = _make_compressor()
        msgs = (
            self._filler(2)
            + _skill_view_pair("call_s", "fresh-skill", size=60000)
            + [{"role": "user", "content": "active ask"}]
        )
        # Tiny token budget → protected region exceeds the soft ceiling and
        # the pressure pass must reclaim the skill body despite protection.
        result, pruned = c._prune_old_tool_results(
            msgs, protect_tail_count=4, protect_tail_tokens=100
        )
        skill_row = result[3]
        assert pruned >= 1
        assert _skill_pruned_marker("fresh-skill") in skill_row["content"]


class TestMarkerSurvivesRealCompress:
    """P2 layer: markers survive a real compress() with a mocked aux LLM."""

    def _mock_response(self, text):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = text
        return response

    def _messages_with_pruned_skill_in_middle(self):
        """Transcript whose compressed middle carries a prune marker row."""
        pruned_row_content = (
            "[skill_view] name=pdf (48,201 chars) " + _skill_pruned_marker("pdf")
        )
        msgs = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Build the PDF report"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_pdf",
                    "type": "function",
                    "function": {
                        "name": "skill_view",
                        "arguments": '{"name":"pdf"}',
                    },
                }],
            },
            {"role": "tool", "tool_call_id": "call_pdf", "content": pruned_row_content},
            {"role": "assistant", "content": "Loaded the skill, working."},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "more work " + "z" * 500},
            {"role": "user", "content": "latest ask"},
            {"role": "assistant", "content": "ack"},
        ]
        return msgs

    def _summary_text_of(self, result):
        for msg in result:
            if ContextCompressor.classify_summary_content(msg.get("content")):
                return msg["content"]
        raise AssertionError(f"no summary message found in {result!r}")

    def test_marker_reinjected_when_summarizer_drops_it(self):
        c = _make_compressor(protect_first_n=1, protect_last_n=2)
        msgs = self._messages_with_pruned_skill_in_middle()
        drop_response = self._mock_response(
            "## Goal\nBuild the PDF report.\n\n## Completed Actions\n"
            "1. Loaded some skills and worked on the report."
        )
        with (
            patch.object(c, "_find_tail_cut_by_tokens", return_value=7),
            patch(
                "agent.context_compressor.call_llm", return_value=drop_response
            ) as mock_call,
        ):
            result = c.compress(msgs, force=True)
        assert mock_call.called
        summary_text = self._summary_text_of(result)
        assert _skill_pruned_marker("pdf") in summary_text
        # Stored iterative-update state carries the marker too.
        assert _skill_pruned_marker("pdf") in c._previous_summary

    def test_marker_not_duplicated_when_summarizer_preserves_it(self):
        c = _make_compressor(protect_first_n=1, protect_last_n=2)
        msgs = self._messages_with_pruned_skill_in_middle()
        keep_response = self._mock_response(
            "## Goal\nBuild the PDF report.\n\n## Pruned Skills\n"
            + _skill_pruned_marker("pdf")
        )
        with (
            patch.object(c, "_find_tail_cut_by_tokens", return_value=7),
            patch("agent.context_compressor.call_llm", return_value=keep_response),
        ):
            result = c.compress(msgs, force=True)
        summary_text = self._summary_text_of(result)
        assert summary_text.count(_skill_pruned_marker("pdf")) == 1

    def test_marker_survives_static_fallback_summary(self):
        c = _make_compressor(protect_first_n=1, protect_last_n=2)
        msgs = self._messages_with_pruned_skill_in_middle()
        with (
            patch.object(c, "_find_tail_cut_by_tokens", return_value=7),
            patch(
                "agent.context_compressor.call_llm",
                side_effect=RuntimeError("no provider"),
            ),
        ):
            result = c.compress(msgs, force=True)
        summary_text = self._summary_text_of(result)
        assert _skill_pruned_marker("pdf") in summary_text
        assert c._last_summary_fallback_used is True

    def test_marker_survives_iterative_recompression(self):
        """Markers in a rehydrated handoff summary survive iterative rewrites.

        On re-compression the previous handoff (carrying the marker) is
        rehydrated into ``_previous_summary``; even when the summarizer's
        iterative update drops the marker, re-injection restores it.
        """
        c = _make_compressor(protect_first_n=1, protect_last_n=2)
        prior_handoff = (
            SUMMARY_PREFIX
            + "\n\n## Goal\nOld work.\n\n## Pruned Skills\n"
            + _skill_pruned_marker("pdf")
        )
        msgs = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": prior_handoff},
        ] + [
            {"role": "assistant" if i % 2 == 0 else "user", "content": f"turn {i} " + "q" * 300}
            for i in range(8)
        ]
        drop_response = self._mock_response("## Goal\nNext task in flight.")
        with (
            patch.object(c, "_find_tail_cut_by_tokens", return_value=8),
            patch("agent.context_compressor.call_llm", return_value=drop_response),
        ):
            result = c.compress(msgs, force=True)
        summary_text = self._summary_text_of(result)
        assert _skill_pruned_marker("pdf") in summary_text


class TestReinjectionBoundsAndRedaction:
    def test_marker_list_is_capped(self):
        names = [f"skill-{i}" for i in range(_MAX_PRUNED_SKILL_MARKERS + 15)]
        out = _reinject_pruned_skill_markers("body", names)
        assert out.count(SKILL_PRUNED_MARKER_PREFIX) == len(names)
        # The cap is applied at the collection sites in _generate_summary /
        # _build_static_fallback_summary; the helper itself is mechanical.

    def test_reinjection_block_is_redacted(self, monkeypatch):
        import agent.redact as redact_mod

        # force=True redaction must win even when redaction is disabled.
        monkeypatch.setattr(redact_mod, "_REDACT_ENABLED", False, raising=False)
        secret = "ghp_" + "a1B2" * 6
        out = _reinject_pruned_skill_markers("body", [f"x {secret}"])
        assert secret not in out


class TestSkillsGuidanceSafetyRule:
    def test_safety_rule_present_with_real_newlines(self):
        from agent.prompt_builder import SKILLS_GUIDANCE

        assert "## Skill Safety Rule" in SKILLS_GUIDANCE
        assert "[SKILL_PRUNED]" in SKILLS_GUIDANCE
        assert "skill_view(name='...')" in SKILLS_GUIDANCE
        # The rule list must use REAL newlines — the original PR hunk risked
        # literal backslash-n escape text rendering into the system prompt.
        assert "\\n" not in SKILLS_GUIDANCE
        assert SKILLS_GUIDANCE.count("\n") >= 6
        for rule in ("UNAVAILABLE", "RELOAD", "WAIT", "DEDUP"):
            assert rule in SKILLS_GUIDANCE
