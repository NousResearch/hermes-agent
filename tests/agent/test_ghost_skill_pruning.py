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

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    SKILL_PRUNED_MARKER_PREFIX,
    SUMMARY_PREFIX,
    ContextCompressor,
    _collect_ghosted_skill_names,
    _extract_pruned_skill_names,
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

    def test_every_loaded_skill_view_is_marked_failed_views_are_not(self):
        """Small skills lose their body to the stub too, so they get the marker; a failed view loaded nothing."""
        summary = _summarize_tool_result(
            "skill_view", '{"name":"docker-management"}', "x" * 1234
        )
        assert summary == "[skill_view] name=docker-management (1,234 chars) " + _skill_pruned_marker("docker-management")
        failed = json.dumps({"success": False, "error": "Skill 'nope' not found. " + "Available: a, b. " * 30})
        failed_stub = _summarize_tool_result("skill_view", '{"name":"nope"}', failed)
        assert SKILL_PRUNED_MARKER_PREFIX not in failed_stub
        # Neither the raw failure nor its stub is listed for reload later, by the collector or the fallback summary.
        for content in (failed, failed_stub):
            turns = _skill_view_pair("call_nope", "nope")
            turns[1]["content"] = content
            assert _collect_ghosted_skill_names(turns) == []
            assert "skill_view(name='nope')" not in _make_compressor()._build_static_fallback_summary(turns)
        # Parallel loads on one assistant row are each listed under their own name.
        pair_one, pair_two = _skill_view_pair("call_1", "one", 300), _skill_view_pair("call_2", "two", 300)
        row = {**pair_one[0], "tool_calls": pair_one[0]["tool_calls"] + pair_two[0]["tool_calls"]}
        assert _collect_ghosted_skill_names([row, pair_one[1], pair_two[1]]) == ["one", "two"]

    def test_marker_extractor_round_trips_the_emitted_marker(self):
        """Emit and check sides share one canonical string.

        The original PR #44166 emitted ``[SKILL_PRUNED:`` but presence-
        checked ``[SKILL_PRUNED]`` — re-injection fired even when the
        marker had survived. Pin the round trip.
        """
        summary = _summarize_tool_result("skill_view", '{"name":"pdf"}', "x" * 6000)
        assert _extract_pruned_skill_names(summary) == ["pdf"]
        assert _skill_pruned_marker("pdf") in summary

    def test_marker_extractor_ignores_template_placeholders(self):
        """Doc/code snippets carrying '…', '...', or '{skill_name}' must not be extracted."""
        doc_snippet = (
            "[SKILL_PRUNED: content lost in compression; reload with skill_view(name='…')]\n"
            "[SKILL_PRUNED: content lost in compression; reload with skill_view(name='...')]\n"
            "[SKILL_PRUNED: content lost in compression; reload with skill_view(name='{skill_name}')]\n"
            + _skill_pruned_marker("productivity/diligence-pipeline")
        )
        assert _extract_pruned_skill_names(doc_snippet) == ["productivity/diligence-pipeline"]

class TestReinjectPrunedSkillMarkers:

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

    def test_reinjected_summary_still_classifies_standalone(self):
        body = _reinject_pruned_skill_markers("## Goal\nwork\n", ["pdf"])
        full = SUMMARY_PREFIX + "\n\n" + body
        assert ContextCompressor.classify_summary_content(full) == "standalone"

    def test_relocates_stray_markers_from_wrong_structured_section(self):
        stray_summary = (
            "## Goal\nInvestigate logs\n\n"
            "## Key Decisions\n- Fixed parser\n"
            + _skill_pruned_marker("rca")
            + "\n"
            + _skill_pruned_marker("always-verify")
            + "\n\n## Errors & Fixes\n- None"
        )
        out = _reinject_pruned_skill_markers(stray_summary, ["rca", "always-verify", "hermes-log-debug"])
        assert "## Pruned Skills" in out
        key_decisions_sec = out.split("## Key Decisions")[1].split("## Errors & Fixes")[0]
        assert SKILL_PRUNED_MARKER_PREFIX not in key_decisions_sec
        pruned_sec = out.split("## Pruned Skills")[1]
        assert _skill_pruned_marker("rca") in pruned_sec
        assert _skill_pruned_marker("always-verify") in pruned_sec
        assert _skill_pruned_marker("hermes-log-debug") in pruned_sec

class TestProtectedSkillPrune:
    """Phase-1 prune must not demote a just-loaded skill (#32106)."""

    def _filler(self, n, start=0):
        out = []
        for i in range(n):
            role = "user" if (start + i) % 2 == 0 else "assistant"
            out.append({"role": role, "content": f"filler {start + i} " + "y" * 400})
        return out

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

    def test_active_user_turn_skill_survives_multi_round_tool_batches_until_next_turn(self):
        """Both polarities: a skill loaded in the active user turn survives >10 tool rows,
        then is demoted once a subsequent user turn + >10 tool rows push it out of scope."""
        c = _make_compressor()
        tool_rounds_active = []
        for i in range(7):
            tool_rounds_active.extend([
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": f"c_{i}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": f"c_{i}", "content": f"round {i} " + "t" * 400},
            ])
        msgs_active = (
            self._filler(2)
            + [{"role": "user", "content": "Execute the readiness audit"}]
            + _skill_view_pair("call_active", "agent-readiness-audit", size=5000)
            + tool_rounds_active
        )
        # Positive polarity: 14 assistant/tool rows (> _SKILL_PRUNE_RECENT_WINDOW=10) after skill_view
        # within the SAME active user turn -> skill body stays intact.
        res_active, _ = c._prune_old_tool_results(msgs_active, protect_tail_count=4)
        assert res_active[4]["content"].startswith("# agent-readiness-audit instructions")
        assert SKILL_PRUNED_MARKER_PREFIX not in res_active[4]["content"]

        # Negative polarity: a new user turn arrives and runs >10 messages without naming the skill ->
        # the old skill from the prior turn is now demoted to [SKILL_PRUNED].
        msgs_next_turn = (
            msgs_active
            + [{"role": "user", "content": "Unrelated follow-up question"}]
            + tool_rounds_active
        )
        res_next, _ = c._prune_old_tool_results(msgs_next_turn, protect_tail_count=4)
        assert _skill_pruned_marker("agent-readiness-audit") in res_next[4]["content"]

    def test_synthetic_mid_turn_user_row_does_not_evict_active_turn_skill(self):
        """Mid-turn runtime injections ([Your active task list...], [IMPORTANT: Context...]) have role=='user'
        but must not advance last_user past the real human turn that loaded the skill."""
        c = _make_compressor()
        tool_rounds = []
        for i in range(7):
            tool_rounds.extend([
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": f"m_{i}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": f"m_{i}", "content": f"round {i} " + "t" * 400},
            ])
        msgs = (
            self._filler(2)
            + [{"role": "user", "content": "Fix everything after sceptic pass"}]
            + _skill_view_pair("call_ahl", "adversarial-hardening-loop", size=5000)
            + [{"role": "user", "content": "[Your active task list was preserved across context compression]\n- [>] t6"}]
            + tool_rounds
        )
        res, _ = c._prune_old_tool_results(msgs, protect_tail_count=4)
        assert res[4]["content"].startswith("# adversarial-hardening-loop instructions")
        assert SKILL_PRUNED_MARKER_PREFIX not in res[4]["content"]

    def test_pressure_demotion_shrinks_non_protected_tools_before_protected_skill(self):
        """Pass-4 sub-pass 4a shrinks ordinary read/search tool outputs first so active skills survive
        when non-protected tool outputs account for the tail pressure."""
        c = _make_compressor()
        msgs = (
            [{"role": "user", "content": "Audit and fix the guards"}]
            + _skill_view_pair("call_skill", "always-verify", size=4000)
            + [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call_read", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"big.py"}'}}],
                },
                {"role": "tool", "tool_call_id": "call_read", "content": "r" * 40000},
                {"role": "assistant", "content": "done reading"},
                {"role": "user", "content": "[IMPORTANT: Context is at 68%]"},
                {"role": "assistant", "content": "continuing"},
                {"role": "user", "content": "[Your active task list was preserved across context compression]"},
                {"role": "assistant", "content": "wrapping up"},
            ]
        )
        result, pruned = c._prune_old_tool_results(
            msgs, protect_tail_count=10, protect_tail_tokens=3000
        )
        assert pruned >= 1
        assert result[2]["content"].startswith("# always-verify instructions")
        assert SKILL_PRUNED_MARKER_PREFIX not in result[2]["content"]
        assert result[4]["content"].startswith("[read_file]")

    def test_pinned_outbound_skill_survives_ordinary_prune_across_turns(self):
        """Pinned outbound skills (outbound-comms, humanizer, google-chat-comms, svensk-avtalsgranskning)
        stay protected across ordinary prune passes so outbound gates never lose their instructions."""
        c = _make_compressor()
        msgs = (
            [{"role": "user", "content": "Draft reply to vendor"}]
            + _skill_view_pair("call_out", "productivity/outbound-comms", size=5000)
            + [{"role": "user", "content": "Now check the database rows first"}]
            + self._filler(14, start=2)
        )
        res, _ = c._prune_old_tool_results(msgs, protect_tail_count=4)
        assert res[2]["content"].startswith("# productivity/outbound-comms instructions")
        assert SKILL_PRUNED_MARKER_PREFIX not in res[2]["content"]

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
            patch.object(c, "_find_tail_cut_by_tokens", return_value=9),
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

    def test_slash_invoked_skill_is_marked_and_quoted_as_typed(self):
        """A /skill turn carries its body inline: compaction lists the skill for reload and quotes only what
        the user typed, never the skill body as "the user's actual words"."""
        from agent.skill_commands import _build_skill_message
        note = ('[IMPORTANT: The user has invoked the "probe-inv" skill, indicating they want '
                "you to follow its instructions. The full skill content is loaded below.]")
        body = "RULE: include INV-HERON.\n" + "skill body prose. " * 600
        invocation = _build_skill_message({"name": "probe-inv", "content": body}, Path("/skills/probe-inv"), note,
                                          user_instruction="remember PLUM-77")
        # A gateway auto-load turn whose skill lists supporting files: the request follows those blocks.
        auto = _build_skill_message(
            {"name": "gw-skill", "content": "gateway skill body", "linked_files": {"references": ["references/gw.md"]},
             "setup_needed": True, "setup_note": "First setup step.\nSecond setup step."},
            Path("/skills/gw-skill"),
            '[IMPORTANT: The "gw-skill" skill is auto-loaded. Follow its instructions for this session.]',
        ) + "\n\nhello from the gateway"
        c = _make_compressor(protect_first_n=1, protect_last_n=2, tail_mode="lean")
        msgs = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": auto},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": invocation},
            {"role": "assistant", "content": "INV-HERON noted. " + "w" * 400},
            {"role": "user", "content": "continue " + "c" * 400},
            {"role": "assistant", "content": "more work " + "z" * 500},
            {"role": "user", "content": "latest ask"},
            {"role": "assistant", "content": "ack"},
        ]
        with (
            patch.object(c, "_find_tail_cut_by_tokens", return_value=9),
            patch("agent.context_compressor.call_llm", return_value=self._mock_response("## Goal\nWork.")),
        ):
            result = c.compress(msgs, force=True)
        summary_text = self._summary_text_of(result)
        assert _skill_pruned_marker("probe-inv") in summary_text
        assert "/probe-inv remember PLUM-77" in summary_text
        assert "skill body prose" not in summary_text
        assert "scripts/foo.js" not in summary_text  # the scaffold footer's example path is not session work
        assert _skill_pruned_marker("gw-skill") in summary_text
        assert "> hello from the gateway" in summary_text and "references/gw.md" not in summary_text


class TestReinjectionBoundsAndRedaction:
        # The cap is applied at the collection sites in _generate_summary /
        # _build_static_fallback_summary; the helper itself is mechanical.

    def test_reinjection_block_is_redacted(self, monkeypatch):
        import agent.redact as redact_mod

        # force=True redaction must win even when redaction is disabled.
        monkeypatch.setattr(redact_mod, "_REDACT_ENABLED", False, raising=False)
        secret = "ghp_" + "a1B2" * 6
        out = _reinject_pruned_skill_markers("body", [f"x {secret}"])
        assert secret not in out

    def test_reinjection_merges_into_existing_pruned_skills_section(self):
        existing = (
            "## Goal\nShip fix.\n\n"
            "## Pruned Skills\n"
            f"{_skill_pruned_marker('rca')}\n\n"
            "## Anchor Index (mechanically extracted, exact)\n"
            "files: foo.py"
        )
        out = _reinject_pruned_skill_markers(existing, ["rca", "always-verify", "hermes-log-debug"])
        assert out.count("## Pruned Skills") == 1
        assert _skill_pruned_marker("rca") in out
        assert _skill_pruned_marker("always-verify") in out
        assert _skill_pruned_marker("hermes-log-debug") in out
        assert out.index(_skill_pruned_marker("hermes-log-debug")) < out.index("## Anchor Index")

    def test_reloaded_skill_in_tail_strips_stale_pruned_marker_from_summary(self):
        from agent.context_compressor_tail import with_skill_safety_rule

        summary = (
            "## Goal\nShip fix.\n\n"
            "## Pruned Skills\n"
            f"{_skill_pruned_marker('pdf')}\n"
            f"{_skill_pruned_marker('docx')}\n\n"
            "## Anchor Index (mechanically extracted, exact)\n"
            "files: foo.py"
        )
        tail = _skill_view_pair("call_reloaded_pdf", "pdf", size=4000)
        updated_summary, _, _ = with_skill_safety_rule(summary, [], tail)
        assert _skill_pruned_marker("pdf") not in updated_summary
        assert _skill_pruned_marker("docx") in updated_summary

        # When all pruned skills are satisfied by the active tail, ## Pruned Skills is dropped cleanly.
        tail_both = tail + _skill_view_pair("call_reloaded_docx", "docx", size=4000)
        fully_cleared, _, _ = with_skill_safety_rule(summary, [], tail_both)
        assert _skill_pruned_marker("pdf") not in fully_cleared
        assert _skill_pruned_marker("docx") not in fully_cleared
        assert "## Pruned Skills" not in fully_cleared

    def test_collect_ghosted_skill_names_ignores_non_skill_tools_and_assistant_prose(self):
        turns = [
            {
                "role": "assistant",
                "content": f"Example marker in prose: {_skill_pruned_marker('fake-prose-skill')}",
                "tool_calls": [
                    {
                        "id": "call_read_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "test.py"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_read_1",
                "content": f"10| {_skill_pruned_marker('fake-file-skill')}",
            },
            *_skill_view_pair("call_real_skill", "real-loaded-skill", size=3000),
        ]
        assert _collect_ghosted_skill_names(turns) == ["real-loaded-skill"]


