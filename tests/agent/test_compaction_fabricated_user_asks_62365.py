"""Regression tests for #62365: context compaction must not fabricate user
requests that never happened in the real transcript.

Root cause: the compaction template pushes the LLM into writing
``User asked: '<verbatim quote>'``. When the actual conversation has no
outstanding user request (or the model can't locate one), it fabricates a
quote to fit the template — the next-turn agent then acts on a request
that was never made.

Fix: post-validation helper ``_strip_fabricated_user_asks`` scans every
``User asked: '<quote>'`` line in the LLM's summary and checks whether
the quoted substring appears in the source turns. Unverifiable quotes
get rewritten to a safe ``None`` fallback so the agent sees "no
outstanding ask" instead of a fabricated one.

Tests below pin:
  1. Verifiable quotes are kept verbatim.
  2. Fabricated quotes (text not in source) get replaced with the fallback.
  3. Whitespace and quote-mark drift is tolerated (real user typos).
  4. Non-quote ``User asked:`` prose (no quoted phrase) is not touched.
  5. Empty source turns or empty summary short-circuit cleanly.
  6. Multiple fabricated lines in one summary are all stripped.
  7. Quoted phrases from tool-call args are detected (full source corpus).
  8. Trivial quotes (< 4 chars) are kept as-is to avoid mangling "ok".
"""

from __future__ import annotations

from agent.context_compressor import (
    _strip_fabricated_user_asks,
    _USER_ASKED_QUOTE_RE,
)


# -----------------------------------------------------------------------
# Direct unit tests for the post-validation helper.
# -----------------------------------------------------------------------


class TestStripFabricatedUserAsks:
    """Pin every clause of the #62365 fix at the helper boundary."""

    def test_verifiable_quote_is_kept_verbatim(self):
        summary = (
            "## Historical Task Snapshot\n"
            "User asked: \"please refactor the auth module\"\n\n"
            "## Goal\nMigrate to JWT."
        )
        source = [
            {"role": "user", "content": "please refactor the auth module"},
            {"role": "assistant", "content": "Working on it."},
        ]
        out = _strip_fabricated_user_asks(summary, source)
        assert 'User asked: "please refactor the auth module"' in out, (
            "Verifiable quotes must be preserved verbatim — the helper "
            "should only strip fabrications, not legitimate asks."
        )

    def test_fabricated_quote_replaced_with_safe_fallback(self):
        """The headline repro from #62365: LLM fabricates a user request
        that never appeared in the actual transcript."""
        summary = (
            "## Historical Task Snapshot\n"
            "User asked: \"пусть подтянет анализ кошельков\"\n\n"
            "## Goal\nWallet analysis."
        )
        # The actual transcript had nothing about wallets — the prior turn
        # was unrelated cleanup work.
        source = [
            {"role": "user", "content": "Показывай, что не так."},
            {"role": "assistant", "content": "Текущее состояние такое..."},
        ]
        out = _strip_fabricated_user_asks(summary, source)
        assert "пусть подтянет" not in out, (
            "Fabricated quote leaked into the rewrite — the post-validation "
            "must catch it"
        )
        assert "no verifiable outstanding user request" in out, (
            "Safe fallback must replace fabricated quotes so the agent "
            "doesn't act on them"
        )

    def test_whitespace_and_quote_drift_tolerated(self):
        """Real user messages are typed with whitespace drift. The verifier
        normalizes whitespace and accepts straight/curly quote variants."""
        summary = 'User asked: "refactor  the   auth   module"'
        source = [{"role": "user", "content": "refactor the auth module"}]
        out = _strip_fabricated_user_asks(summary, source)
        assert 'refactor  the   auth   module' in out, (
            "Real user message with whitespace drift should still match — "
            "otherwise every minor reformat would trip the guard"
        )

    def test_case_insensitive(self):
        """Quotes are matched case-insensitively — a user message with
        capitalization differences should still verify."""
        summary = 'User asked: "Please Refactor The Auth Module"'
        source = [{"role": "user", "content": "please refactor the auth module"}]
        out = _strip_fabricated_user_asks(summary, source)
        assert "Please Refactor" in out

    def test_non_quote_user_asked_prose_not_touched(self):
        """If the LLM writes ``User asked: <no quoted phrase>`` (e.g. prose
        summary, no quotes), the helper must leave it alone — the regex
        requires a quoted phrase."""
        summary = (
            "## Historical Task Snapshot\n"
            "User asked: for clarification on the auth flow\n\n"
        )
        source = [{"role": "user", "content": "different content entirely"}]
        out = _strip_fabricated_user_asks(summary, source)
        # Prose line is preserved — the helper is regex-scoped to quoted phrases
        assert "User asked: for clarification" in out

    def test_empty_source_returns_summary_unchanged(self):
        """No source turns → we have nothing to verify against. Pass through
        rather than strip everything (would mask legitimate None cases)."""
        summary = 'User asked: "anything at all"'
        out = _strip_fabricated_user_asks(summary, [])
        assert 'User asked: "anything at all"' in out

    def test_empty_summary_returns_empty(self):
        assert _strip_fabricated_user_asks("", [{"role": "user", "content": "x"}]) == ""

    def test_multiple_fabricated_lines_all_stripped(self):
        """A summary with several fabricated ``User asked:`` lines must
        have all of them rewritten — partial rewriting leaves landmines."""
        summary = (
            "User asked: \"do thing A\"\n"
            "User asked: \"do thing B\"\n"
            "User asked: \"do thing C\"\n"
        )
        source = [{"role": "user", "content": "no task was requested"}]
        out = _strip_fabricated_user_asks(summary, source)
        assert "do thing A" not in out
        assert "do thing B" not in out
        assert "do thing C" not in out
        # Fallback count: 3
        assert out.count("no verifiable outstanding user request") == 3

    def test_mixed_verifiable_and_fabricated(self):
        """Real ask preserved, fake ask stripped, in the same summary."""
        summary = (
            "User asked: \"refactor auth module\"\n"
            "User asked: \"send the analytics report to investors\"\n"
        )
        source = [
            {"role": "user", "content": "refactor auth module please"},
            {"role": "assistant", "content": "On it."},
        ]
        out = _strip_fabricated_user_asks(summary, source)
        assert 'User asked: "refactor auth module"' in out
        assert "send the analytics report" not in out
        assert "no verifiable outstanding user request" in out

    def test_quote_from_tool_call_args_detected(self):
        """The source-side corpus must include tool-call args. Otherwise a
        verbatim quote the user wrote inside a tool_call.arguments dict
        (e.g. browser_submit with text=...) would be treated as fabricated
        even though it really happened."""
        summary = 'User asked: "click the blue button in the top-right"'
        source = [
            {
                "role": "assistant",
                "content": "I'll click that button for you.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "browser_click",
                            "arguments": 'click the blue button in the top-right',
                        }
                    }
                ],
            },
        ]
        out = _strip_fabricated_user_asks(summary, source)
        assert "click the blue button in the top-right" in out, (
            "Tool-call argument quotes must be locatable in the source corpus"
        )

    def test_trivial_short_quote_kept_as_is(self):
        """Quotes shorter than 4 chars are too brittle to verify reliably.
        Keep as-is to avoid mangling legitimate trivial asks like 'ok',
        'go', 'yes'."""
        summary = 'User asked: "ok"'
        source = [{"role": "user", "content": "something else"}]
        out = _strip_fabricated_user_asks(summary, source)
        assert 'User asked: "ok"' in out

    def test_curly_quotes_detected(self):
        """The regex accepts curly-quote variants since LLMs sometimes
        substitute them for straight quotes."""
        summary = "User asked: \u201cmigrate the database\u201d"
        source = [{"role": "user", "content": "migrate the database"}]
        out = _strip_fabricated_user_asks(summary, source)
        assert "migrate the database" in out, (
            "Curly-quote variants must be recognized so legitimate asks "
            "with smart-quote substitution pass verification"
        )


class TestRegexPattern:
    """Lock the regex shape so a future refactor can't silently widen or
    narrow what counts as a quoted ``User asked:`` phrase."""

    def test_matches_double_quoted(self):
        m = _USER_ASKED_QUOTE_RE.search('User asked: "refactor auth"')
        assert m and m.group(1) == "refactor auth"

    def test_matches_single_quoted(self):
        m = _USER_ASKED_QUOTE_RE.search("User asked: 'refactor auth'")
        assert m and m.group(1) == "refactor auth"

    def test_matches_curly_quoted(self):
        m = _USER_ASKED_QUOTE_RE.search(
            "User asked: \u201crefactor auth\u201d"
        )
        assert m and m.group(1) == "refactor auth"

    def test_matches_case_insensitive(self):
        m = _USER_ASKED_QUOTE_RE.search('USER ASKED: "refactor auth"')
        assert m and m.group(1) == "refactor auth"

    def test_no_match_without_quotes(self):
        """Prose ``User asked: blah`` must NOT match — the regex requires
        an actual quote pair, otherwise we'd rewrite legitimate prose
        summaries."""
        assert _USER_ASKED_QUOTE_RE.search(
            "User asked: for clarification"
        ) is None
