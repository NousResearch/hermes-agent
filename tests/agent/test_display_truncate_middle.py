"""Behavioural tests for ``truncate_middle`` in ``agent/display.py``.

Each test is a small scenario that reads top-to-bottom like a worked
example: the previous preview, the new preview, the budget, and the
expected output.  Together they document what the function does and why
the ``...`` lands where it does.

``truncate_middle`` is *presentation only*: the gateway deduplicates
consecutive progress bubbles on RAW tool identity (see
``gateway/run.py``), so truncation never needs to preserve equality.
"""

import pytest

from agent.display import truncate_middle


# ── Use the whole budget when you can ───────────────────────────────────

class TestPassThroughWhenItFits:
    """Truncation is not a goal in itself.  When the new preview fits
    in the budget, return it unchanged — even if it shares a long
    prefix with the previous preview."""

    def test_short_text_with_no_history_passes_through(self):
        out = truncate_middle("git status", max_len=40)
        assert out == "git status"

    def test_short_text_with_prev_still_passes_through(self):
        # We have a prev, but text still fits — show it in full.
        out = truncate_middle("git diff", max_len=40, prev="git status")
        assert out == "git diff"

    def test_long_but_under_budget_passes_through(self):
        # 39-char curr in a 40-char budget — even though the shared
        # prefix with prev is 29 chars, the reader gets to see the whole
        # thing.
        prev = "cd /opt/myproject/sub/dir && git status"
        curr = "cd /opt/myproject/sub/dir && cat README"
        out = truncate_middle(curr, max_len=40, prev=prev)
        assert out == "cd /opt/myproject/sub/dir && cat README"


# ── Presentation only (dedup moved to raw identity) ─────────────────────

class TestPresentationOnly:
    """``truncate_middle`` no longer round-trips a cached truncation for
    identical re-calls — the gateway dedups on raw identity, so an
    identical re-call may render via the head-tail fallback.  All the
    function must guarantee is determinism and the length budget."""

    def test_identical_recall_is_deterministic_and_bounded(self):
        t = "cd /opt/app && rake db:migrate VERSION=20251201"
        a = truncate_middle(t, max_len=40, prev=t)
        b = truncate_middle(t, max_len=40, prev=t)
        assert a == b
        assert len(a) <= 40


# ── Diff-aware truncation when text doesn't fit ─────────────────────────

class TestRevealsTheDiff:
    """When ``text`` overflows the budget but a ``prev`` is available,
    show everything from the first differing char to the end (mandatory),
    then pack as much of the shared prefix in front as fits."""

    def test_shows_diff_tail_plus_prefix_beginning(self):
        # Path is the same, only the trailing action changed.
        # 39-char curr in a 20-char budget — diff-aware truncation kicks
        # in, showing the action in full and as much of the leading path
        # as fits.
        prev = "cd /opt/myproject/sub/dir && git status"
        curr = "cd /opt/myproject/sub/dir && cat README"

        out = truncate_middle(curr, max_len=20, prev=prev)

        # 10 chars "cat README" + 3 chars "..." + 7 chars beginning prefix = 20
        assert out == "cd /opt...cat README"
        assert len(out) == 20

    def test_path_grew_by_one_segment(self):
        # Tight budget forces truncation; the new "/Y" segment is in
        # the mandatory tail, the beginning of the path provides context.
        prev = "cd /home/user/path/X && ls"
        curr = "cd /home/user/path/X/Y && ls"

        out = truncate_middle(curr, max_len=20, prev=prev)

        # 8 chars "/Y && ls" + 3 chars "..." + 9 chars "cd /home/" = 20
        assert out == "cd /home/.../Y && ls"
        assert len(out) == 20

    def test_action_after_long_shared_path(self):
        # Path is identical, action is the diff; budget is just barely
        # bigger than the diff itself.
        prev = "cd /opt/myproject/sub/dir && git status --short"
        curr = "cd /opt/myproject/sub/dir && cat README"

        out = truncate_middle(curr, max_len=13, prev=prev)

        # The 10-char "cat README" plus "..." takes 13 → no room for any
        # prefix content, just the elision marker.
        assert out == "...cat README"


# ── Fallback when even the diff doesn't fit ─────────────────────────────

class TestFallbackWhenDiffTooBig:
    """If the mandatory "diff onward" tail itself exceeds ``max_len``,
    head-tail-truncate the whole string instead."""

    def test_huge_paste_after_tiny_shared_prefix(self):
        prev = "ab"
        curr = "ab" + ("x" * 100)
        out = truncate_middle(curr, max_len=20, prev=prev)
        assert len(out) == 20
        assert "..." in out

    def test_completely_unrelated_strings(self):
        # No common prefix → no diff-aware path possible.
        prev = "alpha"
        curr = "completely-different-very-long-string-here"
        out = truncate_middle(curr, max_len=15, prev=prev)
        assert len(out) == 15
        assert "..." in out


# ── Defensive edge cases ────────────────────────────────────────────────

class TestEdgeCases:

    def test_zero_max_len_returns_empty(self):
        assert truncate_middle("anything", max_len=0) == ""

    def test_negative_max_len_returns_empty(self):
        assert truncate_middle("anything", max_len=-5) == ""

    def test_no_prev_long_text_uses_head_and_tail_fallback(self):
        out = truncate_middle("a" * 80, max_len=20)
        assert len(out) == 20
        assert "..." in out
        assert out.startswith("a") and out.endswith("a")

    def test_unicode_length_counted_in_codepoints(self):
        # Length budget is in chars (Python str codepoints), not bytes.
        text = "日本語" + "x" * 80
        out = truncate_middle(text, max_len=12)
        assert len(out) == 12

    def test_prefix_too_small_to_replace_with_ellipsis(self):
        # When prefix is only 1-2 chars but full text overflows, we
        # show "..." + tail rather than wasting room on tiny prefix
        # content.  This case exists for parity, not common use.
        prev = "x"
        curr = "x-here-is-a-very-long-tail-that-fills-most-of-the-budget"
        out = truncate_middle(curr, max_len=20, prev=prev)
        # Tail alone is much longer than budget → head-tail fallback.
        assert len(out) == 20


# ── End-to-end against the gateway dedup loop ───────────────────────────

class TestGatewayDedupPipeline:
    """Simulate the producer side of ``gateway/run.py``: dedup on RAW
    identity ``(tool_name, raw preview)`` while ``truncate_middle`` runs
    for display only.  The counts hold regardless of how the previews
    truncate — that is the whole point of keying on raw identity."""

    @staticmethod
    def _simulate(previews, max_len=40):
        """Run the gateway's producer loop.  Return
        ``(distinct_msgs, repeat_ticks)``."""
        last_raw_key = None
        last_display_raw = None
        distinct = 0
        repeats = 0
        for preview in previews:
            raw_key = ("terminal", preview)
            # Display only — its output does NOT drive dedup.
            _ = truncate_middle(preview, max_len, prev=last_display_raw)
            if raw_key == last_raw_key:
                repeats += 1
            else:
                distinct += 1
                last_raw_key = raw_key
                last_display_raw = preview
        return distinct, repeats

    def test_five_true_repeats_collapse_into_one_message_plus_four_ticks(self):
        # The model genuinely re-issued the same command five times.
        previews = ["cd /opt/app && git status"] * 5
        assert self._simulate(previews) == (1, 4)

    def test_five_prefix_sharing_commands_each_show_separately(self):
        # All five commands share `cd /opt/app/sub/dir && ` but the
        # action at the end is different every time.  None should
        # collapse into (×N).
        previews = [
            "cd /opt/app/sub/dir && git status",
            "cd /opt/app/sub/dir && git log -1",
            "cd /opt/app/sub/dir && cat README",
            "cd /opt/app/sub/dir && ls plugins",
            "cd /opt/app/sub/dir && wc -l Gemfile",
        ]
        assert self._simulate(previews) == (5, 0)

    def test_two_repeats_then_change_separates_correctly(self):
        previews = [
            "cd /opt/app && git status",
            "cd /opt/app && git status",  # repeat ← tick
            "cd /opt/app && git status",  # repeat ← tick
            "cd /opt/app && cat README",  # different ← new msg
        ]
        assert self._simulate(previews) == (2, 2)

    def test_long_shared_path_under_tight_budget_distinguishes(self):
        # 20-char budget, 39-char commands.  Even if two previews were to
        # truncate alike, raw-identity keying keeps them distinct.
        previews = [
            "cd /opt/myproject/sub/dir && git status",
            "cd /opt/myproject/sub/dir && cat README",
            "cd /opt/myproject/sub/dir && ls plugins",
        ]
        distinct, repeats = self._simulate(previews, max_len=20)
        assert (distinct, repeats) == (3, 0)
