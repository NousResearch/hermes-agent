"""Regression tests: manual /compress feedback for a would-grow refusal.

The anti-growth guard (#83339 / PR #86700) can refuse the commit AFTER the
compressor produced a candidate. The CLI's feedback previously compared the
returned list against its pre-call snapshot; durable-snapshot adoption can
legitimately change that list, so a REFUSED compression printed
"✅ Compressed: 8 → 14 messages" directly under the refusal warning
(Aug 2026 full-surface CLI QA sweep). The refusal must surface as its own
honest headline.
"""

from types import SimpleNamespace

from agent.manual_compression_feedback import summarize_manual_compression


def _msgs(n):
    return [{"role": "user", "content": f"m{n_i}"} for n_i in range(n)]


def _state(**flags):
    base = {
        "_last_compress_aborted": False,
        "_last_compress_refused_would_grow": False,
        "_last_summary_fallback_used": False,
        "_last_summary_error": None,
        "_last_summary_dropped_count": 0,
    }
    base.update(flags)
    return SimpleNamespace(**base)


class TestWouldGrowRefusalFeedback:
    def test_refusal_reports_preserved_not_compressed(self):
        summary = summarize_manual_compression(
            _msgs(8), _msgs(14), 22025, 22453,
            compression_state=_state(_last_compress_refused_would_grow=True),
        )
        assert summary["refused_would_grow"] is True
        assert "refused" in summary["headline"].lower()
        assert "→" not in summary["headline"]  # never claims a message rewrite
        assert "unchanged" in summary["token_line"]
        assert "no messages were removed" in summary["note"]

    def test_normal_compression_unchanged(self):
        summary = summarize_manual_compression(
            _msgs(10), _msgs(4), 30000, 12000, compression_state=_state(),
        )
        assert summary["refused_would_grow"] is False
        assert summary["headline"].startswith("Compressed: 10 → 4")

    def test_abort_still_wins_its_own_headline(self):
        summary = summarize_manual_compression(
            _msgs(6), _msgs(6), 9000, 9000,
            compression_state=_state(_last_compress_aborted=True),
        )
        assert summary["aborted"] is True
        assert summary["headline"].startswith("Compression aborted")
