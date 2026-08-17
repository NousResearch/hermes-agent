"""Regression tests for cron scheduler delivery hygiene.

Covers two KENSEI CUSTOM fixes in cron/scheduler.py:
1. _strip_verification_leak (already present) — not re-tested here.
2. NEW: raw HTML block stripping from chat delivery body (keeps MEDIA tag).
"""
import re

import pytest

# Import the scheduler module to reach the inline strip logic via a helper.
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from cron import scheduler as S


def _simulate_raw_html_strip(final_response):
    """Mirror of the inline strip added at scheduler.py ~line 3841.

    Removes full <!DOCTYPE>...</html> blocks AND any orphaned stray tags
    from the chat body, preserving MEDIA: tags so attachments still send.
    """
    _raw_html_re = re.compile(r"<!DOCTYPE[^>]*>.*?</html>", re.DOTALL | re.IGNORECASE)
    if _raw_html_re.search(final_response):
        final_response = _raw_html_re.sub("", final_response).strip()
        final_response = re.sub(r"<[^>]+>", "", final_response).strip()
    return final_response


def test_raw_html_stripped_keeps_media():
    resp = (
        "🔀 Mashup Review · 09/07/26\n\n"
        "5 proposals generated.\n\n"
        "MEDIA:/home/kensei/.hermes/runbooks/proposals/mashup-2026-07-09.html\n"
        "<!DOCTYPE html>\n<html lang=\"en\"><head><title>x</title></head>"
        "<body>full report</body></html>"
    )
    out = _simulate_raw_html_strip(resp)
    assert "<!DOCTYPE" not in out, "raw HTML leaked"
    assert "<html" not in out, "raw HTML leaked"
    assert "MEDIA:/home/kensei/.hermes/runbooks/proposals/mashup-2026-07-09.html" in out, "MEDIA tag lost"
    assert "5 proposals generated" in out, "summary lost"


def test_raw_html_strip_noop_when_clean():
    resp = "Summary line\n\nMEDIA:/tmp/x.html"
    assert _simulate_raw_html_strip(resp) == resp


def test_recovers_recent_configured_artifact_when_model_returns_only_verification(tmp_path):
    """A written report must still be delivered when the model drops its MEDIA line."""
    report = tmp_path / "research-paper-synthesis-20260817.html"
    report.write_text("<html><body>report</body></html>", encoding="utf-8")
    job = {
        "name": "research-paper-synthesis-daily",
        "delivery_artifact_glob": str(tmp_path / "research-paper-synthesis-*.html"),
        "delivery_artifact_summary": "📄 Research Paper Synthesis — {date}\nReport attached.",
    }

    recovered = S._recover_configured_artifact_delivery(job, "")

    assert recovered is not None
    assert "📄 Research Paper Synthesis" in recovered
    assert "MEDIA:" + str(report) in recovered


def test_does_not_recover_stale_configured_artifact(tmp_path):
    """A stale report must not be attached to a later failed/empty run."""
    import os
    import time

    report = tmp_path / "research-paper-synthesis-20260801.html"
    report.write_text("<html><body>old</body></html>", encoding="utf-8")
    stale = time.time() - 901
    os.utime(report, (stale, stale))
    job = {
        "delivery_artifact_glob": str(tmp_path / "research-paper-synthesis-*.html"),
        "delivery_artifact_max_age_seconds": 900,
    }

    assert S._recover_configured_artifact_delivery(job, "") is None
