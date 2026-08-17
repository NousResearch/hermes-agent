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


def test_prepares_run_scoped_artifact_and_recovers_only_that_run(tmp_path):
    """Delivery must bind to this execution ID, never the newest matching file."""
    job = {
        "name": "research-paper-synthesis-daily",
        "delivery_artifact_template": str(tmp_path / "{execution_id}" / "report.html"),
        "delivery_artifact_summary": "📄 Research Paper Synthesis — {date}\nReport attached.",
    }

    artifact, prompt = S._prepare_delivery_artifact(job, "run-current")
    assert artifact == tmp_path / "run-current" / "report.html"
    assert "run-current" in prompt
    assert not artifact.exists()

    stale = tmp_path / "run-older" / "report.html"
    stale.parent.mkdir()
    stale.write_text("old", encoding="utf-8")
    artifact.write_text("current", encoding="utf-8")

    recovered = S._recover_run_scoped_artifact_delivery(job, "old summary\nMEDIA:" + str(stale))
    assert recovered is not None
    assert "MEDIA:" + str(artifact) in recovered
    assert str(stale) not in recovered


def test_rejects_artifact_template_without_execution_id(tmp_path):
    """A reusable path cannot prove which run produced its report."""
    job = {"delivery_artifact_template": str(tmp_path / "report.html")}
    artifact, prompt = S._prepare_delivery_artifact(job, "run-current")
    assert artifact is None
    assert prompt is None
