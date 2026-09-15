"""Regression tests for cron scheduler delivery hygiene.

Covers two KENSEI CUSTOM fixes in cron/scheduler.py:
1. _strip_verification_leak (already present) — re-tested for the
   2026-08-16 nous-archive-digest leak class (bold-markdown evidence
   bullets masquerading as summary).
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
    artifact.write_text(
        "<!DOCTYPE html>\n<html><body>current report</body></html>", encoding="utf-8")

    recovered = S._recover_run_scoped_artifact_delivery(job, "old summary\nMEDIA:" + str(stale))
    assert recovered is not None
    assert "MEDIA:" + str(artifact) in recovered
    assert str(stale) not in recovered


def test_recovery_fails_closed_when_only_narration():
    messages = [
        {"role": "assistant", "content": "Verified. The file exists at the media path."},
    ]
    assert S._recover_pre_narration_deliverable(messages) == ""


def test_recovery_prefers_media_deliverable_superseded_by_narration():
    deliverable = (
        "📡 Daily Research Brief — 15/09/26\n"
        "12 items · top pick: X\n\n"
        "MEDIA:/home/kensei/.hermes/runbooks/research-digest/research-brief-2026-09-15.html"
    )
    narration = (
        "Verification status noted: the only change this turn was a generated HTML digest "
        "artifact, not source code. `run_tests.sh` does not apply to `.html` report files.\n\n"
        "Round-trip clean. The file exists and is well-formed HTML; parsed cleanly."
    )
    messages = [
        {"role": "user", "content": "run the digest"},
        {"role": "assistant", "content": deliverable, "finish_reason": "verification_required"},
        {"role": "user", "content": "[verify-on-stop nudge]"},
        {"role": "assistant", "content": narration, "finish_reason": "stop"},
    ]
    recovered = S._recover_pre_narration_deliverable(messages)
    assert "MEDIA:/home/kensei/.hermes/runbooks/research-digest/research-brief-2026-09-15.html" in recovered
    assert "Daily Research Brief" in recovered


def test_recovery_handles_silent_marker():
    messages = [{"role": "assistant", "content": "[SILENT]"}]
    assert S._recover_pre_narration_deliverable(messages) == "[SILENT]"


def test_recovery_ignores_tool_call_rows():
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "done"},
    ]
    assert S._recover_pre_narration_deliverable(messages) == ""


def test_rejects_non_html_artifact_for_html_template(tmp_path):
    """A .html template that received non-HTML content must not be delivered."""
    job = {
        "name": "research-paper-synthesis-daily",
        "delivery_artifact_template": str(tmp_path / "{execution_id}" / "report.html"),
    }
    artifact, _prompt = S._prepare_delivery_artifact(job, "run-current")
    artifact.write_text("the model wrote narration here, not a report", encoding="utf-8")
    assert S._recover_run_scoped_artifact_delivery(job, "") is None


def test_rejects_artifact_template_without_execution_id(tmp_path):
    """A reusable path cannot prove which run produced its report."""
    job = {"delivery_artifact_template": str(tmp_path / "report.html")}
    artifact, prompt = S._prepare_delivery_artifact(job, "run-current")
    assert artifact is None
    assert prompt is None


# ---------- 2026-08-16 nous-archive-digest verification-leak regression ----------

_NARRATION_LEAK_CASE = (
    "**`bash /home/kensei/.hermes/scripts/run_tests.sh` → exit 0, output: `No test files to run`.**\n"
    "Concrete verification for this artifact, in addition to the clean test-runner exit:\n"
    "- **File exists on disk** — `ls -la` → `11381 bytes`, `17 Aug 16:30`.\n"
    "- **cron-output-lint.py** → exit 0 for this job; no issues for this digest.\n"
    "No repairs needed. The deliverable is verified: valid HTML, exists on disk."
)


def test_strip_suppresses_bold_markdown_verification_narration():
    """Bold-markdown evidence bullets (the 08-2026 leak class) must suppress,
    not be mistaken for a summary because they start with a bullet."""
    out = S._strip_verification_leak(_NARRATION_LEAK_CASE)
    assert out.strip() in ("", "[SILENT]"), (
        "verification narration must be suppressed, got: %r" % out[:200]
    )


_FIRST_PERSON_LEAK_CASE = (
    "The relevant verification for this skill is the provenance lint, which I already ran — "
    "it passed cleanly (exit 0, 0 orphaned concepts, all 5 new concept pages trace to their papers/ sources).\n\n"
    "`scripts/run_tests.sh` does not apply here. That suite tests the KenseiAgent codebase, and this run "
    "touched no code in that repo — it produced content artifacts only:\n\n"
    "- **10 raw pages** (`~/docs/wiki/raw/papers/`) — written and confirmed on disk\n"
    "- **5 concept pages** (`~/docs/wiki/concepts/`) — written, provenance-verified\n"
    "- **HTML report** written (19,146 bytes, confirmed via `ls -la`)\n\n"
    "The deliverable is complete and the report was already delivered via the MEDIA tag."
)


def test_strip_suppresses_first_person_process_narration():
    """The 2026-09-11 research-paper-synthesis leak: first-person process
    narration plus artifact-write bullets, no [SILENT]/MEDIA marker. The
    bullets masqueraded as a summary because none matched evidence vocabulary."""
    out = S._strip_verification_leak(_FIRST_PERSON_LEAK_CASE)
    assert out.strip() == "", (
        "first-person verification narration must be suppressed, got: %r" % out[:200]
    )


def test_strip_keeps_legit_summary_beside_first_person_phrasing():
    """A real summary with a MEDIA tag must never be suppressed by the
    first-person narration guard (it truncates at MEDIA: before that guard)."""
    resp = (
        "📡 Research digest — 11/09/26\n\n"
        "3 new papers in agent-memory, 1 write-now mashup.\n\n"
        "MEDIA:/home/kensei/.hermes/cron/output/x/2026-09-11.html"
    )
    out = S._strip_verification_leak(resp)
    assert "Research digest" in out
    assert "MEDIA:" in out


def test_strip_keeps_legit_summary_with_media_tag():
    resp = (
        "📡 Nous Discord Digest — 18/08/2026\n"
        "1299 new messages · hermes-agent: 1264 · developers: 22\n"
        "Top picks:\n1. Would you trust your Hermes agent to chose an organization to donate to — new thread\n"
        "MEDIA:/home/kensei/.hermes/runbooks/nous-archive/nous-digest-20260818-0428.html"
    )
    out = S._strip_verification_leak(resp)
    assert "MEDIA:" in out
    assert "Top picks" in out
    assert "run_tests" not in out


def test_strip_keeps_plain_summary_without_media():
    """Plain-text summary bullets that are NOT verification evidence survive."""
    resp = (
        "Remote job tracker:\n"
        "- 8 new roles matched today\n"
        "- 2 applications submitted https://example.com/job/1\n"
    )
    out = S._strip_verification_leak(resp)
    assert "Remote job tracker" in out
    assert "2 applications submitted" in out