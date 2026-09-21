"""Tests for the cron delivery-layer output envelope."""
import importlib
from pathlib import Path

import pytest

import cron.output_envelope as oe


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Redirect all on-disk state into a tmp dir so tests don't touch live VPS state."""
    monkeypatch.setattr(oe, "DETAIL_DIR", tmp_path / "detail")
    monkeypatch.setattr(oe, "BRIEFING_LOG_DIR", tmp_path / "briefing")
    monkeypatch.setattr(oe, "DEDUP_FILE", tmp_path / "dedup.json")
    yield


JOB = {"id": "j_test", "name": "worker-failure-analysis"}


def test_silent_marker_suppressed():
    d = oe.build_envelope(JOB, "[SILENT]")
    assert d.suppress and d.reason == "silent-marker"


def test_empty_suppressed():
    assert oe.build_envelope(JOB, "   \n  ").suppress


def test_log_severity_suppressed_and_logged():
    d = oe.build_envelope(JOB, "🟢 All healthy, nothing to do")
    assert d.suppress and d.severity == oe.LOG
    logs = list((oe.BRIEFING_LOG_DIR).glob("*.jsonl"))
    assert logs and "healthy" in logs[0].read_text()
    # Carries a compact line for the #cron-outputs audit sink, still suppressed live.
    assert d.text and d.text.startswith("🟢")


def test_fyi_short_passthrough_no_ping():
    d = oe.build_envelope(JOB, "🟡 Ops health\nAll 11 gateways up, disk 8%.")
    assert not d.suppress and d.severity == oe.FYI
    assert "@Sahil" not in d.text
    assert d.text.startswith("🟡")


def test_act_gets_mention_and_action_line():
    content = (
        "🔴 Decision needed · t_cfa93e01\n"
        "StreamMemBench eval blocked since 20 May.\n"
        "Reply: B t_cfa93e01 or 'all defaults'"
    )
    d = oe.build_envelope(JOB, content, success=True)
    assert d.severity == oe.ACT
    assert "@Sahil" in d.text
    assert "Reply:" in d.text


def test_failed_run_defaults_to_act():
    d = oe.build_envelope(JOB, "scan crashed: boom", success=False)
    assert d.severity == oe.ACT


def test_success_no_marker_defaults_to_fyi():
    d = oe.build_envelope(JOB, "Some informational line.", success=True)
    assert d.severity == oe.FYI


def test_large_body_offloaded_to_attachment():
    body = "🟡 Worker Failure Analysis\n" + "\n".join(f"finding {i} detail line" for i in range(40))
    d = oe.build_envelope(JOB, body)
    assert not d.suppress
    assert "MEDIA:" in d.text
    assert "📎" in d.text
    # Channel body must be short: title + summary + attachment refs only.
    assert d.text.count("finding") <= oe.SUMMARY_MAX_LINES
    media_path = [l for l in d.text.splitlines() if l.startswith("MEDIA:")][0][len("MEDIA:"):]
    assert Path(media_path).exists()
    assert "finding 39" in Path(media_path).read_text()


def test_dedup_silences_repeat_within_window():
    msg = "🟡 Same finding\nidentical summary line here"
    first = oe.build_envelope(JOB, msg)
    second = oe.build_envelope(JOB, msg)
    assert not first.suppress
    assert second.suppress and second.reason == "deduped-12h"


def test_dedup_ignores_varying_counts():
    a = oe.build_envelope(JOB, "🟡 Findings\nthere are 3 open items")
    b = oe.build_envelope(JOB, "🟡 Findings\nthere are 7 open items")
    assert not a.suppress
    assert b.suppress  # only digits differ -> same dedup key


def test_sev_text_tag_recognised():
    d = oe.build_envelope(JOB, "[SEV:LOG] routine heartbeat ok")
    assert d.suppress and d.severity == oe.LOG


def test_per_job_severity_config():
    job = {"id": "j2", "name": "noisy", "severity": "LOG"}
    d = oe.build_envelope(job, "no marker here at all")
    assert d.suppress and d.severity == oe.LOG


def test_custom_mention_used():
    d = oe.build_envelope(JOB, "🔴 needs you", mention="<@123>")
    assert "<@123>" in d.text


def test_job_output_cannot_inject_mass_ping():
    d = oe.build_envelope(JOB, "🔴 @everyone build broke <@!42> see <@&7>", success=False)
    assert not d.suppress
    # The literal ping tokens must be neutralised in the channel text.
    assert "@everyone" not in d.text
    assert "<@!42>" not in d.text
    assert "<@&7>" not in d.text
    # The configured mention (the trusted one) still passes through.
    assert "@Sahil" in d.text


def test_detail_file_size_capped(monkeypatch):
    monkeypatch.setattr(oe, "MAX_DETAIL_BYTES", 200)
    body = "🟡 huge\n" + ("x" * 5000)
    d = oe.build_envelope(JOB, body)
    media = [l[6:] for l in d.text.splitlines() if l.startswith("MEDIA:")][0]
    content = Path(media).read_text()
    assert "[truncated]" in content
    # Body is capped to MAX_DETAIL_BYTES; the fixed dark-theme template adds a
    # constant overhead on top, so allow for that rather than asserting < 1KB.
    assert len(content.encode("utf-8")) < 2500


def test_deduped_repeat_writes_no_attachment(tmp_path):
    big = "🟡 wall\n" + "\n".join(f"line {i} of detail" for i in range(40))
    oe.build_envelope(JOB, big)
    before = len(list(oe.DETAIL_DIR.glob("*.html")))
    oe.build_envelope(JOB, big)  # deduped repeat
    after = len(list(oe.DETAIL_DIR.glob("*.html")))
    assert after == before  # no orphaned detail file from the suppressed repeat


def test_links_stay_inline_clickable():
    deals = "🟡 North Hope rig deals\n" + "\n".join(
        f"- RTX 3090 #{i} £620 https://cex.co.uk/item/{i}" for i in range(12)
    )
    d = oe.build_envelope({"id": "j_deal", "name": "rig-deal-scanner"}, deals)
    assert not d.suppress
    # No attachment: the links are in the channel body, clickable.
    assert "MEDIA:" not in d.text
    assert "https://cex.co.uk/item/0" in d.text
    assert "https://cex.co.uk/item/11" in d.text


def test_oversized_links_offload_to_clickable_html():
    deals = "🟡 deals\n" + "\n".join(
        f"- item {i} https://example.com/very/long/path/{i}" for i in range(120)
    )
    d = oe.build_envelope({"id": "j_deal", "name": "rig-deal-scanner"}, deals)
    media = [l[6:] for l in d.text.splitlines() if l.startswith("MEDIA:")][0]
    assert media.endswith(".html")
    html = Path(media).read_text()
    assert '<a href="https://example.com/very/long/path/0">' in html


def test_attachment_is_dark_theme():
    body = "🟡 wall\n" + "\n".join(f"detail line {i}" for i in range(40))
    d = oe.build_envelope(JOB, body)
    media = [l[6:] for l in d.text.splitlines() if l.startswith("MEDIA:")][0]
    html = Path(media).read_text()
    assert "#11100f" in html and "#fbbf24" in html and "#f5f5f4" in html
    assert "color:#1c1c1c" not in html  # no light-mode regression


def test_agent_media_passthrough_no_double_offload():
    # Agent emitted its own attachment; envelope must not create a second one.
    body = (
        "🟡 Mailbox cleaner\nProcessed 14 messages, 2 need replies.\n"
        + "\n".join(f"thread {i} summary line" for i in range(30))
        + "\nMEDIA:/home/kensei/.hermes/runbooks/mailbox/main-cleaner.html"
    )
    before = len(list(oe.DETAIL_DIR.glob("*.html")))
    d = oe.build_envelope(JOB, body)
    after = len(list(oe.DETAIL_DIR.glob("*.html")))
    assert after == before  # no envelope-generated attachment
    assert d.reason == "passthrough-media"
    assert "MEDIA:/home/kensei/.hermes/runbooks/mailbox/main-cleaner.html" in d.text
    assert "thread 29 summary line" not in d.text  # wall not dumped to channel


def test_html_attachment_no_xss_breakout():
    payload = '🟡 t\n' + "\n".join(
        f'see https://evil.com/{i}"><script>alert(1)</script> and more text line {i}'
        for i in range(40)
    )
    d = oe.build_envelope(JOB, payload)
    media = [l[6:] for l in d.text.splitlines() if l.startswith("MEDIA:")][0]
    html = Path(media).read_text()
    # No live script tag and no unescaped attribute breakout survived.
    assert "<script>" not in html
    assert '"><script' not in html
