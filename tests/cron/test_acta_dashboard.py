import base64
import hashlib
import json
import re
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from cron.acta_dashboard import (
    _dashboard_inline_script,
    _feed_lane,
    _is_interactive_html_artifact,
    _outputs_read_state_script,
    _run_browser_uat_preflight,
    acta_dashboard_config,
    apply_feed_preferences,
    attach_artifact_urls,
    attach_run_artifact_urls,
    available_run_dates,
    build_dashboard,
    collect_catalog_outputs,
    collect_run_history,
    collect_situation_items,
    load_release_log,
    publish_catalog_output_artifacts,
    render_acta_detail_report,
    render_interactive_html_detail_report,
    render_archive_index,
    render_catalog_outputs_page,
    render_dashboard,
    render_jobs_page,
    render_outputs_page,
    render_runs_page,
)


def _interactive_frame_source(rendered: str) -> str:
    match = re.search(r'<iframe[^>]+\ssrc="data:text/html;base64,([^"]+)"', rendered)
    assert match, rendered
    return base64.b64decode(match.group(1)).decode("utf-8")


def test_collects_latest_cron_outputs_from_response_section(tmp_path: Path):
    home = tmp_path
    (home / "cron" / "output" / "job1").mkdir(parents=True)
    (home / "cron" / "jobs.json").write_text(
        json.dumps(
            [
                {
                    "id": "job1",
                    "name": "Daily Brief",
                    "schedule": {"display": "0 8 * * *"},
                    "deliver": "telegram",
                    "enabled": True,
                }
            ]
        )
    )
    (home / "cron" / "output" / "job1" / "2026-05-19_08-00-00.md").write_text(
        "# Cron Job: Daily Brief\n\n## Prompt\nsecret prompt\n\n## Response\n\n# Human brief\n\n**Main:** useful output"
    )

    items = collect_situation_items(home)

    assert len(items) == 1
    assert items[0].job_id == "job1"
    assert items[0].status == "fresh"
    assert "Main: useful output" in items[0].excerpt
    assert "secret prompt" not in items[0].excerpt


def test_detail_pages_strip_embedded_html_report_from_markdown(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "details"
    md = tmp_path / "2026-05-20_16-35-00.md"
    md.write_text(
        "## Response\n\n"
        "# Afternoon Operator Closeout\n\n"
        "Useful operator brief.\n\n"
        "<!-- HERMES_HTML_REPORT_START -->\n"
        "<h1>Afternoon Operator Closeout — 2026-05-20</h1>\n"
        "<p>Duplicated report body that should not appear in Acta detail pages.</p>\n"
        "<!-- HERMES_HTML_REPORT_END -->\n"
    )
    item = collect_situation_items.__globals__["CronSituationItem"](
        job_id="job1",
        name="Afternoon Operator Closeout",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=md,
        latest_html=None,
        latest_time=None,
        status="fresh",
        excerpt="Useful operator brief.",
    )
    uploaded = {}

    def fake_publish(path, job, settings):
        uploaded["html"] = Path(path).read_text()
        return "https://acta.example/r/job1/detail.html"

    monkeypatch.setattr("cron.acta_dashboard.publish_html_artifact", fake_publish)

    attach_artifact_urls([item], {"enabled": True}, output_dir)

    assert "Useful operator brief." in uploaded["html"]
    assert "HERMES_HTML_REPORT_START" not in uploaded["html"]
    assert "Duplicated report body" not in uploaded["html"]


def test_interactive_html_detail_report_preserves_controls_inside_acta_shell():
    source = """<!doctype html><html><body>
<button id="run-lane">Run specialist lane</button>
<div id="status">Idle</div>
<script>document.getElementById('run-lane').onclick=function(){document.getElementById('status').textContent='Clicked'};</script>
</body></html>"""

    html = render_interactive_html_detail_report(
        source,
        {"job_id": "acta-specialists", "job_name": "Hermes Agent Lanes", "run_time": "2026-05-26T06:00:00+00:00"},
    )

    assert "ACTA</em> / INTERACTIVE" in html
    assert "Interactive Hermes/Acta output preserved" in html
    assert "sandbox=\"allow-scripts" in html
    assert "src=\"data:text/html;base64," in html
    frame_source = _interactive_frame_source(html)
    assert '<button id="run-lane">Run specialist lane</button>' in frame_source
    assert "document.getElementById('run-lane').onclick" in frame_source
    assert "Hermes Agent Lanes · Acta Interactive Output" in html


def test_interactive_html_detail_report_uses_strict_outer_csp_and_sandbox():
    html = render_interactive_html_detail_report(
        "<button onclick = \"window.open('/x')\">Open</button><script>window.ok=true</script>",
        {"job_id": "daily", "job_name": "Daily", "run_time": "2026-05-26T06:00:00+00:00"},
    )

    csp_match = re.search(r'Content-Security-Policy" content="([^"]+)"', html)
    assert csp_match
    csp = csp_match.group(1)
    assert "script-src 'none'" in csp
    assert "script-src 'unsafe-inline'" not in csp
    assert "frame-src data:" in csp
    frame_source = _interactive_frame_source(html).replace("&#x27;", "'")
    assert "connect-src 'none'" in frame_source
    assert "form-action 'none'" in frame_source
    assert "navigate-to 'none'" in frame_source
    iframe_match = re.search(r"<iframe[^>]+>", html)
    assert iframe_match
    iframe = iframe_match.group(0)
    assert 'sandbox="allow-scripts"' in iframe
    assert "allow-forms" not in iframe
    assert "allow-popups" not in iframe
    assert "allow-popups-to-escape-sandbox" not in iframe
    assert "allow-same-origin" not in iframe
    assert "top-navigation" not in iframe


def test_interactive_html_detail_report_redacts_raw_log_paths_and_secrets():
    source = """<!doctype html><html><body>
<button data-action = "run">Run</button>
<pre>## Prompt: do not leak this raw prompt
Tool call: terminal command: cat /Users/mozzie/.hermes/secrets.env
OPENAI_API_KEY=sk-test-supersecretvalue
Local file: /tmp/private/secret.md</pre>
</body></html>"""

    html = render_interactive_html_detail_report(
        source,
        {"job_id": "daily", "job_name": "Daily", "run_time": "2026-05-26T06:00:00+00:00"},
    )
    frame_source = _interactive_frame_source(html)

    assert "Redacted interactive artifact" in frame_source
    for leaked in (
        "do not leak this raw prompt",
        "Tool call",
        "terminal command",
        "/Users/mozzie",
        "/tmp/private/secret.md",
        "supersecretvalue",
        "OPENAI_API_KEY",
    ):
        assert leaked not in html
        assert leaked not in frame_source


def test_interactive_html_detail_report_redacts_file_url_local_paths():
    html = render_interactive_html_detail_report(
        '<button>Open</button><a href="file:///Users/mozzie/.hermes/secrets.env">secret path</a>',
        {"job_id": "daily", "job_name": "Daily", "run_time": "2026-05-26T06:00:00+00:00"},
    )
    frame_source = _interactive_frame_source(html)

    assert "file:///Users/mozzie" not in html
    assert "file:///Users/mozzie" not in frame_source
    assert "[LOCAL_PATH]" in frame_source


def test_interactive_detector_handles_event_whitespace_but_not_details_only():
    assert _is_interactive_html_artifact('<div onclick = "go()">Go</div>')
    assert _is_interactive_html_artifact('<section data-action = "lane#open">Open</section>')
    assert not _is_interactive_html_artifact('<details><summary>Raw log</summary><pre>Prompt: secret</pre></details>')


def test_html_only_cron_outputs_publish_interactive_wrapper(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "details"
    source_html = tmp_path / "2026-05-26_06-00-00.html"
    source_html.write_text(
        """<!doctype html><html><body>
<h1>Specialist agents</h1>
<button id="alpha">Open Alpha</button>
<script>window.alphaReady=true;</script>
</body></html>""",
        encoding="utf-8",
    )
    item = collect_situation_items.__globals__["CronSituationItem"](
        job_id="acta-specialists",
        name="Acta Hermes agent lanes",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=None,
        latest_html=source_html,
        latest_time=None,
        status="fresh",
        excerpt="Interactive specialist lane artifact.",
    )
    uploaded = {}

    def fake_publish(path, job, settings):
        uploaded["html"] = Path(path).read_text()
        return "https://acta.imperatr.com/r/acta-specialists/detail.html?exp=1&sig=abc"

    monkeypatch.setattr("cron.acta_dashboard.publish_html_artifact", fake_publish)

    linked = attach_artifact_urls([item], {"enabled": True}, output_dir)

    assert linked[0].artifact_url == "https://acta.imperatr.com/r/acta-specialists/detail.html?exp=1&sig=abc"
    assert "ACTA</em> / INTERACTIVE" in uploaded["html"]
    frame_source = _interactive_frame_source(uploaded["html"])
    assert '<button id="alpha">Open Alpha</button>' in frame_source
    assert "window.alphaReady=true" in frame_source
    assert "<article class=\"report-body\">" not in uploaded["html"]


def test_dashboard_escapes_content_and_links_artifact(tmp_path: Path):
    home = tmp_path
    (home / "cron" / "output" / "job1").mkdir(parents=True)
    (home / "cron" / "jobs.json").write_text(
        json.dumps([{"id": "job1", "name": "<b>Bad</b>", "schedule": "daily", "deliver": "local"}])
    )
    (home / "cron" / "output" / "job1" / "2026-05-19.md").write_text("## Response\n\n<script>bad()</script>")

    items = collect_situation_items(home)
    html = render_dashboard(items)

    assert "<script>bad" not in html
    assert "&lt;b&gt;Bad&lt;/b&gt;" in html
    assert "Acta Situation Room" in html


def test_dashboard_uses_acta_imperatr_suite_board_palette(tmp_path: Path):
    (tmp_path / "cron" / "output" / "lead").mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(
        json.dumps([{"id": "lead", "name": "Lead Brief", "schedule": "daily", "deliver": "telegram"}])
    )
    (tmp_path / "cron" / "output" / "lead" / "2026-05-19_10-00-00.md").write_text("## Response\n\nMost important")

    html = render_dashboard(collect_situation_items(tmp_path))

    assert "#03060b" in html
    assert "#756cff" in html
    assert "#23a7ff" in html
    assert "Acta Imperatr situation room" in html
    assert "ACTA</em> / TODAY" in html
    assert "Today’s Brief" in html
    assert "output-summary" in html
    assert "metricrow" not in html
    assert "P2" not in html
    assert "P3" not in html
    assert "Delivery Routes" not in html
    assert "Bloomberg" not in html
    assert "Your cron command center" not in html


def test_dashboard_separates_daily_life_and_development_sprint_feeds(tmp_path: Path):
    jobs = [
        {"id": "news", "name": "Morning Newsletter", "schedule": "daily", "deliver": "telegram"},
        {"id": "vesta", "name": "Vesta Startup Sprint CEO loop", "schedule": "*/30 * * * *", "deliver": "telegram"},
        {"id": "qa", "name": "Weekly app user-testing sweep", "schedule": "weekly", "deliver": "telegram"},
    ]
    for job in jobs:
        (tmp_path / "cron" / "output" / job["id"]).mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(json.dumps(jobs))
    (tmp_path / "cron" / "output" / "news" / "2026-05-19_09-00-00.md").write_text("## Response\n\nDaily signal")
    (tmp_path / "cron" / "output" / "vesta" / "2026-05-19_09-10-00.md").write_text("## Response\n\nSprint signal")
    (tmp_path / "cron" / "output" / "qa" / "2026-05-19_09-20-00.md").write_text("## Response\n\nQA signal")

    items = collect_situation_items(tmp_path)
    lanes = {item.job_id: _feed_lane(item) for item in items}
    html = render_dashboard(items)

    assert lanes == {"news": "daily", "vesta": "dev", "qa": "dev"}
    assert "Output Streams" in html
    assert "Daily life feed" in html
    assert "Development sprint cycles" in html
    assert 'data-feed-lane="daily"' in html
    assert 'class="lane-chip" title="Daily life feed">Daily</span>' in html
    assert 'data-feed-lane="dev"' in html
    assert html.index("Daily life feed") < html.index("Development sprint cycles")
    assert '<details class="feed-section lane-section-dev dev-inbox">' in html
    assert "background by default" in html
    assert "Today’s Brief" in html
    assert "Daily: 1 outputs · News" in html
    assert "Dev: 2 sprint updates · background" in html
    assert '<input class="view-mode-input" id="view-digest" name="acta-view-mode" type="radio" checked>' in html
    assert '<input class="view-mode-input" id="view-trace" name="acta-view-mode" type="radio">' in html
    assert '<div class="mode-switch" role="radiogroup" aria-label="Acta view mode"><label for="view-digest">Digest</label><label for="view-trace">Trace</label></div>' in html


def test_dashboard_dev_only_lead_is_not_duplicated_in_dev_lane_read_keys():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    lead = item_cls(
        job_id="vesta-startup",
        name="Vesta Startup Sprint CEO loop",
        schedule="*/30 * * * *",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Sprint signal.",
        artifact_url="https://acta.imperatr.com/r/vesta-startup/detail.html?exp=1&sig=abc",
    )

    html = render_dashboard([lead], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))
    dev_section = html[html.index('lane-section-dev') :]

    assert _feed_lane(lead) == "dev"
    assert html.count("<h1>Vesta Startup Sprint CEO loop</h1>") == 1
    assert html.count("<h2>Vesta Startup Sprint CEO loop</h2>") == 0
    assert html.count('data-read-key="vesta-startup:') == 1
    assert "0 additional updates · background by default" in dev_section
    assert "No additional outputs in this lane yet." in dev_section
    assert "No visible outputs in this lane yet." not in dev_section


def test_dashboard_digest_uses_real_item_titles_and_escapes_them():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    lead = item_cls(
        job_id="daily-lead",
        name="News & Weather <Lead>",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Daily signal.",
    )
    review = item_cls(
        job_id="qa-review",
        name="QA Smoke Review & Fixes",
        schedule="*/30 * * * *",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_09-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 9, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Needs review before release.",
    )

    html = render_dashboard([lead, review], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))

    assert "Daily: 1 outputs" in html
    assert "Review: 1 item needs review" in html
    assert "News & Weather <Lead>" not in html


def test_dashboard_digest_pluralizes_multiple_review_actions():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    lead = item_cls(
        job_id="daily-lead",
        name="Daily Lead Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Daily signal.",
    )
    top_review = item_cls(
        job_id="qa-review",
        name="QA Smoke Review & Fixes",
        schedule="*/30 * * * *",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_09-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 9, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Needs review before release.",
    )
    second_review = item_cls(
        job_id="security-review",
        name="Security Audit Review",
        schedule="*/30 * * * *",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_08-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 8, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Needs review before deploy.",
    )

    html = render_dashboard([lead, top_review, second_review], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))

    assert "Review: 2 items need review" in html


def test_dashboard_system_only_lead_keeps_system_lane_visible():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    lead = item_cls(
        job_id="acta-refresh",
        name="Situation Room Refresh",
        schedule="*/15 * * * *",
        deliver="local",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Dashboard refreshed.",
    )

    html = render_dashboard([lead], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))
    system_section = html[html.index('lane-section-system') :]

    assert _feed_lane(lead) == "system"
    assert "System/local jobs" in system_section
    assert html.count("<h1>Situation Room Refresh</h1>") == 1
    assert html.count("<h2>Situation Room Refresh</h2>") == 0
    assert "No additional outputs in this lane yet." in system_section
    assert "No visible outputs in this lane yet." not in system_section


def test_feed_lane_classifies_generic_dev_job_names_without_catching_daily_life():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]

    def item(name: str):
        return item_cls(
            job_id=name.lower().replace(" ", "-"),
            name=name,
            schedule="daily",
            deliver="telegram",
            enabled=True,
            latest_md=None,
            latest_html=None,
            latest_time=None,
            status="fresh",
            excerpt="Signal.",
        )

    dev_names = ["Self-healing repair loop", "QA smoke run", "User testing notes", "security audit"]
    daily_names = ["Daily life notes", "News roundup", "Weather brief", "Sports scores", "Lunch ideas"]

    assert {name: _feed_lane(item(name)) for name in dev_names} == {name: "dev" for name in dev_names}
    assert {name: _feed_lane(item(name)) for name in daily_names} == {name: "daily" for name in daily_names}


def test_dashboard_audit_trail_preserves_ordered_recency_across_lanes():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    lead = item_cls(
        job_id="daily-lead",
        name="Daily Lead Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_11-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 11, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Lead daily signal.",
    )
    dev = item_cls(
        job_id="vesta-startup",
        name="Vesta Startup Sprint CEO loop",
        schedule="*/30 * * * *",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Dev signal.",
    )
    daily_followup = item_cls(
        job_id="daily-followup",
        name="Daily Follow Up",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_09-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 9, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Follow-up daily signal.",
    )

    html = render_dashboard([lead, dev, daily_followup], generated_at=datetime(2026, 5, 19, 12, tzinfo=timezone.utc))
    audit_start = html.index('<section class="card"><div class="card-head">Audit Trail')
    audit_section = html[audit_start : html.index("Operator Assist", audit_start)]

    assert "Daily Lead Brief" not in audit_section
    assert audit_section.index("Vesta Startup Sprint CEO loop") < audit_section.index("Daily Follow Up")
    assert "Development sprint cycles · fresh · vesta-startup" in audit_section
    assert "Daily life feed · fresh · daily-followup" in audit_section


def test_collects_telegram_thread_links_from_delivery_targets(tmp_path: Path):
    home = tmp_path
    (home / "cron" / "output" / "job1").mkdir(parents=True)
    (home / "cron" / "jobs.json").write_text(
        json.dumps(
            [
                {
                    "id": "job1",
                    "name": "Operator Brief",
                    "schedule": "daily",
                    "deliver": "telegram:-1003566991387:86",
                }
            ]
        )
    )
    (home / "cron" / "output" / "job1" / "2026-05-19.md").write_text("## Response\n\nUseful")

    item = collect_situation_items(home)[0]

    assert item.telegram_url == "https://t.me/c/3566991387/86"
    jobs_html = render_jobs_page([item])
    assert 'href="https://t.me/c/3566991387/86"' in jobs_html
    assert "THREAD" in jobs_html
    dashboard_html = render_dashboard([item])
    assert ">ASK</a>" in dashboard_html
    assert 'href="https://t.me/c/3566991387/86"' in dashboard_html


def test_collects_public_telegram_username_links_from_delivery_targets(tmp_path: Path):
    (tmp_path / "cron" / "output" / "plain").mkdir(parents=True)
    (tmp_path / "cron" / "output" / "thread").mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(
        json.dumps(
            [
                {"id": "plain", "name": "Plain Channel", "schedule": "daily", "deliver": "telegram:@valid_name"},
                {"id": "thread", "name": "Thread Channel", "schedule": "daily", "deliver": "telegram:@Valid_Name123:987"},
            ]
        )
    )
    (tmp_path / "cron" / "output" / "plain" / "2026-05-19.md").write_text("## Response\n\nPlain")
    (tmp_path / "cron" / "output" / "thread" / "2026-05-19.md").write_text("## Response\n\nThread")

    items = {item.job_id: item for item in collect_situation_items(tmp_path)}

    assert items["plain"].telegram_url == "https://t.me/valid_name"
    assert items["thread"].telegram_url == "https://t.me/Valid_Name123/987"
    jobs_html = render_jobs_page(list(items.values()))
    dashboard_html = render_dashboard(list(items.values()))
    detail_html = render_acta_detail_report(
        "# Thread\n\nPublic username follow-up.",
        {"job_id": "thread", "job_name": "Thread Channel", "run_time": "2026-05-19T10:00:00+00:00"},
        telegram_url=items["thread"].telegram_url,
    )
    assert 'href="https://t.me/valid_name"' in jobs_html
    assert 'href="https://t.me/Valid_Name123/987"' in jobs_html
    assert 'href="https://t.me/valid_name"' in dashboard_html
    assert 'href="https://t.me/Valid_Name123/987"' in detail_html


def test_rejects_invalid_public_telegram_username_links_from_delivery_and_origin_metadata(tmp_path: Path):
    invalid_jobs = [
        {"id": "slash", "name": "Slash", "schedule": "daily", "deliver": "telegram:@bad/name:12"},
        {"id": "protocol", "name": "Protocol", "schedule": "daily", "deliver": "telegram:@https://evil.example:12"},
        {"id": "control", "name": "Control", "schedule": "daily", "deliver": "telegram:@bad\nname:12"},
        {"id": "short", "name": "Short", "schedule": "daily", "deliver": "telegram:@abcd"},
        {"id": "chars", "name": "Chars", "schedule": "daily", "deliver": "telegram:@bad-name:12"},
        {"id": "digit", "name": "Digit", "schedule": "daily", "deliver": "telegram:@1valid:12"},
        {"id": "path", "name": "Path", "schedule": "daily", "deliver": "telegram:@valid_name/../../evil:12"},
        {"id": "thread", "name": "Thread", "schedule": "daily", "deliver": "telegram:@valid_name:not-a-number"},
        {
            "id": "origin",
            "name": "Origin",
            "schedule": "daily",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "@bad/name", "thread_id": "12"},
        },
        {
            "id": "origin_thread",
            "name": "Origin Thread",
            "schedule": "daily",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "@valid_name", "thread_id": "abc"},
        },
    ]
    for job in invalid_jobs:
        (tmp_path / "cron" / "output" / job["id"]).mkdir(parents=True)
        (tmp_path / "cron" / "output" / job["id"] / "2026-05-19.md").write_text("## Response\n\nInvalid")
    (tmp_path / "cron" / "jobs.json").write_text(json.dumps(invalid_jobs))

    items = collect_situation_items(tmp_path)

    assert {item.job_id: item.telegram_url for item in items} == {job["id"]: None for job in invalid_jobs}
    jobs_html = render_jobs_page(items)
    dashboard_html = render_dashboard(items)
    assert "THREAD" not in jobs_html
    assert "ASK TELEGRAM" not in dashboard_html
    for item in items:
        detail_html = render_acta_detail_report(
            "# Invalid\n\nNo follow-up link.",
            {"job_id": item.job_id, "job_name": item.name, "run_time": "2026-05-19T10:00:00+00:00"},
            telegram_url=item.telegram_url,
        )
        assert "Ask follow-up in Telegram" not in detail_html
        assert "https://t.me/" not in detail_html


def test_detail_report_can_link_back_to_telegram_thread():
    html = render_acta_detail_report(
        "# Operator Brief\n\nUseful signal.",
        {"job_id": "job1", "job_name": "Operator Brief", "run_time": "2026-05-19T10:00:00+00:00"},
        telegram_url="https://t.me/c/3566991387/86",
    )

    assert "Ask follow-up in Telegram" in html
    assert 'href="https://t.me/c/3566991387/86"' in html
    assert 'target="_blank" rel="noopener"' in html


def test_detail_report_rejects_telegram_subdomain_followup_link():
    html = render_acta_detail_report(
        "# Operator Brief\n\nUseful signal.",
        {"job_id": "job1", "job_name": "Operator Brief", "run_time": "2026-05-19T10:00:00+00:00"},
        telegram_url="https://evil.t.me/c/3566991387/86",
    )

    assert "Ask follow-up in Telegram" not in html
    assert "evil.t.me" not in html


def test_renderers_reject_telegram_userinfo_spoofing(tmp_path: Path):
    spoofed_urls = [
        "https://evil.t.me@t.me/c/3566991387/86",
        "https://evil.t.me:secret@t.me/c/3566991387/86",
    ]
    CronSituationItem = collect_situation_items.__globals__["CronSituationItem"]

    for telegram_url in spoofed_urls:
        detail_html = render_acta_detail_report(
            "# Operator Brief\n\nUseful signal.",
            {"job_id": "job1", "job_name": "Operator Brief", "run_time": "2026-05-19T10:00:00+00:00"},
            telegram_url=telegram_url,
        )
        assert "Ask follow-up in Telegram" not in detail_html
        assert telegram_url not in detail_html

    item = CronSituationItem(
        job_id="job1",
        name="Operator Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=tmp_path / "2026-05-19.md",
        latest_html=None,
        latest_time=None,
        status="fresh",
        excerpt="Useful signal.",
        telegram_url=spoofed_urls[0],
    )
    dashboard_html = render_dashboard([item])

    assert "ASK TELEGRAM" not in dashboard_html
    assert spoofed_urls[0] not in dashboard_html


def test_build_dashboard_local_only(tmp_path: Path):
    (tmp_path / "cron" / "output" / "job1").mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(json.dumps([{"id": "job1", "name": "Daily"}]))
    (tmp_path / "cron" / "output" / "job1" / "2026-05-19.md").write_text("## Response\n\nOK")

    path, url = build_dashboard(tmp_path, publish=False)

    assert url is None
    assert path.exists()
    assert "Acta Imperatr situation room" in path.read_text()


def _write_publishable_dashboard_fixture(home: Path) -> None:
    (home / "cron" / "output" / "daily").mkdir(parents=True)
    (home / "cron" / "output" / "dev").mkdir(parents=True)
    (home / "cron" / "jobs.json").write_text(
        json.dumps(
            [
                {"id": "daily", "name": "Morning newsletter digest", "schedule": "daily", "deliver": "telegram"},
                {"id": "dev", "name": "Vesta Startup Sprint CEO loop", "schedule": "every 120m", "deliver": "telegram"},
            ]
        )
    )
    (home / "cron" / "output" / "daily" / "2026-05-19_09-00-00.md").write_text("## Response\n\nDaily signal")
    (home / "cron" / "output" / "dev" / "2026-05-19_09-10-00.md").write_text("## Response\n\nSprint signal")


def test_publish_runs_browser_uat_before_uploading_any_artifacts(tmp_path: Path, monkeypatch):
    _write_publishable_dashboard_fixture(tmp_path)
    events = []

    def fake_uat(path: Path, *, artifact_dir: Path, timeout: int = 45) -> None:
        events.append(("uat", path.exists(), artifact_dir.name, timeout))

    def fake_publish(path, job, settings):
        events.append(("publish", settings.get("object_key")))
        return f"https://acta.example/{settings.get('object_key')}"

    monkeypatch.setattr("cron.acta_dashboard._run_browser_uat_preflight", fake_uat)
    monkeypatch.setattr("cron.acta_dashboard.publish_html_artifact", fake_publish)

    path, url = build_dashboard(tmp_path, publish=True)

    assert path.exists()
    assert url == "https://acta.example/public/index.html"
    assert events[0][0] == "uat"
    assert any(event == ("publish", "public/index.html") for event in events)


def test_publish_aborts_without_uploads_when_browser_uat_fails(tmp_path: Path, monkeypatch):
    _write_publishable_dashboard_fixture(tmp_path)
    published = []

    def fake_uat(path: Path, *, artifact_dir: Path, timeout: int = 45) -> None:
        raise RuntimeError("Acta browser UAT preflight failed; publish aborted.")

    def fake_publish(path, job, settings):
        published.append(settings.get("object_key"))
        return "https://acta.example/should-not-publish"

    monkeypatch.setattr("cron.acta_dashboard._run_browser_uat_preflight", fake_uat)
    monkeypatch.setattr("cron.acta_dashboard.publish_html_artifact", fake_publish)

    with pytest.raises(RuntimeError, match="publish aborted"):
        build_dashboard(tmp_path, publish=True)

    assert published == []


def test_publish_skip_uat_preflight_escape_hatch(tmp_path: Path, monkeypatch):
    _write_publishable_dashboard_fixture(tmp_path)
    events = []

    def fake_uat(path: Path, *, artifact_dir: Path, timeout: int = 45) -> None:
        events.append(("uat", str(path)))

    def fake_publish(path, job, settings):
        events.append(("publish", settings.get("object_key")))
        return f"https://acta.example/{settings.get('object_key')}"

    monkeypatch.setattr("cron.acta_dashboard._run_browser_uat_preflight", fake_uat)
    monkeypatch.setattr("cron.acta_dashboard.publish_html_artifact", fake_publish)

    build_dashboard(tmp_path, publish=True, uat_preflight=False)

    assert not any(event[0] == "uat" for event in events)
    assert any(event == ("publish", "public/index.html") for event in events)


def test_publish_skip_uat_preflight_env_escape_hatch(tmp_path: Path, monkeypatch):
    _write_publishable_dashboard_fixture(tmp_path)
    events = []

    def fake_uat(path: Path, *, artifact_dir: Path, timeout: int = 45) -> None:
        events.append(("uat", str(path)))

    def fake_publish(path, job, settings):
        events.append(("publish", settings.get("object_key")))
        return f"https://acta.example/{settings.get('object_key')}"

    monkeypatch.setattr("cron.acta_dashboard._run_browser_uat_preflight", fake_uat)
    monkeypatch.setattr("cron.acta_dashboard.publish_html_artifact", fake_publish)
    monkeypatch.setenv("ACTA_SKIP_BROWSER_UAT", "1")

    build_dashboard(tmp_path, publish=True)

    assert not any(event[0] == "uat" for event in events)
    assert any(event == ("publish", "public/index.html") for event in events)


def test_browser_uat_preflight_timeout_fails_closed(tmp_path: Path, monkeypatch):
    dashboard_path = tmp_path / "dashboard.html"
    dashboard_path.write_text("<html><body>Acta</body></html>")

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"], output="partial browser log")

    monkeypatch.setattr("cron.acta_dashboard.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="timed out; publish aborted") as excinfo:
        _run_browser_uat_preflight(dashboard_path, artifact_dir=tmp_path / "uat", timeout=1)

    assert "partial browser log" in str(excinfo.value)


def test_available_dates_and_date_filtered_items(tmp_path: Path):
    (tmp_path / "cron" / "output" / "job1").mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(json.dumps([{"id": "job1", "name": "Daily"}]))
    (tmp_path / "cron" / "output" / "job1" / "2026-05-18_08-00-00.md").write_text("## Response\n\nOld")
    (tmp_path / "cron" / "output" / "job1" / "2026-05-19_08-00-00.md").write_text("## Response\n\nNew")

    dates = available_run_dates(tmp_path)
    assert [d.isoformat() for d in dates] == ["2026-05-19", "2026-05-18"]
    items = collect_situation_items(tmp_path, run_date=dates[1])
    assert len(items) == 1
    assert items[0].excerpt == "Old"


def test_feed_preferences_excludes_morning_audio_by_default():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    visible = item_cls(
        job_id="daily",
        name="Daily Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=None,
        latest_html=None,
        latest_time=None,
        status="fresh",
        excerpt="Daily",
    )
    audio = item_cls(
        job_id="e9b0a041ced3",
        name="P Morning Audio Briefing",
        schedule="20 7 * * *",
        deliver="telegram",
        enabled=True,
        latest_md=None,
        latest_html=None,
        latest_time=None,
        status="fresh",
        excerpt="Audio",
    )

    assert [item.job_id for item in apply_feed_preferences([visible, audio])] == ["daily"]


def test_dashboard_counts_exclude_morning_audio_by_default():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    visible = item_cls(
        job_id="daily",
        name="Daily Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=None,
        latest_html=None,
        latest_time=None,
        status="fresh",
        excerpt="Daily",
    )
    audio = item_cls(
        job_id="e9b0a041ced3",
        name="P Morning Audio Briefing",
        schedule="20 7 * * *",
        deliver="telegram",
        enabled=True,
        latest_md=None,
        latest_html=None,
        latest_time=None,
        status="fresh",
        excerpt="Audio",
    )

    html = render_dashboard([visible, audio])

    assert "P Morning Audio Briefing" not in html
    assert 'Today <span>1</span>' in html
    assert '<div class="output-summary"><b>1/1</b><span>visible</span><span class="trace-only">0 gaps</span></div>' in html
    assert "metricrow" not in html
    assert "VISIBLE <b>1</b>" in html


def test_feed_preferences_reorder_hide_and_filter_items(tmp_path: Path):
    home = tmp_path
    for job_id in ("sports", "ai", "system"):
        (home / "cron" / "output" / job_id).mkdir(parents=True)
    (home / "cron" / "jobs.json").write_text(
        json.dumps(
            [
                {"id": "sports", "name": "Morning sports briefing", "deliver": "telegram"},
                {"id": "ai", "name": "AI Daily Brief podcast digest", "deliver": "telegram"},
                {"id": "system", "name": "Acta Situation Room refresh", "deliver": "local"},
            ]
        )
    )
    (home / "cron" / "output" / "sports" / "2026-05-19_08-00-00.md").write_text("## Response\n\nSports")
    (home / "cron" / "output" / "ai" / "2026-05-19_09-00-00.md").write_text("## Response\n\nAI")
    (home / "cron" / "output" / "system" / "2026-05-19_10-00-00.md").write_text("## Response\n\nRefresh")

    items = collect_situation_items(home)
    ordered = apply_feed_preferences(
        items,
        {
            "pinned": ["sports"],
            "hidden": ["AI Daily"],
            "show_system": False,
        },
    )

    assert [item.job_id for item in ordered] == ["sports"]


def test_build_dashboard_uses_acta_feed_config(tmp_path: Path):
    for job_id in ("first", "second"):
        (tmp_path / "cron" / "output" / job_id).mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(
        json.dumps(
            [
                {"id": "first", "name": "First Brief", "deliver": "telegram"},
                {"id": "second", "name": "Second Brief", "deliver": "telegram"},
            ]
        )
    )
    (tmp_path / "cron" / "output" / "first" / "2026-05-19_08-00-00.md").write_text("## Response\n\nFirst")
    (tmp_path / "cron" / "output" / "second" / "2026-05-19_09-00-00.md").write_text("## Response\n\nSecond")
    (tmp_path / "config.yaml").write_text(
        "cron:\n  acta_dashboard:\n    pinned:\n      - first\n"
    )

    assert acta_dashboard_config({"cron": {"acta_dashboard": {"pinned": ["first"]}}})["pinned"] == ["first"]
    path, _ = build_dashboard(tmp_path, publish=False)
    html = path.read_text()

    assert html.index("First Brief") < html.index("Second Brief")


def test_mobile_dashboard_does_not_duplicate_lead_in_feed(tmp_path: Path):
    for job_id, name, body in (
        ("lead", "Lead Brief", "Most important"),
        ("second", "Second Brief", "Next item"),
    ):
        (tmp_path / "cron" / "output" / job_id).mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(
        json.dumps(
            [
                {"id": "lead", "name": "Lead Brief", "deliver": "telegram"},
                {"id": "second", "name": "Second Brief", "deliver": "telegram"},
            ]
        )
    )
    (tmp_path / "cron" / "output" / "lead" / "2026-05-19_10-00-00.md").write_text("## Response\n\nMost important")
    (tmp_path / "cron" / "output" / "second" / "2026-05-19_09-00-00.md").write_text("## Response\n\nNext item")

    html = render_dashboard(collect_situation_items(tmp_path), feed_preferences={"pinned": ["lead"]})

    assert html.count("<h1>Lead Brief</h1>") == 1
    assert html.count("<h2>Lead Brief</h2>") == 0
    assert html.count("<h2>Second Brief</h2>") == 1
    assert "maximum-scale=1" in html
    assert "user-scalable=no" in html
    assert ".row-kicker { flex-wrap:wrap; overflow:visible" in html
    assert ".source-line { display:none; }" in html
    assert '#view-digest:checked ~ .mode-switch label[for=\'view-digest\']' in html
    assert ".view-mode-input { position:absolute; width:1px; height:1px; opacity:0; }" in html
    assert ".view-mode-input { position:absolute; opacity:0; pointer-events:none; }" not in html
    assert '#view-trace:checked ~ .mode-switch label[for=\'view-trace\']' in html
    assert "#view-trace:checked ~ .content span.trace-only" in html
    assert "#view-trace:checked ~ .content .source-line.trace-only { display:block !important; }" in html
    assert "#view-trace:checked ~ .content .feed-section-title.trace-only { display:flex !important; }" in html
    assert "#view-trace:checked ~ .content .source-line.trace-only { display:block !important; white-space:normal; overflow-wrap:anywhere; word-break:break-word; overflow:visible; text-overflow:clip; line-height:1.25; }" in html
    assert "source-line.trace-only { display:none !important; }" not in html
    assert "display:initial" not in html
    assert ".feed-section-title span { display:none; }" in html
    assert ".lead { grid-template-columns:1fr; }" in html
    assert ".swipe-content { grid-template-columns:28px minmax(0,1fr); }" in html
    assert "second · telegram · 2026-05-19T09:00:00+00:00" in html
    assert '<span class="attention-chip">Later</span>' in html
    assert '<span class="trace-only">manual</span>' in html
    assert '<div class="mode-switch" role="radiogroup" aria-label="Acta view mode">' in html
    assert '<section class="today-brief"><h2>Today’s Brief</h2>' in html


def test_lead_brief_is_clickable_and_read_state_enabled(tmp_path: Path):
    (tmp_path / "cron" / "output" / "lead").mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(
        json.dumps([{"id": "lead", "name": "Lead Brief", "deliver": "telegram"}])
    )
    (tmp_path / "cron" / "output" / "lead" / "2026-05-19_10-00-00.md").write_text("## Response\n\nMost important")
    item = collect_situation_items(tmp_path)[0]
    linked_item = item.__class__(
        job_id=item.job_id,
        name=item.name,
        schedule=item.schedule,
        deliver=item.deliver,
        enabled=item.enabled,
        latest_md=item.latest_md,
        latest_html=item.latest_html,
        latest_time=item.latest_time,
        status=item.status,
        excerpt=item.excerpt,
        artifact_url="https://acta.imperatr.com/r/lead/detail.html?exp=1&sig=abc",
    )

    html = render_dashboard([linked_item])

    assert '<article class="lead readable unread"' in html
    assert 'data-open-url="https://acta.imperatr.com/r/lead/detail.html?exp=1&amp;sig=abc"' in html
    assert '<a class="row-open-overlay" href="https://acta.imperatr.com/r/lead/detail.html?exp=1&amp;sig=abc" aria-label="Open briefing: Lead Brief"></a>' in html
    assert 'aria-label="Open briefing: Lead Brief"' in html
    assert 'data-read-key="lead:' in html
    assert 'data-read-title="Lead Brief"' in html
    assert '<span class="read-state">UNREAD</span>' in html
    assert 'MARK READ' in html
    assert '<button class="read-toggle" type="button" aria-label="Mark briefing read: Lead Brief">Mark read</button>' in html
    assert '<button class="state-toggle save-toggle" type="button" data-state-action="save" aria-label="Save briefing: Lead Brief">Save</button>' in html
    assert '<button class="state-toggle dismiss-toggle" type="button" data-state-action="dismiss" aria-label="Dismiss briefing: Lead Brief">Dismiss</button>' in html
    assert '<button class="state-toggle later-toggle" type="button" data-state-action="later" aria-label="Read later: Lead Brief">Read later</button>' in html
    assert '<div>UNREAD <b data-unread-count="1">1</b></div>' in html
    assert '<a href="/">TODAY <span class="nav-count" data-unread-count="1">1</span></a>' in html
    assert ".nav-count" in html
    assert ".row-open-overlay:focus-visible" in html
    assert ".brief-row > .swipe-content" in html
    assert ".brief-row > :not(.row-open-overlay)" not in html
    assert ".lead > :not(.row-open-overlay), .brief-row > .swipe-content { pointer-events:none; }" in html
    assert ".lead .ask-label, .brief-row .ask-label, .card-actions, .read-toggle, .state-toggle { pointer-events:auto; }" in html
    assert "el.querySelectorAll('.row-open-overlay')" in html
    assert "script-src 'sha256-" in html
    assert "style-src 'sha256-" in html
    assert "unsafe-inline" not in html
    assert "localStorage" in html
    csp = re.search(r'Content-Security-Policy" content="([^"]+)"', html)
    style = re.search(r"<style>(.*?)</style>", html, re.S)
    script = re.search(r"<script>(.*?)</script>", html, re.S)
    assert csp and style and script
    style_hash = base64.b64encode(hashlib.sha256(style.group(1).encode("utf-8")).digest()).decode("ascii")
    script_hash = base64.b64encode(hashlib.sha256(script.group(1).encode("utf-8")).digest()).decode("ascii")
    assert f"style-src 'sha256-{style_hash}'" in csp.group(1)
    assert f"script-src 'sha256-{script_hash}'" in csp.group(1)
    assert ".style." not in script.group(1)


def test_feed_rows_are_keyboard_accessible_signed_links():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    lead = item_cls(
        job_id="lead",
        name="Lead Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Lead briefing.",
        artifact_url="https://acta.imperatr.com/r/lead/detail.html?exp=1&sig=lead",
    )
    second = item_cls(
        job_id="second",
        name="Second Brief",
        schedule="hourly",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_09-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 9, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Follow-up briefing.",
        artifact_url="https://acta.imperatr.com/r/second/detail.html?exp=1&sig=abc",
    )

    html = render_dashboard([lead, second], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))

    assert '<section class="brief-row readable unread fresh"' in html
    assert 'data-open-url="https://acta.imperatr.com/r/second/detail.html?exp=1&amp;sig=abc"' in html
    assert '<a class="row-open-overlay" href="https://acta.imperatr.com/r/second/detail.html?exp=1&amp;sig=abc" aria-label="Open briefing: Second Brief"></a>' in html
    assert 'aria-label="Open briefing: Second Brief"' in html
    assert 'data-read-title="Second Brief"' in html
    assert '<button class="read-toggle" type="button" aria-label="Mark briefing read: Second Brief">Mark read</button>' in html


def test_dashboard_read_toggle_script_updates_button_labels_and_counts():
    script = _dashboard_inline_script()

    assert "var button=el.querySelector('.read-toggle')" in script
    assert "button.textContent=isRead?'Mark unread':'Mark read';" in script
    assert "button.setAttribute('aria-label', (isRead?'Mark briefing unread: ':'Mark briefing read: ')+title);" in script
    assert "button.addEventListener('click'" in script
    assert "ev.preventDefault();" in script
    assert "ev.stopPropagation();" in script
    assert "setRead(el, el.classList.contains('read') ? false : true);" in script
    assert "setRead(el, true);" in script
    assert "function updateUnreadCount()" in script
    assert "document.querySelectorAll('.readable[data-read-key]')" in script
    assert "document.querySelectorAll('[data-unread-count]')" in script
    assert "updateUnreadCount();" in script


def test_dashboard_action_state_script_persists_save_dismiss_and_read_later():
    script = _dashboard_inline_script()

    assert "var ACTION_KEY='acta:actions:v1';" in script
    assert "function applyActionState(el)" in script
    assert "el.classList.toggle('saved', !!record.saved);" in script
    assert "el.classList.toggle('dismissed', !!record.dismissed);" in script
    assert "el.classList.toggle('read-later', !!record.later);" in script
    assert "button.textContent=active?labels[action].off:labels[action].on;" in script
    assert "button.setAttribute('aria-pressed', active?'true':'false');" in script
    assert "if(action==='save') record.saved=!record.saved;" in script
    assert "if(action==='dismiss') record.dismissed=!record.dismissed;" in script
    assert "if(action==='later') record.later=!record.later;" in script
    assert "button.addEventListener('click', function(ev){" in script
    assert "saveActions();" in script


def test_dashboard_action_buttons_are_signed_row_only():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    signed = item_cls(
        job_id="signed",
        name="Signed Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Valid signed detail.",
        artifact_url="https://acta.imperatr.com/r/signed/detail.html?exp=1&sig=abc",
        telegram_url="javascript:alert(1)",
    )
    unsigned = item_cls(
        job_id="unsigned",
        name="Unsigned Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_09-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 9, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Unsigned same-host detail should stay visible but not clickable.",
        artifact_url="https://acta.imperatr.com/r/unsigned/detail.html?exp=1",
    )
    unsafe = item_cls(
        job_id="unsafe",
        name="Unsafe Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_08-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 8, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Protocol-relative detail must not become a row overlay.",
        artifact_url="//evil.example/path?sig=abc",
    )
    signed_root = item_cls(
        job_id="signed-root",
        name="Signed Root Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_07-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 7, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Root-relative signed detail remains allowed.",
        artifact_url="/r/root/detail.html?exp=1&sig=root",
    )
    dotdot_path = item_cls(
        job_id="dotdot-path",
        name="Dotdot Path Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_06-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 6, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Dot-dot path segment must not become a row overlay.",
        artifact_url="/r/../file.html?exp=1&sig=abc",
    )
    dot_path = item_cls(
        job_id="dot-path",
        name="Dot Path Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_05-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 5, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Dot-only path segment must not become a row overlay.",
        artifact_url="/r/./file.html?exp=1&sig=abc",
    )
    userinfo_url = item_cls(
        job_id="userinfo-url",
        name="Userinfo URL Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_04-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 4, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Signed-looking URL with userinfo must not become a row overlay.",
        artifact_url="https://evil.com@acta.imperatr.com/r/job/file.html?exp=1&sig=abc",
    )

    html = render_dashboard(
        [signed, unsigned, unsafe, signed_root, dotdot_path, dot_path, userinfo_url],
        generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc),
    )

    assert '<div>UNREAD <b data-unread-count="2">2</b></div>' in html
    assert '<span class="nav-count" data-unread-count="2">2</span>' in html
    assert html.count('data-unread-count="2"') == 2
    assert 'data-open-url="https://acta.imperatr.com/r/signed/detail.html?exp=1&amp;sig=abc"' in html
    assert 'href="https://acta.imperatr.com/r/signed/detail.html?exp=1&amp;sig=abc"' in html
    assert 'data-open-url="/r/root/detail.html?exp=1&amp;sig=root"' in html
    assert 'href="/r/root/detail.html?exp=1&amp;sig=root"' in html
    assert "Unsigned Brief" in html
    assert "Unsafe Brief" in html
    assert "Dotdot Path Brief" in html
    assert "Dot Path Brief" in html
    assert "Userinfo URL Brief" in html
    assert "https://acta.imperatr.com/r/unsigned/detail.html?exp=1" not in html
    assert "evil.example" not in html
    assert "/r/../file.html" not in html
    assert "/r/./file.html" not in html
    assert "evil.com@acta.imperatr.com" not in html
    assert "javascript:alert" not in html
    assert "ASK TELEGRAM" not in html
    unsigned_row = next(row for row in re.findall(r'<section class="brief-row[^>]*>.*?</section>', html, re.S) if "Unsigned Brief" in row)
    unsafe_row = next(row for row in re.findall(r'<section class="brief-row[^>]*>.*?</section>', html, re.S) if "Unsafe Brief" in row)
    dotdot_row = next(row for row in re.findall(r'<section class="brief-row[^>]*>.*?</section>', html, re.S) if "Dotdot Path Brief" in row)
    dot_row = next(row for row in re.findall(r'<section class="brief-row[^>]*>.*?</section>', html, re.S) if "Dot Path Brief" in row)
    userinfo_row = next(row for row in re.findall(r'<section class="brief-row[^>]*>.*?</section>', html, re.S) if "Userinfo URL Brief" in row)
    for disabled_row in (unsigned_row, unsafe_row, dotdot_row, dot_row, userinfo_row):
        assert 'aria-disabled="true"' in disabled_row
        assert "readable" not in disabled_row
        assert "data-read-key" not in disabled_row
        assert "data-read-title" not in disabled_row
        assert "row-open-overlay" not in disabled_row
        assert "read-dot" not in disabled_row
        assert "read-state" not in disabled_row
        assert "read-toggle" not in disabled_row
        assert "state-toggle" not in disabled_row
        assert "data-state-action" not in disabled_row
        assert "MARK READ" not in disabled_row


def test_today_dashboard_gates_lead_read_state_to_signed_detail_links():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    lead = item_cls(
        job_id="lead",
        name="Unsigned Lead Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Lead has an artifact candidate but no signed Acta URL.",
        artifact_url="https://acta.imperatr.com/r/lead/detail.html?exp=1",
    )

    html = render_dashboard([lead], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))
    lead_article = re.search(r'<article class="lead"[^>]*>.*?</article>', html, re.S)

    assert lead_article
    assert 'aria-disabled="true"' in lead_article.group(0)
    assert "readable" not in lead_article.group(0)
    assert "data-read-key" not in lead_article.group(0)
    assert "data-read-title" not in lead_article.group(0)
    assert "row-open-overlay" not in lead_article.group(0)
    assert "read-dot" not in lead_article.group(0)
    assert "read-state" not in lead_article.group(0)
    assert "read-toggle" not in lead_article.group(0)
    assert "state-toggle" not in lead_article.group(0)
    assert "data-state-action" not in lead_article.group(0)
    assert "no page" not in lead_article.group(0).lower()
    assert "https://acta.imperatr.com/r/lead/detail.html?exp=1" not in html


def test_today_dashboard_allows_only_safe_absolute_telegram_followup_links():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    lead = item_cls(
        job_id="lead",
        name="Lead Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Lead with valid follow-up.",
        artifact_url="https://acta.imperatr.com/r/lead/detail.html?exp=1&sig=lead",
        telegram_url="https://t.me/c/3566991387/86",
    )
    invalid = item_cls(
        job_id="invalid-followup",
        name="Invalid Follow-up Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_09-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 9, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Invalid follow-up should not render an href.",
        artifact_url="https://acta.imperatr.com/r/invalid/detail.html?exp=1&sig=invalid",
        telegram_url="/c/3566991387/86",
    )
    unsafe = item_cls(
        job_id="unsafe-followup",
        name="Unsafe Follow-up Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_08-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 8, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Unsafe follow-up should not render an href.",
        artifact_url="https://acta.imperatr.com/r/unsafe/detail.html?exp=1&sig=unsafe",
        telegram_url="javascript:alert(1)",
    )
    subdomain = item_cls(
        job_id="subdomain-followup",
        name="Subdomain Follow-up Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_07-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 7, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Telegram subdomain should not render an href.",
        artifact_url="https://acta.imperatr.com/r/subdomain/detail.html?exp=1&sig=subdomain",
        telegram_url="https://evil.t.me/c/3566991387/86",
    )

    html = render_dashboard([lead, invalid, unsafe, subdomain], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))

    assert 'href="https://t.me/c/3566991387/86"' in html
    assert "Invalid Follow-up Brief" in html
    assert "Unsafe Follow-up Brief" in html
    assert "Subdomain Follow-up Brief" in html
    assert 'href="/c/3566991387/86"' not in html
    assert "javascript:alert" not in html
    assert "evil.t.me" not in html


def test_disabled_today_rows_do_not_get_keyboard_open_overlay():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    disabled = item_cls(
        job_id="disabled",
        name="Disabled Brief",
        schedule="daily",
        deliver="telegram",
        enabled=False,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Paused source with an old artifact.",
        artifact_url="https://acta.example/disabled.html",
    )

    html = render_dashboard([disabled], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))

    assert 'aria-disabled="true"' in html
    assert 'data-open-url="https://acta.example/disabled.html"' not in html
    assert 'href="https://acta.example/disabled.html"' not in html
    assert 'aria-label="Open briefing: Disabled Brief"' not in html


def test_read_state_has_cookie_fallback_and_applies_on_open(tmp_path: Path):
    (tmp_path / "cron" / "output" / "lead").mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(
        json.dumps([{"id": "lead", "name": "Lead Brief", "deliver": "telegram"}])
    )
    (tmp_path / "cron" / "output" / "lead" / "2026-05-19_10-00-00.md").write_text("## Response\n\nMost important")

    html = render_dashboard(collect_situation_items(tmp_path))

    assert "COOKIE='acta_read_v1'" in html
    assert "document.cookie" in html
    assert "readFromCookie" in html
    assert "writeToCookie" in html
    assert "function setRead(el, value)" in html
    assert "setRead(el, true);" in html


def test_dashboard_surfaces_confidence_without_changing_read_state():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    now = datetime(2026, 5, 19, 12, tzinfo=timezone.utc)
    lead = item_cls(
        job_id="lead",
        name="Lead Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_11-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 11, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Recent high-confidence signal.",
        artifact_url="https://acta.imperatr.com/r/lead/detail.html?exp=1&sig=lead",
    )
    older = item_cls(
        job_id="older",
        name="Older Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-17_11-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 17, 11, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Older still-visible signal.",
        artifact_url="https://acta.imperatr.com/r/older/detail.html?exp=1&sig=older",
    )
    silent = item_cls(
        job_id="silent",
        name="Silent Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="silent",
        excerpt="No visible response was produced.",
    )

    html = render_dashboard([lead, older, silent], generated_at=now)

    assert '<span class="read-state">UNREAD</span>' in html
    assert 'data-read-key="lead:' in html
    assert 'data-read-key="older:' in html
    assert "CONF HIGH" in html
    assert "CONF MED" in html
    assert "CONF LOW/GAP" in html
    assert '<span class="trace-only confidence-chip">CONF MED</span>' in html
    assert '<span>OUT</span>' in html
    assert "#f5a400" not in html
    assert "amber" not in html.lower()
    assert "generated-file" not in html.lower()


def test_dashboard_includes_pull_to_refresh_affordance(tmp_path: Path):
    (tmp_path / "cron" / "output" / "lead").mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(
        json.dumps([{"id": "lead", "name": "Lead Brief", "deliver": "telegram"}])
    )
    (tmp_path / "cron" / "output" / "lead" / "2026-05-19_10-00-00.md").write_text("## Response\n\nMost important")

    html = render_dashboard(collect_situation_items(tmp_path))

    assert "pull-refresh" in html
    assert "PULL TO REFRESH" in html
    assert "location.reload()" in html


def test_dashboard_exposes_outputs_as_primary_module(tmp_path: Path):
    (tmp_path / "cron" / "output" / "lead").mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(json.dumps([{"id": "lead", "name": "Lead Brief"}]))
    (tmp_path / "cron" / "output" / "lead" / "2026-05-19_10-00-00.md").write_text("## Response\n\nMost important")

    html = render_dashboard(collect_situation_items(tmp_path))

    assert 'href="/outputs"' in html
    assert ">Outputs<" in html or ">OUTPUTS<" in html


def test_mobile_bottom_nav_appears_only_after_top_nav_scrolls_out(tmp_path: Path):
    (tmp_path / "cron" / "output" / "lead").mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(
        json.dumps([{"id": "lead", "name": "Lead Brief", "deliver": "telegram"}])
    )
    (tmp_path / "cron" / "output" / "lead" / "2026-05-19_10-00-00.md").write_text("## Response\n\nMost important")

    html = render_dashboard(collect_situation_items(tmp_path))

    assert '<nav class="date-nav"' in html
    assert '<nav class="mobilebar"><a href="/">TODAY <span class="nav-count" data-unread-count="0">0</span></a><a href="/outputs">OUTPUTS</a><a href="/runs">RUNS</a><a href="/jobs">JOBS</a><a href="/archive">ARCHIVE</a></nav>' in html
    assert '.mobilebar.visible' in html
    assert 'IntersectionObserver' in html
    assert "document.querySelector('.date-nav')" in html
    assert 'background:rgba(2,2,2,.96)' not in html
    assert 'background:linear-gradient(180deg, rgba(7,16,24,.96), rgba(3,6,11,.94)), radial-gradient(circle at 18% 0%, rgba(117,108,255,.28), transparent 42%), radial-gradient(circle at 86% 20%, rgba(35,167,255,.18), transparent 48%)' in html


def test_archive_day_dashboard_has_read_and_action_parity_with_today():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    archive_item = item_cls(
        job_id="daily",
        name="Archived Morning Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_09-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 9, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Archived source-backed briefing.",
        artifact_url="https://acta.imperatr.com/r/daily/detail.html?exp=1&sig=daily",
        telegram_url="https://t.me/imperatr/123",
    )

    html = render_dashboard(
        [archive_item],
        generated_at=datetime(2026, 5, 20, 12, tzinfo=timezone.utc),
        selected_date=date(2026, 5, 19),
        archive_dates=[date(2026, 5, 20), date(2026, 5, 19)],
        archive_day=True,
    )

    assert "ACTA</em> / ARCHIVE DAY" in html
    assert "Day Brief" in html
    assert '<a class="nav-link primary" href="/archive/2026-05-19">Archive day</a>' in html
    assert 'data-read-key="daily:2026-05-19T09:00:00+00:00"' in html
    assert '<span class="read-state">UNREAD</span>' in html
    assert '<button class="read-toggle" type="button" aria-label="Mark briefing read: Archived Morning Brief">Mark read</button>' in html
    assert 'data-state-action="save" aria-label="Save briefing: Archived Morning Brief">Save</button>' in html
    assert 'data-state-action="dismiss" aria-label="Dismiss briefing: Archived Morning Brief">Dismiss</button>' in html
    assert 'data-state-action="later" aria-label="Read later: Archived Morning Brief">Read later</button>' in html
    assert 'class="row-open-overlay" href="https://acta.imperatr.com/r/daily/detail.html?exp=1&amp;sig=daily"' in html
    assert 'class="ask-label" href="https://t.me/imperatr/123"' in html
    assert '<a class="active" href="/archive">ARCHIVE' in html


def test_dashboard_surfaces_production_release_tldr_and_input_needed():
    release_cls = collect_situation_items.__globals__["ActaReleaseNote"]
    release = release_cls(
        date="2026-05-26",
        title="Acta operator feed cleanup",
        shipped=["Split daily feed from dev sprint cycles", "Added real browser UAT publish gate"],
        decisions=["ASK remains secondary", "Today’s Brief becomes Hermes-agent updated"],
        needs_input=["Choose first persistent action state"],
    )

    html = render_dashboard([], generated_at=datetime(2026, 5, 26, tzinfo=timezone.utc), release_notes=[release])

    assert "Release TLDR" in html
    assert "Acta operator feed cleanup" in html
    assert "Split daily feed from dev sprint cycles" in html
    assert "Added real browser UAT publish gate" in html
    assert "Decisions locked" in html
    assert "ASK remains secondary" in html
    assert "Today’s Brief becomes Hermes-agent updated" in html
    assert "Needs your input" in html
    assert "Choose first persistent action state" in html


def test_release_log_loader_sanitizes_and_limits_operator_notes(tmp_path: Path):
    path = tmp_path / "acta-release-log.json"
    path.write_text(
        json.dumps(
            [
                {
                    "date": "2026-05-25",
                    "title": "Older Acta release",
                    "shipped": ["Old note should sort behind latest"],
                    "decisions": [],
                    "needs_input": [],
                },
                {
                    "date": "2026-05-26",
                    "title": "<Acta release>",
                    "shipped": [
                        "One",
                        "Published https://acta.imperatr.com/r/daily/detail.html?exp=1&sig=secret",
                        "AWS https://s3.amazonaws.com/bucket/key?X-Amz-Signature=secret&X-Amz-Credential=cred&X-Amz-Security-Token=tok",
                        "Local file /tmp/private/secret.md and ~/.hermes/private.md with OPENAI_API_KEY=openaisecretvalue AWS_SECRET_ACCESS_KEY=awssecretvalue",
                        "## Prompt leak should be dropped",
                        "Prompt: another trace should be dropped",
                        "Five",
                    ],
                    "needs_input": ["Pick primary action", "terminal command: cat secrets", "Third", "Fourth"],
                    "decisions": ["ASK stays secondary", "Today brief uses Hermes agent", "Third", "Fourth"],
                }
            ]
        ),
        encoding="utf-8",
    )

    releases = load_release_log(path)

    assert len(releases) == 2
    assert releases[0].date == "2026-05-26"
    assert releases[0].title == "<Acta release>"
    assert releases[0].shipped == [
        "One",
        "Published [redacted link]",
        "AWS [redacted link]",
        "Local file [local path removed] and [local path removed] with OPENAI_API_KEY=[redacted] AWS_SECRET_ACCESS_KEY=[redacted]",
    ]
    assert releases[0].needs_input == ["Pick primary action", "Third", "Fourth"]
    assert releases[0].decisions == ["ASK stays secondary", "Today brief uses Hermes agent", "Third"]

    html = render_dashboard([], generated_at=datetime(2026, 5, 26, tzinfo=timezone.utc), release_notes=releases)
    assert "&lt;Acta release&gt;" in html
    assert "https://acta.imperatr.com/r/daily/detail.html" not in html
    assert "/Users/mozzie" not in html
    assert "supersecretvalue" not in html
    assert "supersecrettokenvalue" not in html
    assert "openaisecretvalue" not in html
    assert "awssecretvalue" not in html
    assert "Prompt leak" not in html
    assert "another trace" not in html
    assert "terminal command" not in html
    assert "ASK stays secondary" in html
    assert "[redacted link]" in html
    assert "[local path removed]" in html


def test_detail_report_uses_acta_situation_room_ui():
    html = render_acta_detail_report(
        "# Market Brief\n\nImportant signal.",
        {"job_id": "job1", "job_name": "Market Brief", "run_time": "2026-05-19T10:00:00+00:00"},
        detail_signals={"signed status": "fresh", "signed conf": "CONF HIGH", "signed age": "60m ago"},
    )

    assert "Acta Situation Room" in html
    assert 'href="/outputs"' in html
    assert ">Back<" in html
    assert "width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover, user-scalable=no" in html
    assert "--black:#03060b" in html
    assert "--accent:#756cff" in html
    assert "Important signal." in html
    assert "<b>SIGNED STATUS</b> fresh" in html
    assert "<b>SIGNED CONF</b> CONF HIGH" in html
    assert "<b>SIGNED AGE</b> 60m ago" in html


def test_detail_report_escapes_optional_operator_signals():
    html = render_acta_detail_report(
        "# Safe Brief\n\nVisible signal.",
        {"job_id": "job1", "job_name": "Safe Brief", "run_time": "2026-05-19T10:00:00+00:00"},
        detail_signals={"<script>bad()</script>": "<img src=x onerror=alert(1)>"},
    )

    assert "<script>bad" not in html
    assert "<img src=x" not in html
    assert "SCRIPTBAD/SCRIPT" in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


def test_detail_signal_helpers_are_source_derived_and_time_scoped():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    run_cls = collect_run_history.__globals__["ActaRunItem"]
    now = datetime(2026, 5, 20, 12, tzinfo=timezone.utc)
    cron_item = item_cls(
        job_id="daily",
        name="Daily Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("2026-05-20_11-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 20, 11, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Visible.",
    )
    stale_run = run_cls(
        job_id="daily",
        name="Daily Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        run_id="2026-05-18_08-00-00",
        run_time=datetime(2026, 5, 18, 8, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Visible.",
        source_name="2026-05-18_08-00-00.md",
        has_markdown=True,
        has_html=False,
    )

    assert render_acta_detail_report.__globals__["_cron_detail_signals"](cron_item, now) == {
        "signed status": "fresh",
        "signed conf": "CONF HIGH",
        "signed age": "60m ago",
    }
    assert render_acta_detail_report.__globals__["_run_detail_signals"](stale_run, now) == {
        "signed status": "fresh",
        "signed conf": "CONF MED",
        "signed age": "2d ago",
    }


def test_attach_artifact_urls_reprocesses_markdown_into_acta_detail_ui(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "details"
    md = tmp_path / "2026-05-19_10-00-00.md"
    stale_html = tmp_path / "2026-05-19_10-00-00.html"
    md.write_text("## Response\n\n# Fresh Markdown\n\nUseful briefing")
    stale_html.write_text("<html><body>stale old ui</body></html>")
    item = collect_situation_items.__globals__["CronSituationItem"](
        job_id="job1",
        name="Daily Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=md,
        latest_html=stale_html,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Useful briefing",
    )
    uploaded = {}

    def fake_publish(path, job, settings):
        uploaded["html"] = Path(path).read_text()
        uploaded["path"] = Path(path)
        return "https://acta.example/r/job1/detail.html"

    monkeypatch.setattr("cron.acta_dashboard.publish_html_artifact", fake_publish)

    linked = attach_artifact_urls([item], {"enabled": True}, output_dir)

    assert linked[0].artifact_url == "https://acta.example/r/job1/detail.html"
    assert uploaded["path"].parent == output_dir
    assert "Acta Situation Room" in uploaded["html"]
    assert "Fresh Markdown" in uploaded["html"]
    assert "<b>SIGNED STATUS</b> fresh" in uploaded["html"]
    assert "<b>SIGNED CONF</b> CONF" in uploaded["html"]
    assert "<b>SIGNED AGE</b>" in uploaded["html"]
    assert "stale old ui" not in uploaded["html"]


def test_attach_artifact_urls_wraps_html_only_artifact_in_acta_detail_ui(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "details"
    html_artifact = tmp_path / "2026-05-19_10-00-00.html"
    html_artifact.write_text(
        """
        <html><head><style>:root { --accent:#f5a400; }</style></head>
        <body><h1>Generated files</h1><p>Useful HTML-only signal.</p><p>Bloomberg terminal chrome</p></body></html>
        """
    )
    item = collect_situation_items.__globals__["CronSituationItem"](
        job_id="job1",
        name="Daily Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=None,
        latest_html=html_artifact,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Useful HTML-only signal.",
        telegram_url="https://t.me/c/3566991387/86",
    )
    uploaded = {}

    def fake_publish(path, job, settings):
        uploaded["html"] = Path(path).read_text()
        uploaded["path"] = Path(path)
        return "https://acta.example/r/job1/detail.html"

    monkeypatch.setattr("cron.acta_dashboard.publish_html_artifact", fake_publish)

    linked = attach_artifact_urls([item], {"enabled": True}, output_dir)

    assert linked[0].artifact_url == "https://acta.example/r/job1/detail.html"
    assert uploaded["path"].parent == output_dir
    assert uploaded["path"].name == "job1-2026-05-19_10-00-00.html"
    assert "Acta Situation Room" in uploaded["html"]
    assert "--black:#03060b" in uploaded["html"]
    assert "--accent:#756cff" in uploaded["html"]
    assert "Useful HTML-only signal." in uploaded["html"]
    assert "Ask follow-up in Telegram" in uploaded["html"]
    assert "#f5a400" not in uploaded["html"]
    assert "Generated files" not in uploaded["html"]
    assert "Bloomberg" not in uploaded["html"]


def test_dashboard_links_jobs_to_dedicated_subpage(tmp_path: Path):
    (tmp_path / "cron" / "output" / "active").mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(
        json.dumps([{"id": "active", "name": "Active Brief", "schedule": "0 8 * * *", "deliver": "telegram", "enabled": True}])
    )
    (tmp_path / "cron" / "output" / "active" / "2026-05-19_08-00-00.md").write_text("## Response\n\nActive")

    html = render_dashboard(collect_situation_items(tmp_path))

    assert 'href="/jobs"' in html
    assert 'href="#jobs"' not in html
    assert 'id="jobs"' not in html
    assert "Active Cron Jobs" not in html


def test_archive_index_includes_sticky_navigation_bar():
    html = render_archive_index([date(2026, 5, 20)], generated_at=datetime(2026, 5, 20, tzinfo=timezone.utc))

    assert '<header class="top">' in html
    assert '<a class="ticker" href="/"><em>ACTA</em> / ARCHIVE</a>' in html
    assert '<nav class="nav">' in html
    assert '<a href="/">Today</a>' in html
    assert '<a href="/outputs">Outputs</a>' in html
    assert '<a href="/runs">Runs</a>' in html
    assert '<a href="/jobs">Jobs</a>' in html
    assert '<a class="active" href="/archive">Archive</a>' in html
    assert "position:sticky" in html


def test_archive_index_cards_show_source_signal_summaries_without_fake_tokens():
    summary_cls = collect_situation_items.__globals__["ArchiveDaySummary"]
    day = date(2026, 5, 20)
    summary = summary_cls(
        day=day,
        visible=2,
        silent=1,
        missing=1,
        latest_title="Morning Operator Brief <latest>",
        lane_counts={"daily": 1, "dev": 2, "system": 1},
    )

    html = render_archive_index(
        [day],
        generated_at=datetime(2026, 5, 20, tzinfo=timezone.utc),
        summaries={day: summary},
    )

    assert 'class="archive-card" href="/archive/2026-05-20"' in html
    assert "Visible 2 · Silent 1 · Missing 1" in html
    assert "Latest: Morning Operator Brief &lt;latest&gt;" in html
    assert "Daily 1 · Dev 2 · System 1" in html
    assert "P1" not in html
    assert "P2" not in html
    assert "Acta Day" not in html
    assert "score" not in html.lower()


def test_archive_day_summary_uses_latest_timestamp_title_not_first_item():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    archive_day_summary = collect_situation_items.__globals__["archive_day_summary"]
    older = item_cls(
        job_id="older",
        name="Older Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/older.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 20, 8, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Older signal",
    )
    newer = item_cls(
        job_id="newer",
        name="Newer Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/newer.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 20, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Newer signal",
    )

    summary = archive_day_summary(date(2026, 5, 20), [older, newer])

    assert summary.latest_title == "Newer Brief"


def test_archive_day_summary_falls_back_to_first_item_when_no_timestamps():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    archive_day_summary = collect_situation_items.__globals__["archive_day_summary"]
    first = item_cls(
        job_id="first",
        name="First Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/first.md"),
        latest_html=None,
        latest_time=None,
        status="fresh",
        excerpt="First signal",
    )
    second = item_cls(
        job_id="second",
        name="Second Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/second.md"),
        latest_html=None,
        latest_time=None,
        status="fresh",
        excerpt="Second signal",
    )

    summary = archive_day_summary(date(2026, 5, 20), [first, second])

    assert summary.latest_title == "First Brief"


def test_all_acta_modules_share_compact_v9_shell():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    item = item_cls(
        job_id="lead",
        name="Lead Brief",
        schedule="daily",
        deliver="telegram:-1003566991387:86",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Most important signed briefing.",
        artifact_url="https://acta.imperatr.com/r/lead/detail.html?exp=1&sig=abc",
        telegram_url="https://t.me/c/3566991387/86",
    )
    pages = {
        "today": render_dashboard([item], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc)),
        "jobs": render_jobs_page([item], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc)),
        "archive": render_archive_index([date(2026, 5, 19)], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc)),
        "outputs": render_outputs_page([item], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc)),
        "detail": render_acta_detail_report(
            "# Lead Brief\n\nSignal body.",
            {"job_id": "lead", "job_name": "Lead Brief", "run_time": "2026-05-19T10:00:00+00:00"},
            telegram_url="https://t.me/c/3566991387/86",
        ),
    }

    for name, html in pages.items():
        assert "#03060b" in html, name
        assert "#756cff" in html, name
        assert "#23a7ff" in html, name
        assert "font:720 clamp" in html, name
        assert "font:600 clamp(38px,6.5vw,70px)" not in html, name
        assert "font:600 clamp(34px,6vw,64px)" not in html, name
        assert "background:linear-gradient(135deg,rgba(117,108,255,.10),rgba(255,255,255,.04))" not in html, name
        assert "#f5a400" not in html, name
        assert "amber" not in html.lower(), name
        assert "Bloomberg" not in html, name
        assert "Generated files" not in html, name
        assert "Your cron command center" not in html, name


def test_secondary_acta_surfaces_include_mobile_module_nav_without_stale_markers():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    item = item_cls(
        job_id="lead",
        name="Lead Brief",
        schedule="daily",
        deliver="telegram:-1003566991387:86",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Most important signed briefing.",
        artifact_url="https://acta.imperatr.com/r/lead/detail.html?exp=1&sig=abc",
        telegram_url="https://t.me/c/3566991387/86",
    )
    pages = {
        "jobs": (render_jobs_page([item], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc)), "jobs"),
        "archive": (render_archive_index([date(2026, 5, 19)], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc)), "archive"),
        "outputs": (render_outputs_page([item], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc)), "outputs"),
        "detail": (
            render_acta_detail_report(
                "# Lead Brief\n\nSignal body.",
                {"job_id": "lead", "job_name": "Lead Brief", "run_time": "2026-05-19T10:00:00+00:00"},
                telegram_url="https://t.me/c/3566991387/86",
            ),
            "outputs",
        ),
    }
    expected = {
        "TODAY": "/",
        "OUTPUTS": "/outputs",
        "RUNS": "/runs",
        "JOBS": "/jobs",
        "ARCHIVE": "/archive",
    }
    active_link = {
        "jobs": '<a class="active" href="/jobs">JOBS</a>',
        "archive": '<a class="active" href="/archive">ARCHIVE</a>',
        "outputs": '<a class="active" href="/outputs">OUTPUTS</a>',
    }

    for name, (html, active) in pages.items():
        assert '<nav class="mobilebar" aria-label="Acta mobile module navigation">' in html, name
        for label, href in expected.items():
            assert f'href="{href}"' in html and f'>{label}</a>' in html, name
        assert active_link[active] in html, name
        assert ".mobilebar" in html, name
        assert "#f5a400" not in html, name
        assert "amber" not in html.lower(), name
        assert "generic-dashboard" not in html.lower(), name


def test_jobs_subpage_shows_active_relevant_last_runs(tmp_path: Path):
    for job_id in ("active", "silent", "disabled", "hidden"):
        (tmp_path / "cron" / "output" / job_id).mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(
        json.dumps(
            [
                {"id": "active", "name": "Active Brief", "schedule": {"display": "0 8 * * *"}, "deliver": "telegram", "enabled": True},
                {"id": "silent", "name": "Silent But Active", "schedule": "15 9 * * *", "deliver": "telegram", "enabled": True},
                {"id": "disabled", "name": "Disabled Brief", "schedule": "0 1 * * *", "deliver": "telegram", "enabled": False},
                {"id": "hidden", "name": "Hidden Brief", "schedule": "0 2 * * *", "deliver": "telegram", "enabled": True},
            ]
        )
    )
    (tmp_path / "cron" / "output" / "active" / "2026-05-19_08-00-00.md").write_text("## Response\n\nActive")
    (tmp_path / "cron" / "output" / "silent" / "2026-05-19_09-00-00.md").write_text("## Response\n\n[SILENT]")
    (tmp_path / "cron" / "output" / "disabled" / "2026-05-19_01-00-00.md").write_text("## Response\n\nDisabled")
    (tmp_path / "cron" / "output" / "hidden" / "2026-05-19_02-00-00.md").write_text("## Response\n\nHidden")

    html = render_jobs_page(
        collect_situation_items(tmp_path),
        generated_at=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        feed_preferences={"hidden": ["hidden"], "show_system": False},
    )

    assert "Acta Jobs" in html
    assert "Active Cron Jobs" in html
    assert "Active Brief" in html
    assert "Silent But Active" in html
    assert "0 8 * * *" in html
    assert "15 9 * * *" in html
    assert "LAST RUN" in html
    assert "2026-05-19T08" in html
    assert "SOURCE 2026-05-19_08-00-00.md" in html
    assert "SOURCE 2026-05-19_09-00-00.md" in html
    assert '<span class="confidence-chip">CONF HIGH</span>' in html
    assert '<span class="confidence-chip">CONF LOW/GAP</span>' in html
    assert "<span>fresh</span>" in html
    assert "<span>silent</span>" in html
    assert "Disabled Brief" not in html
    assert "Hidden Brief" not in html


def test_jobs_subpage_source_provenance_escapes_filename_only():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    item = item_cls(
        job_id="escaped",
        name="Escaped Source Brief",
        schedule="daily",
        deliver="local",
        enabled=True,
        latest_md=Path('/tmp/private/<img src=x onerror=alert(1)>.md'),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Escaped filename.",
    )

    html = render_jobs_page([item], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))

    assert "/tmp/private" not in html
    assert "SOURCE &lt;img src=x onerror=alert(1)&gt;.md" in html
    assert "<img src=x" not in html


def _jobs_row_html(html_doc: str, name: str) -> str:
    rows = re.findall(
        r'<div class="job-row\b.*?(?=\n<div class="job-row\b|\n\s*</section>)',
        html_doc,
        re.S,
    )
    return next(row for row in rows if name in row)


def test_jobs_subpage_suppresses_unsafe_thread_links():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    valid = item_cls(
        job_id="valid",
        name="Valid Thread Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Valid thread.",
        telegram_url="https://t.me/c/3566991387/86",
    )
    unsafe = item_cls(
        job_id="unsafe",
        name="Unsafe Thread Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_09-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 9, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Unsafe thread.",
        telegram_url="javascript:alert(1)",
    )
    subdomain = item_cls(
        job_id="subdomain",
        name="Subdomain Thread Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_08-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 8, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Subdomain thread.",
        telegram_url="https://evil.t.me/c/3566991387/86",
    )
    no_thread = item_cls(
        job_id="no-thread",
        name="No Thread Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_07-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 7, tzinfo=timezone.utc),
        status="fresh",
        excerpt="No thread.",
        telegram_url=None,
    )

    html = render_jobs_page([valid, unsafe, subdomain, no_thread], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))

    assert 'href="https://t.me/c/3566991387/86"' in html
    assert "Unsafe Thread Brief" in html
    assert "Subdomain Thread Brief" in html
    assert "No Thread Brief" in html
    assert "javascript:alert" not in html
    assert "evil.t.me" not in html
    valid_row = _jobs_row_html(html, "Valid Thread Brief")
    assert 'class="job-row fresh no-page" data-open-state="no-page"' in valid_row
    assert 'href="https://t.me/c/3566991387/86"' in valid_row
    assert "THREAD" in valid_row
    assert 'aria-disabled="true"' not in valid_row
    assert "data-open-url=" not in valid_row
    assert "row-open-overlay" not in valid_row
    assert '<span class="job-open-state muted">NO PAGE</span>' in valid_row
    unsafe_row = _jobs_row_html(html, "Unsafe Thread Brief")
    no_thread_row = _jobs_row_html(html, "No Thread Brief")
    assert 'aria-disabled="true"' in unsafe_row
    assert 'aria-disabled="true"' in no_thread_row


def test_jobs_subpage_opens_safe_signed_artifact_and_preserves_thread_link():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    item = item_cls(
        job_id="lead",
        name="Lead Brief",
        schedule="daily",
        deliver="telegram:-1003566991387:86",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Most important signed briefing.",
        artifact_url="https://acta.imperatr.com/r/lead/detail.html?exp=1&sig=abc",
        telegram_url="https://t.me/c/3566991387/86",
    )

    html = render_jobs_page([item], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))

    assert 'class="job-row fresh openable" data-open-url="https://acta.imperatr.com/r/lead/detail.html?exp=1&amp;sig=abc"' in html
    assert '<a class="row-open-overlay job-open-overlay" href="https://acta.imperatr.com/r/lead/detail.html?exp=1&amp;sig=abc"' in html
    assert '<span class="job-open-state signed">OPEN/SIGNED</span>' in html
    assert '<span class="confidence-chip">CONF HIGH</span>' in html
    assert "<span>fresh</span>" in html
    assert "<span>60m ago</span>" in html
    assert 'href="https://t.me/c/3566991387/86"' in html
    assert "THREAD" in html
    assert ".job-open-overlay { position:absolute; inset:0; z-index:2; border:0; text-decoration:none; }" in html
    assert ".job-row > :not(.row-open-overlay) { position:relative; }" in html
    assert ".job-main .thread-link { position:relative; z-index:3; pointer-events:auto; }" in html


def test_jobs_subpage_rejects_unsafe_or_unsigned_artifact_urls():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    signed = item_cls(
        job_id="signed",
        name="Signed Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Signed.",
        artifact_url="/r/signed/detail.html?exp=1&sig=abc",
    )
    unsigned = item_cls(
        job_id="unsigned",
        name="Unsigned Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_09-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 9, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Unsigned.",
        artifact_url="https://acta.imperatr.com/r/unsigned/detail.html?exp=1",
    )
    unsafe = item_cls(
        job_id="unsafe",
        name="Unsafe Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_08-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 8, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Unsafe.",
        artifact_url="javascript:alert(1)",
    )

    html = render_jobs_page([signed, unsigned, unsafe], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))

    assert 'data-open-url="/r/signed/detail.html?exp=1&amp;sig=abc"' in html
    assert 'href="/r/signed/detail.html?exp=1&amp;sig=abc"' in html
    assert "javascript:alert" not in html
    assert html.count('data-open-url=') == 1
    assert html.count('class="row-open-overlay job-open-overlay"') == 1
    assert html.count('<span class="job-open-state muted">NO PAGE</span>') == 2
    assert "https://acta.imperatr.com/r/unsigned/detail.html" not in html


def test_outputs_page_uses_v9_shell_and_signed_source_rows():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    item = item_cls(
        job_id="lead",
        name="Lead Brief",
        schedule="daily",
        deliver="telegram:-1003566991387:86",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Most important signed briefing.",
        artifact_url="https://acta.imperatr.com/r/lead/detail.html?exp=1&sig=abc",
        telegram_url="https://t.me/c/3566991387/86",
    )

    html = render_outputs_page([item], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))

    assert "Acta Outputs" in html
    assert "#03060b" in html
    assert "#756cff" in html
    assert "#23a7ff" in html
    assert '<a class="active" href="/outputs">Outputs</a>' in html
    assert 'href="https://acta.imperatr.com/r/lead/detail.html?exp=1&amp;sig=abc"' in html
    assert 'data-open-url="https://acta.imperatr.com/r/lead/detail.html?exp=1&amp;sig=abc"' in html
    assert '<article class="output-row readable unread fresh" data-read-key="output:lead:2026-05-19T10:00:00+00:00"' in html
    assert '<span class="read-state">UNREAD</span>' in html
    assert '<div class="stat">Unread <b data-unread-count="1">1</b></div>' in html
    assert "document.querySelectorAll('.output-row.readable')" in html
    assert "el.querySelectorAll('.output-open-overlay')" in html
    assert "setRead(el, true);" in html
    assert "script-src 'sha256-" in html
    assert "script-src 'unsafe-inline'" not in html
    assert "COOKIE='acta_read_v1'" in html
    assert '<a class="output-open-overlay" href="https://acta.imperatr.com/r/lead/detail.html?exp=1&amp;sig=abc"' in html
    assert '<span class="open">SIGNED</span>' in html
    assert "Open signed" not in html
    assert ".output-row[data-open-url] { cursor:pointer; }" in html
    assert ".output-open-overlay { position:absolute; inset:0; z-index:1; border:0; text-decoration:none; }" in html
    assert 'href="https://t.me/c/3566991387/86"' in html
    assert "ASK" in html
    assert (
        '<a class="followup-meta" href="https://t.me/c/3566991387/86" target="_blank" rel="noopener" '
        'aria-label="Ask follow-up in Telegram" title="Ask follow-up in Telegram">FOLLOW-UP</a>'
        in html
    )
    assert "CONF HIGH" in html
    assert '<span class="confidence-chip">CONF HIGH</span>' in html
    assert "fresh" in html
    assert "daily" in html
    assert "lead" in html
    assert "2026-05-19T10:00:00+00:00" in html
    assert "SOURCE" in html
    assert ".output-meta { flex-wrap:wrap; overflow:visible; }" in html
    assert ".output-meta span:nth-child" not in html
    assert ".output-meta .followup-meta { display:inline-flex; }" in html
    assert "#f5a400" not in html
    assert "Bloomberg" not in html
    assert "generated-file" not in html.lower()
    assert "Generated files" not in html


def test_outputs_page_gates_unsafe_links_and_read_state_to_signed_rows():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    unsafe = item_cls(
        job_id='unsafe" onclick="bad',
        name="Unsafe Brief",
        schedule="daily",
        deliver="telegram:-1003566991387:86",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Unsafe URLs should not become clickable.",
        artifact_url="javascript:alert(1)",
        telegram_url="javascript:alert(2)",
    )

    html = render_outputs_page([unsafe], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))

    assert "javascript:alert" not in html
    assert 'class="output-row fresh" aria-disabled="true"' in html
    assert 'data-read-key="output:unsafe' not in html
    assert '<span class="read-state">UNREAD</span>' not in html
    assert '<span class="muted">No signed link</span>' in html
    assert '<div class="stat">Unread <b data-unread-count="0">0</b></div>' in html
    assert "NO FOLLOW-UP" in html
    assert 'onclick="bad' not in html


def test_outputs_page_rejects_protocol_relative_and_suspicious_artifact_urls():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    for url in (
        "//evil.example/path?sig=abc",
        "///evil.example/path?sig=abc",
        "/\\evil?sig=abc",
        "/ok\npath?sig=abc",
        "/r/job/file.txt?exp=1&sig=abc",
        "/r/../../public/index.html?exp=1&sig=abc",
        "/r/job/file.html?sig=abc",
    ):
        item = item_cls(
            job_id="unsafe",
            name="Unsafe Brief",
            schedule="daily",
            deliver="telegram",
            enabled=True,
            latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
            latest_html=None,
            latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
            status="fresh",
            excerpt="Unsafe URL should not become clickable.",
            artifact_url=url,
        )

        html = render_outputs_page([item], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))

        assert 'class="output-row fresh" aria-disabled="true"' in html
        assert 'data-read-key="output:unsafe' not in html
        assert '<span class="read-state">UNREAD</span>' not in html
        assert '<span class="muted">No signed link</span>' in html
        assert "evil.example" not in html
        assert "file.txt" not in html
        assert "../../public" not in html
        assert "file.html?sig=abc" not in html


def test_outputs_page_requires_signed_acta_artifact_url_for_read_state():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    unsigned = item_cls(
        job_id="unsigned",
        name="Unsigned Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Same-host but unsigned.",
        artifact_url="https://acta.imperatr.com/r/lead/detail.html?exp=1",
    )
    signed_root = item_cls(
        job_id="signed-root",
        name="Signed Root Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Root-relative and signed.",
        artifact_url="/r/lead/detail.html?exp=1&sig=abc",
    )

    html = render_outputs_page([unsigned, signed_root], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))

    assert 'href="https://acta.imperatr.com/r/lead/detail.html?exp=1"' not in html
    assert 'data-read-key="output:unsigned' not in html
    assert 'data-open-url="/r/lead/detail.html?exp=1&amp;sig=abc"' in html
    assert 'data-read-key="output:signed-root' in html
    assert "Signed <b>1</b>" in html


def test_catalog_outputs_import_static_acta_outputs_and_render_persistent_rows(tmp_path: Path):
    outputs = tmp_path / "artifacts" / "acta-outputs"
    outputs.mkdir(parents=True)
    (outputs / "index.html").write_text(
        """
        <article data-id="hermes-agent-lanes-decision-tree" data-href="/outputs/hermes-agent-lanes-decision-tree"
          data-title="Hermes Agent Lanes & Specialist Agents" data-tags="hermes agents decision tree">
          <p>Visual decision tree for when to use Telegram topic lanes and specialist agents.</p>
          <div class="meta">Published 2026-05-24 16:49 UTC</div>
        </article>
        """,
        encoding="utf-8",
    )
    (outputs / "hermes-agent-lanes-decision-tree.html").write_text("<html><title>Fallback</title><body><p>Fallback.</p></body></html>", encoding="utf-8")

    catalog_items = collect_catalog_outputs(tmp_path)
    html = render_catalog_outputs_page(catalog_items, generated_at=datetime(2026, 5, 24, 17, tzinfo=timezone.utc))

    assert [item.id for item in catalog_items] == ["hermes-agent-lanes-decision-tree"]
    assert "Hermes Agent Lanes &amp; Specialist Agents" in html
    assert 'href="/outputs/hermes-agent-lanes-decision-tree"' in html
    assert "Visual decision tree" in html
    assert str(tmp_path) not in html
    assert "latest cron" not in html.lower()
    assert '<a class="active" href="/outputs">Outputs</a>' in html
    assert 'href="/runs"' in html
    assert "script-src 'sha256-" in html


def test_collect_catalog_outputs_preserves_unsafe_and_missing_catalog_hrefs_as_disabled_rows(tmp_path: Path):
    catalog_path = tmp_path / "acta" / "catalog.json"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(
        json.dumps(
            {
                "version": 1,
                "outputs": [
                    {
                        "id": "unsafe-js",
                        "title": "Unsafe JS Output",
                        "href": "javascript:alert(1)",
                        "summary": "Unsafe href must render disabled.",
                        "updated_at": "2026-05-24T16:00:00+00:00",
                    },
                    {
                        "id": "missing-href",
                        "title": "Missing Href Output",
                        "summary": "Missing href must render disabled.",
                        "updated_at": "2026-05-24T16:01:00+00:00",
                    },
                    {
                        "id": "valid-output",
                        "title": "Valid Output",
                        "href": "/outputs/valid-output",
                        "summary": "Valid href must remain openable.",
                        "updated_at": "2026-05-24T16:02:00+00:00",
                        "read": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    catalog_items = collect_catalog_outputs(tmp_path)
    saved_after_first_collect = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_items = collect_catalog_outputs(tmp_path)
    html = render_catalog_outputs_page(catalog_items, generated_at=datetime(2026, 5, 24, 17, tzinfo=timezone.utc))
    rows = re.findall(r"<article class=\"output-row[^>]*>.*?</article>", html, re.S)
    row_by_title = {}
    for row in rows:
        title_match = re.search(r"<b>(.*?)</b>", row, re.S)
        assert title_match is not None
        row_by_title[title_match.group(1)] = row

    assert {item.id: item.href for item in catalog_items} == {
        "valid-output": "/outputs/valid-output",
        "missing-href": "",
        "unsafe-js": "",
    }
    assert {entry["id"]: entry["href"] for entry in saved_after_first_collect["outputs"]} == {
        "valid-output": "/outputs/valid-output",
        "missing-href": "",
        "unsafe-js": "",
    }
    assert '<div class="stat">Unread <b data-unread-count="0">0</b></div>' in html
    assert "javascript:alert" not in html
    for title in ("Unsafe JS Output", "Missing Href Output"):
        row = row_by_title[title]
        assert "readable" not in row
        assert "data-open-url" not in row
        assert "data-read-key" not in row
        assert "output-open-overlay" not in row
        assert '<span class="open">OPEN</span>' not in row
        assert "read-toggle" not in row
        assert 'aria-disabled="true"' in row
        assert "No public link" in row

    valid_row = row_by_title["Valid Output"]
    assert 'data-open-url="/outputs/valid-output"' in valid_row
    assert 'data-read-key="output:valid-output"' in valid_row
    assert '<a class="output-open-overlay" href="/outputs/valid-output"' in valid_row
    assert '<button class="read-toggle" type="button" aria-label="Mark output unread: Valid Output">Mark unread</button>' in valid_row
    assert '<span class="open">OPEN</span>' in valid_row
    assert 'aria-disabled="true"' not in valid_row


def test_catalog_outputs_invalid_hrefs_render_disabled_non_openable_rows():
    item_cls = collect_catalog_outputs.__globals__["ActaOutputItem"]
    catalog_items = [
        item_cls(
            id="unsafe-js",
            title="Unsafe JS Output",
            href="javascript:alert(1)",
            summary="Unsafe href must not become clickable.",
            tags=("security",),
            source_name="catalog",
            created_at="2026-05-24T16:00:00+00:00",
            updated_at="2026-05-24T16:00:00+00:00",
        ),
        item_cls(
            id="unsafe-protocol-relative",
            title="Unsafe Protocol Output",
            href="//evil.example/output",
            summary="Protocol-relative href must not become clickable.",
            tags=(),
            source_name="catalog",
            created_at="2026-05-24T16:00:00+00:00",
            updated_at="2026-05-24T16:00:00+00:00",
        ),
    ]

    html = render_catalog_outputs_page(catalog_items, generated_at=datetime(2026, 5, 24, 17, tzinfo=timezone.utc))
    rows = re.findall(r"<article class=\"output-row[^>]*>.*?</article>", html, re.S)

    assert len(rows) == 2
    assert "javascript:alert" not in html
    assert "//evil.example" not in html
    for row in rows:
        assert "readable" not in row
        assert "data-read-key" not in row
        assert "data-read-initial" not in row
        assert "data-open-url" not in row
        assert 'aria-disabled="true"' in row
        assert "output-open-overlay" not in row
        assert '<span class="open">OPEN</span>' not in row
        assert "No public link" in row


def test_catalog_outputs_valid_hrefs_remain_readable_openable_rows():
    item_cls = collect_catalog_outputs.__globals__["ActaOutputItem"]
    catalog_items = [
        item_cls(
            id="valid-output",
            title="Valid Output",
            href="/outputs/valid-output",
            summary="Valid catalog output.",
            tags=("acta",),
            source_name="catalog",
            created_at="2026-05-24T16:00:00+00:00",
            updated_at="2026-05-24T16:00:00+00:00",
            read=True,
        )
    ]

    html = render_catalog_outputs_page(catalog_items, generated_at=datetime(2026, 5, 24, 17, tzinfo=timezone.utc))
    match = re.search(r"<article class=\"output-row[^>]*>.*?</article>", html, re.S)
    assert match is not None
    row = match.group(0)

    assert '<article class="output-row catalog-output-row readable read fresh"' in row
    assert 'data-read-key="output:valid-output"' in row
    assert 'data-read-initial="true"' in row
    assert 'data-open-url="/outputs/valid-output"' in row
    assert '<a class="output-open-overlay" href="/outputs/valid-output"' in row
    assert '<span class="read-state">READ</span>' in row
    assert '<button class="read-toggle" type="button" aria-label="Mark output unread: Valid Output">Mark unread</button>' in row
    assert '<span class="open">OPEN</span>' in row
    assert 'aria-disabled="true"' not in row


def test_catalog_outputs_read_hydration_falls_back_to_server_initial_state():
    script = _outputs_read_state_script()

    assert "Object.prototype.hasOwnProperty.call(state,k)" in script
    assert "el.dataset.readInitial==='true'" in script
    assert "var isRead=!!state[k];" not in script


def test_catalog_outputs_read_script_updates_aggregate_and_toggle_button():
    script = _outputs_read_state_script()

    assert "function updateUnreadCount()" in script
    assert "document.querySelectorAll('[data-unread-count]')" in script
    assert "el.dataset.unreadCount=String(unread)" in script
    assert "var button=el.querySelector('.read-toggle')" in script
    assert "button.addEventListener('click'" in script
    assert "ev.stopPropagation();" in script
    assert "setRead(el, el.classList.contains('read') ? false : true);" in script
    assert "setRead(el, true);" in script
    assert "updateUnreadCount();" in script


def test_catalog_outputs_unread_stat_and_read_toggle_only_for_safe_rows():
    item_cls = collect_catalog_outputs.__globals__["ActaOutputItem"]
    catalog_items = [
        item_cls(
            id="safe-output",
            title="Safe Output",
            href="/outputs/safe-output",
            summary="Safe catalog output.",
            tags=("acta",),
            source_name="catalog",
            created_at="2026-05-24T16:00:00+00:00",
            updated_at="2026-05-24T16:00:00+00:00",
            read=False,
        ),
        item_cls(
            id="unsafe-output",
            title="Unsafe Output",
            href="javascript:alert(1)",
            summary="Unsafe catalog output.",
            tags=(),
            source_name="catalog",
            created_at="2026-05-24T16:00:00+00:00",
            updated_at="2026-05-24T16:00:00+00:00",
            read=False,
        ),
    ]

    html = render_catalog_outputs_page(catalog_items, generated_at=datetime(2026, 5, 24, 17, tzinfo=timezone.utc))
    rows = re.findall(r"<article class=\"output-row[^>]*>.*?</article>", html, re.S)
    row_by_title = {}
    for row in rows:
        title_match = re.search(r"<b>(.*?)</b>", row, re.S)
        assert title_match is not None
        row_by_title[title_match.group(1)] = row

    assert '<div class="stat">Unread <b data-unread-count="1">1</b></div>' in html
    safe_row = row_by_title["Safe Output"]
    assert 'data-read-title="Safe Output"' in safe_row
    assert '<button class="read-toggle" type="button" aria-label="Mark output read: Safe Output">Mark read</button>' in safe_row
    assert ".output-actions a, .output-actions button { pointer-events:auto; }" in html
    assert ".output-actions button:focus-visible" in html
    assert ".catalog-output-row .output-main p { -webkit-line-clamp:2; line-height:1.32; }" in html
    assert ".catalog-output-row .output-main p { -webkit-line-clamp:3; line-height:1.34; }" in html
    unsafe_row = row_by_title["Unsafe Output"]
    assert "read-toggle" not in unsafe_row
    assert "data-read-key" not in unsafe_row
    assert "data-unread-count" not in unsafe_row


def test_catalog_outputs_local_artifact_base_uses_file_clickable_html_sources():
    item_cls = collect_catalog_outputs.__globals__["ActaOutputItem"]
    catalog_items = [
        item_cls(
            id="decision-tree",
            title="Decision Tree",
            href="/outputs/decision-tree",
            summary="Openable local artifact.",
            tags=("acta",),
            source_name="decision-tree.html",
            created_at="2026-05-24T16:00:00+00:00",
            updated_at="2026-05-24T16:00:00+00:00",
        ),
        item_cls(
            id="encoded-traversal",
            title="Encoded Traversal",
            href="/outputs/encoded-traversal",
            summary="Encoded path separators must not enter local file URLs.",
            tags=(),
            source_name="%2e%2e%2fsecret.html",
            created_at="2026-05-24T16:02:00+00:00",
            updated_at="2026-05-24T16:02:00+00:00",
        ),
    ]

    html = render_catalog_outputs_page(
        catalog_items,
        generated_at=datetime(2026, 5, 24, 17, tzinfo=timezone.utc),
        local_artifact_base="../../../artifacts/acta-outputs",
    )

    assert 'href="../../../artifacts/acta-outputs/decision-tree.html"' in html
    assert 'data-open-url="../../../artifacts/acta-outputs/decision-tree.html"' in html
    assert '<a class="output-open-overlay" href="../../../artifacts/acta-outputs/decision-tree.html"' in html
    assert "../../../artifacts/acta-outputs/%2e%2e%2fsecret.html" not in html
    assert 'href="/outputs/encoded-traversal"' in html


def test_source_outputs_read_toggle_only_for_signed_rows():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    base = dict(
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
    )
    signed = item_cls(
        **base,
        job_id="signed-output",
        name="Signed Source Output",
        excerpt="Signed rows get explicit read controls.",
        artifact_url="https://acta.imperatr.com/r/signed-output/detail.html?exp=1&sig=abc",
        telegram_url="https://t.me/c/3566991387/86",
    )
    unsafe = item_cls(
        **base,
        job_id="unsafe-output",
        name="Unsafe Source Output",
        excerpt="Unsafe rows fail closed.",
        artifact_url="https://evil.example/r/unsafe-output/detail.html?exp=1&sig=abc",
        telegram_url="javascript:alert(1)",
    )

    html = render_outputs_page([signed, unsafe], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))
    rows = re.findall(r"<article class=\"output-row[^>]*>.*?</article>", html, re.S)
    row_by_title = {}
    for row in rows:
        title_match = re.search(r"<b>(.*?)</b>", row, re.S)
        assert title_match is not None
        row_by_title[title_match.group(1)] = row

    assert '<div class="stat">Unread <b data-unread-count="1">1</b></div>' in html
    signed_row = row_by_title["Signed Source Output"]
    assert 'data-read-key="output:signed-output:2026-05-19T10:00:00+00:00"' in signed_row
    assert 'data-read-title="Signed Source Output"' in signed_row
    assert '<span class="read-state">UNREAD</span>' in signed_row
    assert '<button class="read-toggle" type="button" aria-label="Mark output read: Signed Source Output">Mark read</button><span class="open">SIGNED</span>' in signed_row
    assert '<a class="ask" href="https://t.me/c/3566991387/86"' in signed_row
    assert 'data-open-url="https://acta.imperatr.com/r/signed-output/detail.html?exp=1&amp;sig=abc"' in signed_row
    assert '<a class="output-open-overlay" href="https://acta.imperatr.com/r/signed-output/detail.html?exp=1&amp;sig=abc"' in signed_row

    unsafe_row = row_by_title["Unsafe Source Output"]
    assert 'aria-disabled="true"' in unsafe_row
    assert "read-toggle" not in unsafe_row
    assert "data-read-key" not in unsafe_row
    assert "data-read-title" not in unsafe_row
    assert "read-state" not in unsafe_row
    assert "data-open-url" not in unsafe_row
    assert "output-open-overlay" not in unsafe_row
    assert "https://evil.example" not in html


def test_run_history_scans_multiple_files_excludes_acta_and_joins_job_metadata(tmp_path: Path):
    for job_id in ("daily", "htmlonly", "acta-situation-room"):
        (tmp_path / "cron" / "output" / job_id).mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(
        json.dumps(
            [
                {"id": "daily", "name": "Daily Brief", "schedule": {"display": "0 8 * * *"}, "deliver": "telegram:-1003566991387:86"},
                {"id": "htmlonly", "name": "HTML Only", "schedule": "manual", "deliver": "local"},
            ]
        )
    )
    (tmp_path / "cron" / "output" / "daily" / "2026-05-19_08-00-00.md").write_text("## Response\n\nOld daily")
    (tmp_path / "cron" / "output" / "daily" / "2026-05-20_08-00-00.md").write_text("## Response\n\nNew daily")
    (tmp_path / "cron" / "output" / "htmlonly" / "2026-05-20_09-00-00.html").write_text("<html>run</html>")
    (tmp_path / "cron" / "output" / "acta-situation-room" / "2026-05-20_10-00-00.md").write_text("## Response\n\nDashboard")

    runs = collect_run_history(tmp_path)
    html = render_runs_page(runs, generated_at=datetime(2026, 5, 20, 12, tzinfo=timezone.utc))

    assert [run.run_id for run in runs] == ["2026-05-20_09-00-00", "2026-05-20_08-00-00", "2026-05-19_08-00-00"]
    assert "Acta Runs" in html
    assert "Daily Brief" in html
    assert "0 8 * * *" in html
    assert "Old daily" in html and "New daily" in html
    assert "HTML Only" in html and "HTML" in html
    assert '<span class="confidence-chip">CONF HIGH</span>' in html
    assert '<span>fresh</span>' in html
    assert "acta-situation-room" not in html
    assert str(tmp_path) not in html
    assert 'href="https://t.me/c/3566991387/86"' in html
    assert '<a class="active" href="/runs">Runs</a>' in html


def test_run_history_does_not_leak_prompt_when_response_heading_missing(tmp_path: Path):
    (tmp_path / "cron" / "output" / "daily").mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(json.dumps([{"id": "daily", "name": "Daily Brief"}]))
    (tmp_path / "cron" / "output" / "daily" / "2026-05-20_08-00-00.md").write_text(
        "## Prompt\n\nSECRET PROMPT SHOULD NOT RENDER\n\n## Tool Output\n\ninternal trace",
        encoding="utf-8",
    )

    html = render_runs_page(collect_run_history(tmp_path), generated_at=datetime(2026, 5, 20, 12, tzinfo=timezone.utc))

    assert "SECRET PROMPT" not in html
    assert "internal trace" not in html
    assert "No visible response was produced for this run." in html


def test_outputs_and_detail_do_not_leak_prompt_when_response_heading_missing(tmp_path: Path, monkeypatch):
    (tmp_path / "cron" / "output" / "daily").mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(json.dumps([{"id": "daily", "name": "Daily Brief", "deliver": "telegram:-1003566991387:86"}]))
    (tmp_path / "cron" / "output" / "daily" / "2026-05-20_08-00-00.md").write_text(
        "# Cron Job: Daily Brief\n\n"
        "## Prompt\n\nSECRET PROMPT SHOULD NOT RENDER\n\n"
        "## Tool Output\n\ninternal trace /Users/mozzie/private/raw.log",
        encoding="utf-8",
    )
    uploaded: dict[str, str] = {}

    def fake_publish(path, job, settings):
        uploaded["html"] = Path(path).read_text(encoding="utf-8")
        return "https://acta.imperatr.com/r/daily/detail.html?exp=1&sig=abc"

    monkeypatch.setattr("cron.acta_dashboard.publish_html_artifact", fake_publish)

    items = collect_situation_items(tmp_path)
    items = attach_artifact_urls(items, {"enabled": True}, tmp_path / "details")
    outputs_html = render_outputs_page(items, generated_at=datetime(2026, 5, 20, 12, tzinfo=timezone.utc))
    detail_html = uploaded["html"]

    for rendered in (outputs_html, detail_html):
        assert "SECRET PROMPT" not in rendered
        assert "internal trace" not in rendered
        assert "/Users/mozzie/private" not in rendered
        assert "No visible response was produced for this run." in rendered
    assert items[0].status == "silent"
    assert items[0].excerpt == "No visible response was produced for this run."
    assert 'href="https://acta.imperatr.com/r/daily/detail.html?exp=1&amp;sig=abc"' in outputs_html
    assert '<article class="report-body">' in detail_html
    assert 'name="viewport"' in detail_html


def test_run_history_published_rows_open_signed_detail_without_prompt_or_path_leak(tmp_path: Path, monkeypatch):
    (tmp_path / "cron" / "output" / "daily").mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(json.dumps([{"id": "daily", "name": "Daily Brief"}]))
    local_sentinel = "/Users/mozzie/private/raw.log"
    (tmp_path / "cron" / "output" / "daily" / "2026-05-20_08-00-00.md").write_text(
        "# Cron Job: Daily Brief\n\n"
        "## Prompt\n\nSECRET PROMPT SHOULD NOT RENDER\n\n"
        "## Response\n\nVisible response body only.\n\n"
        "## tool output\n\ninternal trace " + local_sentinel,
        encoding="utf-8",
    )
    uploaded: dict[str, str] = {}

    def fake_publish(path, job, settings):
        object_key = str(settings.get("object_key"))
        assert object_key == "r/run-details/daily-2026-05-20_08-00-00.html"
        uploaded[object_key] = Path(path).read_text(encoding="utf-8")
        return f"https://acta.imperatr.com/{object_key}?exp=1&sig=abc"

    monkeypatch.setattr("cron.acta_dashboard.publish_html_artifact", fake_publish)

    runs = attach_run_artifact_urls(collect_run_history(tmp_path), {"enabled": True}, tmp_path / "details")
    html = render_runs_page(runs, generated_at=datetime(2026, 5, 20, 12, tzinfo=timezone.utc))
    detail_html = uploaded["r/run-details/daily-2026-05-20_08-00-00.html"]

    assert 'data-open-url="https://acta.imperatr.com/r/run-details/daily-2026-05-20_08-00-00.html?exp=1&amp;sig=abc"' in html
    assert 'href="https://acta.imperatr.com/r/run-details/daily-2026-05-20_08-00-00.html?exp=1&amp;sig=abc"' in html
    assert '<article class="output-row readable unread fresh" data-read-key="run:daily:2026-05-20_08-00-00" data-read-title="Daily Brief"' in html
    assert '<button class="read-toggle" type="button" aria-label="Mark output read: Daily Brief">Mark read</button><span class="open">SIGNED</span>' in html
    assert re.search(r'<span class="read-state">UNREAD</span><span class="confidence-chip">[^<]+</span>', html)
    assert '<div class="stat">Unread <b data-unread-count="1">1</b></div>' in html
    assert "document.querySelectorAll('.output-row.readable')" in html
    assert "el.querySelectorAll('.output-open-overlay')" in html
    assert "setRead(el, true);" in html
    assert "script-src 'sha256-" in html
    assert "script-src 'unsafe-inline'" not in html
    csp = re.search(r'Content-Security-Policy" content="([^"]+)"', html)
    script = re.search(r"<script>(.*?)</script>", html, re.S)
    assert csp and script
    script_hash = base64.b64encode(hashlib.sha256(script.group(1).encode("utf-8")).digest()).decode("ascii")
    assert f"script-src 'sha256-{script_hash}'" in csp.group(1)
    assert '<span class="open">SIGNED</span>' in html
    assert 'class="output-open-overlay"' in html
    assert "Visible response body only." in detail_html
    assert "<b>SIGNED STATUS</b> fresh" in detail_html
    assert "<b>SIGNED CONF</b> CONF" in detail_html
    assert "<b>SIGNED AGE</b>" in detail_html
    assert "SECRET PROMPT" not in detail_html
    assert "internal trace" not in detail_html
    assert local_sentinel not in detail_html
    assert str(tmp_path) not in html
    assert str(tmp_path) not in detail_html
    assert "2026-05-20_08-00-00.md" in html


def test_run_history_unsafe_or_missing_artifact_url_remains_disabled():
    item_cls = collect_run_history.__globals__["ActaRunItem"]
    base = dict(
        job_id="daily",
        name="Daily Brief",
        schedule="manual",
        deliver="local",
        enabled=True,
        run_id="2026-05-20_08-00-00",
        run_time=datetime(2026, 5, 20, 8, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Visible excerpt.",
        source_name="2026-05-20_08-00-00.md",
        has_markdown=True,
        has_html=False,
    )
    html = render_runs_page(
        [
            item_cls(**base, artifact_url="https://evil.example/run.html?sig=abc"),
            item_cls(**{**base, "job_id": "public", "artifact_url": "https://acta.imperatr.com/public/run-details/daily.html?exp=1&sig=abc"}),
            item_cls(**{**base, "job_id": "missing", "artifact_url": None}),
        ],
        generated_at=datetime(2026, 5, 20, 12, tzinfo=timezone.utc),
    )

    assert "https://evil.example" not in html
    assert "https://acta.imperatr.com/public/run-details/daily.html" not in html
    assert 'data-open-url=' not in html
    assert html.count('aria-disabled="true"') == 3
    assert html.count('<span class="muted">HISTORY</span>') == 3
    assert '<div class="stat">Unread <b data-unread-count="0">0</b></div>' in html
    assert '<span class="open">SIGNED</span>' not in html
    articles = re.findall(r"<article class=\"output-row[^>]*>.*?</article>", html, re.S)
    assert len(articles) == 3
    for article in articles:
        assert 'aria-disabled="true"' in article
        assert "readable" not in article
        assert "unread" not in article
        assert "data-read-key" not in article
        assert "data-read-title" not in article
        assert "read-state" not in article
        assert "read-toggle" not in article


def test_run_history_rejects_symlinked_run_files(tmp_path: Path):
    output_dir = tmp_path / "cron" / "output" / "daily"
    output_dir.mkdir(parents=True)
    secret = tmp_path / "secret.md"
    secret.write_text("## Response\n\nSECRET FILE", encoding="utf-8")
    (output_dir / "2026-05-20_08-00-00.md").symlink_to(secret)

    assert collect_run_history(tmp_path) == []


def test_acta_nav_order_is_today_outputs_runs_jobs_archive():
    html = render_archive_index([date(2026, 5, 20)], generated_at=datetime(2026, 5, 20, tzinfo=timezone.utc))

    assert html.index('href="/">Today') < html.index('href="/outputs">Outputs') < html.index('href="/runs">Runs') < html.index('href="/jobs">Jobs') < html.index('href="/archive">Archive')
    assert "grid-template-columns:repeat(5,1fr)" in html


def test_outputs_page_rejects_http_downgrade_urls():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    item = item_cls(
        job_id="downgrade",
        name="Downgrade Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="HTTP links should not become clickable.",
        artifact_url="http://acta.imperatr.com/r/lead/detail.html?sig=abc",
        telegram_url="http://t.me/c/3566991387/86",
    )

    html = render_outputs_page([item], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))

    assert "http://acta.imperatr.com" not in html
    assert "http://t.me" not in html
    assert 'data-read-key="output:downgrade' not in html
    assert "NO FOLLOW-UP" in html


def test_outputs_page_rejects_root_relative_telegram_urls():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    item = item_cls(
        job_id="root-telegram",
        name="Root Telegram Brief",
        schedule="daily",
        deliver="telegram",
        enabled=True,
        latest_md=Path("/tmp/2026-05-19_10-00-00.md"),
        latest_html=None,
        latest_time=datetime(2026, 5, 19, 10, tzinfo=timezone.utc),
        status="fresh",
        excerpt="Root-relative Telegram should not become clickable.",
        artifact_url="/r/lead/detail.html?exp=1&sig=abc",
        telegram_url="/c/3566991387/86",
    )

    html = render_outputs_page([item], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))

    assert 'href="/c/3566991387/86"' not in html
    assert "NO FOLLOW-UP" in html
    assert 'data-open-url="/r/lead/detail.html?exp=1&amp;sig=abc"' in html


def test_build_dashboard_publishes_outputs_index(tmp_path: Path, monkeypatch):
    (tmp_path / "cron" / "output" / "lead").mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(
        json.dumps([{"id": "lead", "name": "Lead Brief", "deliver": "telegram:-1003566991387:86"}])
    )
    (tmp_path / "cron" / "output" / "lead" / "2026-05-19_10-00-00.md").write_text("## Response\n\nMost important")
    outputs = tmp_path / "artifacts" / "acta-outputs"
    outputs.mkdir(parents=True)
    (outputs / "index.html").write_text(
        '<article data-id="hermes-agent-lanes-decision-tree" data-title="Hermes Agent Lanes & Specialist Agents"><p>Lane decision tree.</p></article>',
        encoding="utf-8",
    )
    (outputs / "hermes-agent-lanes-decision-tree.html").write_text(
        """
        <!doctype html><html><head><style>:root{--amber:#f5a400} body{background:#030302}</style></head>
        <body><header>ACTA / OUTPUTS</header><h1>Lane decision tree</h1><p>Use topic lanes for lightweight routing.</p></body></html>
        """,
        encoding="utf-8",
    )
    (tmp_path / "config.yaml").write_text(
        "cron:\n  html_artifacts:\n    publish:\n      enabled: true\n      endpoint: https://acta.imperatr.com\n"
    )
    published = []

    def fake_publish(path, job, settings):
        published.append({"path": Path(path), "object_key": settings.get("object_key"), "html": Path(path).read_text()})
        if settings.get("object_key") == "public/outputs/index.html":
            return "https://acta.imperatr.com/outputs/"
        if settings.get("object_key") == "public/index.html":
            return "https://acta.imperatr.com/"
        return "https://acta.imperatr.com/r/lead/detail.html?exp=1&sig=abc"

    monkeypatch.setattr("cron.acta_dashboard.publish_html_artifact", fake_publish)

    path, url = build_dashboard(tmp_path, publish=True, uat_preflight=False)

    assert path.exists()
    assert url == "https://acta.imperatr.com/"
    output_publish = next(item for item in published if item["object_key"] == "public/outputs/index.html")
    assert output_publish["path"].name == "outputs.publish.html"
    assert "Acta Outputs" in output_publish["html"]
    assert '<a class="active" href="/outputs">Outputs</a>' in output_publish["html"]
    assert "Hermes Agent Lanes &amp; Specialist Agents" in output_publish["html"]
    assert 'href="/outputs/hermes-agent-lanes-decision-tree"' in output_publish["html"]
    local_outputs_html = (tmp_path / "cron" / "output" / "acta-situation-room" / "outputs.html").read_text(encoding="utf-8")
    assert 'href="../../../artifacts/acta-outputs/hermes-agent-lanes-decision-tree.html"' in local_outputs_html
    assert 'data-open-url="../../../artifacts/acta-outputs/hermes-agent-lanes-decision-tree.html"' in local_outputs_html
    assert "https://acta.imperatr.com/r/lead/detail.html?exp=1&amp;sig=abc" not in output_publish["html"]
    backing_publish = next(item for item in published if item["object_key"] == "public/outputs/hermes-agent-lanes-decision-tree.html")
    assert backing_publish["path"].name == "hermes-agent-lanes-decision-tree.html"
    assert "Lane decision tree" in backing_publish["html"]
    assert "Use topic lanes for lightweight routing." in backing_publish["html"]
    assert "Signed Acta detail. Same Imperatr app shell" in backing_publish["html"]
    assert "#756cff" in backing_publish["html"]
    assert "--amber" not in backing_publish["html"]
    assert "#f5a400" not in backing_publish["html"]
    assert "background:#030302" not in backing_publish["html"]
    runs_publish = next(item for item in published if item["object_key"] == "public/runs/index.html")
    assert runs_publish["path"].name == "runs.html"
    assert "Acta Runs" in runs_publish["html"]
    assert "Lead Brief" in runs_publish["html"]


def test_acta_worker_routes_published_public_surfaces():
    worker = Path("cloudflare/acta/src/index.js").read_text(encoding="utf-8")

    assert 'pathname === "/archive" || pathname === "/archive/"' in worker
    assert 'return "public/archive/index.html"' in worker
    assert 'return `public/archive/${match[1]}.html`' in worker
    assert 'key === "public/archive/index.html"' in worker
    assert 'return `${base}/archive`' in worker
    assert 'key.match(/^public\\/archive\\/([0-9]{4}-[0-9]{2}-[0-9]{2})\\.html$/)' in worker
    assert 'return `${base}/archive/${archiveMatch[1]}`' in worker
    assert '/^public\\/archive\\/[0-9]{4}-[0-9]{2}-[0-9]{2}\\.html$/.test(key)' in worker
    assert 'key === "public/archive/index.html" || /^public\\/archive' not in worker
    assert 'pathname === "/runs" || pathname === "/runs/"' in worker
    assert 'return "public/runs/index.html"' in worker
    assert 'key === "public/runs/index.html"' in worker
    assert 'return `${base}/runs`' in worker
    assert 'return `public/outputs/${outputMatch[1]}.html`' in worker
    assert 'key.match(/^public\\/outputs\\/([A-Za-z0-9._-]+)\\.html$/)' in worker
