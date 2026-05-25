import base64
import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

from cron.acta_dashboard import (
    acta_dashboard_config,
    apply_feed_preferences,
    attach_artifact_urls,
    available_run_dates,
    build_dashboard,
    collect_catalog_outputs,
    collect_run_history,
    collect_situation_items,
    publish_catalog_output_artifacts,
    render_acta_detail_report,
    render_archive_index,
    render_catalog_outputs_page,
    render_dashboard,
    render_jobs_page,
    render_outputs_page,
    render_runs_page,
)


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
    assert "ACTA</em> / OUTPUTS" in html
    assert "no page" in html
    assert "output-summary" in html
    assert "metricrow" not in html
    assert "P2" not in html
    assert "P3" not in html
    assert "Delivery Routes" not in html
    assert "Bloomberg" not in html
    assert "Your cron command center" not in html


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
    assert "ASK TELEGRAM" in dashboard_html
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
    assert '<div class="output-summary"><b>1/1</b><span>fresh</span><span>0 gaps</span></div>' in html
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
    assert ".source-line { display:block" in html
    assert ".source-line { white-space:normal; overflow:visible; text-overflow:clip" in html
    assert ".lead { grid-template-columns:1fr; }" in html
    assert ".swipe-content { grid-template-columns:28px minmax(0,1fr); }" in html
    assert "second · telegram · 2026-05-19T09:00:00+00:00" in html


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
    assert '<span class="read-state">UNREAD</span>' in html
    assert 'MARK READ' in html
    assert ".row-open-overlay:focus-visible" in html
    assert ".brief-row > .swipe-content" in html
    assert ".brief-row > :not(.row-open-overlay)" not in html
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


def test_today_dashboard_requires_signed_acta_artifact_urls_for_open_overlays():
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

    html = render_dashboard([signed, unsigned, unsafe, signed_root], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))

    assert 'data-open-url="https://acta.imperatr.com/r/signed/detail.html?exp=1&amp;sig=abc"' in html
    assert 'href="https://acta.imperatr.com/r/signed/detail.html?exp=1&amp;sig=abc"' in html
    assert 'data-open-url="/r/root/detail.html?exp=1&amp;sig=root"' in html
    assert 'href="/r/root/detail.html?exp=1&amp;sig=root"' in html
    assert "Unsigned Brief" in html
    assert "Unsafe Brief" in html
    assert "https://acta.imperatr.com/r/unsigned/detail.html?exp=1" not in html
    assert "evil.example" not in html
    assert "javascript:alert" not in html
    assert "ASK TELEGRAM" not in html
    unsigned_row = next(row for row in re.findall(r'<section class="brief-row[^>]*>.*?</section>', html, re.S) if "Unsigned Brief" in row)
    unsafe_row = next(row for row in re.findall(r'<section class="brief-row[^>]*>.*?</section>', html, re.S) if "Unsafe Brief" in row)
    for disabled_row in (unsigned_row, unsafe_row):
        assert 'aria-disabled="true"' in disabled_row
        assert "readable" not in disabled_row
        assert "data-read-key" not in disabled_row
        assert "row-open-overlay" not in disabled_row
        assert "read-dot" not in disabled_row
        assert "read-state" not in disabled_row
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
    assert "row-open-overlay" not in lead_article.group(0)
    assert "read-dot" not in lead_article.group(0)
    assert "read-state" not in lead_article.group(0)
    assert "no page" in lead_article.group(0)
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
    assert "state[k]=true; save(); apply(el);" in html


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
    assert '<span class="confidence-chip">CONF MED</span>' in html
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
    assert '<nav class="mobilebar"><a href="/">TODAY</a><a href="/outputs">OUTPUTS</a><a href="/runs">RUNS</a><a href="/jobs">JOBS</a><a href="/archive">ARCHIVE</a></nav>' in html
    assert '.mobilebar.visible' in html
    assert 'IntersectionObserver' in html
    assert "document.querySelector('.date-nav')" in html
    assert 'background:rgba(2,2,2,.96)' not in html
    assert 'background:linear-gradient(180deg, rgba(7,16,24,.96), rgba(3,6,11,.94)), radial-gradient(circle at 18% 0%, rgba(117,108,255,.28), transparent 42%), radial-gradient(circle at 86% 20%, rgba(35,167,255,.18), transparent 48%)' in html


def test_detail_report_uses_acta_situation_room_ui():
    html = render_acta_detail_report(
        "# Market Brief\n\nImportant signal.",
        {"job_id": "job1", "job_name": "Market Brief", "run_time": "2026-05-19T10:00:00+00:00"},
    )

    assert "Acta Situation Room" in html
    assert 'href="/outputs"' in html
    assert ">Back<" in html
    assert "width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover, user-scalable=no" in html
    assert "--black:#03060b" in html
    assert "--accent:#756cff" in html
    assert "Important signal." in html


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
        latest_time=None,
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
    assert '<span class="confidence-chip">CONF HIGH</span>' in html
    assert '<span class="confidence-chip">CONF LOW/GAP</span>' in html
    assert "<span>fresh</span>" in html
    assert "<span>silent</span>" in html
    assert "Disabled Brief" not in html
    assert "Hidden Brief" not in html


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

    html = render_jobs_page([valid, unsafe, subdomain], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))

    assert 'href="https://t.me/c/3566991387/86"' in html
    assert "Unsafe Thread Brief" in html
    assert "Subdomain Thread Brief" in html
    assert "javascript:alert" not in html
    assert "evil.t.me" not in html


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
    assert "document.querySelectorAll('.output-row.readable')" in html
    assert "el.querySelectorAll('.output-open-overlay')" in html
    assert "state[k]=true; save(); apply(el);" in html
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
    assert "NO FOLLOW-UP" in html
    assert 'onclick="bad' not in html


def test_outputs_page_rejects_protocol_relative_and_suspicious_artifact_urls():
    item_cls = collect_situation_items.__globals__["CronSituationItem"]
    for url in ("//evil.example/path?sig=abc", "///evil.example/path?sig=abc", "/\\evil?sig=abc", "/ok\npath?sig=abc"):
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
    assert '<div class="stat">Unread <b>0</b></div>' in html
    assert "javascript:alert" not in html
    for title in ("Unsafe JS Output", "Missing Href Output"):
        row = row_by_title[title]
        assert "readable" not in row
        assert "data-open-url" not in row
        assert "data-read-key" not in row
        assert "output-open-overlay" not in row
        assert '<span class="open">OPEN</span>' not in row
        assert 'aria-disabled="true"' in row
        assert "No public link" in row

    valid_row = row_by_title["Valid Output"]
    assert 'data-open-url="/outputs/valid-output"' in valid_row
    assert 'data-read-key="output:valid-output"' in valid_row
    assert '<a class="output-open-overlay" href="/outputs/valid-output"' in valid_row
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

    assert '<article class="output-row readable read fresh"' in row
    assert 'data-read-key="output:valid-output"' in row
    assert 'data-open-url="/outputs/valid-output"' in row
    assert '<a class="output-open-overlay" href="/outputs/valid-output"' in row
    assert '<span class="read-state">READ</span>' in row
    assert '<span class="open">OPEN</span>' in row
    assert 'aria-disabled="true"' not in row


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
    assert "No visible Markdown response" in html


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
        return "https://acta.imperatr.com/r/daily/detail.html?sig=abc"

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
    assert 'href="https://acta.imperatr.com/r/daily/detail.html?sig=abc"' in outputs_html
    assert '<article class="report-body">' in detail_html
    assert 'name="viewport"' in detail_html


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
        artifact_url="/r/lead/detail.html?sig=abc",
        telegram_url="/c/3566991387/86",
    )

    html = render_outputs_page([item], generated_at=datetime(2026, 5, 19, 11, tzinfo=timezone.utc))

    assert 'href="/c/3566991387/86"' not in html
    assert "NO FOLLOW-UP" in html
    assert 'data-open-url="/r/lead/detail.html?sig=abc"' in html


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

    path, url = build_dashboard(tmp_path, publish=True)

    assert path.exists()
    assert url == "https://acta.imperatr.com/"
    output_publish = next(item for item in published if item["object_key"] == "public/outputs/index.html")
    assert output_publish["path"].name == "outputs.html"
    assert "Acta Outputs" in output_publish["html"]
    assert '<a class="active" href="/outputs">Outputs</a>' in output_publish["html"]
    assert "Hermes Agent Lanes &amp; Specialist Agents" in output_publish["html"]
    assert 'href="/outputs/hermes-agent-lanes-decision-tree"' in output_publish["html"]
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

    assert 'pathname === "/runs" || pathname === "/runs/"' in worker
    assert 'return "public/runs/index.html"' in worker
    assert 'key === "public/runs/index.html"' in worker
    assert 'return `${base}/runs`' in worker
    assert 'return `public/outputs/${outputMatch[1]}.html`' in worker
    assert 'key.match(/^public\\/outputs\\/([A-Za-z0-9._-]+)\\.html$/)' in worker
