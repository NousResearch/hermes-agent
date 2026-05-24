import json
from datetime import date, datetime, timezone
from pathlib import Path

from cron.acta_dashboard import (
    acta_dashboard_config,
    apply_feed_preferences,
    attach_artifact_urls,
    available_run_dates,
    build_dashboard,
    collect_situation_items,
    render_acta_detail_report,
    render_archive_index,
    render_dashboard,
    render_jobs_page,
    render_outputs_page,
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
    assert "open first" in html
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


def test_detail_report_can_link_back_to_telegram_thread():
    html = render_acta_detail_report(
        "# Operator Brief\n\nUseful signal.",
        {"job_id": "job1", "job_name": "Operator Brief", "run_time": "2026-05-19T10:00:00+00:00"},
        telegram_url="https://t.me/c/3566991387/86",
    )

    assert "Ask follow-up in Telegram" in html
    assert 'href="https://t.me/c/3566991387/86"' in html
    assert 'target="_blank" rel="noopener"' in html


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
    assert 'Briefing Packet <span>1</span>' in html
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
        artifact_url="https://acta.example/lead.html",
    )

    html = render_dashboard([linked_item])

    assert '<article class="lead readable unread"' in html
    assert 'data-open-url="https://acta.example/lead.html"' in html
    assert 'data-read-key="lead:' in html
    assert '<span class="read-state">UNREAD</span>' in html
    assert 'MARK READ' in html
    assert "script-src 'unsafe-inline'" in html
    assert "localStorage" in html


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
    assert '<nav class="mobilebar"><a href="/">TODAY</a><a href="/jobs">JOBS</a><a href="/archive">ARCHIVE</a><a href="/outputs">OUTPUTS</a></nav>' in html
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
    assert "Disabled Brief" not in html
    assert "Hidden Brief" not in html


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


def test_build_dashboard_publishes_outputs_index(tmp_path: Path, monkeypatch):
    (tmp_path / "cron" / "output" / "lead").mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(
        json.dumps([{"id": "lead", "name": "Lead Brief", "deliver": "telegram:-1003566991387:86"}])
    )
    (tmp_path / "cron" / "output" / "lead" / "2026-05-19_10-00-00.md").write_text("## Response\n\nMost important")
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
    assert "https://acta.imperatr.com/r/lead/detail.html?exp=1&amp;sig=abc" in output_publish["html"]
