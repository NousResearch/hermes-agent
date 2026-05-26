import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from cron.acta_dashboard import CronSituationItem, collect_situation_items, render_dashboard, render_jobs_page


ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts" / "acta_browser_uat.py"

_spec = importlib.util.spec_from_file_location("acta_browser_uat", HARNESS)
assert _spec is not None and _spec.loader is not None
acta_browser_uat = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = acta_browser_uat
_spec.loader.exec_module(acta_browser_uat)


def _browser_cli_available() -> bool:
    return bool(os.environ.get("ACTA_UAT_AGENT_BROWSER") or shutil.which("agent-browser") or shutil.which("npx"))


def _valid_feed_dom() -> str:
    return """
    <html><body>
      <h1>Output Streams</h1>
      <h2>Daily life feed</h2>
      <section class="brief-row" data-feed-lane="daily"><span class="lane-chip">Daily</span>Morning newsletter</section>
      <h2>Development sprint cycles</h2>
      <section class="brief-row" data-feed-lane="dev"><span class="lane-chip">Dev</span>Operator sprint review</section>
    </body></html>
    """


def _valid_jobs_dom() -> str:
    return """
    <html><body>
      <main>
        <h1>Acta Jobs</h1>
        <p>Source Runs show latest freshness, confidence, and operator actions.</p>
        <article class="job-row">
          <h2>Morning newsletter digest</h2>
          <span class="confidence-chip">CONF HIGH</span>
          <span>LAST RUN 2026-05-25 09:00 UTC</span>
          <span>SCHEDULE daily</span>
          <a>OPEN</a>
          <small>SOURCE cron/output/daily job_id=daily</small>
        </article>
        <article class="job-row">
          <h2>Vesta Startup Sprint CEO loop</h2>
          <span class="confidence-chip">CONF LOW-GAP</span>
          <span>LAST RUN 2026-05-25 08:30 UTC</span>
          <span>SCHEDULE every 120m</span>
          <span>NO PAGE</span>
          <small>SOURCE cron/output/dev job_id=dev</small>
        </article>
      </main>
    </body></html>
    """


def _valid_outputs_dom() -> str:
    return """
    <html><body>
      <main>
        <h1>Outputs</h1>
        <p>Signed source objects in the Persistent catalog.</p>
        <article class="output-row">
          <h2>Morning operator brief</h2>
          <span>CONF HIGH</span>
          <span>SOURCE morning_digest JOB daily ID 2026-05-25_09-00-00</span>
          <span>SCHEDULE daily brief OUTPUT 2h ago</span>
          <a class="output-open-overlay" href="/acta/outputs/morning-operator-brief">OPEN</a>
          <span>UNREAD</span>
          <button>Mark read</button>
        </article>
        <article class="output-row">
          <h2>Catalog-only artifact</h2>
          <span>CATALOG</span>
          <span>SOURCE system ID catalog-17</span>
          <span>PINNED catalog age 1 day ago</span>
          <span>No public link</span>
        </article>
      </main>
    </body></html>
    """


def _valid_archive_dom() -> str:
    return """
    <html><body>
      <main>
        <h1>Previous days.</h1>
        <a class="archive-card" href="/archive/2026-05-24">
          <span>Source-backed day</span><strong>2026-05-24</strong>
          <small>Visible 3 · Silent 1 · Missing 0</small>
          <em>Latest: Morning operator brief</em>
          <small>Daily 2 · Dev 1 · System 1</small>
        </a>
      </main>
    </body></html>
    """


def _archive_day_items() -> list[CronSituationItem]:
    return [
        CronSituationItem(
            job_id="daily",
            name="Archived Morning Brief",
            schedule="daily",
            deliver="telegram",
            enabled=True,
            latest_md=Path("/tmp/daily.md"),
            latest_html=None,
            latest_time=datetime(2026, 5, 19, 9, tzinfo=timezone.utc),
            status="fresh",
            excerpt="Archived source-backed briefing with decisions ready for review.",
            artifact_url="https://acta.imperatr.com/r/daily/detail.html?exp=1&sig=daily",
            telegram_url="https://t.me/imperatr/123",
        ),
        CronSituationItem(
            job_id="acta-startup-sprint",
            name="Acta Startup Sprint CEO loop",
            schedule="every 30m",
            deliver="telegram",
            enabled=True,
            latest_md=Path("/tmp/dev.md"),
            latest_html=None,
            latest_time=datetime(2026, 5, 19, 9, 15, tzinfo=timezone.utc),
            status="fresh",
            excerpt="Development sprint remains in the background lane.",
            artifact_url="https://acta.imperatr.com/r/dev/detail.html?exp=1&sig=dev",
        ),
        CronSituationItem(
            job_id="unsafe",
            name="Unsigned Archive Draft",
            schedule="manual",
            deliver="local",
            enabled=True,
            latest_md=Path("/tmp/unsafe.md"),
            latest_html=None,
            latest_time=datetime(2026, 5, 19, 8, tzinfo=timezone.utc),
            status="fresh",
            excerpt="Visible archive source row without a signed page.",
            artifact_url="https://acta.imperatr.com/r/unsafe/detail.html?exp=1",
        ),
    ]


def _valid_archive_day_dom() -> str:
    return render_dashboard(
        _archive_day_items(),
        generated_at=datetime(2026, 5, 20, 12, tzinfo=timezone.utc),
        selected_date=date(2026, 5, 19),
        archive_dates=[date(2026, 5, 19)],
        archive_day=True,
    )


def test_acta_browser_uat_harness_validates_feed_lane_contract(tmp_path: Path):
    if not _browser_cli_available():
        pytest.skip("agent-browser/npx unavailable; pure validation tests still cover feed contract")

    jobs = [
        {"id": "daily", "name": "Morning newsletter digest", "schedule": "daily", "deliver": "telegram"},
        {"id": "weather", "name": "Daily weather and outfit", "schedule": "daily", "deliver": "telegram"},
        {"id": "dev", "name": "Vesta Startup Sprint CEO loop", "schedule": "every 120m", "deliver": "telegram"},
    ]
    for job in jobs:
        (tmp_path / "cron" / "output" / job["id"]).mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(json.dumps(jobs))
    (tmp_path / "cron" / "output" / "daily" / "2026-05-19_09-00-00.md").write_text("## Response\n\nDaily signal")
    (tmp_path / "cron" / "output" / "weather" / "2026-05-19_08-55-00.md").write_text("## Response\n\nWeather signal")
    (tmp_path / "cron" / "output" / "dev" / "2026-05-19_09-10-00.md").write_text("## Response\n\nSprint signal")
    html_path = tmp_path / "acta.html"
    html_path.write_text(render_dashboard(collect_situation_items(tmp_path)), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(HARNESS), "--html", str(html_path), "--artifact-dir", str(tmp_path / "uat")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=45,
    )

    assert result.returncode == 0, result.stdout
    assert "PASS Acta browser UAT" in result.stdout
    assert (tmp_path / "uat" / "acta-uat.png").exists()


def test_validate_feed_contract_fails_exactly_when_dev_rows_are_commingled():
    dom = """
    <!doctype html><html><body>
      <div class="panel-title"><b>Output Streams</b></div>
      <div class="feed-section lane-section-daily">
        <div class="feed-section-title"><b>Daily life feed</b></div>
        <section class="feed" data-feed-lane="daily">
          <section class="brief-row" data-feed-lane="daily">
            <span class="lane-chip">Daily</span><h2>Morning newsletter digest</h2>
          </section>
          <section class="brief-row" data-feed-lane="daily">
            <span class="lane-chip">Daily</span><h2>QA pipeline canary</h2>
          </section>
        </section>
      </div>
      <div class="feed-section lane-section-dev">
        <div class="feed-section-title"><b>Development sprint cycles</b></div>
        <section class="feed" data-feed-lane="dev">
          <section class="brief-row" data-feed-lane="dev">
            <span class="lane-chip">Dev</span><h2>Operator sprint review</h2>
          </section>
        </section>
      </div>
    </body></html>
    """

    assert acta_browser_uat._validate_feed_contract(dom) == [
        "Development sprint output is commingled into Daily life feed"
    ]


def test_validate_feed_contract_detects_qa_pipeline_canary_in_daily_lane():
    dom = """
    <html><body>
      <h1>Output Streams</h1>
      <h2>Daily life feed</h2>
      <section class="brief-row" data-feed-lane="daily"><span class="lane-chip">Daily</span>QA pipeline canary</section>
      <h2>Development sprint cycles</h2>
      <section class="brief-row" data-feed-lane="dev"><span class="lane-chip">Dev</span>User testing sweep</section>
    </body></html>
    """

    assert "Development sprint output is commingled into Daily life feed" in acta_browser_uat._validate_feed_contract(dom)


def test_validate_feed_contract_counts_lane_tagged_lead_as_daily_row():
    dom = """
    <html><body>
      <h1>Output Streams</h1>
      <article class="lead" data-feed-lane="daily"><span class="lane-chip">Daily</span>Morning newsletter</article>
      <h2>Daily life feed</h2>
      <p class="empty-feed">No additional outputs in this lane yet.</p>
      <h2>Development sprint cycles</h2>
      <section class="brief-row" data-feed-lane="dev"><span class="lane-chip">Dev</span>User testing sweep</section>
    </body></html>
    """

    assert acta_browser_uat._validate_feed_contract(dom) == []


def test_validate_feed_contract_flags_failed_signed_row_action_probe():
    failures = acta_browser_uat._validate_feed_contract(
        _valid_feed_dom(),
        action_state_probe={"skipped": False, "ok": False, "reason": "missing-action-or-overlay"},
    )

    assert failures == ["Signed row action-state browser probe failed: missing-action-or-overlay"]


def test_validate_feed_contract_allows_pages_without_signed_rows_to_skip_action_probe():
    assert acta_browser_uat._validate_feed_contract(
        _valid_feed_dom(),
        action_state_probe={"skipped": True, "reason": "no-readable-row"},
    ) == []


def test_validate_archive_day_contract_requires_browser_action_probe():
    failures = acta_browser_uat._validate_archive_day_contract(
        _valid_archive_day_dom(),
        action_state_probe={"skipped": True, "reason": "no-readable-row"},
    )

    assert "Archive-day signed row action-state browser probe was skipped" in failures


def test_validate_archive_day_contract_accepts_signed_rows_with_actions():
    assert acta_browser_uat._validate_archive_day_contract(
        _valid_archive_day_dom(),
        action_state_probe={"skipped": False, "ok": True, "saveOk": True, "dismissOk": True, "laterOk": True, "overlayOk": True},
    ) == []


def test_validate_archive_day_contract_fails_when_actions_are_missing():
    dom = _valid_archive_day_dom().replace("data-state-action=\"save\"", "data-state-missing=\"save\"")

    failures = acta_browser_uat._validate_archive_day_contract(
        dom,
        action_state_probe={"skipped": False, "ok": False, "reason": "missing-action-or-overlay"},
    )

    assert "Archive-day signed row action-state browser probe failed: missing-action-or-overlay" in failures
    assert any("missing Save action" in failure for failure in failures)


def test_acta_browser_uat_exercises_signed_row_action_buttons_and_overlay(tmp_path: Path):
    if not _browser_cli_available():
        pytest.skip("agent-browser/npx unavailable; pure validation tests still cover action probe failures")

    html_path = tmp_path / "acta-actions.html"
    items = [
        CronSituationItem(
            job_id="daily",
            name="Morning newsletter digest",
            schedule="daily",
            deliver="telegram",
            enabled=True,
            latest_md=Path("/tmp/daily.md"),
            latest_html=None,
            latest_time=None,
            status="fresh",
            excerpt="Daily signed packet with local triage actions.",
            artifact_url="https://acta.imperatr.com/r/daily/detail.html?exp=1&sig=daily",
        ),
        CronSituationItem(
            job_id="dev",
            name="Vesta Startup Sprint CEO loop",
            schedule="every 120m",
            deliver="telegram",
            enabled=True,
            latest_md=Path("/tmp/dev.md"),
            latest_html=None,
            latest_time=None,
            status="fresh",
            excerpt="Development sprint stays in the background lane.",
            artifact_url="https://acta.imperatr.com/r/dev/detail.html?exp=1&sig=dev",
        ),
        CronSituationItem(
            job_id="unsigned",
            name="Unsigned Draft",
            schedule="manual",
            deliver="local",
            enabled=True,
            latest_md=Path("/tmp/unsigned.md"),
            latest_html=None,
            latest_time=None,
            status="fresh",
            excerpt="Visible but no signed action controls.",
            artifact_url="https://acta.imperatr.com/r/unsigned/detail.html?exp=1",
        ),
    ]
    html_path.write_text(render_dashboard(items), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(HARNESS),
            "--html",
            str(html_path),
            "--artifact-dir",
            str(tmp_path / "uat-actions"),
            "--viewport-width",
            "390",
            "--viewport-height",
            "844",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=45,
    )
    report = json.loads((tmp_path / "uat-actions" / "acta-uat-report.json").read_text())

    assert result.returncode == 0, result.stdout
    assert report["action_state_probe"]["ok"] is True
    assert report["action_state_probe"]["saveOk"] is True
    assert report["action_state_probe"]["overlayOk"] is True
    assert report["action_state_probe"]["unsafeHasActions"] is False


def test_acta_browser_uat_archive_day_exercises_signed_row_actions_at_mobile_width(tmp_path: Path):
    if not _browser_cli_available():
        pytest.skip("agent-browser/npx unavailable; pure validation tests still cover archive-day action probe failures")

    html_path = tmp_path / "archive-day.html"
    html_path.write_text(_valid_archive_day_dom(), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(HARNESS),
            "--html",
            str(html_path),
            "--artifact-dir",
            str(tmp_path / "uat-archive-day"),
            "--viewport-width",
            "390",
            "--viewport-height",
            "844",
            "--scenario",
            "archive-day",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=45,
    )

    assert result.returncode == 0, result.stdout
    report = json.loads((tmp_path / "uat-archive-day" / "acta-uat-report.json").read_text())

    assert report["scenario_key"] == "archive-day"
    assert report["persona"] == "mobile Acta operator reviewing a previous-day briefing"
    assert report["viewport"]["width"] == 390
    assert report["readable_rows"] >= 1
    assert report["action_state_probe"]["ok"] is True
    assert report["action_state_probe"]["saveOk"] is True
    assert report["action_state_probe"]["dismissOk"] is True
    assert report["action_state_probe"]["laterOk"] is True
    assert report["action_state_probe"]["overlayOk"] is True
    assert report["action_state_probe"]["unsafeHasActions"] is False


def test_validate_jobs_contract_accepts_operator_useful_source_runs_dom():
    assert acta_browser_uat._validate_jobs_contract(_valid_jobs_dom()) == []


def test_validate_jobs_contract_accepts_real_render_jobs_page_dom(tmp_path: Path):
    (tmp_path / "cron" / "output" / "daily").mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(
        json.dumps(
            [
                {"id": "daily", "name": "Morning newsletter digest", "schedule": "daily", "deliver": "telegram"},
                {"id": "gap", "name": "Missing source run", "schedule": "daily", "deliver": "local"},
            ]
        )
    )
    (tmp_path / "cron" / "output" / "daily" / "2026-05-25_09-00-00.md").write_text(
        "## Response\n\nUseful operator brief"
    )

    dom = render_jobs_page(collect_situation_items(tmp_path))

    assert acta_browser_uat._validate_jobs_contract(dom) == []


def test_validate_jobs_contract_fails_when_confidence_and_action_markers_missing():
    dom = """
    <html><body>
      <h1>Jobs</h1>
      <p>Source Runs</p>
      <article class="job-row">
        <h2>Morning newsletter digest</h2>
        <span>LAST RUN 2026-05-25 09:00 UTC</span>
        <span>SCHEDULE daily</span>
        <small>SOURCE cron/output/daily job_id=daily</small>
      </article>
    </body></html>
    """

    failures = acta_browser_uat._validate_jobs_contract(dom)

    assert "Job row 1 is missing visible confidence chips (CONF HIGH/MED/LOW-GAP)" in failures
    assert "Job row 1 is missing action/status copy (OPEN/SIGNED or NO PAGE)" in failures


def test_validate_jobs_contract_fails_when_markers_exist_outside_job_rows():
    dom = """
    <html><body>
      <h1>Acta Jobs</h1>
      <p>Source Runs <span class="confidence-chip">CONF HIGH</span> LAST RUN SCHEDULE OPEN SOURCE job_id=daily</p>
      <article class="job-row">
        <h2>Morning newsletter digest</h2>
      </article>
    </body></html>
    """

    failures = acta_browser_uat._validate_jobs_contract(dom)

    assert "Job row 1 is missing visible confidence chips (CONF HIGH/MED/LOW-GAP)" in failures
    assert "Job row 1 is missing LAST RUN freshness copy" in failures
    assert "Job row 1 is missing SCHEDULE copy" in failures
    assert "Job row 1 is missing action/status copy (OPEN/SIGNED or NO PAGE)" in failures
    assert "Job row 1 is missing source/provenance copy (SOURCE/job_id)" in failures


def test_validate_jobs_contract_reuses_common_browser_failure_checks():
    failures = acta_browser_uat._validate_jobs_contract(
        _valid_jobs_dom(),
        horizontal_overflow=True,
        console_output="Uncaught Error: boom",
        errors_output="TypeError: failed during render",
    )

    assert "Horizontal overflow detected at mobile viewport" in failures
    assert "Browser console contains error/exception output" in failures
    assert "Browser page errors were reported" in failures


def test_validate_jobs_contract_fails_on_acta_sign_in_wall():
    failures = acta_browser_uat._validate_jobs_contract("<html><body><h1>Sign in to Acta</h1><p>Acta access token</p></body></html>")

    assert failures == [
        "Acta sign-in wall rendered; pass a local --html artifact or validate with authenticated browser storage"
    ]


def test_validate_outputs_contract_accepts_operator_useful_outputs_dom():
    assert acta_browser_uat._validate_outputs_contract(_valid_outputs_dom()) == []


def test_validate_outputs_contract_accepts_production_style_data_open_url():
    dom = """
    <html><body>
      <h1>Outputs</h1>
      <article class="output-row" data-open-url="/acta/outputs/morning-operator-brief">
        <h2>Morning operator brief</h2>
        <span>CONF HIGH</span>
        <span>SOURCE morning_digest JOB daily ID 2026-05-25_09-00-00</span>
        <span>SCHEDULE daily brief OUTPUT 2h ago</span>
        <span>SIGNED</span>
        <span>UNREAD</span>
        <button>Mark read</button>
      </article>
    </body></html>
    """

    assert acta_browser_uat._validate_outputs_contract(dom) == []


def test_validate_outputs_contract_accepts_output_open_overlay_href():
    dom = """
    <html><body>
      <h1>Outputs</h1>
      <article class="output-row">
        <h2>Morning operator brief</h2>
        <span>CONF HIGH</span>
        <span>SOURCE morning_digest JOB daily ID 2026-05-25_09-00-00</span>
        <span>SCHEDULE daily brief OUTPUT 2h ago</span>
        <a class="output-open-overlay" href="/acta/outputs/morning-operator-brief">OPEN</a>
        <span>UNREAD</span>
        <button>Mark read</button>
      </article>
    </body></html>
    """

    assert acta_browser_uat._validate_outputs_contract(dom) == []


def test_validate_outputs_contract_rejects_unrelated_ask_followup_href_as_open_affordance():
    dom = """
    <html><body>
      <h1>Outputs</h1>
      <article class="output-row">
        <h2>Latest artifact</h2>
        <span>CONF HIGH</span>
        <span>SOURCE digest JOB daily ID 2026-05-25_09-00-00</span>
        <span>SCHEDULE daily brief OUTPUT 10m ago</span>
        <span>SIGNED/OPEN</span>
        <a class="followup-meta" href="https://t.me/example">FOLLOW-UP</a>
        <a class="ask" href="https://t.me/example">ASK</a>
        <span>UNREAD</span>
        <button>Mark read</button>
      </article>
    </body></html>
    """

    failures = acta_browser_uat._validate_outputs_contract(dom)

    assert "Output row 1 is missing clickable artifact-open affordance (artifact-open href/data-open-url)" in failures


def test_validate_outputs_contract_rejects_generic_open_class_as_open_affordance():
    dom = """
    <html><body>
      <h1>Outputs</h1>
      <article class="output-row">
        <h2>Latest artifact</h2>
        <span>CONF HIGH</span>
        <span>SOURCE digest JOB daily ID 2026-05-25_09-00-00</span>
        <span>SCHEDULE daily brief OUTPUT 10m ago</span>
        <span>SIGNED/OPEN</span>
        <a class="open" href="https://t.me/example">OPEN THREAD</a>
        <span>UNREAD</span>
        <button>Mark read</button>
      </article>
    </body></html>
    """

    failures = acta_browser_uat._validate_outputs_contract(dom)

    assert "Output row 1 is missing clickable artifact-open affordance (artifact-open href/data-open-url)" in failures


@pytest.mark.parametrize(
    "affordance_html",
    [
        '<a href="">OPEN</a>',
        '<a href="   ">OPEN</a>',
        '<a href="#">OPEN</a>',
        '<a href="javascript:alert(1)">OPEN</a>',
        '<article class="output-row" data-open-url="">',
        '<article class="output-row" data-open-url="   ">',
        '<article class="output-row" data-open-url="#">',
        '<article class="output-row" data-open-url="javascript:alert(1)">',
    ],
)
def test_validate_outputs_contract_rejects_empty_or_invalid_clickable_affordances(affordance_html: str):
    if affordance_html.startswith("<article"):
        row_start = affordance_html
        action_html = "<span>SIGNED</span>"
    else:
        row_start = '<article class="output-row">'
        action_html = affordance_html
    dom = f"""
    <html><body>
      <h1>Outputs</h1>
      {row_start}
        <h2>Latest artifact</h2>
        <span>CONF HIGH</span>
        <span>SOURCE digest JOB daily ID 2026-05-25_09-00-00</span>
        <span>SCHEDULE daily brief OUTPUT 10m ago</span>
        {action_html}
        <span>UNREAD</span>
        <button>Mark read</button>
      </article>
    </body></html>
    """

    failures = acta_browser_uat._validate_outputs_contract(dom)

    assert "Output row 1 is missing clickable artifact-open affordance (artifact-open href/data-open-url)" in failures


def test_validate_outputs_contract_extracts_rows_with_void_tags_inside():
    dom = """
    <html><body>
      <h1>Outputs</h1>
      <article class="output-row" data-open-url="/acta/outputs/morning-operator-brief">
        <h2>Morning operator brief</h2>
        <br><img src="x" alt=""><input type="hidden" value="ignored">
        <span>CONF HIGH</span>
        <span>SOURCE morning_digest JOB daily ID 2026-05-25_09-00-00</span>
        <span>SCHEDULE daily brief OUTPUT 2h ago</span>
        <span>SIGNED</span>
        <span>UNREAD</span>
        <button>Mark read</button>
      </article>
    </body></html>
    """

    assert acta_browser_uat._extract_html_by_class(dom, "output-row") == [
        """<article class="output-row" data-open-url="/acta/outputs/morning-operator-brief">
        <h2>Morning operator brief</h2>
        <br><img src="x" alt=""><input type="hidden" value="ignored">
        <span>CONF HIGH</span>
        <span>SOURCE morning_digest JOB daily ID 2026-05-25_09-00-00</span>
        <span>SCHEDULE daily brief OUTPUT 2h ago</span>
        <span>SIGNED</span>
        <span>UNREAD</span>
        <button>Mark read</button>
      </article>"""
    ]
    assert acta_browser_uat._validate_outputs_contract(dom) == []


def test_validate_outputs_contract_fails_when_action_copy_has_no_clickable_open_affordance():
    dom = """
    <html><body>
      <h1>Outputs</h1>
      <article class="output-row">
        <h2>Latest artifact</h2>
        <span>CONF HIGH</span>
        <span>SOURCE digest JOB daily ID 2026-05-25_09-00-00</span>
        <span>SCHEDULE daily brief OUTPUT 10m ago</span>
        <a>OPEN</a>
        <span>UNREAD</span>
        <button>Mark read</button>
      </article>
    </body></html>
    """

    failures = acta_browser_uat._validate_outputs_contract(dom)

    assert "Output row 1 is missing clickable artifact-open affordance (artifact-open href/data-open-url)" in failures


def test_validate_outputs_contract_fails_when_signed_open_row_lacks_read_state_and_toggle():
    dom = """
    <html><body>
      <h1>Outputs</h1>
      <article class="output-row">
        <h2>Latest artifact</h2>
        <span>CONF MED</span>
        <span>SOURCE digest JOB daily ID 2026-05-25_09-00-00</span>
        <span>SCHEDULE daily brief OUTPUT 10m ago</span>
        <a>SIGNED</a>
      </article>
    </body></html>
    """

    failures = acta_browser_uat._validate_outputs_contract(dom)

    assert "Output row 1 is missing read/unread state" in failures
    assert "Output row 1 is missing Mark read/Mark unread toggle" in failures


def test_validate_outputs_contract_does_not_count_mark_read_as_read_state():
    dom = """
    <html><body>
      <h1>Outputs</h1>
      <article class="output-row">
        <h2>Latest artifact</h2>
        <span>CONF MED</span>
        <span>SOURCE digest JOB daily ID 2026-05-25_09-00-00</span>
        <span>SCHEDULE daily brief OUTPUT 10m ago</span>
        <a>OPEN</a>
        <button>Mark read</button>
      </article>
    </body></html>
    """

    failures = acta_browser_uat._validate_outputs_contract(dom)

    assert "Output row 1 is missing read/unread state" in failures
    assert "Output row 1 is missing Mark read/Mark unread toggle" not in failures


def test_validate_outputs_contract_requires_source_or_job_plus_id_provenance():
    dom = """
    <html><body>
      <h1>Outputs</h1>
      <article class="output-row">
        <h2>Latest artifact</h2>
        <span>CONF MED</span>
        <span>JOB daily</span>
        <span>SCHEDULE daily brief OUTPUT 10m ago</span>
        <a>No signed link</a>
      </article>
    </body></html>
    """

    assert "Output row 1 is missing source/provenance copy (SOURCE or JOB/ID)" in acta_browser_uat._validate_outputs_contract(dom)


def test_validate_outputs_contract_fails_on_raw_log_leakage_but_not_operator_empty_copy():
    assert acta_browser_uat._validate_outputs_contract(_valid_outputs_dom().replace("</main>", "<p>FOLLOW-UP: No visible response</p></main>")) == []

    failures = acta_browser_uat._validate_outputs_contract(
        _valid_outputs_dom().replace("</main>", "<pre>## Prompt\ntool output\n/Users/mozzie/.hermes\napi_key=secret</pre></main>")
    )

    assert "Outputs DOM contains raw prompt/tool/path leakage" in failures


def test_validate_outputs_contract_reuses_common_browser_failure_checks():
    failures = acta_browser_uat._validate_outputs_contract(
        _valid_outputs_dom(),
        horizontal_overflow=True,
        console_output="Uncaught Error: boom",
        errors_output="TypeError: failed during render",
    )

    assert "Horizontal overflow detected at mobile viewport" in failures
    assert "Browser console contains error/exception output" in failures
    assert "Browser page errors were reported" in failures


def test_validate_outputs_contract_fails_on_acta_sign_in_wall_only():
    failures = acta_browser_uat._validate_outputs_contract("<html><body><h1>Sign in to Acta</h1><p>Acta access token</p></body></html>")

    assert failures == [
        "Acta sign-in wall rendered; pass a local --html artifact or validate with authenticated browser storage"
    ]


def test_validate_archive_contract_accepts_source_signal_cards():
    assert acta_browser_uat._validate_archive_contract(_valid_archive_dom()) == []


def test_validate_archive_contract_rejects_broken_archive_cards():
    dom = """
    <html><body>
      <main>
        <h1>Archive</h1>
        <a class="archive-card" href="#"><strong>2026-05-24</strong><span>Acta Day</span></a>
      </main>
    </body></html>
    """

    failures = acta_browser_uat._validate_archive_contract(
        dom,
        horizontal_overflow=True,
        console_output="Uncaught Error: boom",
        errors_output="TypeError: failed during render",
    )

    assert "Archive card 1 has unsafe href: #" in failures
    assert "Archive card 1 is missing safe archive href" in failures
    assert "No archive card includes both safe archive href and numeric Visible/Silent/Missing counts" in failures
    assert "Horizontal overflow detected at mobile viewport" in failures
    assert "Browser console contains error/exception output" in failures
    assert "Browser page errors were reported" in failures


@pytest.mark.parametrize(
    "href",
    [
        "",
        "#",
        "javascript:alert(1)",
        "file:///tmp/acta.html",
        "data:text/html,boom",
        "https://example.com/archive/2026-05-24",
        "//example.com/archive/2026-05-24",
        "/archive/not-a-date",
        "/archive/2026-13-40",
        " /archive/2026-05-24",
    ],
)
def test_validate_archive_contract_rejects_unsafe_archive_hrefs(href: str):
    dom = f"""
    <html><body>
      <main>
        <h1>Archive</h1>
        <a class="archive-card" href="{href}">
          <strong>2026-05-24</strong><small>Visible 1 · Silent 0 · Missing 0</small>
        </a>
      </main>
    </body></html>
    """

    failures = acta_browser_uat._validate_archive_contract(dom)

    display_href = href if href else "<empty>"
    assert f"Archive card 1 has unsafe href: {display_href}" in failures
    assert "Archive card 1 is missing safe archive href" in failures


def test_validate_archive_contract_allows_safe_legacy_card_when_source_signal_card_exists():
    dom = """
    <html><body>
      <main>
        <h1>Archive</h1>
        <a class="archive-card" href="/archive/2026-05-24"><strong>2026-05-24</strong><span>Acta Day</span></a>
        <a class="archive-card" href="/archive/2026-05-25">
          <strong>2026-05-25</strong><small>Visible 3 · Silent 1 · Missing 0</small>
        </a>
      </main>
    </body></html>
    """

    assert acta_browser_uat._validate_archive_contract(dom) == []


def test_validate_archive_contract_requires_numeric_visible_silent_missing_counts():
    dom = """
    <html><body>
      <main>
        <h1>Archive</h1>
        <a class="archive-card" href="/archive/2026-05-24">
          <strong>2026-05-24</strong><small>source counts available</small>
        </a>
      </main>
    </body></html>
    """

    assert acta_browser_uat._validate_archive_contract(dom) == [
        "No archive card includes both safe archive href and numeric Visible/Silent/Missing counts"
    ]


def test_url_target_rejects_userinfo():
    args = type("Args", (), {"html": None, "url": "https://user:secret@example.com/acta"})()

    with pytest.raises(SystemExit, match="--url must not include userinfo"):
        acta_browser_uat._target_url(args)


def test_run_writes_sanitized_report_url_for_http_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    screenshot = tmp_path / "uat" / "acta-uat.png"

    def fake_run_chrome(url: str, artifact_dir: Path, timeout: int, viewport_width: int, viewport_height: int):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        screenshot.write_bytes(b"png")
        return acta_browser_uat.BrowserResult(
            url=url,
            dom=_valid_feed_dom(),
            screenshot=screenshot,
            browser_path=Path("fake-browser"),
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            layout_metrics={"innerWidth": viewport_width, "innerHeight": viewport_height, "scrollWidth": viewport_width},
        )

    monkeypatch.setattr(acta_browser_uat, "_run_chrome", fake_run_chrome)
    args = type(
        "Args",
        (),
        {
            "html": None,
            "url": "https://example.com:8443/acta/dashboard?token=secret#frag",
            "artifact_dir": str(tmp_path / "uat"),
            "timeout": 1,
            "viewport_width": 390,
            "viewport_height": 844,
        },
    )()

    assert acta_browser_uat.run(args) == 0
    report = json.loads((tmp_path / "uat" / "acta-uat-report.json").read_text(encoding="utf-8"))
    assert report["url"] == "https://example.com:8443/acta/dashboard"
    assert report["persona"] == "mobile Acta operator checking dashboard feed lanes"
    assert "mobile" in report["scenario"].lower()
    assert report["viewport"] == {"width": 390, "height": 844}
    assert report["layout_metrics"]["innerWidth"] == 390


def test_run_writes_jobs_report_metadata_for_jobs_scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    screenshot = tmp_path / "uat" / "acta-uat.png"

    def fake_run_chrome(url: str, artifact_dir: Path, timeout: int, viewport_width: int, viewport_height: int):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        screenshot.write_bytes(b"png")
        return acta_browser_uat.BrowserResult(
            url=url,
            dom=_valid_jobs_dom(),
            screenshot=screenshot,
            browser_path=Path("fake-browser"),
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            layout_metrics={"innerWidth": viewport_width, "innerHeight": viewport_height, "scrollWidth": viewport_width},
        )

    monkeypatch.setattr(acta_browser_uat, "_run_chrome", fake_run_chrome)
    args = type(
        "Args",
        (),
        {
            "html": None,
            "url": "https://example.com/acta/jobs?token=secret#frag",
            "artifact_dir": str(tmp_path / "uat"),
            "timeout": 1,
            "viewport_width": 390,
            "viewport_height": 844,
            "scenario": "jobs",
        },
    )()

    assert acta_browser_uat.run(args) == 0
    output = capsys.readouterr().out
    report = json.loads((tmp_path / "uat" / "acta-uat-report.json").read_text(encoding="utf-8"))
    assert "Job rows: 2" in output
    assert report["url"] == "https://example.com/acta/jobs"
    assert report["scenario_key"] == "jobs"
    assert report["persona"] == "mobile Acta operator inspecting Jobs/source-runs freshness and confidence"
    assert "Jobs/source-runs" in report["scenario"]
    assert report["job_rows"] == 2
    assert "daily_rows" not in report
    assert "dev_rows" not in report


def test_run_writes_outputs_report_metadata_for_outputs_scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    screenshot = tmp_path / "uat" / "acta-uat.png"
    artifact_screenshot = tmp_path / "uat" / "output-artifact" / "acta-uat.png"
    opened_urls: list[str] = []

    def fake_run_chrome(url: str, artifact_dir: Path, timeout: int, viewport_width: int, viewport_height: int):
        opened_urls.append(url)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        shot = artifact_screenshot if "morning-operator-brief" in url else screenshot
        shot.write_bytes(b"png")
        return acta_browser_uat.BrowserResult(
            url=url,
            dom="<html><body><main><h1>Morning operator brief</h1><p>Decision-ready output artifact.</p></main></body></html>"
            if "morning-operator-brief" in url
            else _valid_outputs_dom(),
            screenshot=shot,
            browser_path=Path("fake-browser"),
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            layout_metrics={"innerWidth": viewport_width, "innerHeight": viewport_height, "scrollWidth": viewport_width},
        )

    monkeypatch.setattr(acta_browser_uat, "_run_chrome", fake_run_chrome)
    args = type(
        "Args",
        (),
        {
            "html": None,
            "url": "https://example.com/acta/outputs?token=secret#frag",
            "artifact_dir": str(tmp_path / "uat"),
            "timeout": 1,
            "viewport_width": 390,
            "viewport_height": 844,
            "scenario": "outputs",
        },
    )()

    assert acta_browser_uat.run(args) == 0
    output = capsys.readouterr().out
    report = json.loads((tmp_path / "uat" / "acta-uat-report.json").read_text(encoding="utf-8"))
    assert "Output rows: 2" in output
    assert "Opened output artifact: https://example.com/acta/outputs/morning-operator-brief" in output
    assert opened_urls == [
        "https://example.com/acta/outputs?token=secret#frag",
        "https://example.com/acta/outputs/morning-operator-brief",
    ]
    assert report["url"] == "https://example.com/acta/outputs"
    assert report["scenario_key"] == "outputs"
    assert report["persona"] == "mobile Acta operator inspecting Outputs shelf artifacts"
    assert "Outputs shelf" in report["scenario"]
    assert report["output_rows"] == 2
    assert report["opened_output_artifact_url"] == "https://example.com/acta/outputs/morning-operator-brief"
    assert report["output_artifact_screenshot"] == str(artifact_screenshot)
    assert report["output_artifact_horizontal_overflow"] is False
    assert report["output_artifact_failures"] == []
    assert report["action_state_probe"] == {}
    assert "job_rows" not in report
    assert "daily_rows" not in report
    assert "dev_rows" not in report


def test_first_output_artifact_url_prefers_real_artifact_affordance_over_followup_links():
    dom = """
    <html><body>
      <article class="output-row">
        <h2>Signed output</h2>
        <a class="followup" href="https://t.me/c/1/2">FOLLOW-UP</a>
        <a class="output-open-overlay" href="brief.html?sig=abc&token=secret">OPEN</a>
      </article>
    </body></html>
    """

    assert acta_browser_uat._first_output_artifact_url(dom, "file:///tmp/acta/outputs.html") == "file:///tmp/acta/brief.html?sig=abc&token=secret"


@pytest.mark.parametrize(
    "target",
    [
        "//evil.example/artifact.html",
        "https://evil.example/artifact.html",
        "https://user:pass@example.com/artifact.html",
        "file:///Users/mozzie/.hermes/config.yaml",
        "data:text/html,<h1>leak</h1>",
        "ftp://example.com/artifact.html",
        "../secret.html",
        "%2e%2e%2fsecret.html",
        "https://t.me/c/1/2",
        "?token=secret",
        "#artifact",
    ],
)
def test_first_output_artifact_url_rejects_unsafe_or_non_artifact_targets(target: str):
    dom = f"""
    <html><body>
      <article class="output-row" data-open-url="{target}">
        <h2>Signed output</h2><span>OPEN</span>
      </article>
    </body></html>
    """

    assert acta_browser_uat._first_output_artifact_url(dom, "https://example.com/acta/outputs") is None


def test_first_output_artifact_url_allows_same_origin_root_relative_target():
    dom = """
    <html><body>
      <article class="output-row" data-open-url="/acta/outputs/morning-brief.html?sig=abc&token=secret">
        <h2>Signed output</h2><span>OPEN</span>
      </article>
    </body></html>
    """

    assert (
        acta_browser_uat._first_output_artifact_url(dom, "https://example.com/acta/outputs")
        == "https://example.com/acta/outputs/morning-brief.html?sig=abc&token=secret"
    )


def test_report_url_strips_file_query_and_fragment():
    assert (
        acta_browser_uat._report_url("file:///tmp/acta/brief.html?sig=abc&token=secret#frag")
        == "file:///tmp/acta/brief.html"
    )


def test_run_outputs_scenario_fails_when_no_artifact_target_opens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    screenshot = tmp_path / "uat" / "acta-uat.png"

    def fake_run_chrome(url: str, artifact_dir: Path, timeout: int, viewport_width: int, viewport_height: int):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        screenshot.write_bytes(b"png")
        return acta_browser_uat.BrowserResult(
            url=url,
            dom="""
            <html><body><main><h1>Outputs</h1><article class="output-row">
              <h2>Catalog-only artifact</h2><span>CATALOG</span><span>SOURCE system ID catalog-17</span>
              <span>PINNED catalog age 1 day ago</span><span>No public link</span>
            </article></main></body></html>
            """,
            screenshot=screenshot,
            browser_path=Path("fake-browser"),
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            layout_metrics={"innerWidth": viewport_width, "innerHeight": viewport_height, "scrollWidth": viewport_width},
        )

    monkeypatch.setattr(acta_browser_uat, "_run_chrome", fake_run_chrome)
    args = type(
        "Args",
        (),
        {
            "html": None,
            "url": "https://example.com/acta/outputs",
            "artifact_dir": str(tmp_path / "uat"),
            "timeout": 1,
            "viewport_width": 390,
            "viewport_height": 844,
            "scenario": "outputs",
        },
    )()

    assert acta_browser_uat.run(args) == 1
    assert "No actionable Outputs artifact target found/opened" in capsys.readouterr().out


def test_validate_output_artifact_contract_fails_on_raw_log_leakage():
    dom = """
    <html><body><pre>## Prompt\nTool call: terminal\n/Users/mozzie/.hermes/secrets.env\napi_key=secret</pre></body></html>
    """

    assert acta_browser_uat._validate_output_artifact_contract(dom) == [
        "Opened output artifact contains raw prompt/tool/path leakage"
    ]


def test_run_outputs_scenario_fails_when_opened_artifact_leaks_raw_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    screenshot = tmp_path / "uat" / "acta-uat.png"
    artifact_screenshot = tmp_path / "uat" / "output-artifact" / "acta-uat.png"

    def fake_run_chrome(url: str, artifact_dir: Path, timeout: int, viewport_width: int, viewport_height: int):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        shot = artifact_screenshot if "morning-operator-brief" in url else screenshot
        shot.write_bytes(b"png")
        return acta_browser_uat.BrowserResult(
            url=url,
            dom="<html><body><pre>## Prompt\nTool output\n/Users/mozzie/.hermes/config.yaml</pre></body></html>"
            if "morning-operator-brief" in url
            else _valid_outputs_dom(),
            screenshot=shot,
            browser_path=Path("fake-browser"),
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            layout_metrics={"innerWidth": viewport_width, "innerHeight": viewport_height, "scrollWidth": viewport_width},
        )

    monkeypatch.setattr(acta_browser_uat, "_run_chrome", fake_run_chrome)
    args = type(
        "Args",
        (),
        {
            "html": None,
            "url": "https://example.com/acta/outputs",
            "artifact_dir": str(tmp_path / "uat"),
            "timeout": 1,
            "viewport_width": 390,
            "viewport_height": 844,
            "scenario": "outputs",
        },
    )()

    assert acta_browser_uat.run(args) == 1
    output = capsys.readouterr().out
    report = json.loads((tmp_path / "uat" / "acta-uat-report.json").read_text(encoding="utf-8"))
    assert "Opened output artifact contains raw prompt/tool/path leakage" in output
    assert report["output_artifact_failures"] == ["Opened output artifact contains raw prompt/tool/path leakage"]


def test_main_defaults_to_feed_scenario(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured = {}

    def fake_run(args):
        captured["scenario"] = args.scenario
        captured["html"] = args.html
        return 0

    html_path = tmp_path / "acta.html"
    html_path.write_text(_valid_feed_dom(), encoding="utf-8")
    monkeypatch.setattr(acta_browser_uat, "run", fake_run)

    assert acta_browser_uat.main(["--html", str(html_path)]) == 0
    assert captured == {"scenario": "feed", "html": str(html_path)}


def test_main_accepts_outputs_scenario(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured = {}

    def fake_run(args):
        captured["scenario"] = args.scenario
        captured["html"] = args.html
        return 0

    html_path = tmp_path / "acta.html"
    html_path.write_text(_valid_outputs_dom(), encoding="utf-8")
    monkeypatch.setattr(acta_browser_uat, "run", fake_run)

    assert acta_browser_uat.main(["--html", str(html_path), "--scenario", "outputs"]) == 0
    assert captured == {"scenario": "outputs", "html": str(html_path)}


def test_run_chrome_sets_mobile_viewport_clears_and_collects_console_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    commands: list[list[str]] = []

    def fake_agent_browser_command() -> list[str]:
        return ["agent-browser"]

    def fake_run_agent_browser(command: list[str], args: list[str], timeout: int):
        commands.append(args)
        stdout = ""
        if args == ["eval", "document.documentElement.outerHTML"]:
            stdout = json.dumps(_valid_feed_dom())
        elif args and args[0] == "eval":
            stdout = json.dumps({"innerWidth": 390, "innerHeight": 844, "scrollWidth": 390, "bodyScrollWidth": 390, "mobilebarVisible": True})
        elif args and args[0] == "screenshot":
            Path(args[1]).write_bytes(b"png")
        return subprocess.CompletedProcess([*command, *args], 0, stdout)

    monkeypatch.setattr(acta_browser_uat, "_agent_browser_command", fake_agent_browser_command)
    monkeypatch.setattr(acta_browser_uat, "_run_agent_browser", fake_run_agent_browser)

    result = acta_browser_uat._run_chrome("file:///tmp/acta.html", tmp_path, 1, 390, 844)

    assert commands[:4] == [
        ["set", "viewport", "390", "844"],
        ["console", "--clear"],
        ["errors", "--clear"],
        ["open", "file:///tmp/acta.html"],
    ]
    assert ["console"] in commands
    assert ["errors"] in commands
    assert commands[-1] == ["close", "--all"]
    assert result.viewport_width == 390
    assert result.viewport_height == 844
    assert result.layout_metrics["innerWidth"] == 390
    assert result.horizontal_overflow is False


def test_validate_feed_contract_fails_on_mobile_overflow_console_and_page_errors():
    failures = acta_browser_uat._validate_feed_contract(
        _valid_feed_dom(),
        horizontal_overflow=True,
        console_output="Uncaught Error: boom",
        errors_output="TypeError: failed during render",
    )

    assert "Horizontal overflow detected at mobile viewport" in failures
    assert "Browser console contains error/exception output" in failures
    assert "Browser page errors were reported" in failures


def test_validate_feed_contract_ignores_empty_console_and_no_page_errors():
    failures = acta_browser_uat._validate_feed_contract(
        _valid_feed_dom(),
        console_output="No console messages",
        errors_output="No page errors",
    )

    assert failures == []


def test_sanitize_diagnostic_output_redacts_tokens_and_truncates():
    output = "GET https://acta.imperatr.com/r/detail.html?token=abc123secret&api_key=apikeysecret&sig=deadbeef Bearer sk-test-secret-token-1234567890 X-Api-Key: xapikey-secret-123456 public-diagnostic " + "x" * 4100

    sanitized = acta_browser_uat._sanitize_diagnostic_output(output, limit=160)

    assert "abc123secret" not in sanitized
    assert "apikeysecret" not in sanitized
    assert "deadbeef" not in sanitized
    assert "sk-test-secret-token" not in sanitized
    assert "xapikey-secret" not in sanitized
    assert "token=[REDACTED]" in sanitized
    assert "api_key=[REDACTED]" in sanitized
    assert "sig=[REDACTED]" in sanitized
    assert "Bearer [REDACTED]" in sanitized
    assert "X-Api-Key: [REDACTED]" in sanitized
    assert "truncated" in sanitized


def test_run_agent_browser_timeout_redacts_command_and_output(monkeypatch: pytest.MonkeyPatch):
    def fake_subprocess_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["agent-browser", "open", "https://acta.imperatr.com/r/detail.html?token=abc123secret&sig=deadbeef"],
            timeout=1,
            output="X-Api-Key: xapikey-secret-123456",
        )

    monkeypatch.setattr(acta_browser_uat.subprocess, "run", fake_subprocess_run)

    with pytest.raises(RuntimeError) as excinfo:
        acta_browser_uat._run_agent_browser(
            ["agent-browser"],
            ["open", "https://acta.imperatr.com/r/detail.html?token=abc123secret&sig=deadbeef"],
            1,
        )

    message = str(excinfo.value)
    assert "abc123secret" not in message
    assert "deadbeef" not in message
    assert "xapikey-secret" not in message
    assert "token=[REDACTED]" in message
    assert "sig=[REDACTED]" in message
    assert "X-Api-Key: [REDACTED]" in message
