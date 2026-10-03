"""User-facing contract for plain-English cron schedule displays."""

from datetime import datetime

import pytest
from croniter import croniter

import cron.jobs as jobs_mod


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temporary directory."""
    monkeypatch.setattr(jobs_mod, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", tmp_path / "cron" / "output")


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("*/10 * * * *", "Every 10 minutes, every day"),
        ("13 * * * *", "At minute 13 past every hour, every day"),
        (
            "2,17,32,47 * * * *",
            "At minutes 2, 17, 32, and 47 past every hour, every day",
        ),
        ("0 */6 * * *", "Every 6 hours starting at midnight, every day"),
        (
            "15 */2 * * *",
            "At 15 minutes past every 2 hours starting at midnight, every day",
        ),
        (
            "0 10,12,14,16,18 * * *",
            "At 10:00 AM, 12:00 PM, 2:00 PM, 4:00 PM, and 6:00 PM, every day",
        ),
        (
            "0 0-7,20-23 * * *",
            "At the start of each hour from 12:00 AM through 7:59 AM and from 8:00 PM through 11:59 PM, every day",
        ),
        ("0 9 * * 1", "At 9:00 AM every Monday"),
        (
            "0 10 1 1,4,7,10 *",
            "At 10:00 AM on day 1 in January, April, July, and October",
        ),
        ("0 10 12 8 *", "At 10:00 AM on August 12 every year"),
    ],
)
def test_humanize_cron_contract(expr, expected):
    assert jobs_mod.humanize_cron(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected_display", "base", "expected_next"),
    [
        (
            "*/30 7-22 * * *",
            "Every 30 minutes during each hour from 7:00 AM through 10:59 PM, every day",
            datetime(2026, 1, 1, 6, 59),
            datetime(2026, 1, 1, 7, 0),
        ),
        (
            "30 4 1 */3 *",
            "At 4:30 AM on day 1, every 3 months starting in January",
            datetime(2026, 1, 1, 0, 0),
            datetime(2026, 1, 1, 4, 30),
        ),
        (
            "0 9 1 * *",
            "At 9:00 AM on day 1 of every month",
            datetime(2026, 1, 1, 8, 0),
            datetime(2026, 1, 1, 9, 0),
        ),
    ],
)
def test_acceptance_examples_agree_with_croniter_next_fire(
    expr, expected_display, base, expected_next
):
    assert jobs_mod.humanize_cron(expr) == expected_display
    assert croniter(expr, base).get_next(datetime) == expected_next


def test_parse_schedule_uses_humanized_display():
    parsed = jobs_mod.parse_schedule("0 9 1 * *")

    assert parsed == {
        "kind": "cron",
        "expr": "0 9 1 * *",
        "display": "At 9:00 AM on day 1 of every month",
    }


def test_list_jobs_humanizes_legacy_cron_without_mutating_storage(tmp_cron_dir):
    expr = "30 4 1 */3 *"
    jobs_mod.save_jobs(
        [
            {
                "id": "legacy-cron",
                "name": "legacy",
                "prompt": "test",
                "schedule": {"kind": "cron", "expr": expr, "display": expr},
                "schedule_display": expr,
                "enabled": True,
                "state": "scheduled",
            }
        ]
    )

    listed = jobs_mod.list_jobs()[0]

    expected = "At 4:30 AM on day 1, every 3 months starting in January"
    assert listed["schedule"]["display"] == expected
    assert listed["schedule_display"] == expected
    assert jobs_mod.load_jobs()[0]["schedule"]["display"] == expr
    assert jobs_mod.load_jobs()[0]["schedule_display"] == expr

def test_list_jobs_keeps_a_custom_cron_display(tmp_cron_dir):
    jobs_mod.save_jobs(
        [
            {
                "id": "natural",
                "name": "natural",
                "prompt": "test",
                "schedule": {"kind": "cron", "expr": "0 9 * * 1", "display": "every monday 9am"},
                "schedule_display": "every monday 9am",
                "enabled": True,
                "state": "scheduled",
            }
        ]
    )

    assert jobs_mod.list_jobs()[0]["schedule_display"] == "every monday 9am"


@pytest.mark.parametrize("expr", ["0 9 * * MON-FRI", "0 9 * * 1-5", "0 9 L * *"])
def test_unsupported_forms_fall_back_to_the_expression(expr):
    assert jobs_mod.humanize_cron(expr) == expr
