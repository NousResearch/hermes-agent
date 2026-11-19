"""parse_schedule: extended cron fields and monthly-weekday phrases (cron/jobs_schedule_phrases.py)."""

import pytest

from cron.jobs import parse_schedule

pytest.importorskip("croniter")


@pytest.mark.parametrize("expr", ["0 9 * * 1#2", "0 9 * * mon#1", "0 0 L * *", "0 17 * * L5"])
def test_extended_cron_syntax_reaches_croniter(expr):
    """Issue #20858: nth-weekday (#) and last (L) cron fields are croniter's to validate."""
    assert parse_schedule(expr) == {"kind": "cron", "expr": expr, "display": expr}


@pytest.mark.parametrize(
    ("phrase", "expr"),
    [
        ("every second monday at 9am", "0 9 * * 1#2"),
        ("every last friday of the month at 5pm", "0 17 * * L5"),
        ("first and third tuesday at 10:00", "0 10 * * 2#1,2#3"),
        ("every monday and friday at 9am and 5pm", "0 9,17 * * 1,5"),
        ("every monday 9am", "0 9 * * 1"),
    ],
)
def test_monthly_weekday_phrases_compile_to_cron(phrase, expr):
    """'every second monday', 'last friday of the month', two times a day: one cron line."""
    assert parse_schedule(phrase) == {"kind": "cron", "expr": expr, "display": phrase}
