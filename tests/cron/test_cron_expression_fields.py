"""``parse_schedule`` takes a 5-field cron expression: a seconds or year field is refused, with a
5-field replacement suggested only when that replacement is itself accepted."""

import pytest

from cron.jobs import parse_schedule


@pytest.mark.parametrize("expr, hint", [
    ("0 0 9 * * *", "0 9 * * *"),
    ("0 30 9 * * MON-FRI", "30 9 * * MON-FRI"),
    ("0 0 9 * * * 2027", "0 9 * * *"),
    ("0 0 12 * * ?", None),
])
def test_cron_with_seconds_or_year_field_is_rejected_not_reinterpreted(expr, hint):
    """croniter reads a 6th field as seconds; the seconds-first (Quartz/Spring) daily-9am
    ``0 0 9 * * *`` used to be stored as "the 9th of each month at midnight". Any suggested
    replacement must itself be accepted."""
    pytest.importorskip("croniter")
    with pytest.raises(ValueError, match="5 fields") as exc:
        parse_schedule(expr)
    if hint is None:
        assert "drop it" not in str(exc.value)
    else:
        assert f"'{hint}'" in str(exc.value)
        assert parse_schedule(hint)["expr"] == hint
