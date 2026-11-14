"""A running process follows an in-place edit of ``timezone`` in config.yaml.

Regression for #54260: the timezone cache was keyed on the config path alone, which stays equal
across an edit, and ``reset_cache()`` has no production caller. A gateway that resolved its zone
before ``hermes config set timezone ...`` ran in another process kept the old zone until restart,
so unpinned cron jobs fired at the wrong wall-clock hour.
"""
from datetime import datetime

import pytest

import hermes_time


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_TIMEZONE", raising=False)
    hermes_time.reset_cache()
    yield home / "config.yaml"
    hermes_time.reset_cache()


def test_in_place_config_edit_changes_the_resolved_zone(config_path):
    config_path.write_text("timezone: UTC\n", encoding="utf-8")
    assert hermes_time.get_timezone_name() == "UTC"

    config_path.write_text("timezone: Asia/Tokyo\n", encoding="utf-8")

    assert hermes_time.get_timezone_name() == "Asia/Tokyo"
    assert hermes_time.now().utcoffset().total_seconds() == 9 * 3600


def test_unpinned_cron_job_follows_the_edited_zone(config_path):
    from cron.jobs import compute_next_run, parse_schedule

    schedule = parse_schedule("0 14 * * *")
    config_path.write_text("timezone: UTC\n", encoding="utf-8")
    before = datetime.fromisoformat(compute_next_run(schedule))
    assert before.utcoffset().total_seconds() == 0

    config_path.write_text("timezone: Asia/Tokyo\n", encoding="utf-8")
    after = datetime.fromisoformat(compute_next_run(schedule))

    assert after.utcoffset().total_seconds() == 9 * 3600
    assert (after.hour, after.minute) == (14, 0)

