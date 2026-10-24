"""Blueprint content (title, description, slot text, schedule) follows display.language.

English stays the runtime source in ``cron/blueprint_catalog.py``; a locale file overrides only
what it has translated, so the ``blueprint`` surface can ship partially. These tests pin both
halves: the English source is unchanged when no language is asked for, and a translated locale
reaches every user-visible string — including the created job's name.
"""

from __future__ import annotations

import pytest

from cron.blueprint_catalog import (
    CATALOG,
    blueprint_catalog_entry,
    blueprint_form_schema,
    fill_blueprint,
    get_blueprint,
)


def test_no_language_keeps_the_english_source():
    schema = blueprint_form_schema(get_blueprint("morning-brief"))
    assert schema["title"] == "Morning briefing"
    assert schema["description"].startswith("A short daily briefing")
    assert schema["fields"][0]["label"] == "What time?"
    assert blueprint_form_schema(get_blueprint("morning-brief"), None) == schema


def test_zh_localizes_title_description_and_slot_text():
    schema = blueprint_form_schema(get_blueprint("morning-brief"), "zh")
    assert schema["title"] == "晨间简报"
    assert "日程" in schema["description"]
    fields = {f["name"]: f for f in schema["fields"]}
    assert fields["time"]["label"] == "几点？"
    assert fields["time"]["help"].startswith("24 小时制")
    assert fields["deliver"]["label"] == "发送到哪里？"


def test_every_builtin_blueprint_has_a_zh_title():
    """A missing translation is a silent English leak — keep the shipped catalog complete."""
    missing = [b.key for b in CATALOG if blueprint_form_schema(b, "zh")["title"] == b.title]
    assert not missing, f"no zh title for: {missing}"


def test_per_blueprint_slot_label_beats_the_shared_one():
    learn = {f["name"]: f["label"] for f in blueprint_form_schema(get_blueprint("learn-daily"), "zh")["fields"]}
    news = {f["name"]: f["label"] for f in blueprint_form_schema(get_blueprint("news-digest"), "zh")["fields"]}
    assert learn["topic"] == "想学什么？"     # per-blueprint override
    assert news["topic"] == "什么主题？"      # shared blueprint.slot.topic.label


def test_unknown_language_falls_back_to_english():
    schema = blueprint_form_schema(get_blueprint("morning-brief"), "xx")
    assert schema["title"] == "Morning briefing"


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("morning-brief", "每天 08:00"),        # daily at {time}
        ("weekly-review", "周日 18:00"),        # weekday name from a `day` slot
        ("news-digest", "工作日 18:00"),        # preset from a `recurrence` slot
    ],
)
def test_schedule_text_is_localized(key, expected):
    assert blueprint_catalog_entry(get_blueprint(key), "zh")["scheduleHuman"] == expected


def test_schedule_text_stays_english_without_a_language():
    assert blueprint_catalog_entry(get_blueprint("morning-brief"))["scheduleHuman"] == "daily at 08:00"


def test_created_job_name_follows_the_language():
    zh = fill_blueprint(get_blueprint("morning-brief"), {"time": "08:00"}, lang="zh")
    en = fill_blueprint(get_blueprint("morning-brief"), {"time": "08:00"})
    assert zh["name"] == "晨间简报"
    assert en["name"] == "Morning briefing"
    # The prompt is the instruction the model runs, not user-visible chrome: it stays English and
    # its output language is anchored by the cron platform hint instead.
    assert zh["prompt"] == en["prompt"]
