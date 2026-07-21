from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pytest


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_contact_reminder_brief_under_test",
        Path(__file__).resolve().parents[2] / "scripts" / "contact_reminder_brief.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_brief_returns_due_and_upcoming_annual_occasions():
    module = _load_module()
    records = [
        {
            "name": "Finn",
            "relationship": "girlfriend",
            "birthday": "2000-04-10",
            "notes": "Likes playful, low-pressure gestures.",
        },
        {
            "name": "Jeremy",
            "birthday": "1994-05-23",
        },
    ]

    brief = module.build_brief(records, today=date(2026, 4, 8), window_days=7)

    assert brief["silent"] is False
    assert [item["name"] for item in brief["items"]] == ["Finn"]
    assert brief["items"][0]["event"] == "birthday"
    assert brief["items"][0]["days_until"] == 2
    assert brief["items"][0]["turning_age"] == 26
    rendered = module.render_text(brief)
    assert "TL;DR" in rendered
    assert "Finn" in rendered
    assert "2 天後" in rendered
    assert "26 歲" in rendered
    assert "手動傳送" in rendered


def test_build_brief_is_silent_when_nothing_is_within_window():
    module = _load_module()
    records = [{"name": "Jeremy", "birthday": "1994-05-23"}]

    brief = module.build_brief(records, today=date(2026, 4, 8), window_days=7)

    assert brief == {"silent": True, "items": []}
    assert module.render_text(brief) == "[SILENT]"


def test_feb_29_annual_occasion_is_observed_on_feb_28_in_non_leap_year():
    module = _load_module()
    records = [{"name": "Leap Friend", "birthday": "2000-02-29"}]

    brief = module.build_brief(records, today=date(2027, 2, 27), window_days=2)

    assert brief["silent"] is False
    assert brief["items"][0]["date"] == "2027-02-28"
    assert brief["items"][0]["days_until"] == 1
    assert brief["items"][0]["turning_age"] == 27


def test_load_records_accepts_json_mapping_with_contacts_key(tmp_path):
    module = _load_module()
    path = tmp_path / "contacts.json"
    path.write_text(
        json.dumps({"contacts": [{"name": "Finn", "birthday": "2000-04-10"}]}),
        encoding="utf-8",
    )

    assert module.load_records(path) == [{"name": "Finn", "birthday": "2000-04-10"}]


def test_main_prints_text_brief_for_cli_input(tmp_path, capsys):
    module = _load_module()
    path = tmp_path / "contacts.json"
    path.write_text(
        json.dumps({"contacts": [{"name": "Finn", "birthday": "2000-04-10"}]}),
        encoding="utf-8",
    )

    assert module.main(["--input", str(path), "--today", "2026-04-08", "--window-days", "7"]) == 0

    output = capsys.readouterr().out
    assert "TL;DR" in output
    assert "Finn" in output
    assert "手動傳送" in output


def test_main_can_emit_json_brief(tmp_path, capsys):
    module = _load_module()
    path = tmp_path / "contacts.json"
    path.write_text(
        json.dumps({"contacts": [{"name": "Finn", "birthday": "2000-04-10"}]}),
        encoding="utf-8",
    )

    assert module.main([
        "--input",
        str(path),
        "--today",
        "2026-04-08",
        "--window-days",
        "7",
        "--format",
        "json",
    ]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["silent"] is False
    assert payload["items"][0]["name"] == "Finn"


def test_main_rejects_negative_window_days(tmp_path):
    module = _load_module()
    path = tmp_path / "contacts.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(SystemExit):
        module.main(["--input", str(path), "--window-days", "-1"])
