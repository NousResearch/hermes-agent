import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "meeting_prep_brief.py"


def load_module():
    spec = importlib.util.spec_from_file_location("meeting_prep_brief", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_markdown_brief_filters_and_sorts_upcoming_json_meetings(tmp_path):
    module = load_module()
    source = tmp_path / "meetings.json"
    write_json(
        source,
        {
            "meetings": [
                {
                    "title": "Later call",
                    "date": "2026-06-27",
                    "time": "16:00",
                    "attendees": ["Finn"],
                    "goals": ["Book Japan hotels"],
                    "prep": "Check passport dates",
                },
                {
                    "name": "Earlier BD sync",
                    "datetime": "2026-06-24T09:30:00+08:00",
                    "attendees": "Daniel, Yuki",
                    "questions": ["Any blockers?"],
                    "materials": ["Pipeline notes"],
                },
                {"title": "Done meeting", "date": "2026-06-25", "status": "done"},
                {"title": "Too far", "date": "2026-07-10"},
            ]
        },
    )

    brief = module.build_brief([source], today="2026-06-23", days=7)

    assert brief.startswith("# Meeting Prep Brief")
    assert brief.index("Earlier BD sync") < brief.index("Later call")
    assert "Daniel" in brief and "Yuki" in brief
    assert "Done meeting" not in brief
    assert "Too far" not in brief
    assert "Missing prep fields" in brief
    assert "## Quick Actions" in brief


def test_silent_when_no_active_meetings_in_window(tmp_path):
    module = load_module()
    source = tmp_path / "meetings.json"
    write_json(source, [{"title": "Future", "date": "2026-07-15"}])

    assert module.build_brief([source], today="2026-06-23", days=7) == "[SILENT]"


def test_json_output_includes_unparseable_date_warning(tmp_path):
    module = load_module()
    source = tmp_path / "meetings.json"
    write_json(
        source,
        [
            {"title": "No date", "date": "next tuesday"},
            {"title": "Investor chat", "date": "2026-06-25", "goals": "Thesis check"},
        ],
    )

    payload = module.build_payload([source], today="2026-06-23", days=7)

    assert [meeting["title"] for meeting in payload["meetings"]] == ["Investor chat"]
    assert any("No date" in action for action in payload["quick_actions"])


def test_yaml_mapping_input_when_pyyaml_available(tmp_path):
    yaml = pytest.importorskip("yaml")
    module = load_module()
    source = tmp_path / "meetings.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "meetings": [
                    {
                        "title": "House ops",
                        "date": "2026-06-23",
                        "prep": ["AC filter status"],
                        "questions": "Who owns the next cleaning?",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    brief = module.build_brief([source], today="2026-06-23", days=0)

    assert "House ops" in brief
    assert "AC filter status" in brief
    assert "Who owns the next cleaning?" in brief


def test_cli_json_flag_outputs_valid_json(tmp_path, capsys):
    module = load_module()
    source = tmp_path / "meetings.json"
    write_json(source, [{"title": "Review", "date": "2026-06-23", "prep": "Open PR"}])

    exit_code = module.main([str(source), "--today", "2026-06-23", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["generated_for"] == "2026-06-23"
    assert payload["meetings"][0]["title"] == "Review"
