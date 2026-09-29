"""Tests for scripts/personal_reminder_brief.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "personal_reminder_brief.py"
    spec = importlib.util.spec_from_file_location("_personal_reminder_brief", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_reminders_accepts_yaml_and_normalizes_fields(tmp_path):
    module = _load_module()
    data_path = tmp_path / "reminders.yaml"
    data_path.write_text(
        """
reminders:
  - title: Clean fridge
    due: 2026-05-28
    cadence: monthly
    area: home
    action: Throw expired food and wipe shelves.
    source: Joe Operating Manual
  - title: Future item
    due: 2026-07-01
    done: true
""".strip(),
        encoding="utf-8",
    )

    reminders = module.load_reminders(data_path)

    assert [r.title for r in reminders] == ["Clean fridge", "Future item"]
    assert reminders[0].due == date(2026, 5, 28)
    assert reminders[0].cadence == "monthly"
    assert reminders[0].area == "home"
    assert reminders[0].action == "Throw expired food and wipe shelves."
    assert reminders[0].source == "Joe Operating Manual"
    assert reminders[1].completed is True


def test_load_reminders_accepts_json_list(tmp_path):
    module = _load_module()
    data_path = tmp_path / "reminders.json"
    data_path.write_text(
        json.dumps([
            {"title": "AC cleaning", "due": "2026-05-30", "cadence": "quarterly"}
        ]),
        encoding="utf-8",
    )

    reminders = module.load_reminders(data_path)

    assert len(reminders) == 1
    assert reminders[0].title == "AC cleaning"
    assert reminders[0].due == date(2026, 5, 30)


def test_classify_reminders_groups_overdue_today_and_soon():
    module = _load_module()
    reminders = [
        module.Reminder(title="Overdue", due=date(2026, 5, 26)),
        module.Reminder(title="Today", due=date(2026, 5, 28)),
        module.Reminder(title="Soon", due=date(2026, 5, 31)),
        module.Reminder(title="Later", due=date(2026, 6, 20)),
        module.Reminder(title="Done", due=date(2026, 5, 27), completed=True),
    ]

    groups = module.classify_reminders(reminders, today=date(2026, 5, 28), soon_days=7)

    assert [r.title for r in groups.overdue] == ["Overdue"]
    assert [r.title for r in groups.today] == ["Today"]
    assert [r.title for r in groups.soon] == ["Soon"]
    assert groups.has_items is True


def test_render_brief_outputs_joe_style_traditional_chinese_sections():
    module = _load_module()
    reminders = [
        module.Reminder(
            title="Clean fridge",
            due=date(2026, 5, 28),
            cadence="monthly",
            area="home",
            action="Throw expired food and wipe shelves.",
            source="Joe Operating Manual",
        )
    ]
    groups = module.classify_reminders(reminders, today=date(2026, 5, 28), soon_days=7)

    brief = module.render_brief(groups, today=date(2026, 5, 28), silent_if_empty=True)

    assert brief.startswith("## TL;DR")
    assert "Fact / verified" in brief
    assert "Action for Joe" in brief
    assert "Clean fridge" in brief
    assert "2026-05-28" in brief
    assert "Joe Operating Manual" in brief


def test_render_brief_can_return_exact_silent_for_no_actionable_items():
    module = _load_module()
    reminders = [module.Reminder(title="Later", due=date(2026, 7, 1))]
    groups = module.classify_reminders(reminders, today=date(2026, 5, 28), soon_days=7)

    assert module.render_brief(groups, today=date(2026, 5, 28), silent_if_empty=True) == "[SILENT]"


def test_cli_prints_exact_silent_when_requested_and_empty(tmp_path, capsys):
    module = _load_module()
    data_path = tmp_path / "reminders.yaml"
    data_path.write_text(
        "reminders:\n  - title: Later\n    due: 2026-07-01\n",
        encoding="utf-8",
    )

    exit_code = module.main([
        "--input",
        str(data_path),
        "--today",
        "2026-05-28",
        "--silent-if-empty",
    ])

    assert exit_code == 0
    assert capsys.readouterr().out == "[SILENT]\n"
