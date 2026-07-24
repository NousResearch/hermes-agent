from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "reminder_cadence_audit.py"

spec = importlib.util.spec_from_file_location("reminder_cadence_audit", SCRIPT_PATH)
assert spec is not None
assert spec.loader is not None
reminder_cadence_audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reminder_cadence_audit
spec.loader.exec_module(reminder_cadence_audit)


def test_parse_reminders_ignores_guidance_and_templates(tmp_path):
    note = tmp_path / "household.md"
    note.write_text(
        """
# Household

- Keep cadence rows explicit; do not infer vague bullets.
- [ ] Clean fridge | cadence: every 1 month | last: 2026-05-01 | note: Wipe shelves | owner: Joe
- [ ] Template row | cadence: every N months | last: YYYY-MM-DD
- [x] Aircon service | cadence: every 3 months | last: 2026-03-10
- [ ] Buy detergent | priority: low
""".strip(),
        encoding="utf-8",
    )

    reminders = reminder_cadence_audit.parse_files([note])

    assert [item.title for item in reminders] == ["Clean fridge", "Aircon service"]
    assert reminders[0].cadence_count == 1
    assert reminders[0].cadence_unit == "month"
    assert reminders[0].note == "Wipe shelves"
    assert reminders[0].owner == "Joe"
    assert reminders[0].source == str(note)


def test_month_end_calendar_math_clamps_to_valid_day():
    reminder = reminder_cadence_audit.Reminder(
        title="Month-end task",
        cadence_count=1,
        cadence_unit="month",
        last_done=reminder_cadence_audit.parse_date("2026-01-31"),
        source="note.md",
        line=1,
    )

    assert reminder_cadence_audit.next_due_date(reminder).isoformat() == "2026-02-28"


def test_classify_due_upcoming_and_clear_buckets(tmp_path):
    note = tmp_path / "tasks.md"
    note.write_text(
        "\n".join(
            [
                "- [ ] Clean fridge | cadence: every 1 month | last: 2026-05-01",
                "- [ ] Aircon service | cadence: every 3 months | last: 2026-03-18",
                "- [ ] Replace filter | cadence: every 6 months | last: 2026-05-20",
            ]
        ),
        encoding="utf-8",
    )

    reminders = reminder_cadence_audit.parse_files([note])
    report = reminder_cadence_audit.build_report(
        reminders,
        today=reminder_cadence_audit.parse_date("2026-06-15"),
        lookahead_days=14,
    )

    assert [item.reminder.title for item in report.due] == ["Clean fridge"]
    assert report.due[0].days_delta == 14
    assert [item.reminder.title for item in report.upcoming] == ["Aircon service"]
    assert report.upcoming[0].days_delta == -3
    assert [item.reminder.title for item in report.clear] == ["Replace filter"]


def test_markdown_output_is_deterministic(tmp_path):
    note = tmp_path / "tasks.md"
    note.write_text(
        "- [ ] Clean fridge | cadence: every 1 month | last: 2026-05-01 | note: Wipe shelves\n"
        "- [ ] Aircon service | cadence: every 3 months | last: 2026-03-18\n",
        encoding="utf-8",
    )
    report = reminder_cadence_audit.build_report(
        reminder_cadence_audit.parse_files([note]),
        today=reminder_cadence_audit.parse_date("2026-06-15"),
        lookahead_days=14,
    )

    markdown = reminder_cadence_audit.render_markdown(report)

    assert markdown.startswith("# Reminder cadence audit\n")
    assert "## Due now\n- **Clean fridge** — due 2026-06-01 (14d overdue)" in markdown
    assert "cadence: every 1 month" in markdown
    assert "note: Wipe shelves" in markdown
    assert "## Upcoming\n- **Aircon service** — due 2026-06-18 (in 3d)" in markdown


def test_json_output_contains_normalized_records(tmp_path):
    note = tmp_path / "tasks.md"
    note.write_text("- [ ] Clean fridge | cadence: every 1 month | last: 2026-05-01\n", encoding="utf-8")
    report = reminder_cadence_audit.build_report(
        reminder_cadence_audit.parse_files([note]),
        today=reminder_cadence_audit.parse_date("2026-06-15"),
        lookahead_days=14,
    )

    payload = json.loads(reminder_cadence_audit.render_json(report))

    assert payload["today"] == "2026-06-15"
    assert payload["due"][0]["title"] == "Clean fridge"
    assert payload["due"][0]["due_date"] == "2026-06-01"
    assert payload["due"][0]["days_overdue"] == 14


def test_main_reads_stdin_when_no_paths(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "stdin",
        type("FakeStdin", (), {"isatty": lambda self: False, "read": lambda self: "- [ ] Clean fridge | cadence: every 1 month | last: 2026-05-01\n"})(),
    )

    exit_code = reminder_cadence_audit.main(["--today", "2026-06-15", "--lookahead-days", "7"])

    assert exit_code == 0
    assert "Clean fridge" in capsys.readouterr().out
