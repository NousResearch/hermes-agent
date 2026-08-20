from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "life_ops_reminder_brief.py"


def load_module():
    spec = importlib.util.spec_from_file_location("life_ops_reminder_brief", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_formats_overdue_due_today_and_soon_one_off_tasks():
    mod = load_module()
    payload = {
        "items": [
            {"title": "Clean fridge", "due": "2026-06-01", "notes": "Monthly reset"},
            {"title": "Pay credit card", "due": "2026-06-06"},
            {"title": "Buy filters", "due": "2026-06-08"},
            {"title": "Renew passport", "due": "2026-07-01"},
        ]
    }

    report = mod.build_brief(payload, today="2026-06-06", soon_days=7)

    assert "## Overdue" in report
    assert "Clean fridge — due 2026-06-01 (5d overdue) — Monthly reset" in report
    assert "## Due today" in report
    assert "Pay credit card — due today" in report
    assert "## Soon" in report
    assert "Buy filters — due 2026-06-08 (in 2d)" in report
    assert "Renew passport" not in report


def test_recurring_tasks_are_due_from_last_done_plus_interval():
    mod = load_module()
    payload = {
        "items": [
            {"title": "Air-conditioner cleaning", "last_done": "2026-03-06", "every_days": 90},
            {"title": "Replace toothbrush", "last_done": "2026-06-01", "every_days": 90},
        ]
    }

    report = mod.build_brief(payload, today="2026-06-06", soon_days=7)

    assert "Air-conditioner cleaning — due 2026-06-04 (2d overdue)" in report
    assert "Replace toothbrush" not in report


def test_plan_path_caveat_is_included_for_due_item():
    mod = load_module()
    payload = {
        "items": [
            {
                "title": "Book Bali transport",
                "due": "2026-06-07",
                "plan_path": "/Users/hyc/plans/bali.md",
            }
        ]
    }

    report = mod.build_brief(payload, today="2026-06-06", soon_days=7)

    assert "Book Bali transport — due 2026-06-07 (in 1d) — plan: /Users/hyc/plans/bali.md" in report


def test_returns_exact_silent_when_nothing_is_reportable():
    mod = load_module()
    payload = {
        "items": [
            {"title": "Quarterly review", "due": "2026-07-15"},
            {"title": "Air-conditioner cleaning", "last_done": "2026-06-01", "every_days": 90},
        ]
    }

    assert mod.build_brief(payload, today="2026-06-06", soon_days=7) == "[SILENT]"
