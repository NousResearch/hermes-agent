from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "life_ops_brief.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("life_ops_brief", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_brief_surfaces_overdue_and_due_soon_life_ops_items():
    mod = _load_module()
    items = [
        {
            "title": "clean fridge",
            "area": "household",
            "due": "2026-06-01",
            "notes": "monthly reset",
        },
        {
            "title": "aircon cleaning",
            "area": "household",
            "last_done": "2026-03-10",
            "every_days": 90,
        },
        {
            "title": "review portfolio thesis",
            "area": "investment",
            "due": "2026-06-20",
        },
    ]

    brief = mod.build_brief(items, today="2026-06-10", soon_days=7)

    assert brief.startswith("TL;DR")
    assert "逾期 2" in brief
    assert "clean fridge" in brief
    assert "2026-06-01" in brief
    assert "aircon cleaning" in brief
    assert "2026-06-08" in brief
    assert "review portfolio thesis" not in brief


def test_brief_returns_exact_silent_when_nothing_actionable():
    mod = _load_module()
    items = [
        {"title": "future trip planning", "area": "life", "due": "2026-07-01"},
        {"title": "quarterly body check", "area": "health", "last_done": "2026-06-01", "every_days": 30},
    ]

    assert mod.build_brief(items, today="2026-06-10", soon_days=7) == "[SILENT]"


def test_json_cli_smoke_outputs_silent_for_empty_backlog(tmp_path, capsys):
    mod = _load_module()
    backlog = tmp_path / "life_ops.json"
    backlog.write_text("[]", encoding="utf-8")

    code = mod.main(["--input", str(backlog), "--today", "2026-06-10"])

    assert code == 0
    assert capsys.readouterr().out.strip() == "[SILENT]"
