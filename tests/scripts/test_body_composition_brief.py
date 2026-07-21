from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "body_composition_brief.py"


def load_module():
    spec = importlib.util.spec_from_file_location("body_composition_brief", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_records_from_json_file(tmp_path):
    module = load_module()
    source = tmp_path / "body.json"
    source.write_text(
        '[{"date":"2026-05-20","weight_kg":82.4,"body_fat_pct":24.0}]',
        encoding="utf-8",
    )

    records = module.load_records(source)

    assert len(records) == 1
    assert records[0].date.isoformat() == "2026-05-20"
    assert records[0].weight_kg == 82.4
    assert records[0].body_fat_pct == 24.0


def test_render_empty_records_can_suppress_delivery():
    module = load_module()

    assert module.render_brief([], silent_if_empty=True) == "[SILENT]"


def test_summarise_reports_latest_metrics_and_target_gap():
    module = load_module()
    records = [
        module.BodyRecord(date=module.date(2026, 5, 1), weight_kg=83.0, body_fat_pct=24.5),
        module.BodyRecord(date=module.date(2026, 5, 8), weight_kg=82.0, body_fat_pct=23.8),
        module.BodyRecord(date=module.date(2026, 5, 15), weight_kg=81.8, body_fat_pct=23.1),
    ]

    summary = module.summarise(records)

    assert summary["latest"].date.isoformat() == "2026-05-15"
    assert summary["latest_weight"].weight_kg == 81.8
    assert summary["latest_body_fat"].body_fat_pct == 23.1
    assert summary["weight_delta"] == -1.2
    assert summary["body_fat_delta"] == -1.4
    assert summary["target_gap"] == 3.1


def test_render_brief_uses_traditional_chinese_sections_and_actions():
    module = load_module()
    records = [
        module.BodyRecord(date=module.date(2026, 5, 1), weight_kg=83.0, body_fat_pct=24.5),
        module.BodyRecord(date=module.date(2026, 5, 15), weight_kg=81.8, body_fat_pct=23.1),
    ]

    brief = module.render_brief(records, days=30)

    assert "## TL;DR" in brief
    assert "## Fact / verified" in brief
    assert "## Hypothesis" in brief
    assert "## Action for Joe" in brief
    assert "最新紀錄（2026-05-15）：81.8kg，體脂 23.1%" in brief
    assert "距離 20% 還差 3.1 個百分點" in brief
    assert "體重 -1.2kg，體脂 -1.4pct（下降中）" in brief


def test_cli_prints_brief_from_csv(tmp_path, capsys):
    module = load_module()
    source = tmp_path / "body.csv"
    source.write_text(
        "date,weight_kg,body_fat_pct\n2026-05-01,83.0,24.5\n2026-05-15,81.8,23.1\n",
        encoding="utf-8",
    )

    exit_code = module.main(["--input", str(source), "--days", "30"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "## TL;DR" in captured.out
    assert "81.8kg" in captured.out
    assert "未連接任何新資料源、未傳送訊息" in captured.out
