"""Compaction report must preserve scored arms when another arm falls back (#127893)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("recalls", [[None, 60.0, 80.0, 0.0], [None], [], [0.0, 80.0]])
def test_report_renders_all_arms_with_unscored_last(tmp_path, recalls):
    card = [
        {
            "policy": f"arm-{i}",
            "recall_pct": recall,
            "before_tokens": 1000,
            "after_tokens": None if recall is None else 500,
            "compress_seconds": 1.2,
            **({"summary_error": "fixture fallback"} if recall is None else {}),
        }
        for i, recall in enumerate(recalls)
    ]
    source = tmp_path / "scorecard.json"
    source.write_text(json.dumps(card), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[2] / "evals" / "compaction" / "report.py"), str(tmp_path)],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    markdown = (tmp_path / "scorecard.md").read_text(encoding="utf-8")
    rows = [line.split("|")[1:-1] for line in markdown.splitlines() if line.startswith("| arm-")]
    rows = [[cell.strip() for cell in row] for row in rows]
    expected = sorted(card, key=lambda s: (s["recall_pct"] is None, -(s["recall_pct"] or 0)))
    assert [row[0] for row in rows] == [s["policy"] for s in expected]
    for row, summary in zip(rows, expected):
        assert summary["policy"] in result.stdout
        if summary["recall_pct"] is None:
            assert row[1] == row[3] == row[4] == "n/a"
            assert "fixture fallback" in result.stdout
            assert "fixture fallback" in markdown
        else:
            assert row[1:5] == [f"{summary['recall_pct']}%", "1,000", "500", "50.0%"]
    assert json.loads(source.read_text(encoding="utf-8")) == card
