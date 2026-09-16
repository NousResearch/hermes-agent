import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "investment_review_brief.py"


def run_script(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=SCRIPT.parents[1],
    )


def test_json_input_flags_concentration_missing_thesis_and_stale_reviews(tmp_path):
    positions = [
        {
            "name": "BTC",
            "category": "crypto",
            "value": 70000,
            "cost_basis": 50000,
            "thesis": "Institutional adoption keeps BTC as portfolio ballast.",
            "last_reviewed": "2026-01-01",
        },
        {
            "name": "AI Startup Basket",
            "category": "private",
            "value": 20000,
            "cost_basis": 25000,
            "thesis": "",
            "last_reviewed": "2026-05-15",
        },
        {
            "name": "Cash",
            "category": "cash",
            "value": 10000,
            "cost_basis": 10000,
            "thesis": "Dry powder.",
            "last_reviewed": "2026-05-20",
        },
    ]
    path = tmp_path / "portfolio.json"
    path.write_text(json.dumps(positions), encoding="utf-8")

    result = run_script(str(path), "--as-of", "2026-05-31", "--review-after-days", "90")

    assert result.returncode == 0, result.stderr
    assert "# 投資組合 Review Brief" in result.stdout
    assert "總資產：100,000.00" in result.stdout
    assert "BTC — 70.0%" in result.stdout
    assert "AI Startup Basket" in result.stdout
    assert "缺少 thesis" in result.stdout
    assert "BTC（150 天未 review）" in result.stdout
    assert "未實現損益：15,000.00" in result.stdout


def test_silent_if_clear_outputs_exact_silent_marker(tmp_path):
    positions = [
        {
            "name": "Index Fund",
            "value": 5000,
            "cost_basis": 5000,
            "thesis": "Long-term diversified equity exposure.",
            "last_reviewed": "2026-05-01",
        },
        {
            "name": "Cash",
            "value": 3000,
            "cost_basis": 3000,
            "thesis": "Optionality.",
            "last_reviewed": "2026-05-02",
        },
        {
            "name": "Short Bills",
            "value": 2000,
            "cost_basis": 2000,
            "thesis": "Low-risk carry.",
            "last_reviewed": "2026-05-03",
        },
    ]
    path = tmp_path / "clear.json"
    path.write_text(json.dumps(positions), encoding="utf-8")

    result = run_script(str(path), "--as-of", "2026-05-31", "--silent-if-clear")

    assert result.returncode == 0, result.stderr
    assert result.stdout == "[SILENT]\n"


def test_csv_input_is_supported_without_yaml_dependency(tmp_path):
    path = tmp_path / "portfolio.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["name", "category", "value", "cost_basis", "thesis", "last_reviewed"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "name": "ETH",
                "category": "crypto",
                "value": "12000",
                "cost_basis": "10000",
                "thesis": "Settlement asset upside.",
                "last_reviewed": "2026-01-15",
            }
        )

    result = run_script(str(path), "--as-of", "2026-05-31", "--review-after-days", "30")

    assert result.returncode == 0, result.stderr
    assert "ETH" in result.stdout
    assert "crypto" in result.stdout
    assert "136 天未 review" in result.stdout


def test_invalid_empty_input_exits_nonzero_with_actionable_error(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")

    result = run_script(str(path))

    assert result.returncode == 1
    assert "No investment positions found" in result.stderr
