"""Non-finite numeric text must survive XLSX import and editing."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "skills" / "productivity" / "xlsx" / "scripts"
NONFINITE_TEXT = [
    "NaN", "nan", "+NaN", "-nan", "inf", "+Inf", "-Infinity",
    "Infinity", "  NaN  ", "1e309", "-1e309",
]


def run(script, *args):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
        env=dict(os.environ, LC_ALL="C", LANG="C"),
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("flags", [(), ("--plain",), ("--no-infer",)])
def test_csv_nonfinite_text_survives_workbook_and_csv_roundtrip(tmp_path, flags):
    source = tmp_path / "source.csv"
    values = NONFINITE_TEXT + ["42", "3.5", "true", "2026-05-01"]
    with source.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows([["Value"], *[[value] for value in values]])
    workbook = tmp_path / "import.xlsx"
    run("csv_to_xlsx.py", source, workbook, *flags)
    rows = run("xlsx_read.py", workbook, "--json")["rows"]
    assert [row[0] for row in rows[1:1 + len(NONFINITE_TEXT)]] == NONFINITE_TEXT
    expected = values[-4:] if "--no-infer" in flags else [
        42, 3.5, True, "2026-05-01T00:00:00",
    ]
    assert [row[0] for row in rows[-4:]] == expected
    exported = tmp_path / "export.csv"
    run("xlsx_to_csv.py", workbook, exported)
    with exported.open(newline="", encoding="utf-8") as stream:
        exported_rows = list(csv.reader(stream))
    assert [row[0] for row in exported_rows[1:1 + len(NONFINITE_TEXT)]] == NONFINITE_TEXT


def test_edit_nonfinite_text_survives_save_without_changing_finite_types(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"sheets": [{"name": "Data", "rows": [["Value"]]}]}),
                    encoding="utf-8")
    workbook = tmp_path / "edit.xlsx"
    run("xlsx_create.py", spec, workbook)
    values = NONFINITE_TEXT + ["42", "3.5", "false", "2026-05-01", "=SUM(A1:A1)"]
    assignments = [arg for row, value in enumerate(values, 2)
                   for arg in ("--set", f"A{row}={value}")]
    run("xlsx_edit.py", workbook, *assignments)
    rows = run("xlsx_read.py", workbook, "--json")["rows"]
    assert [row[0] for row in rows[1:1 + len(NONFINITE_TEXT)]] == NONFINITE_TEXT
    assert [row[0] for row in rows[-5:]] == [
        42, 3.5, False, "2026-05-01T00:00:00", "=SUM(A1:A1)",
    ]
