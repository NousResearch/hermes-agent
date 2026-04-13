"""Smoke tests for skills/data-science/polars-lazyframes/scripts/polars_run.py."""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("polars")

SCRIPTS = Path(__file__).resolve().parents[2] / "skills" / "data-science" / "polars-lazyframes" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import polars_run  # noqa: E402


def _run(capsys, argv: list[str]) -> dict:
    with mock.patch("sys.argv", ["polars_run"] + argv):
        polars_run.main()
    return json.loads(capsys.readouterr().out.strip())


def test_inspect_and_head_csv(tmp_path, capsys):
    csv = tmp_path / "sample.csv"
    csv.write_text("name,count\nalice,1\nbob,2\n", encoding="utf-8")
    r1 = _run(capsys, ["inspect", str(csv)])
    assert r1["ok"] is True
    assert "name" in r1["schema"]
    assert r1["schema"]["name"] in ("String", "Utf8", "str")
    r2 = _run(capsys, ["head", str(csv), "-n", "1"])
    assert r2["ok"] is True
    assert r2["rows"][0]["name"] == "alice"


def test_convert_to_parquet(tmp_path, capsys):
    csv = tmp_path / "in.csv"
    csv.write_text("x\n3\n", encoding="utf-8")
    out = tmp_path / "out.parquet"
    r = _run(capsys, ["convert", str(csv), str(out)])
    assert r["ok"] is True
    assert out.is_file()
    r2 = _run(capsys, ["inspect", str(out)])
    assert r2["ok"] is True
    assert "x" in r2["schema"]
