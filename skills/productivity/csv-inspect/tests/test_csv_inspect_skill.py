"""Tests for the bundled csv-inspect skill script."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SKILL_DIR = Path(__file__).resolve().parent.parent
_SCRIPT = _SKILL_DIR / "scripts" / "csv_inspect.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("csv_inspect", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(tmp_path, name, content, encoding="utf-8"):
    p = tmp_path / name
    p.write_text(content, encoding=encoding)
    return str(p)


def test_profiles_schema_nulls_and_numbers(mod, tmp_path):
    path = _write(
        tmp_path,
        "data.csv",
        "id,score,when,active\n"
        '1,"1,200.50",2024-01-15,true\n'
        "2,0.5,2024-02-01,false\n"
        "3,,2024-03-10,true\n",
    )
    out = mod.profile(path)
    assert out["rows"] == 3 and out["columns"] == 4
    assert out["delimiter"] == ","

    by_name = {c["name"]: c for c in out["column_profiles"]}
    assert by_name["id"]["type"] == "int"
    assert by_name["score"]["type"] == "float"
    assert by_name["score"]["null_count"] == 1
    assert by_name["score"]["max"] == 1200.50
    assert by_name["when"]["type"] == "date"
    assert by_name["when"]["min_value"] == "2024-01-15"
    assert by_name["active"]["type"] == "bool"


def test_sniffs_tab_and_semicolon(mod, tmp_path):
    tsv = _write(tmp_path, "t.tsv", "a\tb\n1\tx\n2\ty\n")
    assert mod.profile(tsv)["delimiter"] == "\t"

    semi = _write(tmp_path, "s.csv", "a;b\n1;x\n2;y\n")
    assert mod.profile(semi)["delimiter"] == ";"


def test_cp1252_fallback_and_na_values(mod, tmp_path):
    latin = _write(tmp_path, "l.csv", "name,note\nJosé,ok\nAna,NA\n", encoding="cp1252")
    out = mod.profile(latin, na_values=["NA"])
    assert out["encoding"] == "cp1252"
    by_name = {c["name"]: c for c in out["column_profiles"]}
    assert by_name["note"]["null_count"] == 1


def test_no_header_names_columns_positionally(mod, tmp_path):
    path = _write(tmp_path, "nh.csv", "1,x\n2,y\n")
    out = mod.profile(path, no_header=True)
    names = [c["name"] for c in out["column_profiles"]]
    assert names == ["col_1", "col_2"]
    assert out["header_row_present"] is False


def test_head_includes_parsed_rows(mod, tmp_path):
    path = _write(tmp_path, "h.csv", "a,b\n1,x\n2,y\n3,z\n")
    out = mod.profile(path, head=2)
    assert out["head"] == [["1", "x"], ["2", "y"]]


def test_main_emits_valid_json(tmp_path, capsys):
    path = _write(tmp_path, "m.csv", "a\n1\n")
    spec = importlib.util.spec_from_file_location("ci_main", _SCRIPT)
    m2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m2)
    m2.main([path])
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["columns"] == 1
