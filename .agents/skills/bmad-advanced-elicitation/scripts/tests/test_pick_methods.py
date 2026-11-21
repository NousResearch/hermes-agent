# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest>=8.0"]
# ///
"""Tests for pick_methods.py.

Run: uv run scripts/tests/test_pick_methods.py
 or: uv run --with pytest -m pytest scripts/tests/test_pick_methods.py
"""
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pick_methods  # noqa: E402

CSV = """num,category,method_name,description,output_pattern
1,risk,Pre-mortem Analysis,Imagine future failure then work backwards,failure → causes → prevention
2,risk,Assumption Audit,List and stress-test every assumption,list → rate → stress-test
3,core,First Principles Analysis,Rebuild from fundamental truths,assumptions → truths → new approach
4,core,Socratic Questioning,Targeted questions reveal hidden assumptions,questions → revelations → understanding
5,creative,SCAMPER Method,Seven creativity lenses,S→C→A→M→P→E→R
"""

EXTRA = [
    {
        "code": "reg-inversion",
        "category": "domain",
        "method_name": "Regulatory Inversion",
        "description": "Start from the compliance constraint",
        "output_pattern": "constraint → possibility",
    },
    {
        "code": "premortem-lite",
        "category": "risk",
        "method_name": "Pre-mortem Analysis",
        "description": "RETUNED pre-mortem",
        "output_pattern": "failure → prevention",
    },
]


@pytest.fixture
def lib(tmp_path):
    csv_path = tmp_path / "methods.csv"
    csv_path.write_text(CSV, encoding="utf-8")
    return csv_path


def rows(lib):
    return pick_methods.load(lib)


# --- load / merge -----------------------------------------------------------

def test_load_all_fields_present(lib):
    r = rows(lib)
    assert len(r) == 5
    assert r[0]["method_name"] == "Pre-mortem Analysis"
    assert all(set(pick_methods.FIELDS) <= set(row) for row in r)


def test_load_extra_json_literal_and_file(tmp_path, lib):
    literal = pick_methods.load_extra(json.dumps(EXTRA))
    f = tmp_path / "extra.json"
    f.write_text(json.dumps(EXTRA), encoding="utf-8")
    from_file = pick_methods.load_extra(str(f))
    assert literal == from_file
    assert literal[0]["method_name"] == "Regulatory Inversion"
    assert literal[0]["num"] == ""  # missing fields normalize to empty
    assert literal[0]["code"] == "reg-inversion"  # code survives loading


def test_merge_extra_replaces_by_name_and_appends(lib):
    merged = pick_methods.merge_extra(rows(lib), pick_methods.load_extra(json.dumps(EXTRA)))
    assert len(merged) == 6  # 5 shipped, 1 replaced in place, 1 appended
    premortem = next(r for r in merged if r["method_name"] == "Pre-mortem Analysis")
    assert premortem["description"] == "RETUNED pre-mortem"
    assert premortem["num"] == "1"  # replacement inherits the shipped num
    appended = next(r for r in merged if r["method_name"] == "Regulatory Inversion")
    assert appended["num"] == "6"  # appended extras get the next free num
    assert dict(pick_methods.categories(merged))["domain"] == 1  # new category is first-class


def test_extras_are_addressable_by_num(lib):
    merged = pick_methods.merge_extra(rows(lib), pick_methods.load_extra(json.dumps(EXTRA)))
    found, missing = pick_methods.find(merged, ["6", "1"])
    assert [r["method_name"] for r in found] == ["Regulatory Inversion", "Pre-mortem Analysis"]
    assert missing == []


# --- categories / filter / find / exclude -----------------------------------

def test_categories_counts_sorted(lib):
    assert pick_methods.categories(rows(lib)) == [("core", 2), ("creative", 1), ("risk", 2)]


def test_filter_is_case_insensitive(lib):
    got = pick_methods.filter_cats(rows(lib), ["RISK"])
    assert {r["method_name"] for r in got} == {"Pre-mortem Analysis", "Assumption Audit"}


def test_filter_none_returns_all(lib):
    assert len(pick_methods.filter_cats(rows(lib), None)) == 5


def test_find_by_name_num_and_missing(lib):
    found, missing = pick_methods.find(rows(lib), ["scamper method", "3", "Nope"])
    assert [r["method_name"] for r in found] == ["SCAMPER Method", "First Principles Analysis"]
    assert missing == ["Nope"]


def test_exclude_skips_named(lib):
    got = pick_methods.exclude(rows(lib), ["pre-mortem analysis", "SCAMPER Method"])
    assert {r["method_name"] for r in got} == {
        "Assumption Audit", "First Principles Analysis", "Socratic Questioning",
    }


# --- spread sampling ---------------------------------------------------------

def test_spread_hits_distinct_categories(lib):
    for seed in range(20):
        picks = pick_methods.spread_sample(rows(lib), 3, random.Random(seed))
        assert len({r["category"] for r in picks}) == 3


def test_spread_wraps_when_categories_run_out(lib):
    picks = pick_methods.spread_sample(rows(lib), 5, random.Random(0))
    assert len(picks) == 5
    assert len({r["method_name"] for r in picks}) == 5  # no duplicates


def test_spread_clamps_to_pool(lib):
    assert len(pick_methods.spread_sample(rows(lib), 99, random.Random(0))) == 5


# --- CLI ---------------------------------------------------------------------

def run(args, lib, capsys):
    code = pick_methods.main(["--file", str(lib), *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_categories(lib, capsys):
    code, out, _ = run(["categories"], lib, capsys)
    assert code == 0
    assert "risk\t2" in out


def test_cli_list_requires_scope(lib, capsys):
    code, _, err = run(["list"], lib, capsys)
    assert code == 2
    assert "--category" in err


def test_cli_list_category_and_all(lib, capsys):
    code, out, _ = run(["list", "--category", "core"], lib, capsys)
    assert code == 0 and len(out.strip().splitlines()) == 2
    assert "Socratic Questioning" in out and "SCAMPER" not in out
    code, out, _ = run(["list", "--all"], lib, capsys)
    assert code == 0 and "SCAMPER" in out


def test_cli_show_found_and_missing(lib, capsys):
    code, out, err = run(["show", "Assumption Audit", "Ghost"], lib, capsys)
    assert code == 0
    assert "stress-test" in out
    assert "not found: Ghost" in err
    code, _, _ = run(["show", "Ghost"], lib, capsys)
    assert code == 1


def test_cli_random_spread_exclude(lib, capsys):
    code, out, _ = run(
        ["random", "-n", "3", "--spread", "--exclude", "SCAMPER Method"], lib, capsys
    )
    assert code == 0
    lines = [ln for ln in out.strip().splitlines() if ln]
    assert len(lines) == 3
    assert "SCAMPER" not in out


def test_cli_random_clamps_and_empty_pool(lib, capsys):
    code, out, _ = run(["random", "-n", "99"], lib, capsys)
    assert code == 0 and len(out.strip().splitlines()) == 5
    code, _, err = run(["random", "--category", "nope"], lib, capsys)
    assert code == 1 and "no methods match" in err


def test_cli_extra_inline_json(lib, capsys):
    code, out, _ = run(
        ["--extra", json.dumps(EXTRA), "list", "--category", "domain"], lib, capsys
    )
    assert code == 0 and "Regulatory Inversion" in out


def test_cli_bad_extra_and_missing_file(tmp_path, lib, capsys):
    code, _, err = run(["--extra", str(tmp_path / "gone.json"), "categories"], lib, capsys)
    assert code == 2 and "--extra" in err
    code = pick_methods.main(["--file", str(tmp_path / "gone.csv"), "categories"])
    assert code == 2


def test_cli_json_output(lib, capsys):
    code, out, _ = run(["--json", "show", "1"], lib, capsys)
    assert code == 0
    data = json.loads(out)
    assert data[0]["method_name"] == "Pre-mortem Analysis"


# --- shipped catalog integration ----------------------------------------------

def test_shipped_catalog_loads_clean():
    shipped = pick_methods.DEFAULT_FILE
    assert shipped.is_file(), f"shipped catalog missing: {shipped}"
    r = pick_methods.load(shipped)
    assert len(r) >= 60
    for row in r:
        assert row["category"] and row["method_name"] and row["description"], row


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
