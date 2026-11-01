"""Behaviour contracts for optional-skills/security/data-share-review/scripts/data_share.py.

The script's promise is that it reports counts and metadata and never echoes a
value from the file it inspects, while still catching identifiers that survive
a column-level transform (an address quoted in a notes field).
"""
import csv
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (Path(__file__).resolve().parents[2]
          / "optional-skills/security/data-share-review/scripts/data_share.py")
EMAIL = "jane.doe@example.com"
KEY_SHAPED = "sk-" + "a1b2c3d4e5" * 3


@pytest.fixture(scope="module")
def ds():
    return SimpleNamespace(**runpy.run_path(str(SCRIPT)))


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _run(ds, capsys, *argv) -> tuple[int, dict, str]:
    code = ds.main(list(argv))
    out = capsys.readouterr().out
    return code, json.loads(out), out


@pytest.fixture
def export(tmp_path):
    return _write_csv(tmp_path / "export.csv", [
        {"customer_id": "C-10001", "email": EMAIL, "postcode": "94110",
         "notes": f"customer wrote from {EMAIL} and pasted api_key={KEY_SHAPED}"},
        {"customer_id": "C-10002", "email": "bob@example.org", "postcode": "94110", "notes": "no contact"},
    ])


def test_pseudonymized_column_still_leaks_through_free_text_and_scan_never_echoes_it(ds, capsys, tmp_path, export):
    out_dir = tmp_path / "share"
    out_dir.mkdir()
    shipped = out_dir / "share.csv"
    code, _, _ = _run(ds, capsys, "pseudonymize", str(export), str(shipped),
                      "--columns", "customer_id,email", "--key-file", str(tmp_path / "keys" / "vendor.key"))
    assert code == 0
    assert EMAIL not in next(csv.DictReader(shipped.open(encoding="utf-8")))["email"]

    code, report, raw = _run(ds, capsys, "scan", str(shipped), "--originals", str(export),
                             "--columns", "email,customer_id")
    assert code == 0
    assert report["original_values"] == {"email": 1, "customer_id": 0}
    assert report["totals"]["email"] == 1 and report["totals"]["secret"] == 1
    assert report["per_column"] == {"notes": {"email": 1, "secret": 1}}
    assert report["all_zero"] is False
    assert EMAIL not in raw and KEY_SHAPED not in raw


def test_pseudonyms_are_stable_per_key_and_differ_across_keys(ds, capsys, tmp_path, export):
    (tmp_path / "out").mkdir()

    def pseudonyms(key_name, out_name):
        out = tmp_path / "out" / out_name
        _run(ds, capsys, "pseudonymize", str(export), str(out), "--columns", "email",
             "--drop", "notes", "--key-file", str(tmp_path / "keys" / key_name))
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert "notes" not in rows[0]
        return [r["email"] for r in rows]

    first, again, other = pseudonyms("a.key", "1.csv"), pseudonyms("a.key", "2.csv"), pseudonyms("b.key", "3.csv")
    assert first == again
    assert first != other
    assert all(p.startswith("p_") for p in first)


def test_key_inside_the_shipped_folder_is_refused_before_anything_is_written(ds, capsys, tmp_path, export):
    out_dir = tmp_path / "share"
    out_dir.mkdir()
    code, report, _ = _run(ds, capsys, "pseudonymize", str(export), str(out_dir / "share.csv"),
                           "--columns", "email", "--key-file", str(out_dir / "nested" / "k.key"))
    assert code == 2 and "outside" in report["error"]
    assert list(out_dir.iterdir()) == []


def test_empty_key_file_from_an_interrupted_run_is_refused_before_the_export_opens(ds, capsys, tmp_path, export):
    key = tmp_path / "keys" / "vendor.key"
    key.parent.mkdir()
    key.write_text("", encoding="utf-8")
    shipped = tmp_path / "share" / "share.csv"
    shipped.parent.mkdir()
    code, report, _ = _run(ds, capsys, "pseudonymize", str(export), str(shipped),
                           "--columns", "email", "--key-file", str(key))
    assert code == 2 and "key" in report["error"]
    assert not shipped.exists()


def test_clean_file_reports_all_zero_and_k_anonymity_groups(ds, capsys, tmp_path):
    clean = _write_csv(tmp_path / "clean.csv", [
        {"region": "west", "plan": "pro", "orders": "3"},
        {"region": "west", "plan": "pro", "orders": "5"},
        {"region": "east", "plan": "free", "orders": "1"},
    ])
    _, report, _ = _run(ds, capsys, "scan", str(clean))
    assert report["all_zero"] is True
    _, report, _ = _run(ds, capsys, "scan", str(clean), "--quasi", "region,plan", "--k", "2")
    assert report["quasi_identifier"]["rows_in_groups_below_k"] == 1
    assert report["all_zero"] is False


@pytest.mark.parametrize("name,content", [
    ("dup_header.csv", f"notes,notes\n{EMAIL},ok\n"),
    ("wide_row.csv", f"notes\nok,{EMAIL}\n"),
    ("dup_key.jsonl", f'{{"notes": "{EMAIL}", "notes": "ok"}}\n'),
])
def test_cells_that_would_hide_from_the_per_column_scan_are_refused(ds, capsys, tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    code, report, raw = _run(ds, capsys, "scan", str(path))
    assert code == 2 and "all_zero" not in report
    assert EMAIL not in raw


@pytest.mark.parametrize("extra", [
    ["--quasi", "post_code", "--k", "2"],
    ["--quasi", "region", "--k", "1"],
    ["--originals", "SRC"],
])
def test_a_requested_check_that_cannot_run_is_refused_not_reported_clean(ds, capsys, tmp_path, extra):
    shipped = _write_csv(tmp_path / "shipped.csv", [{"region": "west", "postcode": f"9411{i}"} for i in range(3)])
    argv = [str(shipped) if a == "SRC" else a for a in extra]
    code, report, _ = _run(ds, capsys, "scan", str(shipped), *argv)
    assert code == 2 and "all_zero" not in report


def test_short_original_name_in_kept_free_text_is_counted(ds, capsys, tmp_path):
    src = _write_csv(tmp_path / "src.csv", [{"name": "Amy", "notes": "Called Amy back"},
                                            {"name": "Bob", "notes": "no answer"}])
    out_dir = tmp_path / "share"
    out_dir.mkdir()
    shipped = out_dir / "share.csv"
    _run(ds, capsys, "pseudonymize", str(src), str(shipped), "--drop", "name",
         "--key-file", str(tmp_path / "k.key"))
    code, report, raw = _run(ds, capsys, "scan", str(shipped), "--originals", str(src), "--columns", "name")
    assert code == 0
    assert report["original_values"] == {"name": 1} and report["all_zero"] is False
    assert "Amy" not in raw


def test_identifier_shaped_header_is_counted_and_never_echoed(ds, capsys, tmp_path):
    pivot = tmp_path / "pivot.csv"
    pivot.write_text(f"month,{EMAIL}\n2026-09,3\n2026-10,5\n", encoding="utf-8")
    code, report, raw = _run(ds, capsys, "scan", str(pivot))
    assert code == 0
    assert report["totals"]["email"] == 1 and report["all_zero"] is False
    assert report["header"] == {"email": 1}
    outputs = [raw]
    for argv in (["inventory", str(pivot)], ["record", str(pivot), "--destination", "vendor"]):
        code, _, raw = _run(ds, capsys, *argv)
        assert code == 0
        outputs.append(raw)
    assert all(EMAIL not in out for out in outputs)


def test_parse_error_reports_only_its_type(ds, capsys, tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(f'{{"email": "{EMAIL}"}}\n{{"email": "{EMAIL}", oops}}\n', encoding="utf-8")
    code, report, raw = _run(ds, capsys, "scan", str(bad))
    assert code == 2 and report == {"error": "JSONDecodeError"}
    assert EMAIL not in raw
