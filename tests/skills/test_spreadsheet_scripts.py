"""Tests for the spreadsheet skill helper scripts."""

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import scripts as modules via importlib (they live outside the package tree)
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "skills" / "data-science" / "spreadsheet" / "scripts"


def _import_script(name):
    """Import a script from the spreadsheet skill's scripts/ directory."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Only import if openpyxl/pandas are available
openpyxl = pytest.importorskip("openpyxl")
pandas = pytest.importorskip("pandas")

read_sheet_mod = _import_script("read_sheet")
write_sheet_mod = _import_script("write_sheet")
apply_formula_mod = _import_script("apply_formula")
pivot_mod = _import_script("pivot")
chart_mod = _import_script("chart")
diff_mod = _import_script("diff")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_xlsx(path, rows, sheet_name="Sheet1"):
    """Create a minimal xlsx file from a list of row lists (first row = headers)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    for row in rows:
        ws.append(row)
    wb.save(str(path))


def _create_csv(path, rows, delimiter=","):
    """Create a minimal CSV from a list of row lists (first row = headers)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        for row in rows:
            writer.writerow(row)


# ===========================================================================
# read_sheet tests
# ===========================================================================


class TestReadSheet:
    def test_read_xlsx_basic(self, tmp_path):
        path = tmp_path / "test.xlsx"
        _create_xlsx(path, [["Name", "Score"], ["Alice", 95], ["Bob", 87]])

        result = read_sheet_mod.read_xlsx(str(path))
        assert len(result) == 2
        assert result[0]["Name"] == "Alice"
        assert result[0]["Score"] == 95
        assert result[1]["Name"] == "Bob"

    def test_read_xlsx_specific_sheet(self, tmp_path):
        path = tmp_path / "multi.xlsx"
        wb = openpyxl.Workbook()
        ws1 = wb.active
        ws1.title = "Data"
        ws1.append(["X", "Y"])
        ws1.append([1, 2])
        ws2 = wb.create_sheet("Other")
        ws2.append(["A", "B"])
        ws2.append([3, 4])
        wb.save(str(path))

        result = read_sheet_mod.read_xlsx(str(path), sheet="Other")
        assert len(result) == 1
        assert result[0]["A"] == 3

    def test_read_xlsx_with_range(self, tmp_path):
        path = tmp_path / "range.xlsx"
        _create_xlsx(path, [
            ["A", "B", "C"],
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])

        result = read_sheet_mod.read_xlsx(str(path), cell_range="A1:B3")
        assert len(result) == 2
        assert "C" not in result[0]
        assert result[0]["A"] == 1
        assert result[1]["A"] == 4

    def test_read_csv(self, tmp_path):
        path = tmp_path / "test.csv"
        _create_csv(path, [["Name", "Score"], ["Alice", "95"], ["Bob", "87"]])

        result = read_sheet_mod.read_csv(str(path))
        assert len(result) == 2
        assert result[0]["Name"] == "Alice"

    def test_read_tsv(self, tmp_path):
        path = tmp_path / "test.tsv"
        _create_csv(path, [["Name", "Score"], ["Alice", "95"]], delimiter="\t")

        result = read_sheet_mod.read_csv(str(path), delimiter="\t")
        assert len(result) == 1
        assert result[0]["Name"] == "Alice"

    def test_read_sheet_dispatch_xlsx(self, tmp_path):
        path = tmp_path / "dispatch.xlsx"
        _create_xlsx(path, [["Col"], ["val"]])
        result = read_sheet_mod.read_sheet(str(path))
        assert len(result) == 1

    def test_read_sheet_dispatch_csv(self, tmp_path):
        path = tmp_path / "dispatch.csv"
        _create_csv(path, [["Col"], ["val"]])
        result = read_sheet_mod.read_sheet(str(path))
        assert len(result) == 1

    def test_read_empty_xlsx(self, tmp_path):
        path = tmp_path / "empty.xlsx"
        wb = openpyxl.Workbook()
        wb.save(str(path))
        result = read_sheet_mod.read_xlsx(str(path))
        # Empty sheet — only has the default empty row
        assert isinstance(result, list)

    def test_format_markdown(self):
        rows = [{"Name": "Alice", "Score": 95}]
        md = read_sheet_mod.format_markdown(rows)
        assert "| Name | Score |" in md
        assert "| Alice | 95 |" in md

    def test_format_markdown_empty(self):
        assert read_sheet_mod.format_markdown([]) == "(empty)"


# ===========================================================================
# write_sheet tests
# ===========================================================================


class TestWriteSheet:
    def test_write_xlsx_roundtrip(self, tmp_path):
        data = [{"Name": "Alice", "Score": 95}, {"Name": "Bob", "Score": 87}]
        path = tmp_path / "out.xlsx"
        write_sheet_mod.write_xlsx(data, str(path))

        result = read_sheet_mod.read_xlsx(str(path))
        assert len(result) == 2
        assert result[0]["Name"] == "Alice"
        assert result[1]["Score"] == 87

    def test_write_csv_roundtrip(self, tmp_path):
        data = [{"Name": "Alice", "Score": "95"}]
        path = tmp_path / "out.csv"
        write_sheet_mod.write_csv(data, str(path))

        result = read_sheet_mod.read_csv(str(path))
        assert len(result) == 1
        assert result[0]["Name"] == "Alice"

    def test_write_xlsx_named_sheet(self, tmp_path):
        data = [{"X": 1}]
        path = tmp_path / "named.xlsx"
        write_sheet_mod.write_xlsx(data, str(path), sheet_name="Results")

        wb = openpyxl.load_workbook(str(path))
        assert "Results" in wb.sheetnames
        wb.close()

    def test_write_empty(self, tmp_path):
        path = tmp_path / "empty.xlsx"
        write_sheet_mod.write_xlsx([], str(path))
        assert path.exists()

    def test_write_sheet_dispatch(self, tmp_path):
        data = [{"A": 1}]
        xlsx_path = tmp_path / "dispatch.xlsx"
        csv_path = tmp_path / "dispatch.csv"

        write_sheet_mod.write_sheet(data, str(xlsx_path))
        write_sheet_mod.write_sheet(data, str(csv_path))

        assert xlsx_path.exists()
        assert csv_path.exists()


# ===========================================================================
# apply_formula tests
# ===========================================================================


class TestApplyFormula:
    def test_single_formula(self, tmp_path):
        path = tmp_path / "formulas.xlsx"
        _create_xlsx(path, [["Val"], [10], [20], [30]])

        apply_formula_mod.apply_formula(str(path), "A5", "=SUM(A2:A4)")

        wb = openpyxl.load_workbook(str(path))
        ws = wb.active
        assert ws["A5"].value == "=SUM(A2:A4)"
        wb.close()

    def test_formula_specific_sheet(self, tmp_path):
        path = tmp_path / "multi.xlsx"
        wb = openpyxl.Workbook()
        ws1 = wb.active
        ws1.title = "Data"
        ws1.append(["V"])
        ws1.append([5])
        ws2 = wb.create_sheet("Calc")
        ws2.append(["R"])
        wb.save(str(path))

        apply_formula_mod.apply_formula(str(path), "A2", "=Data!A2*2", sheet="Calc")

        wb = openpyxl.load_workbook(str(path))
        assert wb["Calc"]["A2"].value == "=Data!A2*2"
        wb.close()

    def test_batch_formulas(self, tmp_path):
        path = tmp_path / "batch.xlsx"
        _create_xlsx(path, [["A", "B"], [1, 2], [3, 4]])

        formulas = [
            {"cell": "A4", "formula": "=SUM(A2:A3)"},
            {"cell": "B4", "formula": "=SUM(B2:B3)"},
        ]
        apply_formula_mod.apply_formulas_batch(str(path), formulas)

        wb = openpyxl.load_workbook(str(path))
        ws = wb.active
        assert ws["A4"].value == "=SUM(A2:A3)"
        assert ws["B4"].value == "=SUM(B2:B3)"
        wb.close()


# ===========================================================================
# pivot tests
# ===========================================================================


class TestPivot:
    def test_basic_pivot(self, tmp_path):
        path = tmp_path / "sales.xlsx"
        _create_xlsx(path, [
            ["Category", "Revenue"],
            ["A", 100],
            ["B", 200],
            ["A", 150],
        ])

        result = pivot_mod.create_pivot(str(path), rows="Category", values="Revenue", aggfunc="sum")
        assert len(result) == 2
        # Find category A
        cat_a = next(r for r in result if r["Category"] == "A")
        assert cat_a["Revenue"] == 250

    def test_pivot_with_cols(self, tmp_path):
        path = tmp_path / "pivot_cols.xlsx"
        _create_xlsx(path, [
            ["Cat", "Month", "Val"],
            ["A", "Jan", 10],
            ["A", "Feb", 20],
            ["B", "Jan", 30],
        ])

        result = pivot_mod.create_pivot(
            str(path), rows="Cat", cols="Month", values="Val", aggfunc="sum"
        )
        assert len(result) == 2

    def test_pivot_count(self, tmp_path):
        path = tmp_path / "count.xlsx"
        _create_xlsx(path, [
            ["Category", "Item"],
            ["A", "x"],
            ["A", "y"],
            ["B", "z"],
        ])

        result = pivot_mod.create_pivot(
            str(path), rows="Category", values="Item", aggfunc="count"
        )
        cat_a = next(r for r in result if r["Category"] == "A")
        assert cat_a["Item"] == 2

    def test_pivot_csv_input(self, tmp_path):
        path = tmp_path / "data.csv"
        _create_csv(path, [["Cat", "Val"], ["A", "10"], ["B", "20"]])

        result = pivot_mod.create_pivot(str(path), rows="Cat", values="Val", aggfunc="sum")
        assert len(result) == 2


# ===========================================================================
# chart tests
# ===========================================================================


class TestChart:
    def test_bar_chart(self, tmp_path):
        path = tmp_path / "chart_data.xlsx"
        _create_xlsx(path, [
            ["Month", "Revenue"],
            ["Jan", 100],
            ["Feb", 200],
            ["Mar", 150],
        ])

        output = tmp_path / "chart_out.xlsx"
        chart_mod.create_chart(str(path), "bar", "Month", "Revenue", output=str(output), title="Test")

        wb = openpyxl.load_workbook(str(output))
        assert "Chart" in wb.sheetnames
        wb.close()

    def test_line_chart(self, tmp_path):
        path = tmp_path / "line.xlsx"
        _create_xlsx(path, [["X", "Y"], [1, 10], [2, 20]])
        chart_mod.create_chart(str(path), "line", "X", "Y")

        wb = openpyxl.load_workbook(str(path))
        assert "Chart" in wb.sheetnames
        wb.close()

    def test_scatter_chart(self, tmp_path):
        path = tmp_path / "scatter.xlsx"
        _create_xlsx(path, [["Weight", "Height"], [60, 170], [70, 175]])
        chart_mod.create_chart(str(path), "scatter", "Weight", "Height")

        wb = openpyxl.load_workbook(str(path))
        assert "Chart" in wb.sheetnames
        wb.close()

    def test_chart_invalid_column(self, tmp_path):
        path = tmp_path / "bad_col.xlsx"
        _create_xlsx(path, [["A", "B"], [1, 2]])

        with pytest.raises(ValueError, match="not found"):
            chart_mod.create_chart(str(path), "bar", "Missing", "B")

    def test_chart_invalid_type(self, tmp_path):
        path = tmp_path / "bad_type.xlsx"
        _create_xlsx(path, [["A", "B"], [1, 2]])

        with pytest.raises(ValueError, match="Unsupported"):
            chart_mod.create_chart(str(path), "radar", "A", "B")


# ===========================================================================
# diff tests
# ===========================================================================


class TestDiff:
    def test_identical_files(self, tmp_path):
        a = tmp_path / "a.xlsx"
        b = tmp_path / "b.xlsx"
        _create_xlsx(a, [["H"], [1]])
        _create_xlsx(b, [["H"], [1]])

        changes = diff_mod.diff_sheets(str(a), str(b))
        assert changes == []

    def test_changed_cell(self, tmp_path):
        a = tmp_path / "a.xlsx"
        b = tmp_path / "b.xlsx"
        _create_xlsx(a, [["H"], [1]])
        _create_xlsx(b, [["H"], [2]])

        changes = diff_mod.diff_sheets(str(a), str(b))
        assert len(changes) == 1
        assert changes[0]["cell"] == "A2"
        assert changes[0]["type"] == "changed"
        assert changes[0]["old"] == 1
        assert changes[0]["new"] == 2

    def test_added_row(self, tmp_path):
        a = tmp_path / "a.xlsx"
        b = tmp_path / "b.xlsx"
        _create_xlsx(a, [["H"], [1]])
        _create_xlsx(b, [["H"], [1], [2]])

        changes = diff_mod.diff_sheets(str(a), str(b))
        assert any(ch["type"] == "added" for ch in changes)

    def test_removed_cell(self, tmp_path):
        a = tmp_path / "a.xlsx"
        b = tmp_path / "b.xlsx"
        _create_xlsx(a, [["H"], [1], [2]])
        _create_xlsx(b, [["H"], [1]])

        changes = diff_mod.diff_sheets(str(a), str(b))
        assert any(ch["type"] == "removed" for ch in changes)

    def test_diff_csv(self, tmp_path):
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        _create_csv(a, [["H"], ["old"]])
        _create_csv(b, [["H"], ["new"]])

        changes = diff_mod.diff_sheets(str(a), str(b))
        assert len(changes) == 1
        assert changes[0]["type"] == "changed"

    def test_diff_markdown_format(self):
        changes = [{"cell": "A1", "type": "changed", "old": 1, "new": 2}]
        md = diff_mod.format_markdown(changes)
        assert "| A1 |" in md
        assert "changed" in md

    def test_diff_markdown_no_changes(self):
        assert "No differences" in diff_mod.format_markdown([])


# ===========================================================================
# Utility function tests
# ===========================================================================


class TestUtilities:
    def test_col_letter_to_index(self):
        assert read_sheet_mod._col_letter_to_index("A") == 0
        assert read_sheet_mod._col_letter_to_index("B") == 1
        assert read_sheet_mod._col_letter_to_index("Z") == 25
        assert read_sheet_mod._col_letter_to_index("AA") == 26

    def test_col_index_to_letter(self):
        assert diff_mod._col_index_to_letter(0) == "A"
        assert diff_mod._col_index_to_letter(1) == "B"
        assert diff_mod._col_index_to_letter(25) == "Z"
        assert diff_mod._col_index_to_letter(26) == "AA"

    def test_parse_range(self):
        result = read_sheet_mod._parse_range("A1:D20")
        assert result == (0, 0, 19, 3)

    def test_parse_range_invalid(self):
        with pytest.raises(ValueError):
            read_sheet_mod._parse_range("invalid")
