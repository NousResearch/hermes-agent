"""dcf-model validate_dcf.py: the WACC range check must actually run.

openpyxl is not a test dependency, so ``openpyxl.load_workbook`` is replaced by a
double exposing only the Workbook/Worksheet surface openpyxl itself provides
(``sheetnames``, ``wb[name]``, ``name in wb``, ``ws.iter_rows``, ``ws.cell``).
openpyxl's Workbook has no ``.get()``.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SCRIPT = (Path(__file__).resolve().parents[2]
          / "optional-skills/finance/dcf-model/scripts/validate_dcf.py")


class _Sheet:
    def __init__(self, cells: dict[tuple[int, int], object]):
        self._cells = cells

    def cell(self, row, column):
        return SimpleNamespace(row=row, column=column, value=self._cells.get((row, column)))

    def iter_rows(self, max_row, max_col):
        for r in range(1, max_row + 1):
            yield [self.cell(r, c) for c in range(1, max_col + 1)]


class _Workbook:
    def __init__(self, sheets: dict[str, _Sheet]):
        self._sheets = sheets
        self.sheetnames = list(sheets)

    def __getitem__(self, name):
        if name not in self._sheets:
            raise KeyError(f"Worksheet {name} does not exist.")
        return self._sheets[name]

    def __contains__(self, name):
        return name in self._sheets


def _validator(monkeypatch, tmp_path, sheets):
    fake = ModuleType("openpyxl")
    fake.load_workbook = lambda path, data_only=False: _Workbook(sheets)
    monkeypatch.setitem(sys.modules, "openpyxl", fake)
    model = tmp_path / "model.xlsx"
    model.write_bytes(b"")
    return runpy.run_path(str(SCRIPT))["DCFModelValidator"](str(model))


@pytest.mark.parametrize("sheets, expected", [
    ({"DCF": _Sheet({}), "WACC": _Sheet({(3, 1): "WACC", (3, 2): 0.25})},
     "WACC (25.00%) is outside typical range"),
    ({"DCF": _Sheet({(5, 1): "WACC", (5, 2): 0.09})},
     "✓ WACC (9.00%) in reasonable range"),
])
def test_wacc_range_check_reads_wacc_sheet_or_falls_back_to_dcf(monkeypatch, tmp_path, sheets, expected):
    validator = _validator(monkeypatch, tmp_path, sheets)
    validator._check_wacc_range()
    messages = validator.warnings + validator.info
    assert any(expected in m for m in messages), messages
    assert not any("Could not validate WACC range" in m for m in validator.warnings)
