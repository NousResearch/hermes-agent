"""Long pdf-skill table cells must wrap inside the page."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("reportlab")
pypdf = pytest.importorskip("pypdf")

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "productivity"
    / "pdf"
    / "scripts"
    / "pdf_create.py"
)

SPEC = {
    "title": "Table overflow repro",
    "elements": [
        {
            "type": "table",
            "header": True,
            "rows": [
                ["Item", "Description", "Qty"],
                [
                    "1",
                    "This is a very long description that contains several words "
                    "and should automatically wrap inside a normal table cell so "
                    "that it does not run past the page margin",
                    "3",
                ],
                ["2", "Tom & Jerry <qty>", "1"],
            ],
        }
    ],
}

# Negative control: the same spec through a raw-strings Table, the pre-fix
# rendering mechanism. reportlab never wraps plain strings, so the long
# description column widens past the right margin; Paragraph flowables (the
# fix) keep it inside. Pinning the overflow here keeps the fixed-run margin
# assertion from passing for the wrong reason (e.g. an accidentally empty
# text layer or a spec too narrow to overflow at all).
BASELINE_SCRIPT = """\
import json
import sys

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

with open(sys.argv[1], encoding="utf-8") as fh:
    spec = json.load(fh)
rows = next(el["rows"] for el in spec["elements"] if el["type"] == "table")
doc = SimpleDocTemplate(sys.argv[2], pagesize=A4)
table = Table(rows, repeatRows=1)
table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "grey")]))
doc.build([table])
"""


def _extract(pdf_path: Path) -> tuple[str, float]:
    page = pypdf.PdfReader(str(pdf_path)).pages[0]
    origins: list[float] = []

    def visitor(text, _cm, tm, _font, _size):
        if text and text.strip():
            origins.append(float(tm[4]))

    text = page.extract_text(visitor_text=visitor) or ""
    assert origins, "expected text positions"
    return text, max(origins)


def test_table_cells_stay_inside_the_page(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    out = tmp_path / "table.pdf"
    spec_path.write_text(json.dumps(SPEC), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(spec_path), "-o", str(out)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    width = float(pypdf.PdfReader(str(out)).pages[0].mediabox.width)
    text, fixed_max_origin = _extract(out)

    assert "Qty" in text
    # Exact string only: without escape() reportlab parses "<qty>" as an XML
    # tag and silently drops it, so a bare "Tom & Jerry" match would pass
    # even with the escaping removed.
    assert "Tom & Jerry <qty>" in text

    baseline = tmp_path / "baseline.py"
    baseline_pdf = tmp_path / "baseline.pdf"
    baseline.write_text(BASELINE_SCRIPT, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(baseline), str(spec_path), str(baseline_pdf)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    _, baseline_max_origin = _extract(baseline_pdf)

    # Right margin is 72 pt. The raw-strings baseline parks text past it;
    # the Paragraph fix must keep every text run inside the margin.
    assert baseline_max_origin > width - 72
    assert fixed_max_origin < width - 72
