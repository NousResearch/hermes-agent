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


def test_table_cells_stay_inside_the_page(tmp_path: Path) -> None:
    spec = {
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
    spec_path = tmp_path / "spec.json"
    out = tmp_path / "table.pdf"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(spec_path), "-o", str(out)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    page = pypdf.PdfReader(str(out)).pages[0]
    width = float(page.mediabox.width)
    origins: list[float] = []

    def visitor(text, _cm, tm, _font, _size):
        if text and text.strip():
            origins.append(float(tm[4]))

    text = page.extract_text(visitor_text=visitor) or ""
    assert "Qty" in text
    assert "Tom & Jerry <qty>" in text or "Tom & Jerry" in text
    # Right margin is 72 pt. An unwrapped row parks the last column past the page.
    assert origins, "expected text positions"
    assert max(origins) < width - 72
