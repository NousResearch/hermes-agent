"""Each extracted occurrence receives the requested rotation once."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "skills/productivity/pdf/scripts/pdf_split.py"


@pytest.mark.parametrize("initial,rotation", [(0, 90), (90, 90), (0, -90), (90, 0)])
def test_repeated_page_rotation_does_not_mutate_source(tmp_path, initial, rotation):
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    fixture = subprocess.run(
        [sys.executable, "-c", """
from pypdf import PdfWriter
import sys
writer = PdfWriter()
writer.add_blank_page(width=100, height=200).rotate(int(sys.argv[2]))
writer.add_blank_page(width=300, height=400)
writer.write(sys.argv[1])
""", str(source), str(initial)], capture_output=True, text=True, timeout=30, check=False,
    )
    assert fixture.returncode == 0, fixture.stderr
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), "--pages", "1,2,1",
         "--rotate", str(rotation), "--output", str(output)],
        capture_output=True, text=True, encoding="utf-8", timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    inspected = subprocess.run(
        [sys.executable, "-c", """
from pypdf import PdfReader
import json, sys
pages = PdfReader(sys.argv[1]).pages
print(json.dumps({'rotations': [p.rotation % 360 for p in pages],
                  'widths': [float(p.mediabox.width) for p in pages],
                  'source_rotation': PdfReader(sys.argv[2]).pages[0].rotation % 360}))
""", str(output), str(source)], capture_output=True, text=True, timeout=30, check=False,
    )
    assert inspected.returncode == 0, inspected.stderr
    data = json.loads(inspected.stdout)
    assert data["rotations"] == [(initial + rotation) % 360, rotation % 360,
                                  (initial + rotation) % 360]
    assert data["widths"] == [100, 300, 100]
    assert data["source_rotation"] == initial
