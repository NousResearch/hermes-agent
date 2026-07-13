import importlib.util
import runpy
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "skills/productivity/ocr-and-documents/scripts/extract_pymupdf.py"
)
SPEC = importlib.util.spec_from_file_location("extract_pymupdf", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
extract_pymupdf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extract_pymupdf)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("3", [3]),
        ("1-5", [1, 2, 3, 4, 5]),
    ],
)
def test_parse_pages_accepts_valid_values(value, expected):
    assert extract_pymupdf.parse_pages(["document.pdf", "--pages", value]) == expected


@pytest.mark.parametrize(
    "page_args",
    [
        ["--pages"],
        ["--pages", "1-2-3"],
        ["--pages", "abc"],
        ["--pages", "1-x"],
    ],
)
def test_invalid_pages_exit_cleanly(page_args, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "document.pdf", *page_args])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(SCRIPT), run_name="__main__")

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "ERROR:" in captured.err
    assert "Traceback" not in captured.err
