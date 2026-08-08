# MIT License. End-to-end tests for the docx skill.
"""Pytest suite proving create / read / edit / template round-trips.

Runs the scripts as subprocesses (argparse CLIs) and also verifies the
outputs with python-docx directly. Stdlib + python-docx only; all
fixtures are generated on the fly; no network.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest
from docx import Document

SKILL = Path(__file__).resolve().parent.parent
SCRIPTS = SKILL / "scripts"

NON_ASCII = "Фамилия — ‘test’"


def make_png(path: Path) -> None:
    """Write a tiny valid 2x2 red PNG using only stdlib."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * 2 for _ in range(2))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    path.write_bytes(png)


def run(script: str, *args: str) -> dict:
    env = dict(os.environ)
    env["LC_ALL"] = "C"  # prove no locale-default text reads
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        capture_output=True, env=env)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    return json.loads(proc.stdout.decode("utf-8"))


@pytest.fixture(scope="module")
def workdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("docxskill")


@pytest.fixture(scope="module")
def created(workdir: Path) -> Path:
    """Create a document exercising every create feature."""
    png = workdir / "pic.png"
    make_png(png)
    spec = {
        "page": {"width_mm": 210, "height_mm": 297,
                 "margins_mm": {"top": 25, "bottom": 25,
                                "left": 20, "right": 20}},
        "header": "Report header",
        "footer": "Page footer",
        "styles": [{"name": "FancyNote", "base": "Normal", "font": "Arial",
                    "size_pt": 11, "italic": True, "color": "1F4E79"}],
        "blocks": [
            {"type": "heading", "text": "Main Title", "level": 1},
            {"type": "heading", "text": "Section One", "level": 2},
            {"type": "paragraph", "runs": [
                {"text": "plain "},
                {"text": "boldbit", "bold": True},
                {"text": " italicbit", "italic": True},
                {"text": " underbit", "underline": True}]},
            {"type": "paragraph", "text": "Styled note.",
             "style": "FancyNote"},
            {"type": "bullet_list", "items": ["alpha", "beta"]},
            {"type": "numbered_list", "items": ["first", "second"]},
            {"type": "table", "header": ["Name", "Qty"],
             "rows": [["Widget", "3"], ["Gadget", "5"]],
             "style": "Table Grid", "header_bold": True},
            {"type": "image", "path": str(png), "width_mm": 30},
            {"type": "page_break"},
            {"type": "paragraph", "text": "After the break."},
        ],
    }
    spec_path = workdir / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    out = workdir / "created.docx"
    res = run("docx_create.py", spec_path, out)
    assert res["ok"] and out.exists()
    return out


class TestCreateAndRead:
    def test_text_roundtrip(self, created: Path):
        text = run("docx_read.py", created, "--text")
        body = "\n".join(text["body"])
        for expected in ("Main Title", "plain boldbit italicbit underbit",
                         "Styled note.", "alpha", "second",
                         "After the break."):
            assert expected in body
        assert text["tables"] == [[["Name", "Qty"], ["Widget", "3"],
                                   ["Gadget", "5"]]]
        assert "Report header" in text["headers"]
        assert "Page footer" in text["footers"]

    def test_structure(self, created: Path):
        st = run("docx_read.py", created, "--structure")
        outline = [(h["level"], h["text"]) for h in st["outline"]]
        assert (1, "Main Title") in outline
        assert (2, "Section One") in outline
        assert st["table_count"] == 1
        assert st["tables"][0] == {"rows": 3, "cols": 2}

    def test_styles_used(self, created: Path):
        styles = run("docx_read.py", created, "--styles")["styles"]
        for s in ("Heading 1", "FancyNote", "List Bullet", "List Number",
                  "Table Grid"):
            assert s in styles

    def test_images_extracted(self, created: Path, workdir: Path):
        outdir = workdir / "media"
        res = run("docx_read.py", created, "--images", outdir)
        assert len(res["images"]) == 1
        img = Path(res["images"][0])
        assert img.read_bytes().startswith(b"\x89PNG")

    def test_run_formatting_persisted(self, created: Path):
        doc = Document(str(created))
        para = next(p for p in doc.paragraphs if "boldbit" in p.text)
        flags = {r.text.strip(): (r.bold, r.italic, r.underline)
                 for r in para.runs if r.text.strip()}
        assert flags["boldbit"][0] is True
        assert flags["italicbit"][1] is True
        assert flags["underbit"][2] is True

    def test_page_setup(self, created: Path):
        sec = Document(str(created)).sections[0]
        assert round(sec.page_width.mm) == 210
        assert round(sec.top_margin.mm) == 25

    def test_revisions_detection(self, created: Path):
        rev = run("docx_read.py", created, "--revisions")
        assert rev["has_tracked_changes"] is False
        assert rev["comments"] is False


class TestEdit:
    def test_replace_preserves_formatting(self, created: Path, workdir: Path):
        out = workdir / "edited.docx"
        res = run("docx_edit.py", "replace", created, "--find", "boldbit",
                  "--replace", "REPLACED", "-o", out)
        assert res["replacements"] == 1
        doc = Document(str(out))
        para = next(p for p in doc.paragraphs if "REPLACED" in p.text)
        run_ = next(r for r in para.runs if "REPLACED" in r.text)
        assert run_.bold is True  # formatting survived

    def test_set_cell(self, created: Path, workdir: Path):
        out = workdir / "cell.docx"
        run("docx_edit.py", "set-cell", created, "--table", "0", "--row",
            "1", "--col", "1", "--text", "99", "-o", out)
        assert Document(str(out)).tables[0].cell(1, 1).text == "99"

    def test_insert_and_delete(self, created: Path, workdir: Path):
        out = workdir / "ins.docx"
        run("docx_edit.py", "insert", created, "--index", "0", "--text",
            "Inserted first", "-o", out)
        doc = Document(str(out))
        assert doc.paragraphs[0].text == "Inserted first"
        out2 = workdir / "del.docx"
        run("docx_edit.py", "delete", out, "--index", "0", "-o", out2)
        assert Document(str(out2)).paragraphs[0].text != "Inserted first"

    def test_apply_style(self, created: Path, workdir: Path):
        out = workdir / "styled.docx"
        doc = Document(str(created))
        idx = next(i for i, p in enumerate(doc.paragraphs)
                   if p.text == "After the break.")
        run("docx_edit.py", "style", created, "--index", str(idx),
            "--style", "Heading 2", "-o", out)
        doc2 = Document(str(out))
        assert doc2.paragraphs[idx].style.name == "Heading 2"


class TestTemplate:
    def test_fill_everywhere_non_ascii(self, workdir: Path):
        # Build a template: tokens in body, split runs, table, header, footer.
        tpl = workdir / "tpl.docx"
        doc = Document()
        doc.sections[0].header.paragraphs[0].text = "H: {{name}}"
        doc.sections[0].footer.paragraphs[0].text = "F: {{date}}"
        p = doc.add_paragraph()
        p.add_run("Dear {{na")          # token split across runs
        p.add_run("me}}, hello.")
        t = doc.add_table(rows=1, cols=2)
        t.cell(0, 0).text = "{{name}}"
        t.cell(0, 1).text = "{{ date }}"   # spaced variant
        doc.add_paragraph("Unfilled: {{missing}}")
        doc.save(str(tpl))

        values = workdir / "values.json"
        values.write_text(
            json.dumps({"name": NON_ASCII, "date": "2026-08-08"},
                       ensure_ascii=False), encoding="utf-8")
        out = workdir / "filled.docx"
        res = run("docx_template.py", tpl, values, out)
        assert res["ok"] is True
        assert res["unfilled_tokens"] == ["missing"]

        text = run("docx_read.py", out, "--text")
        assert f"Dear {NON_ASCII}, hello." in text["body"]
        assert text["tables"][0][0] == [NON_ASCII, "2026-08-08"]
        assert f"H: {NON_ASCII}" in text["headers"]
        assert "F: 2026-08-08" in text["footers"]

    def test_strict_fails_on_unfilled(self, workdir: Path):
        tpl = workdir / "tpl2.docx"
        doc = Document()
        doc.add_paragraph("{{gone}}")
        doc.save(str(tpl))
        values = workdir / "empty.json"
        values.write_text("{}", encoding="utf-8")
        env = dict(os.environ, LC_ALL="C", PYTHONIOENCODING="utf-8")
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "docx_template.py"), str(tpl),
             str(values), str(workdir / "out2.docx"), "--strict"],
            capture_output=True, env=env)
        assert proc.returncode == 1
        payload = json.loads(proc.stdout.decode("utf-8"))
        assert payload["unfilled_tokens"] == ["gone"]
