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


# --------------------------------------------------------------- new parity

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def q(tag: str) -> str:
    return f"{{{W}}}{tag}"


def run_raw(script: str, *args: str):
    env = dict(os.environ, LC_ALL="C", PYTHONIOENCODING="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        capture_output=True, env=env)


def _add_ins(para, rev_id: int, text: str, author="Editor"):
    from lxml import etree
    ins = etree.SubElement(para._p, q("ins"))
    ins.set(q("id"), str(rev_id))
    ins.set(q("author"), author)
    ins.set(q("date"), "2026-01-02T03:04:05Z")
    r = etree.SubElement(ins, q("r"))
    t = etree.SubElement(r, q("t"))
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def _add_del(para, rev_id: int, text: str, author="Editor"):
    from lxml import etree
    dele = etree.SubElement(para._p, q("del"))
    dele.set(q("id"), str(rev_id))
    dele.set(q("author"), author)
    dele.set(q("date"), "2026-01-02T03:04:05Z")
    r = etree.SubElement(dele, q("r"))
    t = etree.SubElement(r, q("delText"))
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


@pytest.fixture()
def tracked(tmp_path: Path) -> Path:
    """Doc with a tracked insertion + deletion in body AND in a table."""
    doc = Document()
    p = doc.add_paragraph("Base ")
    _add_ins(p, 1, "ADDED")
    _add_del(p, 2, "REMOVED")
    table = doc.add_table(rows=1, cols=1)
    cp = table.cell(0, 0).paragraphs[0]
    cp.add_run("Cell ")
    _add_ins(cp, 3, "CELLADD")
    _add_del(cp, 4, "CELLGONE")
    path = tmp_path / "tracked.docx"
    doc.save(str(path))
    return path


class TestRevisions:
    def test_list(self, tracked: Path):
        res = run("docx_revisions.py", "list", tracked)
        revs = {r["id"]: r for r in res["revisions"]}
        assert len(revs) == 4
        assert revs["1"] == {"id": "1", "author": "Editor",
                             "date": "2026-01-02T03:04:05Z",
                             "type": "insertion", "text": "ADDED"}
        assert revs["2"]["type"] == "deletion"
        assert revs["2"]["text"] == "REMOVED"
        assert revs["3"]["text"] == "CELLADD"  # inside table
        assert revs["4"]["type"] == "deletion"

    def test_accept_all(self, tracked: Path, tmp_path: Path):
        out = tmp_path / "acc.docx"
        res = run("docx_revisions.py", "accept-all", tracked, "-o", out)
        assert res["resolved"] == 4
        doc = Document(str(out))
        assert doc.paragraphs[0].text == "Base ADDED"
        assert doc.tables[0].cell(0, 0).text == "Cell CELLADD"
        assert run("docx_revisions.py", "list", out)["revisions"] == []

    def test_reject_all(self, tracked: Path, tmp_path: Path):
        out = tmp_path / "rej.docx"
        run("docx_revisions.py", "reject-all", tracked, "-o", out)
        doc = Document(str(out))
        assert doc.paragraphs[0].text == "Base REMOVED"
        assert doc.tables[0].cell(0, 0).text == "Cell CELLGONE"

    def test_accept_single_by_id(self, tracked: Path, tmp_path: Path):
        out = tmp_path / "one.docx"
        res = run("docx_revisions.py", "accept", tracked, "--id", "1",
                  "-o", out)
        assert res["resolved"] == 1
        doc = Document(str(out))
        assert doc.paragraphs[0].text == "Base ADDED"  # del 2 unresolved
        remaining = run("docx_revisions.py", "list", out)["revisions"]
        assert sorted(r["id"] for r in remaining) == ["2", "3", "4"]

    def test_reject_single_by_id(self, tracked: Path, tmp_path: Path):
        out = tmp_path / "rone.docx"
        run("docx_revisions.py", "reject", tracked, "--id", "2", "-o", out)
        doc = Document(str(out))
        assert doc.paragraphs[0].text == "Base REMOVED"  # ins 1 unresolved

    def test_unknown_id_fails(self, tracked: Path, tmp_path: Path):
        proc = run_raw("docx_revisions.py", "accept", tracked, "--id",
                       "999", "-o", tmp_path / "x.docx")
        assert proc.returncode == 1


class TestComments:
    @pytest.fixture()
    def base(self, tmp_path: Path) -> Path:
        doc = Document()
        doc.add_paragraph("The quarterly revenue rose sharply.")
        doc.add_paragraph("Second paragraph.")
        path = tmp_path / "base.docx"
        doc.save(str(path))
        return path

    def test_add_list_delete(self, base: Path, tmp_path: Path):
        out = tmp_path / "com.docx"
        res = run("docx_comments.py", "add", base, "--target",
                  "quarterly revenue", "--text", "Needs a source",
                  "--author", "Reviewer", "--initials", "R", "-o", out)
        assert res["ok"] is True
        cid = res["comment_id"]

        listed = run("docx_comments.py", "list", out)["comments"]
        assert len(listed) == 1
        c = listed[0]
        assert c["id"] == cid
        assert c["author"] == "Reviewer"
        assert c["text"] == "Needs a source"
        assert c["anchored_text"] == "quarterly revenue"
        assert c["date"]

        # document text unchanged by anchoring
        text = run("docx_read.py", out, "--text")
        assert "The quarterly revenue rose sharply." in text["body"]

        out2 = tmp_path / "nocom.docx"
        run("docx_comments.py", "delete", out, "--id", cid, "-o", out2)
        assert run("docx_comments.py", "list", out2)["comments"] == []
        text2 = run("docx_read.py", out2, "--text")
        assert "The quarterly revenue rose sharply." in text2["body"]

    def test_xml_fallback_path(self, base: Path, tmp_path: Path):
        out = tmp_path / "xmlcom.docx"
        res = run("docx_comments.py", "add", base, "--target",
                  "Second paragraph", "--text", "fallback note",
                  "--author", "Bot", "--xml", "-o", out)
        assert res["native_api"] is False
        listed = run("docx_comments.py", "list", out)["comments"]
        assert listed[0]["text"] == "fallback note"
        assert listed[0]["anchored_text"] == "Second paragraph"
        # file still opens cleanly
        assert Document(str(out)).paragraphs[1].text == "Second paragraph."

    def test_missing_target_fails(self, base: Path, tmp_path: Path):
        proc = run_raw("docx_comments.py", "add", base, "--target",
                       "not present", "--text", "x", "-o",
                       tmp_path / "y.docx")
        assert proc.returncode == 1


class TestValidate:
    def test_healthy_file_passes(self, created: Path):
        res = run("docx_validate.py", created)
        assert res["ok"] is True
        assert all(i["severity"] != "error" for i in res["issues"])

    def test_not_a_zip(self, tmp_path: Path):
        bad = tmp_path / "bad.docx"
        bad.write_bytes(b"this is not a zip file")
        proc = run_raw("docx_validate.py", bad)
        assert proc.returncode == 1
        rep = json.loads(proc.stdout.decode("utf-8"))
        assert rep["issues"][0]["code"] == "not-a-zip"

    def test_dangling_rel_and_empty_image(self, created: Path,
                                          tmp_path: Path):
        import shutil
        import zipfile
        broken = tmp_path / "broken.docx"
        shutil.copy(created, broken)
        # rebuild the zip: drop the image part, zero out nothing else
        src = zipfile.ZipFile(str(created))
        with zipfile.ZipFile(str(broken), "w") as dst:
            for item in src.infolist():
                if item.filename.startswith("word/media/"):
                    dst.writestr(item.filename, b"")  # empty image
                else:
                    dst.writestr(item, src.read(item.filename))
        proc = run_raw("docx_validate.py", broken)
        assert proc.returncode == 1
        rep = json.loads(proc.stdout.decode("utf-8"))
        codes = {i["code"] for i in rep["issues"]}
        assert "empty-image" in codes

    def test_missing_style(self, tmp_path: Path):
        import zipfile
        doc = Document()
        doc.add_paragraph("styled", style="Heading 1")
        path = tmp_path / "styles.docx"
        doc.save(str(path))
        # rewrite document.xml to reference a style id that doesn't exist
        src = zipfile.ZipFile(str(path))
        broken = tmp_path / "badstyle.docx"
        with zipfile.ZipFile(str(broken), "w") as dst:
            for item in src.infolist():
                data = src.read(item.filename)
                if item.filename == "word/document.xml":
                    data = data.replace(b'w:val="Heading1"',
                                        b'w:val="GhostStyle"')
                dst.writestr(item, data)
        proc = run_raw("docx_validate.py", broken)
        assert proc.returncode == 1
        rep = json.loads(proc.stdout.decode("utf-8"))
        assert any(i["code"] == "missing-style" and "GhostStyle"
                   in i["detail"] for i in rep["issues"])


class TestNormalize:
    def test_merges_split_runs(self, tmp_path: Path):
        doc = Document()
        p = doc.add_paragraph()
        p.add_run("Hel")            # identical (no) formatting, split
        p.add_run("lo wo")
        p.add_run("rld")
        b = p.add_run("BOLD1")
        b.bold = True
        b2 = p.add_run("BOLD2")
        b2.bold = True
        i = p.add_run("ital")
        i.italic = True
        path = tmp_path / "split.docx"
        doc.save(str(path))

        out = tmp_path / "norm.docx"
        res = run("docx_edit.py", "normalize", path, "-o", out)
        assert res["runs_merged"] == 3  # 2 plain merges + 1 bold merge

        doc2 = Document(str(out))
        para = doc2.paragraphs[0]
        assert para.text == "Hello worldBOLD1BOLD2ital"
        assert [r.text for r in para.runs] == \
            ["Hello world", "BOLD1BOLD2", "ital"]
        assert para.runs[1].bold is True
        assert para.runs[2].italic is True


class TestFields:
    def test_toc_and_page_numbers_via_edit(self, created: Path,
                                           tmp_path: Path):
        out = tmp_path / "fields.docx"
        run("docx_edit.py", "toc", created, "--index", "0", "-o", out)
        run("docx_edit.py", "page-numbers", out)

        import zipfile
        doc_xml = zipfile.ZipFile(str(out)).read(
            "word/document.xml").decode("utf-8")
        assert "TOC \\o" in doc_xml
        assert "fldChar" in doc_xml
        footer_names = [n for n in zipfile.ZipFile(str(out)).namelist()
                        if n.startswith("word/footer")]
        footers = "".join(zipfile.ZipFile(str(out)).read(n).decode("utf-8")
                          for n in footer_names)
        assert "PAGE" in footers and "NUMPAGES" in footers
        # still a valid document
        assert run("docx_validate.py", out)["ok"] is True

    def test_toc_and_footer_in_create_spec(self, tmp_path: Path):
        spec = {
            "footer_page_numbers": True,
            "blocks": [
                {"type": "toc"},
                {"type": "heading", "text": "Chapter", "level": 1},
            ],
        }
        spec_path = tmp_path / "fspec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        out = tmp_path / "fcreate.docx"
        run("docx_create.py", spec_path, out)

        import zipfile
        z = zipfile.ZipFile(str(out))
        assert "TOC \\o" in z.read("word/document.xml").decode("utf-8")
        footers = "".join(z.read(n).decode("utf-8") for n in z.namelist()
                          if n.startswith("word/footer"))
        assert "NUMPAGES" in footers
