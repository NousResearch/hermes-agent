"""End-to-end tests for the pdf skill helper scripts. No network required."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def run(script: str, *args: str, expect: int = 0) -> subprocess.CompletedProcess:
    env = dict(os.environ, LC_ALL="C", LANG="C", PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    assert proc.returncode == expect, f"{script} {args}: rc={proc.returncode}\n{proc.stderr}"
    return proc


@pytest.fixture(scope="module")
def workdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("pdfwork")


@pytest.fixture(scope="module")
def sample_image(workdir: Path) -> Path:
    from PIL import Image
    img_path = workdir / "sample.png"
    img = Image.new("RGB", (120, 80), (30, 120, 200))
    img.save(img_path)
    return img_path


@pytest.fixture(scope="module")
def report_pdf(workdir: Path, sample_image: Path) -> Path:
    spec = {
        "title": "Quarterly Example Report",
        "author": "example-author",
        "elements": [
            {"type": "heading", "text": "Quarterly Example Report", "level": 1},
            {"type": "paragraph", "text": "This is the introduction paragraph with a marker UNIQUEMARK42."},
            {"type": "table", "rows": [["Region", "Units"], ["North", "1250"], ["South", "980"]], "header": True},
            {"type": "image", "path": str(sample_image), "width": 200},
            {"type": "pagebreak"},
            {"type": "heading", "text": "Appendix", "level": 2},
            {"type": "paragraph", "text": "Second page content."},
        ],
    }
    spec_path = workdir / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    out = workdir / "report.pdf"
    run("pdf_create.py", str(spec_path), "-o", str(out))
    assert out.exists() and out.stat().st_size > 500
    return out


def test_create_and_meta(report_pdf: Path):
    meta = json.loads(run("pdf_read.py", str(report_pdf), "--meta").stdout)
    assert meta["page_count"] == 2
    assert meta["encrypted"] is False
    assert meta["likely_scanned_pages"] == []
    assert "Quarterly Example Report" in meta["metadata"].get("Title", "")


def test_extract_text(report_pdf: Path):
    data = json.loads(run("pdf_read.py", str(report_pdf), "--text").stdout)
    assert data["page_count"] == 2
    assert "UNIQUEMARK42" in data["pages"][0]
    assert "Appendix" in data["pages"][1]
    assert "Page 1" in data["pages"][0]  # page number footer


def test_extract_tables(report_pdf: Path, workdir: Path):
    csv_dir = workdir / "csvs"
    data = json.loads(run("pdf_read.py", str(report_pdf), "--tables",
                          "--csv-dir", str(csv_dir)).stdout)
    assert data["table_count"] >= 1
    rows = data["tables"][0]["rows"]
    assert rows[0] == ["Region", "Units"]
    assert ["North", "1250"] in rows
    csv_files = list(csv_dir.glob("*.csv"))
    assert csv_files and "Region" in csv_files[0].read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def form_pdf(workdir: Path) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    out = workdir / "form.pdf"
    c = canvas.Canvas(str(out), pagesize=A4)
    form = c.acroForm
    c.drawString(72, 760, "Example Form")
    form.textfield(name="surname", x=72, y=700, width=300, height=20, value="")
    form.checkbox(name="agree", x=72, y=660, buttonStyle="check")
    form.radio(name="color", value="red", x=72, y=620, selected=False)
    form.radio(name="color", value="blue", x=110, y=620, selected=True)
    form.choice(name="size", x=72, y=580, width=120, height=20,
                options=["small", "large"], value="small")
    c.save()
    return out


def test_form_fill_unicode_roundtrip(form_pdf: Path, workdir: Path):
    surname = "Фамилия — ‘test’"
    values = {"surname": surname, "agree": True, "color": "/red", "size": "large"}
    fields_json = workdir / "values.json"
    fields_json.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")
    filled = workdir / "filled.pdf"
    run("pdf_fill_form.py", str(form_pdf), "--fields-json", str(fields_json),
        "-o", str(filled))
    data = json.loads(run("pdf_read.py", str(filled), "--fields").stdout)
    fields = data["fields"]
    assert fields["surname"]["value"] == surname
    assert fields["agree"]["value"] in ("/Yes", "/On", "True", "/1")
    assert fields["color"]["value"] == "/red"
    assert fields["size"]["value"] == "large"


def test_merge_split_rotate(report_pdf: Path, workdir: Path):
    merged = workdir / "merged.pdf"
    out = json.loads(run("pdf_merge.py", str(report_pdf), str(report_pdf),
                         "-o", str(merged), "--bookmarks").stdout)
    assert out["page_count"] == 4

    part = workdir / "part.pdf"
    out = json.loads(run("pdf_split.py", str(merged), "--pages", "2-3",
                         "--rotate", "90", "-o", str(part)).stdout)
    assert out["page_count"] == 2
    meta = json.loads(run("pdf_read.py", str(part), "--meta").stdout)
    assert meta["page_count"] == 2
    assert all(p["rotation"] % 360 == 90 for p in meta["pages"])


def test_watermark(report_pdf: Path, workdir: Path):
    # Build the stamp at mid-page so its text does not overlap existing
    # headings (overlapping glyphs confuse text extraction).
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    stamp = workdir / "stamp.pdf"
    c = canvas.Canvas(str(stamp), pagesize=A4)
    c.setFont("Helvetica", 40)
    c.drawString(200, 400, "DRAFT")
    c.save()
    stamped = workdir / "stamped.pdf"
    run("pdf_watermark.py", str(report_pdf), "--stamp", str(stamp), "-o", str(stamped))
    data = json.loads(run("pdf_read.py", str(stamped), "--text").stdout)
    assert all("DRAFT" in page for page in data["pages"])


def test_encrypt_decrypt_roundtrip(report_pdf: Path, workdir: Path):
    enc = workdir / "enc.pdf"
    run("pdf_secure.py", str(report_pdf), "--encrypt", "-o", str(enc),
        "--user-password", "your-password")
    meta = json.loads(run("pdf_read.py", str(enc), "--meta").stdout)
    assert meta["encrypted"] is True

    dec = workdir / "dec.pdf"
    run("pdf_secure.py", str(enc), "--decrypt", "-o", str(dec),
        "--password", "your-password")
    data = json.loads(run("pdf_read.py", str(dec), "--text").stdout)
    assert "UNIQUEMARK42" in data["pages"][0]


def test_compress(report_pdf: Path, workdir: Path):
    out = workdir / "compressed.pdf"
    run("pdf_split.py", str(report_pdf), "--pages", "1-2", "--compress", "-o", str(out))
    data = json.loads(run("pdf_read.py", str(out), "--text").stdout)
    assert "UNIQUEMARK42" in data["pages"][0]


# ---------------------------------------------------------------- form creation

FORM_SPEC = {
    "title": "Example Intake Form",
    "page_size": "A4",
    "fields": [
        {"name": "surname", "type": "text", "page": 1, "label": "Surname",
         "label_box": [72, 700, 150, 714], "entry_box": [160, 696, 400, 716]},
        {"name": "agree", "type": "checkbox", "page": 1, "label": "I agree",
         "label_box": [72, 660, 150, 674], "entry_box": [160, 658, 176, 674]},
        {"name": "color", "type": "radio", "page": 1, "label": "Color",
         "label_box": [72, 620, 150, 634], "entry_box": [160, 616, 400, 636],
         "options": ["red", "blue"], "value": "blue"},
        {"name": "size", "type": "dropdown", "page": 1, "label": "Size",
         "label_box": [72, 580, 150, 594], "entry_box": [160, 576, 300, 596],
         "options": ["small", "large"], "value": "small"},
    ],
}


@pytest.fixture(scope="module")
def built_form(workdir: Path) -> Path:
    spec_path = workdir / "formspec.json"
    spec_path.write_text(json.dumps(FORM_SPEC), encoding="utf-8")
    out = workdir / "built_form.pdf"
    result = json.loads(run("pdf_make_form.py", str(spec_path), "-o", str(out)).stdout)
    assert len(result["fields"]) == 4
    return out


def test_make_form_lists_all_fields(built_form: Path):
    data = json.loads(run("pdf_read.py", str(built_form), "--fields").stdout)
    fields = data["fields"]
    assert set(fields) == {"surname", "agree", "color", "size"}
    assert fields["surname"]["type"] == "text"
    assert fields["agree"]["options"] == ["/Off", "/Yes"]
    assert fields["color"]["value"] == "/blue"  # pre-selected radio
    assert set(fields["size"]["options"]) == {"small", "large"}
    # label text is drawn on the page, not just stored in the widget
    text = json.loads(run("pdf_read.py", str(built_form), "--text").stdout)
    assert "Surname" in text["pages"][0] and "Size" in text["pages"][0]


def test_make_form_fill_roundtrip(built_form: Path, workdir: Path):
    values = {"surname": "Smith", "agree": True, "color": "/red", "size": "large"}
    vals = workdir / "builtvals.json"
    vals.write_text(json.dumps(values), encoding="utf-8")
    filled = workdir / "built_filled.pdf"
    run("pdf_fill_form.py", str(built_form), "--fields-json", str(vals), "-o", str(filled))
    fields = json.loads(run("pdf_read.py", str(filled), "--fields").stdout)["fields"]
    assert fields["surname"]["value"] == "Smith"
    assert fields["agree"]["value"] == "/Yes"
    assert fields["color"]["value"] == "/red"
    assert fields["size"]["value"] == "large"


# ------------------------------------------------------------- layout validation

def test_form_layout_valid_spec(workdir: Path):
    spec_path = workdir / "layout_ok.json"
    spec_path.write_text(json.dumps(FORM_SPEC), encoding="utf-8")
    report = json.loads(run("pdf_form_layout.py", str(spec_path)).stdout)
    assert report["ok"] is True
    assert report["errors"] == 0
    assert all(f["ok"] for f in report["fields"])


def test_form_layout_detects_problems(workdir: Path):
    bad = {
        "page_size": "A4",
        "fields": [
            # out of bounds (x1 beyond A4 width)
            {"name": "wide", "type": "text", "page": 1, "label": "Wide",
             "label_box": [10, 700, 60, 714], "entry_box": [70, 696, 900, 716]},
            # two overlapping entry boxes
            {"name": "one", "type": "text", "page": 1, "label": "One",
             "label_box": [10, 600, 60, 614], "entry_box": [70, 596, 300, 616]},
            {"name": "two", "type": "text", "page": 1, "label": "Two",
             "label_box": [10, 560, 60, 574], "entry_box": [200, 600, 400, 620]},
            # label far away from its entry
            {"name": "lost", "type": "text", "page": 1, "label": "Lost",
             "label_box": [10, 100, 60, 114], "entry_box": [400, 500, 500, 520]},
            # too small
            {"name": "tiny", "type": "text", "page": 1,
             "entry_box": [10, 50, 14, 54]},
        ],
    }
    spec_path = workdir / "layout_bad.json"
    spec_path.write_text(json.dumps(bad), encoding="utf-8")
    proc = run("pdf_form_layout.py", str(spec_path), expect=1)
    report = json.loads(proc.stdout)
    assert report["ok"] is False
    by_name = {f["name"]: f for f in report["fields"]}
    assert any("bounds" in p for p in by_name["wide"]["problems"])
    assert any("overlaps field" in p for p in by_name["two"]["problems"])
    assert any("from its entry" in p for p in by_name["lost"]["problems"])
    assert any("minimum size" in p for p in by_name["tiny"]["problems"])
    assert by_name["one"]["ok"]  # first of the overlapping pair reports clean


# ------------------------------------------------- rasterization (overlay, pages)

def _raster_available() -> bool:
    import shutil
    try:
        import pypdfium2  # noqa: F401
        return True
    except ImportError:
        return shutil.which("pdftoppm") is not None


def test_form_layout_overlay(built_form: Path, workdir: Path):
    spec_path = workdir / "formspec.json"
    out_png = workdir / "overlay.png"
    report = json.loads(run("pdf_form_layout.py", str(spec_path), "--pdf", str(built_form),
                            "--render-overlay", str(out_png)).stdout)
    overlay = report["overlay"]
    if _raster_available():
        assert overlay["rendered"] is True
        assert out_png.exists() and out_png.stat().st_size > 1000
        from PIL import Image
        with Image.open(out_png) as img:
            assert img.width > 100 and img.height > 100
    else:
        assert overlay["rendered"] is False
        assert overlay["missing"]  # install hints present


def test_form_layout_overlay_blank_page(workdir: Path):
    # No --pdf: overlay is drawn on a blank page, PIL-only, always renders.
    spec_path = workdir / "formspec.json"
    out_png = workdir / "overlay_blank.png"
    report = json.loads(run("pdf_form_layout.py", str(spec_path),
                            "--render-overlay", str(out_png)).stdout)
    assert report["overlay"]["rendered"] is True
    assert out_png.exists()


def test_page_image_export(report_pdf: Path, workdir: Path):
    out_dir = workdir / "pageimgs"
    result = json.loads(run("pdf_page_image.py", str(report_pdf), "--pages", "1-2",
                            "--dpi", "72", "--out-dir", str(out_dir)).stdout)
    if _raster_available():
        assert result["rendered"] is True
        assert len(result["files"]) == 2
        from PIL import Image
        with Image.open(result["files"][0]) as img:
            # A4 at 72 dpi is ~595x842 px
            assert 500 < img.width < 700
    else:
        assert result["rendered"] is False
        assert result["missing"]


def test_page_image_bad_range(report_pdf: Path, workdir: Path):
    if not _raster_available():
        pytest.skip("no rasterizer available")
    run("pdf_page_image.py", str(report_pdf), "--pages", "9",
        "--out-dir", str(workdir / "nope"), expect=4)


# --------------------------------------------------------------------- stamping

def test_stamp_text(report_pdf: Path, workdir: Path):
    out = workdir / "stamp_text.pdf"
    run("pdf_stamp.py", str(report_pdf), "-o", str(out),
        "--text", "STAMPMARK77", "--x", "150", "--y", "500",
        "--font-size", "30", "--color", "#cc0000", "--pages", "1")
    data = json.loads(run("pdf_read.py", str(out), "--text").stdout)
    assert "STAMPMARK77" in data["pages"][0]
    assert "STAMPMARK77" not in data["pages"][1]  # only page 1 stamped


def test_stamp_text_rotated_opacity(report_pdf: Path, workdir: Path):
    out = workdir / "stamp_rot.pdf"
    run("pdf_stamp.py", str(report_pdf), "-o", str(out),
        "--text", "DRAFT", "--x", "150", "--y", "400", "--font-size", "60",
        "--rotation", "45", "--opacity", "0.3")
    # Rotated glyphs confuse pdfplumber's line grouping; verify via pypdf.
    from pypdf import PdfReader
    text = PdfReader(str(out)).pages[0].extract_text()
    assert "DRAFT" in text


def test_stamp_image(report_pdf: Path, sample_image: Path, workdir: Path):
    out = workdir / "stamp_img.pdf"
    run("pdf_stamp.py", str(report_pdf), "-o", str(out),
        "--image", str(sample_image), "--x", "400", "--y", "60",
        "--width", "100", "--pages", "2")
    from pypdf import PdfReader
    before = PdfReader(str(report_pdf))
    after = PdfReader(str(out))

    def image_xobjects(page):
        res = page.get("/Resources", {})
        xo = res.get("/XObject")
        if xo is None:
            return 0
        return sum(1 for k in xo if xo[k].get("/Subtype") == "/Image")

    assert image_xobjects(after.pages[1]) > image_xobjects(before.pages[1])
    assert image_xobjects(after.pages[0]) == image_xobjects(before.pages[0])


# --------------------------------------------------------- metadata + attachments

def test_meta_set_and_clear(report_pdf: Path, workdir: Path):
    out = workdir / "meta_set.pdf"
    run("pdf_meta.py", str(report_pdf), "--set-meta", "-o", str(out),
        "--title", "Retitled Example", "--author", "example-author",
        "--subject", "Testing", "--keywords", "alpha, beta")
    meta = json.loads(run("pdf_read.py", str(out), "--meta").stdout)["metadata"]
    assert meta["Title"] == "Retitled Example"
    assert meta["Author"] == "example-author"
    assert meta["Subject"] == "Testing"
    assert meta["Keywords"] == "alpha, beta"

    cleared = workdir / "meta_clear.pdf"
    run("pdf_meta.py", str(out), "--clear-meta", "-o", str(cleared))
    meta = json.loads(run("pdf_read.py", str(cleared), "--meta").stdout)["metadata"]
    assert "Title" not in meta or meta.get("Title") in ("", None)


def test_attachments_roundtrip(report_pdf: Path, workdir: Path):
    payload = workdir / "payload.txt"
    payload.write_text("attachment payload UNIQUEATTACH99\n", encoding="utf-8")
    with_att = workdir / "with_att.pdf"
    run("pdf_meta.py", str(report_pdf), "--attach", str(payload), "-o", str(with_att))

    listing = json.loads(run("pdf_meta.py", str(with_att), "--list-attachments").stdout)
    assert listing["attachment_count"] == 1
    assert listing["attachments"] == ["payload.txt"]

    ext_dir = workdir / "extracted"
    result = json.loads(run("pdf_meta.py", str(with_att),
                            "--extract-attachments", str(ext_dir)).stdout)
    assert len(result["extracted"]) == 1
    extracted = Path(result["extracted"][0])
    assert "UNIQUEATTACH99" in extracted.read_text(encoding="utf-8")


def test_attach_preserves_existing_metadata(report_pdf: Path, workdir: Path):
    """#94870: attaching a file must not drop the document's existing
    metadata — writer.append(reader) copies pages only, so the Info dict has
    to be re-applied."""
    titled = workdir / "meta_titled.pdf"
    run("pdf_meta.py", str(report_pdf), "--set-meta",
        "--title", "Preserve Me", "--author", "author-x", "-o", str(titled))
    meta = json.loads(run("pdf_read.py", str(titled), "--meta").stdout)["metadata"]
    assert meta["Title"] == "Preserve Me"

    payload = workdir / "keep_meta_payload.txt"
    payload.write_text("keep-meta payload\n", encoding="utf-8")
    with_att = workdir / "meta_kept.pdf"
    run("pdf_meta.py", str(titled), "--attach", str(payload), "-o", str(with_att))

    meta = json.loads(run("pdf_read.py", str(with_att), "--meta").stdout)["metadata"]
    assert meta.get("Title") == "Preserve Me", "attach dropped existing metadata"
    assert meta.get("Author") == "author-x"
    listing = json.loads(run("pdf_meta.py", str(with_att), "--list-attachments").stdout)
    assert listing["attachments"] == ["keep_meta_payload.txt"]


def test_attachments_survive_encrypt_decrypt(workdir: Path):
    """#94870: embedded attachments must survive an encrypt→decrypt round
    trip — both pdf_secure paths rebuild via writer.append(reader), which
    drops embedded files."""
    pytest.importorskip("cryptography")
    payload = workdir / "secure_payload.txt"
    payload.write_text("secure payload UNIQUESPEC42\n", encoding="utf-8")
    with_att = workdir / "secure_in.pdf"
    spec = {"title": "Secure Doc", "elements": [{"type": "paragraph", "text": "body"}]}
    spec_path = workdir / "secure_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    run("pdf_create.py", str(spec_path), "-o", str(with_att))
    run("pdf_meta.py", str(with_att), "--attach", str(payload), "-o", str(with_att))

    encrypted = workdir / "secure_enc.pdf"
    run("pdf_secure.py", str(with_att), "--encrypt",
        "--user-password", "pw", "-o", str(encrypted))
    decrypted = workdir / "secure_dec.pdf"
    run("pdf_secure.py", str(encrypted), "--decrypt", "--password", "pw",
        "-o", str(decrypted))

    listing = json.loads(run("pdf_meta.py", str(decrypted), "--list-attachments").stdout)
    assert listing["attachments"] == ["secure_payload.txt"], (
        "attachment lost across encrypt/decrypt"
    )
    meta = json.loads(run("pdf_read.py", str(decrypted), "--meta").stdout)["metadata"]
    assert meta.get("Title") == "Secure Doc"


XMP_PACKET = (
    b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
    b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b'<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
    b'<dc:title><rdf:Alt><rdf:li xml:lang="x-default">XMPTitle</rdf:li>'
    b'</rdf:Alt></dc:title></rdf:Description>'
    b'</rdf:RDF></x:xmpmeta>\n<?xpacket end="w"?>'
)


def _write_pdf_with_xmp_and_dup_attachments(path):
    """Build a source PDF carrying an XMP packet plus two payloads sharing
    one attachment name — both shapes that writer.append() drops."""
    import io

    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import NameObject, StreamObject
    w = pypdf.PdfWriter()
    w.add_blank_page(200, 200)
    w.add_metadata({"/Title": "InfoTitle"})
    stream = StreamObject()
    stream.set_data(XMP_PACKET)
    stream.update({
        NameObject("/Type"): NameObject("/Metadata"),
        NameObject("/Subtype"): NameObject("/XML"),
    })
    w._root_object[NameObject("/Metadata")] = stream
    w.add_attachment("dup.bin", b"one")
    w.add_attachment("dup.bin", b"two")
    buf = io.BytesIO()
    w.write(buf)
    Path(path).write_bytes(buf.getvalue())


def test_set_meta_preserves_xmp_and_all_same_name_attachments(workdir: Path):
    """#94870 review: the preserve block must also carry the catalog's XMP
    metadata stream and *every* payload under a shared attachment name —
    writer.append() drops both."""
    pypdf = pytest.importorskip("pypdf")
    src = workdir / "xmp_src.pdf"
    _write_pdf_with_xmp_and_dup_attachments(src)

    out = workdir / "xmp_kept.pdf"
    run("pdf_meta.py", str(src), "--set-meta", "--title", "New", "-o", str(out))

    r = pypdf.PdfReader(str(out))
    assert r.xmp_metadata is not None, "XMP packet dropped by --set-meta"
    assert "XMPTitle" in (r.xmp_metadata.dc_title or {}).values()
    payloads = r.attachments.get("dup.bin")
    assert sorted(bytes(p) for p in payloads) == [b"one", b"two"], (
        "same-name attachment payloads truncated"
    )
    assert (r.metadata or {}).get("/Title") == "New"


def test_xmp_and_same_name_attachments_survive_encrypt_decrypt(workdir: Path):
    """#94870 review: pdf_secure rebuilds via append() too — the XMP packet
    and every payload under a shared name must survive the round trip."""
    pypdf = pytest.importorskip("pypdf")
    pytest.importorskip("cryptography")
    src = workdir / "xmp_secure_src.pdf"
    _write_pdf_with_xmp_and_dup_attachments(src)

    encrypted = workdir / "xmp_enc.pdf"
    run("pdf_secure.py", str(src), "--encrypt", "--user-password", "pw",
        "-o", str(encrypted))
    decrypted = workdir / "xmp_dec.pdf"
    run("pdf_secure.py", str(encrypted), "--decrypt", "--password", "pw",
        "-o", str(decrypted))

    r = pypdf.PdfReader(str(decrypted))
    assert r.xmp_metadata is not None, "XMP packet dropped by encrypt/decrypt"
    assert "XMPTitle" in (r.xmp_metadata.dc_title or {}).values()
    payloads = r.attachments.get("dup.bin")
    assert sorted(bytes(p) for p in payloads) == [b"one", b"two"], (
        "same-name attachment payloads truncated across encrypt/decrypt"
    )
