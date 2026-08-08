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
