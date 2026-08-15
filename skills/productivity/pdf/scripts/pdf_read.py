#!/usr/bin/env python3
"""Read a PDF: per-page text, tables, metadata, or form fields. JSON to stdout."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys


def _reconfigure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _need(module: str, package: str):
    try:
        return __import__(module)
    except ImportError:
        print(f"Missing dependency: install with 'python3 -m pip install {package}'", file=sys.stderr)
        raise SystemExit(2)


def read_text(path: str, password: str | None) -> dict:
    pdfplumber = _need("pdfplumber", "pdfplumber")
    pages = []
    with pdfplumber.open(path, password=password) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return {"page_count": len(pages), "pages": pages}


def read_tables(path: str, password: str | None, csv_dir: str | None) -> dict:
    pdfplumber = _need("pdfplumber", "pdfplumber")
    result = []
    written = []
    with pdfplumber.open(path, password=password) as pdf:
        for pageno, page in enumerate(pdf.pages, start=1):
            for tidx, table in enumerate(page.extract_tables()):
                result.append({"page": pageno, "index": tidx, "rows": table})
                if csv_dir:
                    os.makedirs(csv_dir, exist_ok=True)
                    csv_path = os.path.join(csv_dir, f"page{pageno}_table{tidx}.csv")
                    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                        csv.writer(fh).writerows([[c if c is not None else "" for c in row] for row in table])
                    written.append(csv_path)
    out = {"table_count": len(result), "tables": result}
    if csv_dir:
        out["csv_files"] = written
    return out


def read_meta(path: str, password: str | None) -> dict:
    pypdf = _need("pypdf", "pypdf")
    reader = pypdf.PdfReader(path)
    encrypted = reader.is_encrypted
    if encrypted:
        if password is None or not reader.decrypt(password):
            return {"encrypted": True, "note": "Provide --password to read metadata of an encrypted file."}
    meta = {}
    if reader.metadata:
        for key, value in reader.metadata.items():
            meta[str(key).lstrip("/")] = str(value)
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        box = page.mediabox
        pages.append({
            "page": idx,
            "width": float(box.width),
            "height": float(box.height),
            "rotation": int(page.get("/Rotate", 0)),
        })
    # scanned-page heuristic: no extractable text but page has images
    likely_scanned = []
    try:
        pdfplumber = _need("pdfplumber", "pdfplumber")
        with pdfplumber.open(path, password=password) as pdf:
            for pageno, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if not text and page.images:
                    likely_scanned.append(pageno)
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - heuristic only
        print(f"Warning: scanned-page check failed: {exc}", file=sys.stderr)
    out = {
        "encrypted": encrypted,
        "page_count": len(reader.pages),
        "metadata": meta,
        "pages": pages,
        "likely_scanned_pages": likely_scanned,
    }
    if likely_scanned:
        out["note"] = ("Image-only pages detected: no text layer to extract. "
                       "Use the ocr-and-documents skill for OCR.")
    return out


FIELD_TYPES = {"/Tx": "text", "/Btn": "button", "/Ch": "choice", "/Sig": "signature"}


def read_fields(path: str, password: str | None) -> dict:
    pypdf = _need("pypdf", "pypdf")
    reader = pypdf.PdfReader(path)
    if reader.is_encrypted:
        if password is None or not reader.decrypt(password):
            print("File is encrypted; pass --password.", file=sys.stderr)
            raise SystemExit(3)
    fields = reader.get_fields() or {}
    out = {}
    for name, field in fields.items():
        ftype = FIELD_TYPES.get(str(field.get("/FT")), str(field.get("/FT")))
        value = field.get("/V")
        states = field.get("/_States_")
        entry = {"type": ftype, "value": None if value is None else str(value)}
        if states:
            entry["options"] = [str(s) for s in states]
        out[name] = entry
    return {"field_count": len(out), "fields": out}


def main() -> int:
    _reconfigure_stdio()
    parser = argparse.ArgumentParser(description="Extract text, tables, metadata, or form fields from a PDF.")
    parser.add_argument("pdf", help="Input PDF path")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text", action="store_true", help="Per-page text as JSON")
    mode.add_argument("--tables", action="store_true", help="Tables as JSON (optionally CSV via --csv-dir)")
    mode.add_argument("--meta", action="store_true", help="Metadata, page sizes, encrypted/scanned flags")
    mode.add_argument("--fields", action="store_true", help="AcroForm fields with types and values")
    parser.add_argument("--csv-dir", help="Also write each table as a CSV file into this directory")
    parser.add_argument("--password", help="Password for encrypted PDFs")
    args = parser.parse_args()

    if args.text:
        result = read_text(args.pdf, args.password)
    elif args.tables:
        result = read_tables(args.pdf, args.password, args.csv_dir)
    elif args.meta:
        result = read_meta(args.pdf, args.password)
    else:
        result = read_fields(args.pdf, args.password)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
