#!/usr/bin/env python3
"""
pdf_client.py - PDF CLI tool for the Hermes Agent project.
Requires: pypdf >= 3.0 (pip install pypdf)
"""

import argparse
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _require_pypdf():
    try:
        from pypdf import PdfReader, PdfWriter  # noqa: F401
    except ImportError:
        print(
            "Error: 'pypdf' is not installed.\n"
            "Install it with:  pip install pypdf",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_json(data):
    print(json.dumps(data, indent=2, default=str))


def parse_pages(pages_str, total_pages):
    """
    Parse a pages string like '1,3,5', '1-5', or 'all'.
    Returns a sorted list of 0-based page indices.
    """
    if pages_str is None or pages_str.lower() == "all":
        return list(range(total_pages))

    indices = set()
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            bounds = part.split("-", 1)
            try:
                start = int(bounds[0].strip())
                end = int(bounds[1].strip())
            except ValueError:
                print(f"Error: Invalid page range '{part}'.", file=sys.stderr)
                sys.exit(1)
            for p in range(start, end + 1):
                if 1 <= p <= total_pages:
                    indices.add(p - 1)
        else:
            try:
                p = int(part)
            except ValueError:
                print(f"Error: Invalid page number '{part}'.", file=sys.stderr)
                sys.exit(1)
            if 1 <= p <= total_pages:
                indices.add(p - 1)

    return sorted(indices)


def open_reader(path):
    """Open a PdfReader, attempting empty password for encrypted PDFs."""
    from pypdf import PdfReader

    if not os.path.exists(path):
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    reader = PdfReader(path)
    if reader.is_encrypted:
        try:
            result = reader.decrypt("")
            if result == 0:
                print(
                    f"Error: '{path}' is encrypted and requires a password.",
                    file=sys.stderr,
                )
                sys.exit(1)
        except Exception as exc:
            print(f"Error decrypting '{path}': {exc}", file=sys.stderr)
            sys.exit(1)
    return reader


def extract_page_text(page):
    """Extract text from a page; return empty string for scanned pages."""
    try:
        text = page.extract_text()
        return text if text else ""
    except Exception:
        return ""


def pts_to_mm(pts):
    """Convert PDF points to millimetres."""
    return round(pts * 0.352778, 2)


def page_has_images(page):
    """Detect images via /XObject resources."""
    try:
        resources = page.get("/Resources")
        if resources is None:
            return False
        xobject = resources.get("/XObject")
        if xobject is None:
            return False
        for key in xobject:
            obj = xobject[key]
            if obj.get("/Subtype") == "/Image":
                return True
        return False
    except Exception:
        return False


def build_pages_label(pages_str, indices):
    """Build a short label like '1-5' or '1,3,7' for output filename."""
    if pages_str and pages_str.lower() != "all":
        # Sanitise for filename
        return re.sub(r"[^\d,\-]", "", pages_str)
    # Build from indices (1-based)
    nums = [str(i + 1) for i in indices]
    return ",".join(nums)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_extract(args):
    _require_pypdf()
    reader = open_reader(args.file)
    total = len(reader.pages)
    indices = parse_pages(args.pages, total)

    if args.output == "json":
        results = []
        for idx in indices:
            page = reader.pages[idx]
            text = extract_page_text(page)
            results.append(
                {
                    "page_num": idx + 1,
                    "text": text,
                    "char_count": len(text),
                }
            )
        print_json(results)
    else:
        # txt: plain text to stdout
        parts = []
        for idx in indices:
            page = reader.pages[idx]
            text = extract_page_text(page)
            parts.append(text)
        print("\n".join(parts))


def cmd_metadata(args):
    _require_pypdf()
    reader = open_reader(args.file)
    meta = reader.metadata or {}

    def _get(key):
        val = meta.get(key)
        return str(val) if val is not None else None

    file_size_kb = round(os.path.getsize(args.file) / 1024, 2)

    result = {
        "title": _get("/Title"),
        "author": _get("/Author"),
        "subject": _get("/Subject"),
        "creator": _get("/Creator"),
        "producer": _get("/Producer"),
        "creation_date": _get("/CreationDate"),
        "modification_date": _get("/ModDate"),
        "pages": len(reader.pages),
        "encrypted": reader.is_encrypted,
        "file_size_kb": file_size_kb,
    }
    print_json(result)


def cmd_info(args):
    _require_pypdf()
    reader = open_reader(args.file)
    meta = reader.metadata or {}

    def _get(key):
        val = meta.get(key)
        return str(val) if val is not None else None

    file_size_kb = round(os.path.getsize(args.file) / 1024, 2)

    metadata = {
        "title": _get("/Title"),
        "author": _get("/Author"),
        "subject": _get("/Subject"),
        "creator": _get("/Creator"),
        "producer": _get("/Producer"),
        "creation_date": _get("/CreationDate"),
        "modification_date": _get("/ModDate"),
        "pages": len(reader.pages),
        "encrypted": reader.is_encrypted,
        "file_size_kb": file_size_kb,
    }

    page_info = []
    total_words = 0
    doc_has_images = False

    for idx, page in enumerate(reader.pages):
        mediabox = page.mediabox
        width_mm = pts_to_mm(float(mediabox.width))
        height_mm = pts_to_mm(float(mediabox.height))

        text = extract_page_text(page)
        word_count = len(text.split()) if text.strip() else 0
        total_words += word_count

        has_img = page_has_images(page)
        if has_img:
            doc_has_images = True

        page_info.append(
            {
                "page_num": idx + 1,
                "width_mm": width_mm,
                "height_mm": height_mm,
                "word_count": word_count,
                "has_images": has_img,
            }
        )

    result = {
        "metadata": metadata,
        "page_details": page_info,
        "estimated_word_count": total_words,
        "has_images": doc_has_images,
    }
    print_json(result)


def cmd_search(args):
    _require_pypdf()
    reader = open_reader(args.file)

    try:
        pattern = re.compile(args.query, re.IGNORECASE)
    except re.error as exc:
        print(f"Error: Invalid regex pattern: {exc}", file=sys.stderr)
        sys.exit(1)

    matches = []
    for idx, page in enumerate(reader.pages):
        text = extract_page_text(page)
        if not text:
            continue
        lines = text.splitlines()
        for line_num, line in enumerate(lines, start=1):
            if pattern.search(line):
                matches.append(
                    {
                        "page": idx + 1,
                        "line_number": line_num,
                        "context": line.strip(),
                    }
                )

    print_json(matches)


def cmd_split(args):
    _require_pypdf()
    from pypdf import PdfWriter

    reader = open_reader(args.file)
    total = len(reader.pages)
    indices = parse_pages(args.pages, total)

    if not indices:
        print("Error: No valid pages selected.", file=sys.stderr)
        sys.exit(1)

    label = build_pages_label(args.pages, indices)
    base = os.path.splitext(os.path.basename(args.file))[0]
    out_filename = f"{base}_pages_{label}.pdf"

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, out_filename)
    else:
        out_dir = os.path.dirname(os.path.abspath(args.file))
        out_path = os.path.join(out_dir, out_filename)

    writer = PdfWriter()
    for idx in indices:
        writer.add_page(reader.pages[idx])

    with open(out_path, "wb") as fh:
        writer.write(fh)

    result = {
        "output_file": out_path,
        "pages_extracted": [i + 1 for i in indices],
        "page_count": len(indices),
    }
    print_json(result)


def cmd_merge(args):
    _require_pypdf()
    from pypdf import PdfWriter

    writer = PdfWriter()
    files_merged = []

    for path in args.files:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}", file=sys.stderr)
            sys.exit(1)
        reader = open_reader(path)
        for page in reader.pages:
            writer.add_page(page)
        files_merged.append({"file": path, "pages": len(reader.pages)})

    out_path = args.output if args.output else "merged.pdf"

    with open(out_path, "wb") as fh:
        writer.write(fh)

    total_pages = sum(f["pages"] for f in files_merged)
    result = {
        "output_file": out_path,
        "files_merged": files_merged,
        "total_pages": total_pages,
    }
    print_json(result)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="pdf_client",
        description="PDF CLI tool — requires pypdf (pip install pypdf)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # extract
    p_extract = subparsers.add_parser(
        "extract", help="Extract text from a PDF file."
    )
    p_extract.add_argument("file", help="Path to the PDF file.")
    p_extract.add_argument(
        "--pages",
        default=None,
        help="Pages to extract: '1,3,5', '1-5', or 'all' (default: all).",
    )
    p_extract.add_argument(
        "--output",
        choices=["txt", "json"],
        default="txt",
        help="Output format: txt (default) or json.",
    )

    # metadata
    p_meta = subparsers.add_parser(
        "metadata", help="Extract PDF metadata."
    )
    p_meta.add_argument("file", help="Path to the PDF file.")

    # info
    p_info = subparsers.add_parser(
        "info", help="Full document info: metadata + page dimensions + word count."
    )
    p_info.add_argument("file", help="Path to the PDF file.")

    # search
    p_search = subparsers.add_parser(
        "search", help="Search for a text pattern in a PDF."
    )
    p_search.add_argument("file", help="Path to the PDF file.")
    p_search.add_argument("query", help="Search query (supports regex).")

    # split
    p_split = subparsers.add_parser(
        "split", help="Extract specific pages to a new PDF."
    )
    p_split.add_argument("file", help="Path to the PDF file.")
    p_split.add_argument(
        "--pages",
        default=None,
        help="Pages to extract: '1-5' or '1,3,7'. Required.",
    )
    p_split.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Directory to save the output PDF (default: same as input).",
    )

    # merge
    p_merge = subparsers.add_parser(
        "merge", help="Merge multiple PDFs into one."
    )
    p_merge.add_argument(
        "files",
        nargs="+",
        help="Two or more PDF files to merge.",
    )
    p_merge.add_argument(
        "--output",
        default="merged.pdf",
        help="Output filename (default: merged.pdf).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "extract": cmd_extract,
        "metadata": cmd_metadata,
        "info": cmd_info,
        "search": cmd_search,
        "split": cmd_split,
        "merge": cmd_merge,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
