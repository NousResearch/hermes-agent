#!/usr/bin/env python3
"""Extract text from documents using pymupdf. Lightweight (~25MB), no models.

Usage:
    python extract_pymupdf.py document.pdf
    python extract_pymupdf.py document.pdf --markdown
    python extract_pymupdf.py document.pdf --pages 0-4
    python extract_pymupdf.py document.pdf --images output_dir/
    python extract_pymupdf.py document.pdf --tables
    python extract_pymupdf.py document.pdf --metadata
"""
import sys
import json

def extract_text(path, pages=None):
    import pymupdf
    doc = pymupdf.open(path)
    page_range = range(len(doc)) if pages is None else pages
    for i in page_range:
        if i < len(doc):
            print(f"\n--- Page {i+1}/{len(doc)} ---\n")
            print(doc[i].get_text())

def extract_markdown(path, pages=None):
    import pymupdf4llm
    md = pymupdf4llm.to_markdown(path, pages=pages)
    print(md)

def extract_tables(path):
    import pymupdf
    doc = pymupdf.open(path)
    for i, page in enumerate(doc):
        tables = page.find_tables()
        for j, table in enumerate(tables.tables):
            print(f"\n--- Page {i+1}, Table {j+1} ---\n")
            df = table.to_pandas()
            print(df.to_markdown(index=False))

def extract_images(path, output_dir):
    import pymupdf
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(path)
    count = 0
    for i, page in enumerate(doc):
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = pymupdf.Pixmap(doc, xref)
            if pix.n >= 5:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            out_path = f"{output_dir}/page{i+1}_img{img_idx+1}.png"
            pix.save(out_path)
            count += 1
    print(f"Extracted {count} images to {output_dir}/")

def show_metadata(path):
    import pymupdf
    doc = pymupdf.open(path)
    print(json.dumps({
        "pages": len(doc),
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
        "subject": doc.metadata.get("subject", ""),
        "creator": doc.metadata.get("creator", ""),
        "producer": doc.metadata.get("producer", ""),
        "format": doc.metadata.get("format", ""),
    }, indent=2))

def parse_pages(args):
    if "--pages" not in args:
        return None

    idx = args.index("--pages")
    if idx + 1 >= len(args):
        raise ValueError("--pages requires a value")

    value = args[idx + 1]
    try:
        if "-" not in value:
            return [int(value)]

        parts = value.split("-")
        if len(parts) != 2 or not all(parts):
            raise ValueError
        start, end = map(int, parts)
        return list(range(start, end + 1))
    except ValueError:
        raise ValueError(
            f"invalid --pages value {value!r}; expected N or START-END"
        ) from None

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    path = args[0]
    try:
        pages = parse_pages(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if "--metadata" in args:
        show_metadata(path)
    elif "--tables" in args:
        extract_tables(path)
    elif "--images" in args:
        idx = args.index("--images")
        output_dir = args[idx + 1] if idx + 1 < len(args) else "./images"
        extract_images(path, output_dir)
    elif "--markdown" in args:
        extract_markdown(path, pages=pages)
    else:
        extract_text(path, pages=pages)
