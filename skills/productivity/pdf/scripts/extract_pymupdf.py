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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract text/tables/images/metadata from documents using pymupdf (lightweight, no models)."
    )
    parser.add_argument("path", help="Document to read")
    parser.add_argument("--pages", help="Page selection: N or START-END (0-indexed)")
    parser.add_argument("--markdown", action="store_true", help="Markdown output via pymupdf4llm")
    parser.add_argument("--tables", action="store_true", help="Extract tables as markdown")
    parser.add_argument("--images", nargs="?", const="./images", metavar="OUTPUT_DIR",
                        help="Extract embedded images to OUTPUT_DIR (default ./images)")
    parser.add_argument("--metadata", action="store_true", help="Show document metadata as JSON")
    args = parser.parse_args()

    pages = None
    if args.pages:
        if "-" in args.pages:
            start, end = args.pages.split("-")
            pages = list(range(int(start), int(end) + 1))
        else:
            pages = [int(args.pages)]

    if args.metadata:
        show_metadata(args.path)
    elif args.tables:
        extract_tables(args.path)
    elif args.images is not None:
        extract_images(args.path, args.images)
    elif args.markdown:
        extract_markdown(args.path, pages=pages)
    else:
        extract_text(args.path, pages=pages)
