#!/usr/bin/env python3
"""
Split a book text file into chapter files.

Usage:
    python split_book.py book.txt output_dir/
    python split_book.py book.txt output_dir/ --pattern "CHAPTER\s+[0-9]+"
    python split_book.py book.txt output_dir/ --by-pages 50  # split every N pages (if page markers exist)
"""

import argparse
import re
import sys
from pathlib import Path


def split_by_chapter(text, pattern=None):
    """Split text by chapter markers. Returns list of (chapter_title, content) tuples."""
    if pattern is None:
        # Common patterns: CHAPTER 1, Chapter I, CHAPTER ONE
        patterns = [
            r'(?:^|\n)\s*(CHAPTER\s+[0-9IVXLC]+)\s*\n',
            r'(?:^|\n)\s*(Chapter\s+[0-9]+)\s*\n',
            r'(?:^|\n)\s*(PART\s+[0-9IVXLC]+)\s*\n',
            r'(?:^|\n)\s*(BOOK\s+[0-9IVXLC]+)\s*\n',
        ]
        for pat in patterns:
            matches = list(re.finditer(pat, text))
            if len(matches) >= 2:
                pattern = pat
                break
        
        if pattern is None:
            print("No chapter markers found. Falling back to page-based split.", file=sys.stderr)
            return None
    
    parts = re.split(pattern, text)
    if len(parts) < 3:
        return None
    
    chapters = []
    # parts[0] = front matter
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1] if i + 1 < len(parts) else ""
        chapters.append((title, content))
    
    return chapters


def split_by_pages(text, pages_per_chunk):
    """Split by form feed characters (page breaks in pdftotext output)."""
    pages = text.split('\f')
    chunks = []
    current = []
    for i, page in enumerate(pages):
        current.append(page)
        if (i + 1) % pages_per_chunk == 0:
            chunks.append((f"Pages_{i+2-pages_per_chunk}-{i+1}", "\n".join(current)))
            current = []
    if current:
        start = len(pages) - len(current) + 1
        chunks.append((f"Pages_{start}-{len(pages)}", "\n".join(current)))
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Split a book into chapter files")
    parser.add_argument("input", help="Input text file")
    parser.add_argument("output_dir", help="Output directory for chapter files")
    parser.add_argument("--pattern", help="Regex pattern for chapter markers")
    parser.add_argument("--by-pages", type=int, help="Split every N pages (if page breaks exist)")
    parser.add_argument("--prefix", default="ch", help="Filename prefix")
    args = parser.parse_args()
    
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    text = in_path.read_text(encoding="utf-8")
    
    if args.by_pages:
        chapters = split_by_pages(text, args.by_pages)
    else:
        chapters = split_by_chapter(text, args.pattern)
        if chapters is None:
            chapters = split_by_pages(text, 50)
    
    if not chapters:
        print("Could not split book.", file=sys.stderr)
        sys.exit(1)
    
    for i, (title, content) in enumerate(chapters):
        safe_title = re.sub(r'[^\w]', '_', title)[:50]
        filename = f"{args.prefix}_{i+1:03d}_{safe_title}.txt"
        out_path = out_dir / filename
        out_path.write_text(content, encoding="utf-8")
        print(f"Wrote: {filename} ({len(content)} chars)")
    
    print(f"\nSplit into {len(chapters)} files in {out_dir}")


if __name__ == "__main__":
    main()
