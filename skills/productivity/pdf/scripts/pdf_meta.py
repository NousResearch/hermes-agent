#!/usr/bin/env python3
"""Document metadata and file attachments for PDFs (pypdf).

Modes (one required):
  --set-meta            set metadata keys given via --title/--author/...
  --clear-meta          drop all document info metadata
  --attach FILE         embed a file attachment
  --list-attachments    list embedded attachment names
  --extract-attachments DIR  write all attachments into DIR

Metadata note: values are stored in the classic DocInfo dictionary
(Title/Author/Subject/Keywords). XMP metadata, if present, is not
rewritten and may disagree in sophisticated viewers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Set/clear PDF metadata; manage attachments.")
    parser.add_argument("pdf", help="Input PDF path")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--set-meta", action="store_true", help="Set metadata fields")
    mode.add_argument("--clear-meta", action="store_true", help="Remove all DocInfo metadata")
    mode.add_argument("--attach", metavar="FILE", help="Embed FILE as an attachment")
    mode.add_argument("--list-attachments", action="store_true", help="List attachment names")
    mode.add_argument("--extract-attachments", metavar="DIR", help="Extract attachments into DIR")
    parser.add_argument("-o", "--output", help="Output PDF (required for write modes)")
    parser.add_argument("--title")
    parser.add_argument("--author")
    parser.add_argument("--subject")
    parser.add_argument("--keywords")
    parser.add_argument("--password", help="Password if the input is encrypted")
    args = parser.parse_args()

    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        print("Missing dependency: install with 'python3 -m pip install pypdf'", file=sys.stderr)
        return 2

    reader = PdfReader(args.pdf)
    if reader.is_encrypted:
        if args.password is None or not reader.decrypt(args.password):
            print("Error: input is encrypted; pass --password", file=sys.stderr)
            return 3

    if args.list_attachments:
        names = list(reader.attachments.keys())
        json.dump({"attachment_count": len(names), "attachments": names}, sys.stdout,
                  ensure_ascii=False, indent=2)
        print()
        return 0

    if args.extract_attachments:
        out_dir = Path(args.extract_attachments)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for name, contents in reader.attachments.items():
            data = contents[0] if isinstance(contents, list) else contents
            safe = os.path.basename(name) or "attachment.bin"
            target = out_dir / safe
            with open(target, "wb") as fh:
                fh.write(bytes(data))
            written.append(str(target))
        json.dump({"extracted": written}, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0

    if not args.output:
        print("Error: -o/--output is required for write modes", file=sys.stderr)
        return 4

    writer = PdfWriter()
    writer.append(reader)
    # append() copies pages/outlines but neither the Info dictionary nor
    # embedded attachments — re-apply both so composeable operations stop
    # losing document state (#94870). Mode branches below still override
    # (--set-meta merges over preserved values; --clear-meta wipes them).
    if reader.metadata:
        writer.add_metadata(dict(reader.metadata))
    # append() also drops the catalog's XMP metadata stream (the packet most
    # producer tools write) — handing the parsed object back re-attaches the
    # stream to the new catalog (#94870).
    if reader.xmp_metadata is not None:
        writer.xmp_metadata = reader.xmp_metadata
    for name, contents in (reader.attachments or {}).items():
        # attachments maps a name to a *list* of payloads — a PDF may embed
        # several files under one name; re-attach all of them, not just the
        # first (#94870).
        for data in contents if isinstance(contents, list) else [contents]:
            writer.add_attachment(name, bytes(data))

    if args.set_meta:
        meta = {}
        for key, value in ((f"/{k.capitalize()}", getattr(args, k))
                           for k in ("title", "author", "subject", "keywords")):
            if value is not None:
                meta[key] = value
        if not meta:
            print("Error: --set-meta needs at least one of --title/--author/--subject/--keywords",
                  file=sys.stderr)
            return 4
        writer.add_metadata(meta)
        result = {"output": args.output, "set": {k.lstrip("/"): v for k, v in meta.items()}}
    elif args.clear_meta:
        writer.metadata = None
        result = {"output": args.output, "cleared": True}
    else:  # --attach
        attach_path = Path(args.attach)
        with open(attach_path, "rb") as fh:
            writer.add_attachment(attach_path.name, fh.read())
        result = {"output": args.output, "attached": attach_path.name}

    with open(args.output, "wb") as fh:
        writer.write(fh)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
