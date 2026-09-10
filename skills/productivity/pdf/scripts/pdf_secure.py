#!/usr/bin/env python3
"""Encrypt or decrypt a PDF with passwords (AES-256 via pypdf).

Note: permission flags set at encryption time are advisory — viewers may honor
them, but any PDF library can strip them. Only the user password gates content.
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Encrypt/decrypt PDFs (pypdf, AES-256).")
    parser.add_argument("pdf", help="Input PDF path")
    parser.add_argument("-o", "--output", required=True, help="Output PDF path")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--encrypt", action="store_true", help="Encrypt the PDF")
    mode.add_argument("--decrypt", action="store_true", help="Remove encryption (password required)")
    parser.add_argument("--user-password", help="User (open) password for --encrypt")
    parser.add_argument("--owner-password", help="Owner password for --encrypt (defaults to user password)")
    parser.add_argument("--password", help="Known password for --decrypt")
    args = parser.parse_args()

    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        print("Missing dependency: install with 'python3 -m pip install pypdf'", file=sys.stderr)
        return 2

    def _reapply_document_state(writer: PdfWriter, source: PdfReader) -> None:
        # append() copies pages/outlines but neither the Info dictionary, the
        # catalog's XMP metadata stream, nor embedded attachments — re-apply
        # all three so encrypt/decrypt round trips stop losing document
        # state (#94870).
        try:
            if source.metadata:
                writer.add_metadata(dict(source.metadata))
        except Exception:
            pass
        try:
            # The xmp_metadata parsing property raises on AES-encrypted
            # sources even after decrypt(), so copy the /Metadata stream
            # itself — byte-level, no XML round-trip — which works for
            # encrypted and plain sources alike (#94870).
            from pypdf.generic import NameObject

            _md = source.root_object.get("/Metadata")
            if _md is not None:
                writer._root_object[NameObject("/Metadata")] = _md.clone(writer)
        except Exception:
            pass
        try:
            for name, contents in (source.attachments or {}).items():
                # attachments maps a name to a *list* of payloads — a PDF
                # may embed several files under one name; re-attach all of
                # them, not just the first (#94870).
                for data in contents if isinstance(contents, list) else [contents]:
                    writer.add_attachment(name, bytes(data))
        except Exception:
            pass

    reader = PdfReader(args.pdf)
    if args.encrypt:
        if not args.user_password:
            print("Error: --encrypt requires --user-password", file=sys.stderr)
            return 2
        if reader.is_encrypted:
            print("Error: input already encrypted; decrypt first", file=sys.stderr)
            return 3
        writer = PdfWriter()
        writer.append(reader)
        _reapply_document_state(writer, reader)
        writer.encrypt(
            user_password=args.user_password,
            owner_password=args.owner_password or args.user_password,
            algorithm="AES-256",
        )
        action = "encrypted"
    else:
        if not reader.is_encrypted:
            print("Error: input is not encrypted", file=sys.stderr)
            return 3
        if args.password is None or not reader.decrypt(args.password):
            print("Error: wrong or missing --password", file=sys.stderr)
            return 4
        writer = PdfWriter()
        writer.append(reader)
        _reapply_document_state(writer, reader)
        action = "decrypted"

    with open(args.output, "wb") as fh:
        writer.write(fh)
    print(json.dumps({"output": args.output, "action": action, "page_count": len(reader.pages)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
