#!/usr/bin/env python3
"""Inspect a .goodnotes file: verify it is a ZIP, list members, identify attachments.

Usage:
    python3 inspect_goodnotes.py <file.goodnotes> [member]

With a member name, prints that member's size and first bytes (hex) instead of the full listing.
"""
import sys
import zipfile

MAGIC = [
    (b"\x89PNG\r\n\x1a\n", "PNG image"),
    (b"\xff\xd8\xff", "JPEG image"),
    (b"%PDF", "PDF document"),
    (b"ID3", "MP3 audio (ID3 tag)"),
    (b"\x00\x00\x00\x18ftyp", "MP4/M4A audio/video"),
    (b"OggS", "Ogg audio"),
    (b"fLaC", "FLAC audio"),
    (b"PK\x03\x04", "ZIP archive"),
]


def sniff(data: bytes) -> str:
    for magic, name in MAGIC:
        if data.startswith(magic):
            return name
    return "unknown"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: inspect_goodnotes.py <file.goodnotes> [member]", file=sys.stderr)
        return 2
    path = sys.argv[1]
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        print(f"NOT a valid ZIP container: {path}")
        return 1

    names = z.namelist()
    print(f"Valid ZIP container: {path}")
    print(f"Members: {len(names)}")

    if "schema.pb" in names:
        print(f"schema.pb bytes: {z.read('schema.pb').hex()}")
    else:
        print("schema.pb: MISSING (not a GoodNotes notebook?)")

    if len(sys.argv) >= 3:
        member = sys.argv[2]
        if member not in names:
            print(f"member not found: {member}", file=sys.stderr)
            return 1
        data = z.read(member)
        print(f"{member}: {len(data)} bytes")
        print(f"first 32 bytes: {data[:32].hex()}")
        return 0

    print("\nContents:")
    for info in z.infolist():
        kind = ""
        if info.filename.startswith("attachments/"):
            try:
                kind = " -> " + sniff(z.read(info.filename)[:16])
            except Exception:
                kind = " -> (unreadable)"
        print(f"  {info.filename}  ({info.file_size} bytes){kind}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
