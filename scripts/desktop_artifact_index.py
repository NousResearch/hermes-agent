#!/usr/bin/env python3
"""Build and merge Desktop prebuilt artifact index documents (schema_version 1).

Used by the ``desktop-prebuilt-artifacts`` workflow. Stdlib only so CI can
run it without installing Hermes.

Commands:
  fragment   write one matrix-job fragment
  merge      union fragments; last-writer-wins on (commit, platform, arch)
  zip-dir    zip an unpacked electron-builder dir so the unpacked name is
             the zip root (``linux-unpacked/...``, ``win-unpacked/...``)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

SCHEMA_VERSION = 1
DEFAULT_WINDOW = 64


def build_fragment(
    *,
    commit: str,
    platform: str,
    arch: str,
    url: str,
    sha256: str,
    tag: str = "",
    filename: str = "",
    compatibility_window: int = DEFAULT_WINDOW,
) -> dict:
    commit = commit.strip().lower()
    sha256 = sha256.strip().lower()
    return {
        "schema_version": SCHEMA_VERSION,
        "compatibility_window": int(compatibility_window),
        "artifacts": [
            {
                "commit": commit,
                "tag": tag,
                "platform": platform,
                "arch": arch,
                "url": url,
                "sha256": sha256,
                "filename": filename,
            }
        ],
    }


def merge_index_docs(docs: list[dict]) -> dict:
    by_id: dict[tuple[str, str, str], dict] = {}
    window = DEFAULT_WINDOW
    schema = SCHEMA_VERSION
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        try:
            schema = max(schema, int(doc.get("schema_version") or SCHEMA_VERSION))
        except (TypeError, ValueError):
            pass
        try:
            window = max(window, int(doc.get("compatibility_window") or 0))
        except (TypeError, ValueError):
            pass
        rows = doc.get("artifacts") or []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = (
                str(row.get("commit") or "").lower(),
                str(row.get("platform") or ""),
                str(row.get("arch") or ""),
            )
            if not all(key):
                continue
            by_id[key] = row
    return {
        "schema_version": schema,
        "compatibility_window": window,
        "artifacts": list(by_id.values()),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_unpacked_dir(unpacked: Path, dest_zip: Path) -> None:
    """Zip *unpacked* so the archive root is ``unpacked.name/``."""
    unpacked = unpacked.resolve()
    if not unpacked.is_dir():
        raise FileNotFoundError(f"unpacked dir missing: {unpacked}")
    dest_zip = dest_zip.resolve()
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    root_name = unpacked.name
    with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(unpacked.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(unpacked)
            zf.write(path, arcname=str(Path(root_name) / rel))


def _cmd_fragment(args: argparse.Namespace) -> int:
    doc = build_fragment(
        commit=args.commit,
        platform=args.platform,
        arch=args.arch,
        url=args.url,
        sha256=args.sha256,
        tag=args.tag or "",
        filename=args.filename or "",
        compatibility_window=args.window,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    docs = []
    for raw in args.inputs:
        path = Path(raw)
        docs.append(json.loads(path.read_text(encoding="utf-8")))
    merged = merge_index_docs(docs)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return 0


def _cmd_zip_dir(args: argparse.Namespace) -> int:
    zip_unpacked_dir(Path(args.unpacked), Path(args.dest))
    print(sha256_file(Path(args.dest)))
    return 0


def _cmd_sha256(args: argparse.Namespace) -> int:
    print(sha256_file(Path(args.path)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_frag = sub.add_parser("fragment", help="write one artifact-index fragment")
    p_frag.add_argument("--commit", required=True)
    p_frag.add_argument("--platform", required=True)
    p_frag.add_argument("--arch", required=True)
    p_frag.add_argument("--url", required=True)
    p_frag.add_argument("--sha256", required=True)
    p_frag.add_argument("--tag", default="")
    p_frag.add_argument("--filename", default="")
    p_frag.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p_frag.add_argument("--output", required=True)
    p_frag.set_defaults(func=_cmd_fragment)

    p_merge = sub.add_parser("merge", help="merge fragment JSON files")
    p_merge.add_argument("inputs", nargs="+")
    p_merge.add_argument("--output", required=True)
    p_merge.set_defaults(func=_cmd_merge)

    p_zip = sub.add_parser("zip-dir", help="zip an unpacked dir; print sha256")
    p_zip.add_argument("unpacked")
    p_zip.add_argument("dest")
    p_zip.set_defaults(func=_cmd_zip_dir)

    p_hash = sub.add_parser("sha256", help="print sha256 of a file")
    p_hash.add_argument("path")
    p_hash.set_defaults(func=_cmd_sha256)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
