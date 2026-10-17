#!/usr/bin/env python3
"""Enumerate local note sources and report which are new or changed.

Walks the Obsidian vault (and any extra plaintext source dirs) for .md/.txt
notes, hashes each, diffs against the notes-extract state cache, and emits JSON
the model reads to decide what to extract. Output dirs (People/, Ideas-Projects/)
and dotfiles are skipped so generated files are never re-ingested. Read-only:
this script never writes state — upsert_entry.py owns that.

Usage (invoke through the `terminal` tool):
  python notes_scan.py --vault "/path/to/vault"            # new + changed notes
  python notes_scan.py --vault "/path" --sources "/a,/b"   # extra source dirs
  python notes_scan.py --vault "/path" --all               # include unchanged
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _state import load_state, sha256_text, source_id as make_source_id  # noqa: E402

SKIP_DIRS = {"People", "Ideas-Projects"}
NOTE_SUFFIXES = {".md", ".txt"}


def iter_notes(roots: list[Path]):
    """Yield note files under each root, skipping output + hidden dirs."""
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in NOTE_SUFFIXES or not path.is_file():
                continue
            rel_parts = path.relative_to(root).parts
            if any(p.startswith(".") or p in SKIP_DIRS for p in rel_parts):
                continue
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            yield path


def scan(vault: Path, extra_sources: list[Path], state: dict, include_unchanged: bool) -> list[dict]:
    known = {sid: rec.get("sha") for sid, rec in state.get("sources", {}).items()}
    out = []
    for path in iter_notes([vault, *extra_sources]):
        text = path.read_text(encoding="utf-8", errors="replace")
        sha = sha256_text(text)
        sid = make_source_id(vault, path)
        if sid not in known:
            status = "new"
        elif known[sid] != sha:
            status = "changed"
        else:
            status = "unchanged"
        if status == "unchanged" and not include_unchanged:
            continue
        out.append({
            "path": str(path),
            "link": path.stem,           # Obsidian wikilink target (basename, no ext)
            "source_id": sid,
            "sha": sha,
            "status": status,
            "text": text,
        })
    return out


def resolve_sources(args) -> tuple[Path, list[Path]]:
    vault_arg = args.vault or os.environ.get("OBSIDIAN_VAULT_PATH", "").strip()
    if not vault_arg:
        vault_arg = str(Path.home() / "Documents" / "Obsidian Vault")
    vault = Path(vault_arg)
    raw = args.sources or os.environ.get("NOTES_EXTRACT_SOURCES", "")
    extra = [Path(s.strip()) for s in raw.split(",") if s.strip()]
    return vault, extra


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Report new/changed local notes as JSON.")
    p.add_argument("--vault", default="")
    p.add_argument("--sources", default="", help="Comma-separated extra source dirs")
    p.add_argument("--all", action="store_true", help="Include unchanged notes")
    args = p.parse_args(argv)

    vault, extra = resolve_sources(args)
    state = load_state(vault)
    results = scan(vault, extra, state, include_unchanged=args.all)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
