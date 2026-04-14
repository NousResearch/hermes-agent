#!/usr/bin/env python3
"""Ingest and extract ALL PDFs from known inbound + workspace locations.

Creates a canonical library under memory/inbound_pdfs/library_<YYYY-MM-DD>/
- Dedupes by sha256
- Copies canonical PDF as <sha256>.pdf
- Extracts text via pdftotext into <sha256>.txt
- Writes:
  - INDEX.md (human)
  - MANIFEST.json (machine)
  - per-doc summary stubs (optional)

Safe to re-run: idempotent on sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

ROOT = Path("/home/yeqiuqiu/clawd-architect")

SOURCES = [
    ROOT / "memory" / "inbound_pdfs",
    Path.home() / ".clawdbot" / "media" / "inbound",
    Path.home() / ".openclaw" / "media" / "inbound",
    ROOT,
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_pdfs() -> Iterable[Path]:
    seen = set()
    for src in SOURCES:
        if not src.exists():
            continue
        if src.name == "inbound":
            it = src.glob("*.pdf")
        elif src.name == "inbound_pdfs":
            it = src.rglob("*.pdf")
        elif src == ROOT:
            it = src.rglob("*.pdf")
        else:
            it = src.rglob("*.pdf")
        for p in it:
            if p.suffix.lower() != ".pdf":
                continue
            # avoid scanning huge node_modules etc by simple path skip
            sp = str(p)
            if "/node_modules/" in sp or "/.venv" in sp or "/state/" in sp:
                continue
            if p in seen:
                continue
            seen.add(p)
            yield p


def run_pdftotext(pdf: Path, out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    # -layout preserves columns somewhat
    cmd = ["pdftotext", "-layout", str(pdf), str(out_txt)]
    subprocess.run(cmd, check=True)


@dataclass
class Doc:
    sha256: str
    pdf_path: str
    txt_path: str
    bytes: int
    mtime: float
    sources: list[str]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max", type=int, default=0, help="max unique pdfs (0=all)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else ROOT / "memory" / "inbound_pdfs" / f"library_{args.date}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Doc] = {}
    src_map: dict[str, list[str]] = {}

    # compute sha and source mapping
    pdfs = list(iter_pdfs())
    for p in pdfs:
        try:
            sh = sha256_file(p)
        except Exception:
            continue
        src_map.setdefault(sh, []).append(str(p))

    shas = sorted(src_map.keys(), key=lambda sh: max(Path(x).stat().st_mtime for x in src_map[sh]), reverse=True)
    if args.max and args.max > 0:
        shas = shas[: args.max]

    for sh in shas:
        # pick most recent as canonical source path
        sources = src_map[sh]
        canonical = max(sources, key=lambda x: Path(x).stat().st_mtime)
        pdf_out = out_dir / f"{sh}.pdf"
        txt_out = out_dir / f"{sh}.txt"

        if not pdf_out.exists():
            shutil.copy2(canonical, pdf_out)

        if not txt_out.exists() or txt_out.stat().st_size == 0:
            try:
                run_pdftotext(pdf_out, txt_out)
            except subprocess.CalledProcessError:
                # leave placeholder
                txt_out.write_text("[pdftotext failed]\n")

        st = pdf_out.stat()
        manifest[sh] = Doc(
            sha256=sh,
            pdf_path=str(pdf_out.relative_to(ROOT)),
            txt_path=str(txt_out.relative_to(ROOT)),
            bytes=st.st_size,
            mtime=st.st_mtime,
            sources=sources,
        )

    # write manifest
    mf_path = out_dir / "MANIFEST.json"
    mf_path.write_text(json.dumps({k: asdict(v) for k, v in manifest.items()}, indent=2, sort_keys=True) + "\n")

    # write index
    idx = out_dir / "INDEX.md"
    lines = []
    lines.append(f"# PDF Library Ingest — {args.date}\n")
    lines.append(f"Unique PDFs: **{len(manifest)}**  ")
    lines.append(f"Generated: `{time.strftime('%Y-%m-%dT%H:%M:%S%z')}`\n")
    lines.append("## Entries\n")
    for sh, doc in sorted(manifest.items(), key=lambda kv: kv[1].mtime, reverse=True):
        lines.append(f"- `{sh}`\n  - pdf: `{doc.pdf_path}`\n  - txt: `{doc.txt_path}`\n  - size: {doc.bytes}\n  - sources: {len(doc.sources)}")
    idx.write_text("\n".join(lines) + "\n")

    print(f"OK: wrote {idx}")
    print(f"OK: wrote {mf_path}")
    print(f"OK: unique={len(manifest)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
