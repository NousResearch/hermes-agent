#!/usr/bin/env python3
"""Build a unified PDF corpus manifest for the architect library.

Sources included:
1) All PDFs under /home/yeqiuqiu/clawd-architect/memory
2) All PDF references found in MEMORY.md (including wildcard refs)
3) Analysis markdown references found in MEMORY.md (exported separately)

Outputs:
- JSON manifest consumable by scripts/pdf_skill_pipeline.py ingest --manifest-json
- JSON list of analysis refs from MEMORY.md
- Markdown summary
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEMORY_MD = REPO_ROOT / "MEMORY.md"
DEFAULT_LIBRARY_ROOT = REPO_ROOT / "memory"
DEFAULT_OUT_DIR = REPO_ROOT / "memory" / "pdf_skill_pipeline" / "input"


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def collect_repo_pdfs(library_root: Path) -> Set[Path]:
    return {p.resolve() for p in library_root.rglob("*.pdf") if p.is_file()}


def parse_memory_md_paths(memory_md: Path) -> Dict[str, Set[Path]]:
    text = read_text(memory_md)

    # Backticked refs first (common in MEMORY.md)
    backticked = re.findall(r"`([^`]+)`", text)

    pdf_paths: Set[Path] = set()
    analysis_paths: Set[Path] = set()
    text_paths: Set[Path] = set()

    candidates = list(backticked)
    # Also catch plain absolute pdf links not wrapped in backticks
    candidates.extend(re.findall(r"(/home/yeqiuqiu/[^\s`]+\.pdf)", text))

    for c in candidates:
        c = c.strip()
        if not c:
            continue

        # Resolve path relative to repo when needed.
        base = Path(c)
        if not base.is_absolute():
            base = (REPO_ROOT / base).resolve()

        # wildcard expansion support (e.g., file_28---*.pdf)
        if "*" in str(base) or "?" in str(base) or "[" in str(base):
            matches = [p.resolve() for p in base.parent.glob(base.name)] if base.parent.exists() else []
            for m in matches:
                if m.suffix.lower() == ".pdf":
                    pdf_paths.add(m)
                if "analysis" in m.name.lower() and m.suffix.lower() == ".md":
                    analysis_paths.add(m)
                if m.suffix.lower() == ".txt":
                    text_paths.add(m)
            continue

        if base.suffix.lower() == ".pdf" and base.exists():
            pdf_paths.add(base)
        elif base.suffix.lower() == ".md" and "analysis" in base.name.lower() and base.exists():
            analysis_paths.add(base)
        elif base.suffix.lower() == ".txt" and base.exists():
            text_paths.add(base)

    # Explicit analysis refs pattern to avoid misses
    for m in re.findall(r"`([^`]*analysis[^`]*\.md)`", text, flags=re.I):
        p = Path(m)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if p.exists():
            analysis_paths.add(p)

    return {
        "pdf_paths": pdf_paths,
        "analysis_paths": analysis_paths,
        "text_paths": text_paths,
    }


def parse_pdf_to_text_mappings(memory_md: Path) -> Dict[Path, Path]:
    """Heuristic parser for Source PDF / Extracted text adjacency in MEMORY.md."""
    text = read_text(memory_md)
    lines = text.splitlines()

    source_re = re.compile(r"Source PDF:\s*`([^`]+\.pdf)`", re.I)
    text_re = re.compile(r"Extracted text:\s*`([^`]+\.txt)`", re.I)

    mappings: Dict[Path, Path] = {}
    last_pdf: Path | None = None
    last_pdf_line = -999

    for i, line in enumerate(lines):
        sm = source_re.search(line)
        if sm:
            p = Path(sm.group(1))
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            if p.exists():
                last_pdf = p
                last_pdf_line = i
            continue

        tm = text_re.search(line)
        if tm and last_pdf is not None and (i - last_pdf_line) <= 8:
            t = Path(tm.group(1))
            if not t.is_absolute():
                t = (REPO_ROOT / t).resolve()
            if t.exists():
                mappings[last_pdf] = t

    return mappings


def load_title_text_hints_from_json_manifests(library_root: Path) -> Dict[Path, Dict[str, str]]:
    """Load known title/text hints from JSON manifests already present in memory/inbound_pdfs."""
    hints: Dict[Path, Dict[str, str]] = {}
    for mf in library_root.rglob("*MANIFEST*.json"):
        try:
            data = json.loads(mf.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for row in data:
            if not isinstance(row, dict):
                continue
            pdf = row.get("pdf")
            if not pdf:
                continue
            p = Path(str(pdf)).expanduser()
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            else:
                p = p.resolve()
            title = row.get("title_hint")
            text_path = row.get("text_path")
            rec = hints.setdefault(p, {})
            if isinstance(title, str) and title.strip():
                rec["title_hint"] = title.strip()
            if isinstance(text_path, str) and text_path.strip():
                t = Path(text_path).expanduser()
                if not t.is_absolute():
                    t = (REPO_ROOT / t).resolve()
                else:
                    t = t.resolve()
                if t.exists():
                    rec["text_path"] = str(t)
    return hints


def derive_title_from_text_path(text_path: Path) -> str:
    stem = text_path.stem
    # Strip common date suffixes for readability.
    stem = re.sub(r"_\d{4}-\d{2}-\d{2}$", "", stem)
    return stem.replace("_", " ").strip() or text_path.stem


def build_manifest_rows(
    repo_pdfs: Set[Path],
    memory_pdf_refs: Set[Path],
    pdf_text_map: Dict[Path, Path],
    manifest_hints: Dict[Path, Dict[str, str]],
) -> List[dict]:
    all_pdfs = sorted(repo_pdfs.union(memory_pdf_refs))
    rows: List[dict] = []

    for p in all_pdfs:
        row = {
            "pdf": str(p),
            "title_hint": p.stem,
            "source": "repo_memory_scan" if p in repo_pdfs else "memory_md_reference",
        }

        hinted = manifest_hints.get(p, {})
        if hinted.get("title_hint"):
            row["title_hint"] = hinted["title_hint"]

        txt = None
        if hinted.get("text_path"):
            t = Path(hinted["text_path"])
            if t.exists():
                txt = t

        if txt is None:
            mapped = pdf_text_map.get(p)
            if mapped and mapped.exists():
                txt = mapped

        if txt is None:
            sibling = p.with_suffix(".txt")
            if sibling.exists():
                txt = sibling

        if txt is not None:
            row["text_path"] = str(txt)
            if row.get("title_hint") == p.stem:
                row["title_hint"] = derive_title_from_text_path(txt)

        rows.append(row)

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Build architect unified PDF corpus manifest")
    ap.add_argument("--memory-md", default=str(DEFAULT_MEMORY_MD))
    ap.add_argument("--library-root", default=str(DEFAULT_LIBRARY_ROOT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--tag", default=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"))
    args = ap.parse_args()

    memory_md = Path(args.memory_md).expanduser().resolve()
    library_root = Path(args.library_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    parsed = parse_memory_md_paths(memory_md)
    repo_pdfs = collect_repo_pdfs(library_root)
    pdf_text_map = parse_pdf_to_text_mappings(memory_md)
    manifest_hints = load_title_text_hints_from_json_manifests(library_root)

    rows = build_manifest_rows(repo_pdfs, parsed["pdf_paths"], pdf_text_map, manifest_hints)

    manifest_path = out_dir / f"architect_unified_pdf_manifest_{args.tag}.json"
    analysis_path = out_dir / f"architect_memory_analysis_refs_{args.tag}.json"
    summary_path = out_dir / f"architect_unified_pdf_manifest_{args.tag}.md"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    analysis_rows = []
    for p in sorted(parsed["analysis_paths"]):
        analysis_rows.append(
            {
                "path": str(p),
                "title_hint": p.stem,
                "exists": p.exists(),
                "source": "memory_md",
            }
        )

    with analysis_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": now_utc(),
                "memory_md": str(memory_md),
                "analysis_count": len(analysis_rows),
                "analysis_refs": analysis_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    lines = [
        f"# Unified PDF Corpus Manifest ({args.tag})",
        "",
        f"Generated: {now_utc()}",
        f"Memory file: `{memory_md}`",
        f"Repo library root: `{library_root}`",
        "",
        "## Counts",
        f"- Repo PDFs: {len(repo_pdfs)}",
        f"- MEMORY.md PDF refs (resolved): {len(parsed['pdf_paths'])}",
        f"- Unified manifest rows: {len(rows)}",
        f"- Manifest hint records loaded: {len(manifest_hints)}",
        f"- MEMORY.md analysis refs: {len(analysis_rows)}",
        "",
        "## Outputs",
        f"- Manifest JSON: `{manifest_path}`",
        f"- Analysis refs JSON: `{analysis_path}`",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(str(manifest_path))
    print(str(analysis_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
