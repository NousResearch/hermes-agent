#!/usr/bin/env python3
"""Structural checks for golden prose transcripts (stdlib only).

Usage:
  python scripts/check_transcripts.py
  python scripts/check_transcripts.py --dir evals/transcripts
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import paths as ng_paths  # noqa: E402


def _default_transcripts_dir() -> Path:
    try:
        return ng_paths.evals_dir() / "transcripts"
    except FileNotFoundError:
        return SKILL_ROOT / "evals" / "transcripts"


DEFAULT_DIR = _default_transcripts_dir()

REQUIRED_SECTIONS = ("## OPEN", "## ALIGN", "## ASCEND", "## CLEAR", "## SEAL")
REQUIRED_FRONTMATTER = ("id:", "scenario:", "profiles:", "research_mode:", "expected_promotion_max:")
CLAIM_LABEL = re.compile(
    r"\b(OBSERVED|INFERRED|SPECULATIVE|NOT_COMPUTABLE)\b"
)
SKIP_NAMES = {"README.md", "rubric.md"}


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    meta: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line and not line[0].isspace():
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, text[end + 5 :]


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    if not meta:
        errors.append(f"{path.name}: missing YAML frontmatter")
        return errors

    for key in ("id", "scenario", "profiles", "research_mode", "expected_promotion_max"):
        if key not in meta:
            errors.append(f"{path.name}: frontmatter missing {key}")

    for sec in REQUIRED_SECTIONS:
        if sec not in body and sec not in text:
            errors.append(f"{path.name}: missing section {sec}")

    if not CLAIM_LABEL.search(text):
        errors.append(f"{path.name}: no claim labels OBSERVED|INFERRED|SPECULATIVE|NOT_COMPUTABLE")

    mode = (meta.get("research_mode") or "").lower()
    if mode == "offline":
        # Fail only if a claim line invents from prior as OBSERVED (not doctrine warnings)
        for line in text.splitlines():
            low = line.lower()
            if "forbidden" in low or "cannot" in low or "must not" in low or "gate" in low:
                continue
            if "model prior" in low and "OBSERVED" in line and "— `OBSERVED`" in line.replace("—", "—"):
                errors.append(f"{path.name}: offline claim invents OBSERVED from model prior: {line.strip()[:80]}")
            if re.search(r"from model prior.*`OBSERVED`|`OBSERVED`.*from model prior", line, re.I):
                if "not" not in low and "forbidden" not in low:
                    errors.append(f"{path.name}: offline claim invents OBSERVED from model prior")

    # Authority
    if "advisory_only" not in text and "advisory only" not in text.lower():
        errors.append(f"{path.name}: SEAL/body should state advisory_only")

    # Private scenarios should show DataRequest when sensitivity private is mentioned
    if re.search(r"sensitivity:\s*private", text) and "DataRequest" not in text and "data_request" not in text.lower():
        # allow yaml blocks with field: without header word
        if "blocks_promotion" not in text:
            errors.append(f"{path.name}: private sensitivity without DataRequest-like block")

    if "wayfinder" in text.lower() and "product_intent_changes_require_neon_genie_review" not in text:
        if "wayfinder_handoff" in (meta.get("profiles") or ""):
            errors.append(
                f"{path.name}: wayfinder handoff should set product_intent_changes_require_neon_genie_review"
            )

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check golden prose transcripts")
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DIR,
        help="Transcripts directory",
    )
    args = parser.parse_args(argv)

    tdir: Path = args.dir if args.dir.is_absolute() else SKILL_ROOT / args.dir
    if not tdir.is_dir():
        print(f"FAIL: transcripts dir missing: {tdir}", file=sys.stderr)
        return 1

    files = sorted(p for p in tdir.glob("*.md") if p.name not in SKIP_NAMES)
    if len(files) < 3:
        print(f"FAIL: expected at least 3 transcripts, found {len(files)}", file=sys.stderr)
        return 1

    errors: list[str] = []
    for path in files:
        errors.extend(check_file(path))

    if errors:
        print("FAIL: transcript checks", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(f"PASS: transcript checks ({len(files)} files)")
    for path in files:
        print(f"  - {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
