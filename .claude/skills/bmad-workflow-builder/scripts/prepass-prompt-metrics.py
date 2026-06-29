#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["tiktoken"]
# ///
"""Deterministic prompt-metrics pre-pass for the Analyze scanners.

Reads SKILL.md, root-level prompt files, and references, and emits one compact
JSON object the LLM scanners read instead of the raw files. Length is reported
as tiktoken token counts via count_tokens (cl100k_base, chars//4 fallback);
there is no line-count gate anywhere in this script.

What it surfaces per file:
  - token count and the counting method (tiktoken or fallback)
  - frontmatter facts (name, description, description length, angle-bracket flag)
  - section inventory (heading level + title)
  - structural signals scanners care about: tables, fenced blocks, defensive
    padding, meta-explanation, back-references, config header, progression cues

Budgets the scanners compare against: SKILL.md ~1500-2500 tokens,
multi-branch reference ~4500, single-purpose reference ~9000.

Usage:
  prepass-prompt-metrics.py <skill-dir> [--output FILE]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Reuse the single length metric rather than reimplementing token counting.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from count_tokens import count_tokens
except Exception:  # pragma: no cover - count_tokens ships alongside this script
    def count_tokens(text: str) -> tuple[int, str]:
        return len(text) // 4, "fallback"


WASTE_PATTERNS = [
    (r"\b[Mm]ake sure (?:to|you)\b", "defensive-padding", 'Defensive: "make sure to/you"'),
    (r"\b[Dd]on'?t forget (?:to|that)\b", "defensive-padding", 'Defensive: "don\'t forget"'),
    (r"\b[Rr]emember (?:to|that)\b", "defensive-padding", 'Defensive: "remember to/that"'),
    (r"\b[Bb]e sure to\b", "defensive-padding", 'Defensive: "be sure to"'),
    (r"\b[Pp]lease ensure\b", "defensive-padding", 'Defensive: "please ensure"'),
    (r"\b[Ii]t is important (?:to|that)\b", "defensive-padding", 'Defensive: "it is important"'),
    (r"\b[Yy]ou are an AI\b", "meta-explanation", 'Meta: "you are an AI"'),
    (r"\b[Aa]s a language model\b", "meta-explanation", 'Meta: "as a language model"'),
    (r"\b[Aa]s an AI assistant\b", "meta-explanation", 'Meta: "as an AI assistant"'),
    (r"\b[Tt]his (?:workflow|skill|process) is designed to\b", "meta-explanation", 'Meta: "this is designed to"'),
    (r"\b[Tt]he purpose of this (?:section|step) is\b", "meta-explanation", 'Meta: "the purpose of this is"'),
]

BACKREF_PATTERNS = [
    (r"\bas described above\b", 'Back-reference: "as described above"'),
    (r"\bas mentioned (?:above|in|earlier)\b", 'Back-reference: "as mentioned above/earlier"'),
    (r"\bsee (?:above|the overview)\b", 'Back-reference: "see above/the overview"'),
    (r"\brefer to (?:the )?(?:above|overview|SKILL)\b", 'Back-reference: "refer to above/overview"'),
]

ALLCAPS_PATTERN = re.compile(r"\b(?:ALWAYS|NEVER|MUST|DO NOT|CRITICAL|REQUIRED)\b")
NUMBERED_PREFIX = re.compile(r"^\d{2}[-_]")


def split_frontmatter(content: str) -> tuple[dict, str]:
    """Return (frontmatter dict, body). Empty dict when there is no frontmatter."""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, content
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}, content
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, "\n".join(lines[end + 1:])


def count_tables(content: str) -> tuple[int, int]:
    count = rows = 0
    in_table = False
    for line in content.split("\n"):
        if re.match(r"^\s*\|", line):
            if not in_table:
                count += 1
                in_table = True
            rows += 1
        else:
            in_table = False
    return count, rows


def count_fenced(content: str) -> int:
    blocks = 0
    in_block = False
    for line in content.split("\n"):
        if line.strip().startswith("```"):
            in_block = not in_block
            if in_block:
                blocks += 1
    return blocks


def grep(content: str, lines: list[str], patterns, ignore_case: bool = False) -> list[dict]:
    flags = re.IGNORECASE if ignore_case else 0
    hits = []
    for entry in patterns:
        pattern, *rest = entry
        if len(rest) == 2:
            category, label = rest
        else:
            category, label = None, rest[0]
        for m in re.finditer(pattern, content, flags):
            ln = content[: m.start()].count("\n") + 1
            hit = {"line": ln, "pattern": label, "context": lines[ln - 1].strip()[:100]}
            if category:
                hit["category"] = category
            hits.append(hit)
    return hits


def scan_file(filepath: Path, rel_path: str) -> dict:
    content = filepath.read_text(encoding="utf-8")
    lines = content.split("\n")
    meta, body = split_frontmatter(content)
    tokens, method = count_tokens(content)

    sections = [
        {"level": len(m.group(1)), "title": m.group(2).strip()}
        for m in (re.match(r"^(#{2,4})\s+(.+)$", ln) for ln in lines)
        if m
    ]

    table_count, table_rows = count_tables(content)
    allcaps = len(ALLCAPS_PATTERN.findall(content))

    data = {
        "file": rel_path,
        "tokens": tokens,
        "token_method": method,
        "sections": sections,
        "table_count": table_count,
        "table_rows": table_rows,
        "fenced_block_count": count_fenced(content),
        "allcaps_directive_count": allcaps,
        "numbered_prefix_filename": bool(NUMBERED_PREFIX.match(filepath.name)),
        "waste_patterns": grep(content, lines, WASTE_PATTERNS),
        "back_references": grep(content, lines, BACKREF_PATTERNS, ignore_case=True),
    }

    if meta:
        desc = meta.get("description", "")
        data["frontmatter"] = {
            "name": meta.get("name", ""),
            "description": desc,
            "description_chars": len(desc),
            "description_has_angle_brackets": "<" in desc or ">" in desc,
            "keys": sorted(meta.keys()),
        }
    return data


def scan(skill_path: Path) -> dict:
    files_data = []

    skill_md = skill_path / "SKILL.md"
    if skill_md.exists():
        d = scan_file(skill_md, "SKILL.md")
        d["is_skill_md"] = True
        files_data.append(d)

    for f in sorted(skill_path.iterdir()):
        if f.is_file() and f.suffix == ".md" and f.name != "SKILL.md":
            d = scan_file(f, f.name)
            d["is_skill_md"] = False
            files_data.append(d)

    references = {}
    ref_dir = skill_path / "references"
    if ref_dir.exists():
        for f in sorted(ref_dir.iterdir()):
            if f.is_file() and f.suffix in (".md", ".json", ".yaml", ".yml"):
                tokens, method = count_tokens(f.read_text(encoding="utf-8"))
                references[f.name] = {
                    "tokens": tokens,
                    "token_method": method,
                    "numbered_prefix_filename": bool(NUMBERED_PREFIX.match(f.name)),
                }

    skill_md_data = next((f for f in files_data if f.get("is_skill_md")), None)

    return {
        "scanner": "prompt-metrics-prepass",
        "script": "prepass-prompt-metrics.py",
        "version": "2.0.0",
        "skill_path": str(skill_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "budgets": {
            "skill_md_tokens": [1500, 2500],
            "multi_branch_reference_tokens": 4500,
            "single_purpose_reference_tokens": 9000,
        },
        "skill_md": {
            "tokens": skill_md_data["tokens"] if skill_md_data else 0,
            "token_method": skill_md_data["token_method"] if skill_md_data else "fallback",
            "section_count": len(skill_md_data["sections"]) if skill_md_data else 0,
            "frontmatter": skill_md_data.get("frontmatter") if skill_md_data else None,
        },
        "aggregate": {
            "total_files_scanned": len(files_data),
            "total_tokens": sum(f["tokens"] for f in files_data),
            "total_waste_patterns": sum(len(f["waste_patterns"]) for f in files_data),
            "total_back_references": sum(len(f["back_references"]) for f in files_data),
            "files_with_numbered_prefix": sum(
                1 for f in files_data if f["numbered_prefix_filename"]
            ) + sum(1 for r in references.values() if r["numbered_prefix_filename"]),
        },
        "reference_sizes": references,
        "files": files_data,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Token-based prompt metrics for the Analyze scanners")
    p.add_argument("skill_path", type=Path, help="path to the skill directory to scan")
    p.add_argument("--output", "-o", type=Path, help="write JSON to a file instead of stdout")
    args = p.parse_args(argv)

    if not args.skill_path.is_dir():
        print(f"error: {args.skill_path} is not a directory", file=sys.stderr)
        return 2

    output = json.dumps(scan(args.skill_path), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
        print(f"results written to {args.output}", file=sys.stderr)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
