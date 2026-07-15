"""Semantic delta accumulation for edge working-memory scratchpads.

Merges new findings without wiping history, dedupes file-path bullets, and
marks checklist items done — all keyword-oriented, minimal prose.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set

# File-ish tokens: `path`, bare posix paths
_PATHISH = re.compile(
    r"(?:`|\b)((?:[\w.-]+/)+[\w.-]+\.(?:py|js|ts|tsx|go|rs|java|kt|c|h|hpp|cpp|md|yml|yaml|json|toml)|(?:[\w.-]+/)+[\w.-]+)(?:`|$|\s|\))",
    re.I,
)


def _norm_bullet(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip().lower())


def _extract_paths(text: str) -> Set[str]:
    return {m.group(1) for m in _PATHISH.finditer(text or "")}


def dedupe_bullets_by_path(lines: List[str]) -> List[str]:
    """Keep first bullet per file-ish path token; drop obvious duplicates."""
    seen_paths: Set[str] = set()
    out: List[str] = []
    for line in lines:
        paths = _extract_paths(line)
        if paths:
            if all(p in seen_paths for p in paths):
                continue
            for p in paths:
                seen_paths.add(p)
        out.append(line)
    return out


def _section_slice(text: str, header: str, stop_prefixes: tuple[str, ...]) -> tuple[int, int]:
    start = text.find(header)
    if start == -1:
        return -1, -1
    rest = text[start + len(header) :]
    end_rel = len(rest)
    for sp in stop_prefixes:
        j = rest.find(sp)
        if j != -1 and j < end_rel:
            end_rel = j
    return start, start + len(header) + end_rel


def accumulate_scratchpad_delta(
    scratchpad: str,
    *,
    add_facts: Optional[Iterable[str]] = None,
    add_faults: Optional[Iterable[str]] = None,
    complete_task_substrings: Optional[Iterable[str]] = None,
    dedupe_paths: bool = True,
) -> str:
    """Apply semantic deltas; preserves unrelated sections and history."""
    base = scratchpad or ""
    facts_needles = ("**Facts:**", "**Validated Facts:**")
    faults_needles = ("**Faults:**", "**Faults & Blockers:**")
    next_needle = "### NEXT"

    # --- Facts ---
    f_needle = next((n for n in facts_needles if n in base), None)
    if f_needle and add_facts:
        s, e = _section_slice(base, f_needle, ("**Faults", "### NEXT", "\n### "))
        if s != -1:
            chunk = base[s:e]
            body = chunk[len(f_needle) :]
            existing_lines = [ln for ln in body.splitlines() if ln.strip()]
            norm_existing = {_norm_bullet(ln) for ln in existing_lines}
            new_lines = list(existing_lines)
            for fact in add_facts:
                bullet = fact if str(fact).lstrip().startswith("-") else f"- {fact}"
                bn = _norm_bullet(bullet)
                if bn and bn not in norm_existing:
                    new_lines.append(bullet.strip())
                    norm_existing.add(bn)
            if dedupe_paths:
                new_lines = dedupe_bullets_by_path(new_lines)
            rebuilt = f_needle + "\n" + "\n".join(new_lines) + "\n"
            base = base[:s] + rebuilt + base[e:]

    # --- Faults ---
    fl_needle = next((n for n in faults_needles if n in base), None)
    if fl_needle and add_faults:
        s, e = _section_slice(base, fl_needle, ("### NEXT", "\n### "))
        if s != -1:
            chunk = base[s:e]
            body = chunk[len(fl_needle) :]
            existing_lines = [ln for ln in body.splitlines() if ln.strip()]
            norm_existing = {_norm_bullet(ln) for ln in existing_lines}
            new_lines = list(existing_lines)
            for fault in add_faults:
                bullet = fault if str(fault).lstrip().startswith("-") else f"- {fault}"
                bn = _norm_bullet(bullet)
                if bn and bn not in norm_existing:
                    new_lines.append(bullet.strip())
                    norm_existing.add(bn)
            if dedupe_paths:
                new_lines = dedupe_bullets_by_path(new_lines)
            rebuilt = fl_needle + "\n" + "\n".join(new_lines) + "\n"
            base = base[:s] + rebuilt + base[e:]

    # --- NEXT checkbox completion ---
    if complete_task_substrings and next_needle in base:
        s, e = _section_slice(base, next_needle, ("\n### ",))
        if s != -1:
            segment = base[s:e]
            for sub in complete_task_substrings:
                sub_s = str(sub).strip()
                if not sub_s:
                    continue
                lines = segment.splitlines()
                out_ln: List[str] = []
                for ln in lines:
                    if (
                        sub_s.lower() in ln.lower()
                        and ln.strip().startswith("- [ ]")
                    ):
                        out_ln.append(ln.replace("- [ ]", "- [x]", 1))
                    else:
                        out_ln.append(ln)
                segment = "\n".join(out_ln)
            base = base[:s] + segment + base[e:]

    return base.rstrip() + "\n" if base.strip() else base
