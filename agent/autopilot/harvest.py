"""Harvest novel deception phrasings from autopilot ADR logs.

The ADR decision log records every caught deception (kind='deception') with the
flags that fired and the response context. This scans those logs and surfaces:

  * which categories fire most (so you know where the model pushes hardest), and
  * candidate NEW phrasings — lines from flagged responses that don't yet match
    any dictionary pattern — so you can promote the good ones into
    deception_patterns.yaml (or your local overlay) and grow the dictionary.

This is the "learn from the model as it evolves" loop: the model keeps inventing
new excuses, the ADR captures them, and this turns them into dictionary entries.

Pure stdlib + the shipped dictionary; no model call. Safe to run anytime.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from agent.autopilot import deception


def _default_adr_root() -> Path:
    root = os.environ.get("HERMES_WORKSPACE", "").strip() or os.getcwd()
    return Path(root) / ".hermes" / "autopilot" / "adr"


def _iter_adr_files(adr_root: Path) -> Iterable[Path]:
    if adr_root.is_file():
        yield adr_root
        return
    if adr_root.is_dir():
        yield from sorted(adr_root.glob("AUTOPILOT-*.md"))


_SECTION_RE = re.compile(r"^## .+ — (?P<kind>\w+)\s*$")

# ADR field-label / scaffolding noise to exclude from novel-phrase candidates.
_HARVEST_STOPWORDS = (
    "reviewer", "rationale", "verdict", "goal", "options considered", "chosen path",
    "gap found / why not passing", "required to pass", "caught deception",
    "deception-detector", "sent for verification", "council", "aux", "options-fallback",
    "continue — re-inject with the caught behavior named",
)


def _parse_sections(text: str) -> list[dict]:
    """Split an ADR markdown file into its decision sections."""
    sections: list[dict] = []
    cur: dict | None = None
    for line in text.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            if cur is not None:
                sections.append(cur)
            cur = {"kind": m.group("kind"), "lines": []}
        elif cur is not None:
            cur["lines"].append(line)
    if cur is not None:
        sections.append(cur)
    return sections


def _candidate_phrases(text: str) -> list[str]:
    """Break a flagged response/rationale into short candidate phrases (clauses)."""
    out: list[str] = []
    for chunk in re.split(r"[.\n;:]", text.lower()):
        c = chunk.strip(" -•\t\"'")
        if not (8 <= len(c) <= 80):
            continue
        if c in _HARVEST_STOPWORDS or any(c.startswith(sw) for sw in _HARVEST_STOPWORDS):
            continue
        # skip pure flag-name lists (e.g. "await_user, effort_excuse")
        if re.fullmatch(r"[a-z_]+(,\s*[a-z_]+)*", c):
            continue
        out.append(c)
    return out


def harvest(adr_root: Path | None = None, *, top: int = 25) -> dict:
    """Scan ADR logs and return a harvest report.

    Returns a dict with: files scanned, category_counts (how often each fired),
    and novel_phrases (Counter of clauses from flagged sections that do NOT match
    any existing dictionary pattern — these are the promotion candidates).
    """
    adr_root = adr_root or _default_adr_root()
    d = deception.load_dictionary(force=True)
    all_patterns: list[str] = []
    for cat in d.categories:
        all_patterns.extend(d.patterns(cat))

    category_counts: Counter = Counter()
    novel: Counter = Counter()
    files = list(_iter_adr_files(adr_root))

    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for sec in _parse_sections(text):
            if sec["kind"] != "deception":
                continue
            body = "\n".join(sec["lines"])
            # tally which flags fired (recorded as "caught deception: a, b, c")
            for m in re.finditer(r"caught deception:\s*([a-z_,\s]+)", body):
                for flag in m.group(1).split(","):
                    flag = flag.strip()
                    if flag:
                        category_counts[flag] += 1
            # candidate novel phrasings: clauses not already covered by a pattern
            for phrase in _candidate_phrases(body):
                if not any(p in phrase for p in all_patterns):
                    novel[phrase] += 1

    return {
        "files_scanned": len(files),
        "adr_root": str(adr_root),
        "category_counts": dict(category_counts.most_common()),
        "novel_phrases": dict(novel.most_common(top)),
    }


def format_report(report: dict) -> str:
    """Human-readable harvest report for the CLI."""
    lines: list[str] = []
    lines.append(f"Deception harvest — scanned {report['files_scanned']} ADR file(s)")
    lines.append(f"  ADR root: {report['adr_root']}")
    cc = report.get("category_counts") or {}
    if cc:
        lines.append("\nCategories caught (most frequent first):")
        for cat, n in cc.items():
            lines.append(f"  {n:>5}  {cat}")
    else:
        lines.append("\nNo deception records found yet (run some autopilot with autopilot.adr on).")
    novel = report.get("novel_phrases") or {}
    if novel:
        lines.append("\nCandidate NEW phrasings (not yet in the dictionary — promote the real tells):")
        for phrase, n in novel.items():
            lines.append(f"  {n:>3}×  {phrase!r}")
        lines.append(
            "\nTo grow the dictionary: add the genuine excuses above to the right category in\n"
            "agent/autopilot/deception_patterns.yaml (to contribute upstream) or your local\n"
            "~/.hermes/autopilot/deception-patterns.local.yaml (private)."
        )
    return "\n".join(lines)
