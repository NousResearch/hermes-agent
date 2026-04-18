"""Semantic-tier readers and brain CLI wrappers.

Uses ripgrep when available for fast searches; falls back to pure-Python
substring scan when rg is missing. All shell-outs have explicit timeouts
so a misbehaving brain directory can't stall Hermes.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


# Tokens we never want to search on. Common English filler plus short
# function words that would match everything.
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "into",
    "what", "which", "when", "where", "why", "how", "are", "was",
    "were", "been", "being", "have", "has", "had", "but", "not",
    "can", "will", "would", "should", "could", "may", "might",
    "any", "all", "some", "you", "your", "our", "their", "its",
    "about", "there", "also", "just", "then", "than", "more",
    "most", "like", "only", "such", "each", "these", "those",
    "get", "got", "make", "made", "take", "took", "give",
})


def _tokenize(query: str, min_len: int = 3) -> List[str]:
    """Split ``query`` into lowercase content words for search.

    Keeps alphanumerics of ``min_len`` or more, drops stopwords, dedupes
    while preserving first-seen order so the user's emphasis survives.
    """
    raw = re.findall(r"[\w'-]+", (query or "").lower())
    seen: List[str] = []
    for w in raw:
        clean = w.strip("-'")
        if len(clean) < min_len:
            continue
        if clean in _STOPWORDS:
            continue
        if clean in seen:
            continue
        seen.append(clean)
    return seen


def _run(cmd: List[str], timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )


def brain_python(brain_path: Path) -> str:
    """Resolve the agentic-stack venv interpreter, fall back to system."""
    venv = brain_path / ".venv" / "bin" / "python"
    if venv.exists():
        return str(venv)
    return "python3"


def read_review_queue(brain_path: Path) -> str:
    """Read REVIEW_QUEUE.md. Returns empty string if missing or empty."""
    path = brain_path / "memory" / "working" / "REVIEW_QUEUE.md"
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    if not text or "no pending" in text.lower():
        return ""
    return text


def search(brain_path: Path, query: str, max_results: int = 5) -> Dict:
    """Search semantic tier with ripgrep, fall back to Python if rg missing.

    Multi-word queries are tokenized; stopwords dropped; remaining words
    are OR-ed together so any content word matches. Hits are deduped by
    (file, line) so the same line isn't returned once per matching word.
    """
    q = query.strip()
    if not q:
        return {"ok": False, "error": "empty query"}

    sem = brain_path / "memory" / "semantic"
    if not sem.is_dir():
        return {"ok": False, "error": f"semantic tier missing at {sem}"}

    tokens = _tokenize(q)
    if not tokens:
        # Fall back to the full raw query escaped literally so obscure
        # all-short-words queries still have a chance.
        tokens = [re.escape(q)]
    else:
        tokens = [re.escape(t) for t in tokens]
    pattern = "|".join(tokens)

    rg = shutil.which("rg")
    hits: List[Dict] = []
    seen: set = set()
    try:
        if rg:
            result = _run(
                [
                    rg, "--json", "--max-count", "3", "--smart-case",
                    "--type", "md", pattern, str(sem),
                ],
                timeout=8,
            )
            for line in result.stdout.splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") != "match":
                    continue
                data = rec.get("data", {})
                path_obj = data.get("path", {}) or {}
                fpath = path_obj.get("text", "")
                lnum = data.get("line_number", 0)
                key = (fpath, lnum)
                if key in seen:
                    continue
                seen.add(key)
                hits.append({
                    "file": fpath,
                    "line": lnum,
                    "text": (data.get("lines", {}) or {}).get("text", "").strip(),
                })
                if len(hits) >= max_results:
                    break
        else:
            # Pure-Python fallback: any token match, deduped by (file, line).
            token_lowers = [t.lower() for t in _tokenize(q) or [q.lower()]]
            for p in sem.rglob("*.md"):
                try:
                    lines = p.read_text(encoding="utf-8").splitlines()
                except Exception:
                    continue
                for i, line in enumerate(lines, 1):
                    lo = line.lower()
                    if any(t in lo for t in token_lowers):
                        key = (str(p), i)
                        if key in seen:
                            continue
                        seen.add(key)
                        hits.append({"file": str(p), "line": i, "text": line.strip()})
                        if len(hits) >= max_results:
                            break
                if len(hits) >= max_results:
                    break
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "search timed out"}
    except Exception as e:
        return {"ok": False, "error": f"search failed: {e}"}

    return {"ok": True, "hits": hits[:max_results], "tool": "rg" if rg else "python"}


def review_queue(brain_path: Path) -> Dict:
    """Return pending candidates via list_candidates.py --format json."""
    tool = brain_path / "tools" / "list_candidates.py"
    if not tool.exists():
        return {"ok": False, "error": f"list_candidates.py missing at {tool}"}
    try:
        result = _run(
            [brain_python(brain_path), str(tool), "--format", "json"], timeout=10
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "list_candidates timed out"}
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip() or "list_candidates failed"}
    try:
        items = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        items = []
    return {"ok": True, "pending": len(items), "items": items}


def graduate(brain_path: Path, candidate_id: str, rationale: str) -> Dict:
    tool = brain_path / "tools" / "graduate.py"
    if not tool.exists():
        return {"ok": False, "error": "graduate.py missing"}
    result = _run(
        [
            brain_python(brain_path), str(tool), candidate_id,
            "--rationale", rationale,
        ],
        timeout=15,
    )
    return {
        "ok": result.returncode == 0,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def reject(brain_path: Path, candidate_id: str, reason: str) -> Dict:
    tool = brain_path / "tools" / "reject.py"
    if not tool.exists():
        return {"ok": False, "error": "reject.py missing"}
    result = _run(
        [
            brain_python(brain_path), str(tool), candidate_id,
            "--reason", reason,
        ],
        timeout=15,
    )
    return {
        "ok": result.returncode == 0,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def log_via_cli(
    brain_path: Path,
    skill: str,
    action: str,
    outcome: str,
    importance: int = 5,
    reflection: str = "",
    success: bool = True,
) -> Dict:
    """Explicit log via memory_reflect.py. Prefer sync_turn for auto-logging;
    this is for tool-call overrides."""
    tool = brain_path / "tools" / "memory_reflect.py"
    if not tool.exists():
        return {"ok": False, "error": "memory_reflect.py missing"}
    cmd = [
        brain_python(brain_path), str(tool),
        skill, action[:200], outcome[:500],
        "--importance", str(max(1, min(importance, 10))),
    ]
    if reflection:
        cmd.extend(["--note", reflection[:500]])
    if not success:
        cmd.append("--fail")
    result = _run(cmd, timeout=10)
    return {
        "ok": result.returncode == 0,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
