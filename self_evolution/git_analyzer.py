"""
Self Evolution Plugin — Git Analysis
=====================================

Analyzes git commit history for the dream consolidation engine.

Uses a single batched ``git log --stat --name-only`` call instead of
25+ individual subprocess invocations.

Extracted from reflection_engine.py for single-responsibility.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Dict

from self_evolution.models import CodeChangeAnalysis, CommitInfo

logger = logging.getLogger(__name__)


def analyze_code_changes(hours: int = 24) -> CodeChangeAnalysis:
    """Analyze git commits from the previous period.

    Uses a single batched git log call with --stat --name-only
    instead of 25+ individual subprocess calls.
    """
    project_root = str(Path(__file__).resolve().parent.parent)

    cutoff_epoch = time.time() - (hours * 3600)
    cutoff_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cutoff_epoch))

    try:
        # Single batched call: format + shortstat + name-only
        result = subprocess.run(
            ["git", "log",
             "--format=COMMITSTART%h%n%s%n%an%n%at%n%b%nENDHEADER",
             "--shortstat", "--name-only",
             "--no-merges", f"--since={cutoff_date}", "-15"],
            capture_output=True, text=True, timeout=30,
            cwd=project_root,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return CodeChangeAnalysis()

        commits = _parse_batched_output(result.stdout)
        if not commits:
            return CodeChangeAnalysis()

        # Aggregate stats
        total_ins = sum(c.insertions for c in commits)
        total_del = sum(c.deletions for c in commits)
        total_files = sum(c.files_changed for c in commits)
        authors = list(dict.fromkeys(c.author for c in commits))

        # Categorize by conventional commit prefix
        categories: Dict[str, int] = {}
        for c in commits:
            cat = _categorize_commit(c.subject)
            categories[cat] = categories.get(cat, 0) + 1

        # Extract top-level module areas
        all_files = []
        for c in commits:
            all_files.extend(c.file_list)
        areas = list(dict.fromkeys(
            f.split("/")[0] for f in all_files
            if "/" in f and not f.startswith(".")
        ))[:10]

        return CodeChangeAnalysis(
            commits=commits,
            total_commits=len(commits),
            total_insertions=total_ins,
            total_deletions=total_del,
            total_files_changed=total_files,
            authors=authors,
            change_categories=categories,
            areas_touched=areas,
        )

    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        logger.debug("git analysis unavailable", exc_info=True)
        return CodeChangeAnalysis()


def _parse_batched_output(stdout: str) -> list:
    """Parse the batched git log output into CommitInfo objects."""
    commits = []
    raw_commits = stdout.split("COMMITSTART")
    for raw in raw_commits:
        raw = raw.strip()
        if not raw:
            continue

        header_end = raw.find("ENDHEADER")
        if header_end < 0:
            continue
        header = raw[:header_end].strip()
        lines = header.split("\n")
        if len(lines) < 4:
            continue

        hash_short = lines[0].strip()
        subject = lines[1].strip()
        author = lines[2].strip()
        try:
            timestamp = float(lines[3].strip())
        except ValueError:
            continue
        body = "\n".join(lines[4:]).strip()[:500]

        # After ENDHEADER: shortstat line(s) + file list
        rest = raw[header_end + len("ENDHEADER"):].strip()

        files_changed = 0
        insertions = 0
        deletions = 0
        file_list = []
        stat_done = False
        for rline in rest.split("\n"):
            rline = rline.strip()
            if not rline:
                continue
            if not stat_done and ("files changed" in rline or "file changed" in rline
                                  or "insertion" in rline or "deletion" in rline):
                files_changed = _parse_int(r'(\d+) files? changed', rline)
                insertions = _parse_int(r'(\d+) insertion', rline)
                deletions = _parse_int(r'(\d+) deletion', rline)
                stat_done = True
                continue
            if "/" in rline or "." in rline:
                file_list.append(rline)

        commits.append(CommitInfo(
            hash_short=hash_short,
            subject=subject,
            body=body,
            author=author,
            timestamp=timestamp,
            files_changed=files_changed,
            insertions=insertions,
            deletions=deletions,
            file_list=file_list[:20],
        ))

    return commits


# ── Helpers ───────────────────────────────────────────────────────────────


def _parse_int(pattern: str, text: str) -> int:
    """Extract first integer matching regex pattern from text."""
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0


def _categorize_commit(subject: str) -> str:
    """Categorize commit by conventional commit prefix."""
    s = subject.lower()
    for prefix in ("feat", "fix", "refactor", "test", "docs", "chore", "perf", "style", "ci", "build"):
        if s.startswith(prefix):
            return prefix
    return "other"
