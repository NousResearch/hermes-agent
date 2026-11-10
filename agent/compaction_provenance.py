"""Harness-authored provenance block for compaction summaries.

A summary is a snapshot of claims taken at one instant; the workspace it describes keeps moving. The
summarizer can record *what* was true when the turns were written, but it cannot record *which checkout*
those turns described, and it is not a reliable narrator for it either: asked for a commit or a branch it
will invent one. So the provenance facts are read from the workspace by the harness and appended after
the model's text, the same way ``[SKILL_PRUNED]`` markers are re-injected.

The block is deliberately short and lives in the summary body (a mid-conversation message), never in
``SUMMARY_PREFIX`` or the system prompt, so prompt-cache stability is untouched.

Scope note: this cannot prove a claim in the summary was true at the recorded HEAD. It only makes
staleness detectable, which is the difference between "unknown age" and "recorded at a commit you can
compare against".
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROVENANCE_HEADING = "## Session Provenance"

# A summary that already carries the block is refreshed, never duplicated: the iterative-update path
# feeds the previous summary back to the summarizer, so a copied block would otherwise pin a stale HEAD.
_STALE_TAIL = (
    "Claims that were verified against a different HEAD may be stale. Re-check with a tool call "
    "before relying on them."
)
_NOT_A_REPO = "not a git repository"
_UNKNOWN = "unknown"

_GIT_TIMEOUT_SECONDS = 2.0
_GIT_CACHE_TTL_SECONDS = 30.0

_git_cache: dict[str, tuple[float, tuple[Optional[str], Optional[str]]]] = {}
_git_cache_lock = threading.Lock()


def _git(cwd: str, *args: str) -> Optional[str]:
    """Run one read-only git command in ``cwd``; ``None`` on any failure (missing git, not a repo, timeout)."""
    env = dict(os.environ)
    env.setdefault("GIT_OPTIONAL_LOCKS", "0")  # never take a lock from a summarizer hook
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    try:
        completed = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_SECONDS, env=env, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def detect_git_state(cwd: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Return ``(branch, short_head)`` for *cwd*, or ``(None, None)`` outside a repository.

    Cached per directory for a few seconds: compaction can run twice in a burst (a prepared pass plus a
    live pass) and must not pay two subprocess round-trips for the same answer. A failure is cached too,
    so a non-repository workspace is probed once, not once per compaction.
    """
    if not cwd:
        return None, None
    key = os.path.normcase(os.path.abspath(cwd))
    now = time.monotonic()
    with _git_cache_lock:
        cached = _git_cache.get(key)
        if cached is not None and now - cached[0] < _GIT_CACHE_TTL_SECONDS:
            return cached[1]
    branch = _git(cwd, "rev-parse", "--abbrev-ref", "HEAD")
    head = _git(cwd, "rev-parse", "--short", "HEAD") if branch else None
    state = (branch, head)
    with _git_cache_lock:
        _git_cache[key] = (now, state)
    return state


def _clear_git_cache() -> None:
    """Test/session helper: forget cached git state."""
    with _git_cache_lock:
        _git_cache.clear()


def resolve_workspace() -> Optional[str]:
    """Best-effort working directory for the current turn, without creating one.

    ``scoped_session_cwd`` carries a declared logical cwd (remote/container paths included);
    ``resolve_context_cwd`` is the validated fallback. Resolution failure is expected outside a real
    session (tests, one-shot tooling) and must never block compaction.
    """
    try:
        from agent.runtime_cwd import scoped_session_cwd, resolve_context_cwd

        declared = scoped_session_cwd()
        if declared:
            return declared
        resolved = resolve_context_cwd()
        return str(resolved) if resolved else None
    except Exception:  # health: allow BLE001 -- provenance is best-effort by contract
        return None


def _heading_spans(text: str) -> list[tuple[int, int]]:
    """Span(s) of any existing provenance section: heading line through the next ``## `` heading."""
    spans: list[tuple[int, int]] = []
    lines = text.splitlines(keepends=True)
    offset = 0
    start: Optional[int] = None
    for line in lines:
        stripped = line.strip()
        if stripped == PROVENANCE_HEADING:
            start = offset
        elif start is not None and stripped.startswith("## "):
            spans.append((start, offset))
            start = None
        offset += len(line)
    if start is not None:
        spans.append((start, len(text)))
    return spans


def strip_provenance_section(summary: str) -> str:
    """Remove every provenance block, leaving the surrounding text intact."""
    for start, end in reversed(_heading_spans(summary)):
        summary = summary[:start].rstrip() + ("\n\n" if summary[end:].strip() else "") + summary[end:].lstrip()
    return summary


def _format_section(cwd: Optional[str], branch: Optional[str], head: Optional[str], stamp: str) -> str:
    if not cwd:
        location = f"- Working directory: {_UNKNOWN}"
    else:
        location = f"- Working directory: {cwd}"
    if branch:
        git_line = f"- Git branch: {branch}"
        if head:
            git_line += f" (HEAD {head})"
    else:
        git_line = f"- Git: {_NOT_A_REPO if cwd else _UNKNOWN}"
    return (
        f"\n\n{PROVENANCE_HEADING}\n"
        f"Compacted {stamp} and recorded by the harness from the workspace, not by the summarizer.\n"
        f"{location}\n{git_line}\n{_STALE_TAIL}"
    )


def append_provenance_section(summary: str, *, cwd: Optional[str] = None, now=None) -> str:
    """Return *summary* with one fresh, harness-authored provenance block at the end.

    ``cwd``/``now`` are injectable for tests. On any internal failure the original summary is returned
    unchanged: a missing provenance block degrades traceability, a raised exception costs the whole
    compaction it was embedded in.
    """
    try:
        if not isinstance(summary, str):
            return summary
        workspace = cwd if cwd is not None else resolve_workspace()
        if now is None:
            try:
                from hermes_time import now as _hermes_now

                now = _hermes_now()
            except Exception:  # health: allow BLE001 -- clock failure must not block compaction
                now = None
        try:
            stamp = now.isoformat(timespec="seconds") if now is not None else _UNKNOWN
        except Exception:  # health: allow BLE001
            stamp = _UNKNOWN
        branch, head = detect_git_state(workspace)
        body = strip_provenance_section(summary).rstrip()
        return body + _format_section(workspace, branch, head, stamp)
    except Exception:  # health: allow BLE001 -- never fail a compaction over bookkeeping
        logger.debug("compaction provenance block skipped", exc_info=True)
        return summary
