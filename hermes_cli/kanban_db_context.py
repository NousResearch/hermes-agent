"""Worker-context rendering for ``build_worker_context``: the capped markdown sections a spawned
worker reads about its task (header, attachments, prior attempts, parent handoffs, role
history, comments).

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached late-bound via
``_kb`` so monkeypatching ``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import TYPE_CHECKING, Any, Optional

import hermes_cli.kanban_db as _kb

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Attachment, Comment, Task

# build_worker_context() caps, sized for a ~100k-char prompt with headroom.
_CTX_MAX_PRIOR_ATTEMPTS = 10      # most recent N prior runs shown in full
_CTX_MAX_COMMENTS       = 30      # most recent N comments shown in full
_CTX_MAX_FIELD_BYTES    = 4 * 1024   # per summary/error/metadata/result
_CTX_MAX_BODY_BYTES     = 8 * 1024   # per task.body (opening post)
_CTX_MAX_COMMENT_BYTES  = 2 * 1024   # per comment

# A clean-exit protocol violation means the run did its work and skipped the
# paperwork — the guidance already tells the retry to "verify it and report the
# result via kanban_complete". Verifying it started with REDISCOVERING it: nothing
# handed the retry what the previous run had changed, so each attempt re-derived
# the same state. Measured on one install: 117 such runs across 49 tasks, 5.0 runs
# per task, and 4 tasks that never recovered at all.
_CTX_EVIDENCE_MAX_COMMITS = 5
_CTX_EVIDENCE_TIMEOUT_S = 10


def _relative_age(ts: Optional[int], now: Optional[int] = None) -> str:
    """``just now`` / ``18h ago`` / ``3d ago``; "" for a missing/invalid ts. An LLM
    reads a bare absolute timestamp as current fact — the relative age is what
    prompts a worker to re-verify stale sibling work."""
    try:
        ts = int(ts)
    except (TypeError, ValueError):
        return ""
    if now is None:
        now = int(time.time())
    delta = now - ts
    if delta < 60:  # includes negative = clock skew across machines; never claim "in the future"
        return "just now"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"


def _ctx_cap(s: Optional[str], limit: int = _CTX_MAX_FIELD_BYTES) -> str:
    """Truncate to ``limit`` chars with a visible ellipsis."""
    if not s:
        return ""
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[:limit] + f"… [truncated, {len(s) - limit} chars omitted]"


def _ctx_stamp(ts: int, now: int) -> str:
    """``YYYY-MM-DD HH:MM`` plus a relative age when one is available."""
    disp = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
    age = _relative_age(ts, now)
    return f"{disp}, {age}" if age else disp


def _ctx_metadata_line(metadata: Any) -> Optional[str]:
    if not metadata:
        return None
    try:
        return f"_metadata_: `{_ctx_cap(json.dumps(metadata, ensure_ascii=False, sort_keys=True))}`"
    except Exception:
        return None


def _ctx_tail(items: list, cap: int, noun: str) -> tuple[list, Optional[str]]:
    """Keep the newest ``cap`` items; describe the omitted head, if any."""
    omitted = max(0, len(items) - cap)
    if not omitted:
        return items, None
    return items[-cap:], (
        f"_({omitted} earlier {noun}{'s' if omitted != 1 else ''} "
        f"omitted; showing most recent {cap})_"
    )


def _ctx_header(lines: list[str], task: Task) -> None:
    lines.append(f"# Kanban task {task.id}: {task.title}")
    lines.append("")
    lines.append(f"Assignee: {task.assignee or '(unassigned)'}")
    lines.append(f"Status:   {task.status}")
    if task.tenant:
        lines.append(f"Tenant:   {task.tenant}")
    lines.append(f"Workspace: {task.workspace_kind} @ {task.workspace_path or '(unresolved)'}")
    if task.max_runtime_seconds is not None:
        terminal_timeout = _kb._worker_terminal_timeout_env(
            task.max_runtime_seconds, os.environ.get("TERMINAL_TIMEOUT"),
        )
        effective_terminal_timeout = terminal_timeout or os.environ.get("TERMINAL_TIMEOUT")
        lines.append(f"Max runtime: {task.max_runtime_seconds}s")
        if effective_terminal_timeout:
            lines.append(f"Terminal timeout: {effective_terminal_timeout}s")
    if task.branch_name:
        lines.append(f"Branch:   {task.branch_name}")
    lines.append("")
    if task.body and task.body.strip():
        lines.append("## Body")
        lines.append(_ctx_cap(task.body, _CTX_MAX_BODY_BYTES))
        lines.append("")


def _ctx_attachments(lines: list[str], attachments: list[Attachment]) -> None:
    """Absolute on-disk paths so the worker's file tools read them directly
    (remote terminal backends need the attachments dir mounted)."""
    if not attachments:
        return
    lines.append("## Attachments")
    lines.append(
        "Files attached to this task. Read them with the file/terminal "
        "tools at the absolute paths below:"
    )
    for att in attachments:
        size_kb = max(1, (att.size + 1023) // 1024) if att.size else 0
        size_str = f", {size_kb} KB" if size_kb else ""
        ctype = f", {att.content_type}" if att.content_type else ""
        lines.append(f"- `{att.filename}`{ctype}{size_str} → `{att.stored_path}`")
    lines.append("")

def _run_was_protocol_violation(run) -> bool:
    """True when this closed run ended as a clean-exit-without-verdict.

    Two readings because the two sources age differently: the durable marker
    ``detect_crashed_workers`` copies into the run metadata, and the error text
    for runs recorded before that marker existed.
    """
    meta = getattr(run, "metadata", None)
    if isinstance(meta, dict) and meta.get("protocol_violation"):
        return True
    return "protocol violation" in (getattr(run, "error", "") or "")


def _protocol_violation_evidence(task, run) -> list[str]:
    """What the previous run left behind in the workspace, named for the retry.

    Read-only, time-bounded, and silent on ANY failure: a worker prompt that
    cannot be built is worse than one without this section, and this is a
    convenience on top of guidance that already works.

    ``--since`` filters on COMMITTER date, which is what "landed during that run"
    means here — a commit authored earlier and committed by that run still counts,
    and that is the case worth catching (an agent that stages for a while and
    commits at the end).
    """
    path = (getattr(task, "workspace_path", None) or "").strip()
    kind = getattr(task, "workspace_kind", None)
    started = getattr(run, "started_at", None)
    if not path or kind not in {"dir", "worktree"} or not started:
        return []
    try:
        import subprocess

        since = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(int(started)))

        def _git(*args: str) -> str:
            out = subprocess.run(
                ("git", "-C", path) + args,
                capture_output=True, text=True, check=False,
                timeout=_CTX_EVIDENCE_TIMEOUT_S,
            )
            return out.stdout.strip() if out.returncode == 0 else ""

        if not _git("rev-parse", "--is-inside-work-tree"):
            return []
        commits = _git("log", f"--since={since}", "--oneline",
                       f"-n{_CTX_EVIDENCE_MAX_COMMITS}")
        dirty = [ln for ln in _git("status", "--porcelain").splitlines() if ln.strip()]
    except Exception:
        return []

    if not commits and not dirty:
        # Nothing landed. Saying so is the useful half: it tells the retry the
        # work is still to do, instead of leaving it to hunt for work that was
        # never there.
        return ["_The previous run left no commit and no uncommitted change in "
                f"`{path}` — treat the work as still to do._"]

    out = ["**What the previous run left behind** (it exited without reporting, "
           "so this is unverified — confirm before claiming it):"]
    if commits:
        out.append(f"Commits in `{path}` since that run started:")
        out.extend(f"- `{ln}`" for ln in commits.splitlines())
    if dirty:
        shown = dirty[:_CTX_EVIDENCE_MAX_COMMITS]
        out.append(f"Uncommitted in `{path}`: {len(dirty)} file(s)"
                   + (" (first few)" if len(dirty) > len(shown) else ""))
        out.extend(f"- `{ln}`" for ln in shown)
    return out


def _ctx_prior_attempts(lines: list[str], conn: sqlite3.Connection, task_id: str, now: int) -> None:
    """Closed runs on this task (the active run is this worker), newest
    ``_CTX_MAX_PRIOR_ATTEMPTS`` in full, older ones as a one-line marker."""
    all_prior = [r for r in _kb.list_runs(conn, task_id) if r.ended_at is not None]
    shown, omitted_note = _ctx_tail(all_prior, _CTX_MAX_PRIOR_ATTEMPTS, "attempt")
    if not shown:
        return
    first_shown_idx = len(all_prior) - len(shown) + 1
    lines.append("## Prior attempts on this task")
    if omitted_note:
        lines.append(omitted_note)
    for offset, run in enumerate(shown):
        profile = run.profile or "(unknown)"
        outcome = run.outcome or run.status
        lines.append(
            f"### Attempt {first_shown_idx + offset} — {outcome} ({profile}, {_ctx_stamp(run.started_at, now)})"
        )
        if run.summary and run.summary.strip():
            lines.append(_ctx_cap(run.summary))
        if run.error and run.error.strip():
            lines.append(f"_error_: {_ctx_cap(run.error)}")
        meta_line = _ctx_metadata_line(run.metadata)
        if meta_line:
            lines.append(meta_line)
        lines.append("")
    # The most recent closed attempt ended clean-but-verdictless: name what it
    # left in the workspace instead of asking the retry to go and find it.
    if shown and _run_was_protocol_violation(shown[-1]):
        evidence = _protocol_violation_evidence(_kb.get_task(conn, task_id), shown[-1])
        if evidence:
            lines.append("")
            lines.extend(evidence)
            lines.append("")


def _ctx_parent_results(lines: list[str], conn: sqlite3.Connection, task_id: str, now: int) -> None:
    """Done-parent handoffs: newest ``completed`` run's summary+metadata,
    falling back to ``task.result`` for pre-runs-table data. Stamped with a
    relative age so the worker re-verifies stale upstream results."""
    parent_rows = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id", (task_id,),
    ).fetchall()
    wrote_header = False
    for pid in (r["parent_id"] for r in parent_rows):
        pt = _kb.get_task(conn, pid)
        if not pt or pt.status != "done":
            continue
        runs = [r for r in _kb.list_runs(conn, pid) if r.outcome == "completed"]
        runs.sort(key=lambda r: r.started_at, reverse=True)
        run = runs[0] if runs else None
        if not wrote_header:
            lines.append("## Parent task results")
            lines.append(
                "_Handoffs from upstream tasks, captured when each parent "
                "completed (see age below). These are point-in-time "
                "snapshots, not live state — if a result drives your "
                "current work and it's not recent, re-verify against the "
                "source before acting on it as current._"
            )
            wrote_header = True
        done_ts = run.ended_at if run is not None and run.ended_at else (pt.completed_at or None)
        age = _relative_age(done_ts, now)
        lines.append(f"### {pid}" + (f" (completed {age})" if age else ""))
        if run is not None and run.summary and run.summary.strip():
            lines.append(_ctx_cap(run.summary))
        elif pt.result:
            lines.append(_ctx_cap(pt.result))
        else:
            lines.append("(no result recorded)")
        meta_line = _ctx_metadata_line(run.metadata) if run is not None else None
        if meta_line:
            lines.append(meta_line)
        lines.append("")


def _ctx_role_history(lines: list[str], conn: sqlite3.Connection, task: Task, now: int) -> None:
    """The assignee's 5 most recent completed runs on OTHER tasks — implicit
    role continuity without wiring anything into SOUL.md / MEMORY.md."""
    if not task.assignee:
        return
    role_rows = conn.execute(
        "SELECT t.id, t.title, r.summary, r.ended_at "
        "FROM task_runs r JOIN tasks t ON r.task_id = t.id "
        "WHERE r.profile = ? AND r.task_id != ? "
        "  AND r.outcome = 'completed' "
        "ORDER BY r.ended_at DESC LIMIT 5", (task.assignee, task.id),
    ).fetchall()
    if not role_rows:
        return
    lines.append(f"## Recent work by @{task.assignee}")
    for row in role_rows:
        first = _kb._first_line(row["summary"], 200) or "(no summary)"
        lines.append(
            f"- {row['id']} — {row['title']} ({_ctx_stamp(_kb._to_epoch(row['ended_at']) or 0, now)}): {first}"
        )
    lines.append("")


def _ctx_comments(lines: list[str], comments: list[Comment], now: int) -> None:
    """Newest ``_CTX_MAX_COMMENTS`` comments. The explicit "comment from
    worker" framing stops an operator-controlled HERMES_PROFILE like
    "hermes-system" being read as a system directive above an
    attacker-influenceable body (defense-in-depth)."""
    shown, omitted_note = _ctx_tail(comments, _CTX_MAX_COMMENTS, "comment")
    if not shown:
        return
    lines.append("## Comment thread")
    if omitted_note:
        lines.append(omitted_note)
    for c in shown:
        # Render author with explicit "comment from worker" framing so operator-controlled HERMES_PROFILE
        # values like "hermes-system" or "operator" can't be misread by the next worker as a system
        # directive above the (attacker-influenceable) comment body. Defense-in-depth — the LLM-controlled
        # author-forgery surface was already closed in #22435. See #22452.
        safe_author = (c.author or "").replace("`", "")
        lines.append(f"comment from worker `{safe_author}` at {_ctx_stamp(c.created_at, now)}:")
        lines.append(_ctx_cap(c.body, _CTX_MAX_COMMENT_BYTES))
        lines.append("")
