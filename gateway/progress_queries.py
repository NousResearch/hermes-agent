"""Deterministic, read-only Kanban progress replies for trusted chat sources.

This is deliberately a gateway edge helper, rather than a model prompt or a
Kanban tool.  It reads only cards that explicitly subscribed the current
platform/chat/thread source and produces a bounded response without creating
work, following paths, or interpreting prose as authority.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from gateway.configured_board import configured_board_db_path


MAX_PROGRESS_RESPONSE_CHARS = 1_600
_MAX_GRAPH_TASKS = 24
_MAX_NEXT_TASKS = 3
_MAX_AMBIGUOUS_ROOTS = 3
_MAX_RECEIPT_CHARS = 280
_MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024
_SNAPSHOT_COPY_CHUNK_BYTES = 1024 * 1024
_SNAPSHOT_ATTEMPTS = 3
_BOARD_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_TASK_ID_RE = re.compile(r"\bt_[a-z0-9]+\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]{3,}")
_SECRET_KEY_PATTERN = (
    r"[a-z0-9_-]*(?:(?:access|private)[_-]?key|secret|token|password|credential|"
    r"api[_-]?key|database[_-]?url|connection[_-]?string|authorization)[a-z0-9_-]*"
)
_SECRET_LINE_RE = re.compile(rf"(?i)^(\s*(?:{_SECRET_KEY_PATTERN})\s*(?:=|:)\s*).*$")
_SECRET_RE = re.compile(
    rf"(?i)\b(?:{_SECRET_KEY_PATTERN})"
    r"\s*(?:=|:)\s*(?:bearer\s+)?[^\n]*"
)
_BARE_BEARER_RE = re.compile(r"(?i)\bbearer(?:\s+[^\s,;]+)?")
_URI_USERINFO_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/@\s:]+:[^/@\s]+@")
_GITHUB_TOKEN_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{8,}\b")
_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----",
    re.IGNORECASE,
)
_PATH_RE = re.compile(
    r"(?:(?:[A-Za-z]:)?[\\/](?:Users|home|private|tmp|var|etc|root|opt)[^\s,;]*)"
)
_PROSE_COMMIT_RE = re.compile(r"\b[0-9a-f]{7,64}\b", re.IGNORECASE)
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$", re.IGNORECASE)
_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,119}$")
_STOP_WORDS = frozenset({
    "about",
    "also",
    "and",
    "could",
    "did",
    "does",
    "else",
    "from",
    "have",
    "how",
    "need",
    "progress",
    "project",
    "status",
    "that",
    "the",
    "this",
    "what",
    "when",
    "where",
    "with",
    "work",
    "would",
    "your",
})
_TOPIC_WORD = r"[a-z0-9][a-z0-9_'./:-]*"
_TOPIC = rf"{_TOPIC_WORD}(?:\s+{_TOPIC_WORD})*"
_STRUCTURAL_TOPIC_TAIL_RE = re.compile(r"(?:^|\s)(?:and|while|then|so|to)(?:\s|$)")
_READ_ONLY_PROGRESS_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        rf"how did (?P<topic>{_TOPIC}) go(?: and what else do we need to do)?\?",
        rf"how(?:'s| is| are) (?P<topic>{_TOPIC}) going(?:,? (?:and )?how much "
        rf"(?:more )?(?:work )?do we (?:have|need) to do(?: (?:until|till) we can "
        rf"(?:send|open|submit) (?:a )?(?:pr|pull request))?)?\??",
        rf"(?:can|could|would) you update me on (?P<topic>{_TOPIC}) (?:progress|status)\?",
        rf"give me an update on (?P<topic>{_TOPIC})",
        rf"what remains (?P<topic>{_TOPIC})\?",
        rf"what's left (?P<topic>{_TOPIC})\?",
        rf"where are we (?P<topic>{_TOPIC})\?",
        rf"status of (?P<topic>{_TOPIC})\?",
        rf"progress on (?P<topic>{_TOPIC})\?",
        rf"is (?P<topic>{_TOPIC}) complete\?",
    )
)
_GENERIC_PROGRESS_RE = re.compile(r"^how did it go\?$", re.IGNORECASE)
_NONTERMINAL_STATUSES = frozenset({"triage", "scheduled", "todo", "ready", "review"})
_FAILED_RUN_STATES = frozenset({
    "spawn_failed",
    "gave_up",
    "failed",
    "timed_out",
    "crashed",
})


@dataclass(frozen=True)
class ProgressSource:
    """Trusted ingress identity used for exact subscription matching."""

    platform: str
    chat_id: str
    thread_id: Optional[str] = None
    reply_to_message_id: Optional[str] = None


@dataclass(frozen=True)
class ProgressQueryResult:
    """A handled reply is safe to send without invoking the normal agent."""

    handled: bool
    response: str
    reason: str


@dataclass(frozen=True)
class _ParsedProgressQuery:
    """A recognized status template and its untrusted topic text."""

    normalized: str
    topic: Optional[str]


class _SnapshotChangedError(OSError):
    """The captured source identity or byte extent changed during copy."""


def _parse_progress_query(request: object) -> Optional[_ParsedProgressQuery]:
    """Parse a complete status template without deciding topic authority."""
    if not isinstance(request, str):
        return None
    normalized = " ".join(request.casefold().replace("’", "'").split())
    if not normalized or len(normalized) > 4_000:
        return None
    if _GENERIC_PROGRESS_RE.fullmatch(normalized):
        return _ParsedProgressQuery(normalized, None)
    for pattern in _READ_ONLY_PROGRESS_PATTERNS:
        match = pattern.fullmatch(normalized)
        if match:
            topic = match.group("topic")
            if _STRUCTURAL_TOPIC_TAIL_RE.search(topic):
                return None
            return _ParsedProgressQuery(normalized, topic)
    return None


def is_progress_query(request: object) -> bool:
    """Recognize complete project-status templates, independent of authority."""
    return _parse_progress_query(request) is not None


def _safe_text(value: object, *, limit: int = _MAX_RECEIPT_CHARS) -> str:
    raw = str(value or "").replace("\x00", " ")
    try:
        from agent.redact import redact_sensitive_text

        canonical = redact_sensitive_text(
            raw,
            force=True,
            redact_url_credentials=True,
        )
    except Exception:
        return "[redacted]"
    if not isinstance(canonical, str):
        return "[redacted]"
    raw = _GITHUB_TOKEN_RE.sub("[redacted]", canonical)
    raw = _PRIVATE_KEY_BLOCK_RE.sub("[redacted]", raw)
    redacted_lines = []
    for line in raw.splitlines() or [raw]:
        line = _SECRET_LINE_RE.sub(lambda match: match.group(1) + "[redacted]", line)
        line = _SECRET_RE.sub("[redacted]", line)
        line = _BARE_BEARER_RE.sub("[redacted]", line)
        line = _URI_USERINFO_RE.sub(r"\1[userinfo redacted]@", line)
        redacted_lines.append(line)
    text = " ".join("\n".join(redacted_lines).split())
    text = _PATH_RE.sub("[path redacted]", text)
    # A hash mentioned in a free-form worker receipt is not provenance. Only
    # the explicit structured metadata fields below may identify a commit.
    text = _PROSE_COMMIT_RE.sub("[unverified commit]", text)
    return text[:limit].rstrip()


def _safe_structured_identifier(value: object) -> Optional[str]:
    candidate = str(value or "").strip()
    if (
        not candidate
        or _GITHUB_TOKEN_RE.search(candidate)
        or _PRIVATE_KEY_BLOCK_RE.search(candidate)
    ):
        return None
    try:
        from agent.redact import redact_sensitive_text

        canonical = redact_sensitive_text(
            candidate,
            force=True,
            redact_url_credentials=True,
        )
    except Exception:
        return None
    return candidate if canonical == candidate else None


def _safe_commit(value: object) -> Optional[str]:
    candidate = _safe_structured_identifier(value)
    if candidate is None:
        return None
    return candidate if _COMMIT_RE.fullmatch(candidate) else None


def _safe_branch(value: object) -> Optional[str]:
    candidate = _safe_structured_identifier(value)
    if candidate is None:
        return None
    if (
        not _BRANCH_RE.fullmatch(candidate)
        or ".." in candidate
        or candidate.startswith("/")
    ):
        return None
    return candidate


def _source_subscription_rows(conn, source: ProgressSource) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT task_id, delivery_metadata FROM kanban_notify_subs
         WHERE platform = ? AND chat_id = ? AND thread_id = ?
         ORDER BY task_id ASC
        """,
        (source.platform, source.chat_id, source.thread_id or ""),
    ).fetchall()
    return list(rows)


def _regular_file_signature(path: Path) -> tuple[int, int, int, int]:
    file_stat = path.lstat()
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise OSError("Kanban snapshot source must be a regular non-symlink file")
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
    )


def _optional_file_signature(path: Path) -> Optional[tuple[int, int, int, int]]:
    try:
        return _regular_file_signature(path)
    except FileNotFoundError:
        return None


def _snapshot_source_state(
    source: Path,
) -> tuple[tuple[int, int, int, int], Optional[tuple[int, int, int, int]]]:
    return _regular_file_signature(source), _optional_file_signature(
        Path(str(source) + "-wal")
    )


def _copy_snapshot_file(
    source: Path,
    destination: Path,
    expected: tuple[int, int, int, int],
) -> None:
    """Copy the captured byte extent, then perform one bounded EOF probe."""
    expected_size = expected[2]
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise _SnapshotChangedError(
            "Kanban snapshot source changed before copy"
        ) from exc
    try:
        opened_stat = os.fstat(descriptor)
        opened_signature = (
            opened_stat.st_dev,
            opened_stat.st_ino,
            opened_stat.st_size,
            opened_stat.st_mtime_ns,
        )
        if opened_signature != expected or not stat.S_ISREG(opened_stat.st_mode):
            raise _SnapshotChangedError("Kanban snapshot source changed before copy")
        remaining = expected_size
        with destination.open("xb") as writer:
            while remaining:
                chunk = os.read(descriptor, min(_SNAPSHOT_COPY_CHUNK_BYTES, remaining))
                if not chunk:
                    raise _SnapshotChangedError(
                        "Kanban snapshot source changed during copy"
                    )
                if writer.write(chunk) != len(chunk):
                    raise OSError("Kanban snapshot destination write was incomplete")
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise _SnapshotChangedError(
                    "Kanban snapshot source changed during copy"
                )

        closed_stat = os.fstat(descriptor)
        closed_signature = (
            closed_stat.st_dev,
            closed_stat.st_ino,
            closed_stat.st_size,
            closed_stat.st_mtime_ns,
        )
        try:
            path_signature = _regular_file_signature(source)
        except OSError as exc:
            raise _SnapshotChangedError(
                "Kanban snapshot source changed after copy"
            ) from exc
        if closed_signature != expected or path_signature != expected:
            raise _SnapshotChangedError("Kanban snapshot source changed after copy")
    finally:
        os.close(descriptor)
    if destination.stat().st_size != expected_size:
        raise _SnapshotChangedError("Kanban snapshot source changed during copy")


@contextmanager
def _open_existing_board_readonly(board: str):
    """Query a validated private DB+WAL snapshot without opening the source DB."""
    source = configured_board_db_path(board)
    wal_source = Path(str(source) + "-wal")

    for _attempt in range(_SNAPSHOT_ATTEMPTS):
        before = _snapshot_source_state(source)
        total_size = before[0][2] + (before[1][2] if before[1] is not None else 0)
        if total_size > _MAX_SNAPSHOT_BYTES:
            raise OSError("Kanban snapshot exceeds the bounded copy limit")

        with tempfile.TemporaryDirectory(prefix="hermes-progress-") as temp_dir:
            try:
                snapshot = Path(temp_dir) / "kanban.db"
                _copy_snapshot_file(source, snapshot, before[0])
                if before[1] is not None:
                    _copy_snapshot_file(
                        wal_source, Path(str(snapshot) + "-wal"), before[1]
                    )
                if _snapshot_source_state(source) != before:
                    raise _SnapshotChangedError(
                        "Kanban board changed during snapshot copy"
                    )

            except _SnapshotChangedError:
                continue

            conn = sqlite3.connect(snapshot.as_uri() + "?mode=ro", uri=True)
            try:
                conn.row_factory = sqlite3.Row
                checks = conn.execute("PRAGMA quick_check").fetchall()
                if not checks or any(str(row[0]).casefold() != "ok" for row in checks):
                    raise sqlite3.DatabaseError(
                        "Kanban snapshot integrity check failed"
                    )
                yield conn
                return
            finally:
                conn.close()
    raise OSError("Kanban board changed during bounded snapshot attempts")


def _roots_for_tasks(conn, task_ids: Iterable[str]) -> list[str]:
    """Walk parent links so child subscriptions still recall their root graph."""
    from hermes_cli import kanban_db as kb

    roots: set[str] = set()
    for task_id in task_ids:
        seen: set[str] = set()
        frontier = [task_id]
        terminal: set[str] = set()
        while frontier and len(seen) < _MAX_GRAPH_TASKS:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            parents = kb.parent_ids(conn, current)
            if not parents:
                terminal.add(current)
            else:
                frontier.extend(parents)
        roots.update(terminal)
    return sorted(roots)


def _graph_task_ids(conn, root_id: str) -> tuple[list[str], bool]:
    """Return bounded graph IDs and whether unvisited descendants remain."""
    from hermes_cli import kanban_db as kb

    ordered: list[str] = []
    seen: set[str] = set()
    frontier = [root_id]
    while frontier and len(ordered) < _MAX_GRAPH_TASKS:
        task_id = frontier.pop(0)
        if task_id in seen:
            continue
        seen.add(task_id)
        ordered.append(task_id)
        frontier.extend(kb.child_ids(conn, task_id))
    truncated = any(task_id not in seen for task_id in frontier)
    return ordered, truncated


def _query_terms(request: str) -> set[str]:
    without_card_ids = _TASK_ID_RE.sub(" ", request.casefold())
    without_card_ids = re.sub(r"\bburn\s+down(s?)\b", r"burndown\1", without_card_ids)
    return {
        word for word in _WORD_RE.findall(without_card_ids) if word not in _STOP_WORDS
    }


def _root_has_trusted_linkage(
    conn,
    root_id: str,
    source: ProgressSource,
    subscription_rows: list[sqlite3.Row],
) -> bool:
    """Bind a generic follow-up to structured reply provenance only."""
    from hermes_cli import kanban_db as kb

    graph_ids = set(_graph_task_ids(conn, root_id)[0])
    reply_id = str(source.reply_to_message_id or "")
    if not reply_id:
        return False
    for row in subscription_rows:
        if str(row["task_id"]) not in graph_ids:
            continue
        try:
            metadata = json.loads(row["delivery_metadata"] or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        if isinstance(metadata, dict) and any(
            str(metadata.get(key) or "") == reply_id
            for key in (
                "origin_message_id",
                "message_id",
                "reply_to_message_id",
                "prompt_message_id",
            )
        ):
            return True
    for task_id in graph_ids:
        task = kb.get_task(conn, task_id)
        key = str(task.idempotency_key or "") if task is not None else ""
        if (
            task is not None
            and task.created_by == "specialist-routing"
            and key.startswith("specialist-routing:")
            and key.rsplit(":", 1)[-1] == reply_id
        ):
            return True
    return False


def _select_root(
    conn,
    root_ids: list[str],
    query: _ParsedProgressQuery,
    *,
    source: ProgressSource,
    subscription_rows: list[sqlite3.Row],
):
    """Return one fully topic-bound root, or ambiguity. Never guess ties."""
    from hermes_cli import kanban_db as kb

    topic = query.topic or ""
    explicit_ids = {match.casefold() for match in _TASK_ID_RE.findall(topic)}
    terms = _query_terms(topic)
    ranked = []
    for root_id in root_ids:
        graph_ids, _truncated = _graph_task_ids(conn, root_id)
        tasks = [kb.get_task(conn, task_id) for task_id in graph_ids]
        tasks = [task for task in tasks if task is not None]
        graph_id_set = {task.id.casefold() for task in tasks}
        title_words = (
            set().union(*(_query_terms(task.title) for task in tasks))
            if tasks
            else set()
        )
        ids_explained = explicit_ids <= graph_id_set
        terms_explained = terms <= title_words
        score = len(terms) + (100 * len(explicit_ids))
        if not (score and ids_explained and terms_explained):
            score = 0
        if score:
            root = kb.get_task(conn, root_id)
            if root is not None:
                ranked.append((score, root.created_at, root))

    if not ranked:
        if query.topic is None:
            linked = [
                kb.get_task(conn, root_id)
                for root_id in root_ids
                if _root_has_trusted_linkage(conn, root_id, source, subscription_rows)
            ]
            linked = [task for task in linked if task is not None]
            if len(linked) == 1:
                return "resolved", linked[0]
            if len(linked) > 1:
                return "ambiguous", linked
        return "no_match", None
    top_score = max(item[0] for item in ranked)
    top = [item for item in ranked if item[0] == top_score]
    if len(top) != 1:
        return "ambiguous", [
            item[2] for item in sorted(top, key=lambda item: (-item[1], item[2].id))
        ]
    return "resolved", top[0][2]


def _task_receipt(
    conn, task
) -> tuple[Optional[tuple[int, int, int]], Optional[str], list[str]]:
    """Read the newest safe receipt key plus structured commit identities."""
    from hermes_cli import kanban_db as kb

    receipt = None
    receipt_key = None
    commits: list[str] = []
    branch = _safe_branch(task.branch_name)
    if branch:
        commits.append(f"branch {branch}")
    for run in kb.list_runs(conn, task.id):
        candidate = _safe_text(run.summary or run.error)
        key = (
            run.ended_at if run.ended_at is not None else run.started_at,
            run.started_at,
            run.id,
        )
        if candidate and (receipt_key is None or key > receipt_key):
            receipt = candidate
            receipt_key = key
        metadata = run.metadata if isinstance(run.metadata, dict) else {}
        commit = _safe_commit(metadata.get("commit") or metadata.get("commit_sha"))
        if commit:
            commits.append(f"commit {commit}")
        run_branch = _safe_branch(metadata.get("branch") or metadata.get("branch_name"))
        if run_branch:
            commits.append(f"branch {run_branch}")
    return receipt_key, receipt, list(dict.fromkeys(commits))


def _bounded(response: str) -> str:
    return (
        response
        if len(response) <= MAX_PROGRESS_RESPONSE_CHARS
        else response[:MAX_PROGRESS_RESPONSE_CHARS].rstrip()
    )


def _format_multiple_progress(conn, roots: Iterable[object]) -> str:
    """Summarize source-authorized matching roots without requiring card IDs."""
    selected = list(roots)[:_MAX_AMBIGUOUS_ROOTS]
    if not selected:
        return ""
    heading = f"I found {len(selected)} matching subscribed workstreams."
    per_root_limit = max(
        240,
        (MAX_PROGRESS_RESPONSE_CHARS - len(heading) - len(selected)) // len(selected),
    )
    summaries = []
    for root in selected:
        summary = _format_progress(conn, root)
        if len(summary) > per_root_limit:
            summary = summary[:per_root_limit].rstrip()
        summaries.append(summary)
    return _bounded(" ".join((heading, *summaries)))


def _failed_run_attempt_count(conn, tasks: Iterable[object]) -> int:
    """Count failed execution attempts once per run row, including retries.

    ``task_runs.outcome`` is the semantic authority when present; ``status``
    is its legacy fallback. A row whose outcome and status both name failure
    still contributes one attempt, while separate retry rows each contribute
    their own attempt.
    """
    from hermes_cli import kanban_db as kb

    failed = 0
    for task in tasks:
        for run in kb.list_runs(conn, task.id):
            state = str(run.outcome or run.status or "").casefold()
            failed += state in _FAILED_RUN_STATES
    return failed


def _format_progress(conn, root) -> str:
    from hermes_cli import kanban_db as kb

    graph_ids, truncated = _graph_task_ids(conn, root.id)
    tasks = [kb.get_task(conn, task_id) for task_id in graph_ids]
    tasks = [task for task in tasks if task is not None]
    completed = sum(task.status == "done" for task in tasks)
    failed_attempts = _failed_run_attempt_count(conn, tasks)
    blocked = sum(task.status == "blocked" for task in tasks)
    running = sum(task.status == "running" for task in tasks)
    next_tasks = [task for task in tasks if task.status in _NONTERMINAL_STATUSES]
    next_tasks.sort(key=lambda task: (task.priority * -1, task.created_at, task.id))

    failure_label = "failed attempt" if failed_attempts == 1 else "failed attempts"
    lines = [
        f"Progress for `{root.id}` — {_safe_text(root.title, limit=160)}: "
        f"{completed} completed, {failed_attempts} {failure_label}, "
        f"{blocked} blocked, {running} running."
    ]
    if truncated:
        lines.append(
            f"Scope: counts and receipt cover only the first {len(tasks)} tasks "
            "in this bounded graph traversal."
        )
    if next_tasks:
        next_text = "; ".join(
            f"`{task.id}` {_safe_text(task.title, limit=100)} ({task.status})"
            for task in next_tasks[:_MAX_NEXT_TASKS]
        )
        next_label = (
            f"Next (partial; first {len(tasks)} tasks only)" if truncated else "Next"
        )
        lines.append(f"{next_label}: {next_text}.")

    receipt = None
    identifiers: list[str] = []
    receipt_task_id = None
    receipt_key = None
    latest_comment = None
    for task in tasks:
        task_receipt_key, task_receipt, task_identifiers = _task_receipt(conn, task)
        if task_receipt and (receipt_key is None or task_receipt_key > receipt_key):
            receipt_key = task_receipt_key
            receipt = task_receipt
            receipt_task_id = task.id
        identifiers.extend(task_identifiers)
        for comment in kb.list_comments(conn, task.id):
            candidate_key = (comment.created_at, comment.id)
            if latest_comment is None or candidate_key > (
                latest_comment.created_at,
                latest_comment.id,
            ):
                latest_comment = comment
    if receipt and receipt_task_id:
        if truncated:
            lines.append(
                f"Newest receipt within first {len(tasks)} tasks for "
                f"`{receipt_task_id}`: {receipt}"
            )
        else:
            lines.append(f"Latest receipt for `{receipt_task_id}`: {receipt}")
    if identifiers:
        lines.append(
            "Structured metadata: " + ", ".join(dict.fromkeys(identifiers)) + "."
        )
    if latest_comment is not None:
        if truncated:
            lines.append(
                f"Newest note within first {len(tasks)} tasks: "
                + _safe_text(latest_comment.body)
                + "."
            )
        else:
            lines.append("Latest note: " + _safe_text(latest_comment.body) + ".")
    return _bounded(" ".join(lines))


def resolve_progress_query(
    request: object,
    *,
    source: ProgressSource,
    board: object,
) -> ProgressQueryResult:
    """Resolve a project-status question from one configured board, or fall through.

    The query executes no model/tool/git action and makes no Kanban mutation.
    Only a valid configured board and exact trusted source subscription enter the
    lookup path. Database failures deliberately preserve ordinary chat.
    """
    parsed = _parse_progress_query(request)
    if parsed is None:
        return ProgressQueryResult(False, "", "irrelevant")
    if (
        not isinstance(board, str)
        or not _BOARD_RE.fullmatch(board)
        or not source.platform
        or not source.chat_id
    ):
        return ProgressQueryResult(False, "", "unavailable")
    try:
        with _open_existing_board_readonly(board) as conn:
            subscription_rows = _source_subscription_rows(conn, source)
            subscribed_ids = [str(row["task_id"]) for row in subscription_rows]
            root_ids = _roots_for_tasks(conn, subscribed_ids)
            if not root_ids:
                return ProgressQueryResult(False, "", "no_match")
            selection, value = _select_root(
                conn,
                root_ids,
                parsed,
                source=source,
                subscription_rows=subscription_rows,
            )
            if selection == "no_match":
                return ProgressQueryResult(False, "", "no_match")
            if selection == "ambiguous":
                response = _format_multiple_progress(conn, value)
                from gateway.vault_reports import append_vault_context, load_live_config

                return ProgressQueryResult(
                    True,
                    append_vault_context(
                        load_live_config(),
                        response,
                        board=board,
                        root_task_ids=[task.id for task in value],
                    ),
                    "resolved_multiple",
                )
            response = _format_progress(conn, value)
            from gateway.vault_reports import append_vault_context, load_live_config

            return ProgressQueryResult(
                True,
                append_vault_context(
                    load_live_config(), response, board=board, root_task_ids=[value]
                ),
                "resolved",
            )
    except Exception:
        return ProgressQueryResult(False, "", "unavailable")
