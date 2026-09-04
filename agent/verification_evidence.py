"""Coding verification evidence ledger.

This module records what the agent actually proved while working in a code
workspace. It is deliberately passive: it never decides to run a suite, never
blocks completion, and never upgrades targeted checks into "repo green".
"""

from __future__ import annotations

import json
import shlex
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from hermes_constants import get_hermes_home


_DB_LOCK = threading.Lock()
_MAX_OUTPUT_SUMMARY_CHARS = 2000
_MAX_EVIDENCE_AGE_DAYS = 30
_MAX_EVENTS_PER_SESSION_ROOT = 100
_MAX_TOTAL_UNREFERENCED_EVENTS = 10_000
_AD_HOC_SCRIPT_NAME_PREFIXES = ("hermes-verify-", "hermes-ad-hoc-")
_VERIFY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _ShellSegment:
    tokens: list[str]
    following_operator: str | None = None


@dataclass(frozen=True)
class VerificationEvidence:
    """A classified command result worth recording."""

    command: str
    canonical_command: str
    kind: str
    scope: str
    status: str
    exit_code: int
    cwd: str
    root: str
    session_id: str
    output_summary: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _retention_cutoff() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=_MAX_EVIDENCE_AGE_DAYS)).isoformat()


def _db_path() -> Path:
    return get_hermes_home() / "verification_evidence.db"


def _connect() -> sqlite3.Connection:
    from hermes_state import apply_wal_with_fallback

    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        apply_wal_with_fallback(conn, db_label="verification_evidence.db")
        conn.execute("PRAGMA busy_timeout=5000")
        _ensure_schema(conn)
    except Exception:
        # A PRAGMA/DDL failure after a successful connect() must not leak the
        # just-opened connection back to the caller.
        conn.close()
        raise
    return conn


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, and ALWAYS close it.

    ``sqlite3.Connection.__enter__``/``__exit__`` only commit or roll back the
    transaction; they do not close the connection. Using ``with _connect()``
    alone therefore leaks a connection — and its WAL/SHM file descriptors — on
    every call, deferring the close to the garbage collector, which over a
    long-running process can exhaust ``RLIMIT_NOFILE`` (the cron-ledger sibling
    of this bug was #69567 / PR #69594).
    """
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS verification_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            session_id TEXT NOT NULL,
            cwd TEXT NOT NULL,
            root TEXT NOT NULL,
            command TEXT NOT NULL,
            canonical_command TEXT NOT NULL,
            kind TEXT NOT NULL,
            scope TEXT NOT NULL,
            status TEXT NOT NULL,
            exit_code INTEGER NOT NULL,
            output_summary TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS verification_state (
            session_id TEXT NOT NULL,
            root TEXT NOT NULL,
            last_event_id INTEGER,
            last_edit_at TEXT,
            changed_paths_json TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY (session_id, root)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_verification_events_session_root
        ON verification_events(session_id, root, id DESC)
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(_VERIFY_SCHEMA_VERSION),),
    )
    conn.commit()


def _split_shell_segments(command: str, *, posix: bool = True) -> list[_ShellSegment]:
    """Tokenize top-level shell commands while preserving their control operators."""
    raw_segments: list[tuple[str, str | None]] = []
    start = 0
    quote: str | None = None
    escaped = False
    index = 0

    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if quote:
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue

        operator = None
        if command.startswith(("&&", "||", "|&"), index):
            operator = command[index:index + 2]
        elif char == "\n":
            operator = ";"
        elif char in ";|":
            operator = char
        elif (
            char == "&"
            and (index == 0 or command[index - 1] not in "<>")
            and not command.startswith(("&>", "&>>"), index)
        ):
            operator = char

        if operator is None:
            index += 1
            continue

        raw = command[start:index].strip()
        if not raw:
            return []
        raw_segments.append((raw, operator))
        index += 1 if char == "\n" else len(operator)
        start = index

    if quote or escaped:
        return []
    trailing = command[start:].strip()
    if trailing:
        raw_segments.append((trailing, None))
    elif raw_segments and raw_segments[-1][1] not in {";"}:
        return []

    segments: list[_ShellSegment] = []
    for raw, operator in raw_segments:
        try:
            tokens = shlex.split(raw, posix=posix)
        except ValueError:
            return []
        if not tokens:
            return []
        segments.append(_ShellSegment(tokens=tokens, following_operator=operator))
    return segments


def _exit_status_is_attributable(
    segments: list[_ShellSegment], match_index: int, exit_code: int
) -> bool:
    """Whether the shell's status proves the matched segment's own status."""
    if not segments or not 0 <= match_index < len(segments):
        return False
    if any(segment.following_operator == "&" for segment in segments):
        return False

    sequence_start = 0
    for index, segment in enumerate(segments[:-1]):
        if segment.following_operator == ";":
            sequence_start = index + 1
    if match_index < sequence_start:
        return False

    sequence = segments[sequence_start:]
    operators = [segment.following_operator for segment in sequence[:-1]]
    if any(operator in {"|", "|&", "||"} for operator in operators):
        return False
    if len(sequence) == 1:
        return True
    return int(exit_code) == 0 and all(operator == "&&" for operator in operators)


def _clean_token(token: str) -> str:
    token = token.strip()
    while token.startswith("./"):
        token = token[2:]
    return token


def _canonical_tokens(canonical: str) -> list[str]:
    try:
        return [_clean_token(t) for t in shlex.split(canonical) if t]
    except ValueError:
        return []


def _find_subsequence(tokens: list[str], needle: list[str]) -> Optional[int]:
    if not tokens or not needle or len(needle) > len(tokens):
        return None
    cleaned = [_clean_token(t) for t in tokens]
    for idx in range(0, len(cleaned) - len(needle) + 1):
        if cleaned[idx:idx + len(needle)] == needle:
            return idx
    return None


def _strip_command_prefix(tokens: list[str]) -> list[str]:
    """Remove harmless command prefixes before matching canonical commands."""
    remaining = list(tokens)
    if remaining and remaining[0] == "env":
        remaining = remaining[1:]
    while remaining and "=" in remaining[0] and not remaining[0].startswith("-"):
        remaining = remaining[1:]
    while remaining and remaining[0] in {"command", "time", "noglob"}:
        remaining = remaining[1:]
    return remaining


def _equivalent_needles(needle: list[str]) -> list[list[str]]:
    """Return command spellings equivalent to the detected canonical command."""
    candidates = [needle]
    if len(needle) >= 3 and needle[1] == "run":
        package_manager = needle[0]
        script_name = needle[2]
        if package_manager in {"npm", "pnpm", "yarn", "bun"}:
            candidates.append([package_manager, script_name])
    if len(needle) == 1 and "/" in needle[0]:
        candidates.extend([["bash", needle[0]], ["sh", needle[0]]])
    if needle == ["pytest"]:
        candidates.extend(
            [
                ["python", "-m", "pytest"],
                ["python3", "-m", "pytest"],
                ["uv", "run", "pytest"],
                ["poetry", "run", "pytest"],
                ["pipenv", "run", "pytest"],
            ]
        )
    return candidates


def _find_canonical_match(
    command: str,
    canonical_commands: list[str],
    exit_code: int,
) -> Optional[tuple[str, list[str]]]:
    """Return ``(canonical, trailing_args)`` for the first detected command."""

    segments = _split_shell_segments(command)
    for canonical in canonical_commands:
        needle = _canonical_tokens(canonical)
        if not needle:
            continue
        for index, segment in enumerate(segments):
            candidate_tokens = _strip_command_prefix(segment.tokens)
            for candidate in _equivalent_needles(needle):
                if (
                    candidate_tokens[:len(candidate)] == candidate
                    and _exit_status_is_attributable(segments, index, exit_code)
                ):
                    return canonical, candidate_tokens[len(candidate):]
    return None


def _kind_for_command(canonical: str) -> str:
    lowered = canonical.lower()
    if any(word in lowered for word in ("lint", "eslint", "ruff")):
        return "lint"
    if any(word in lowered for word in ("typecheck", "tsc", "mypy", "pyright", "ty")):
        return "typecheck"
    if "build" in lowered:
        return "build"
    if "fmt" in lowered or "format" in lowered:
        return "format"
    if "check" in lowered and "test" not in lowered:
        return "check"
    return "test"


def _looks_like_target(arg: str) -> bool:
    if not arg or arg.startswith("-") or "=" in arg:
        return False
    return (
        "/" in arg
        or "\\" in arg
        or "::" in arg
        or arg.endswith((".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go", ".java"))
        or arg.startswith(("test_", "tests", "spec", "__tests__"))
    )


def _scope_for_args(args: list[str]) -> str:
    return "targeted" if any(_looks_like_target(arg) for arg in args) else "full"


def _is_under_temp_dir(token: str) -> bool:
    if not token or token.startswith("-"):
        return False
    try:
        path = Path(token).expanduser()
        if not path.is_absolute():
            return False
        resolved = path.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        return resolved == temp_root or temp_root in resolved.parents
    except Exception:
        return False


def _is_under_root(token: str, root: str | Path | None) -> bool:
    if not root:
        return False
    try:
        path = Path(token).expanduser().resolve()
        root_path = Path(root).expanduser().resolve()
        return path == root_path or root_path in path.parents
    except Exception:
        return False


def _is_temp_script_path(token: str, root: str | Path | None) -> bool:
    try:
        name = Path(token).expanduser().name
    except Exception:
        return False
    return (
        name.startswith(_AD_HOC_SCRIPT_NAME_PREFIXES)
        and _is_under_temp_dir(token)
        and not _is_under_root(token, root)
    )


def _ad_hoc_script_args(tokens: list[str], root: str | Path | None) -> Optional[list[str]]:
    candidate_tokens = _strip_command_prefix(tokens)
    if not candidate_tokens:
        return None
    command = candidate_tokens[0]
    if _is_temp_script_path(command, root):
        return candidate_tokens[1:]
    if command in {"python", "python3", "node", "bash", "sh", "ruby", "perl"}:
        for idx, token in enumerate(candidate_tokens[1:], start=1):
            if token == "--":
                continue
            if _is_temp_script_path(token, root):
                return candidate_tokens[idx + 1:]
            if not token.startswith("-"):
                return None
    return None


def _find_ad_hoc_match(
    command: str,
    root: str | Path | None,
    exit_code: int = 0,
) -> Optional[list[str]]:
    # Try both posix=True (default) and posix=False (Windows backslash paths)
    # so ad-hoc verification scripts with backslash paths are matched on Windows.
    for posix in (True, False):
        segments = _split_shell_segments(command, posix=posix)
        for index, segment in enumerate(segments):
            trailing_args = _ad_hoc_script_args(segment.tokens, root)
            if trailing_args is not None and _exit_status_is_attributable(
                segments, index, exit_code
            ):
                return trailing_args
    return None


def _summarize_output(output: str) -> str:
    text = (output or "").strip()
    if len(text) <= _MAX_OUTPUT_SUMMARY_CHARS:
        return text
    head = _MAX_OUTPUT_SUMMARY_CHARS // 3
    tail = _MAX_OUTPUT_SUMMARY_CHARS - head
    return (
        text[:head]
        + f"\n... [{len(text) - _MAX_OUTPUT_SUMMARY_CHARS} chars omitted] ...\n"
        + text[-tail:]
    )


def _prune_old_events(conn: sqlite3.Connection, *, session_id: str, root: str) -> None:
    """Bound ledger growth without deleting the current state pointer."""
    cutoff = _retention_cutoff()
    conn.execute(
        """
        DELETE FROM verification_events
        WHERE session_id = ?
          AND root = ?
          AND id NOT IN (
              SELECT id FROM verification_events
              WHERE session_id = ? AND root = ?
              ORDER BY id DESC
              LIMIT ?
          )
        """,
        (session_id, root, session_id, root, _MAX_EVENTS_PER_SESSION_ROOT),
    )
    conn.execute(
        """
        DELETE FROM verification_state
        WHERE (
            last_edit_at IS NOT NULL
            AND last_edit_at < ?
        )
        OR (
            last_edit_at IS NULL
            AND last_event_id IN (
                SELECT id FROM verification_events
                WHERE created_at < ?
            )
        )
        """,
        (cutoff, cutoff),
    )
    conn.execute(
        """
        DELETE FROM verification_events
        WHERE created_at < ?
          AND id NOT IN (
              SELECT last_event_id FROM verification_state
              WHERE last_event_id IS NOT NULL
          )
        """,
        (cutoff,),
    )
    conn.execute(
        """
        DELETE FROM verification_events
        WHERE id NOT IN (
            SELECT id FROM verification_events
            ORDER BY id DESC
            LIMIT ?
        )
          AND id NOT IN (
              SELECT last_event_id FROM verification_state
              WHERE last_event_id IS NOT NULL
          )
        """,
        (_MAX_TOTAL_UNREFERENCED_EVENTS,),
    )


def classify_verification_command(
    command: str,
    *,
    cwd: str | Path | None = None,
    session_id: str | None = None,
    exit_code: int = 0,
    output: str = "",
) -> Optional[VerificationEvidence]:
    """Classify a terminal command as verification evidence, if applicable."""

    if not command or not isinstance(command, str):
        return None
    try:
        from agent.coding_context import project_facts_for

        facts = project_facts_for(cwd)
    except Exception:
        facts = None
    if not facts:
        return None

    verify_commands = list(facts.get("verifyCommands") or [])
    match = _find_canonical_match(command, verify_commands, int(exit_code))
    is_ad_hoc = False
    if match is None and not verify_commands:
        ad_hoc_args = _find_ad_hoc_match(command, facts.get("root"), int(exit_code))
        if ad_hoc_args is not None:
            match = ("ad-hoc verification script", ad_hoc_args)
            is_ad_hoc = True
    if match is None:
        return None

    canonical, trailing_args = match
    return VerificationEvidence(
        command=command,
        canonical_command=canonical,
        kind="ad_hoc" if is_ad_hoc else _kind_for_command(canonical),
        scope="targeted" if is_ad_hoc else _scope_for_args(trailing_args),
        status="passed" if int(exit_code) == 0 else "failed",
        exit_code=int(exit_code),
        cwd=str(Path(cwd or ".").resolve()),
        root=str(facts.get("root") or Path(cwd or ".").resolve()),
        session_id=str(session_id or "default"),
        output_summary=_summarize_output(output),
    )


def record_terminal_result(
    *,
    command: str,
    cwd: str | Path | None,
    session_id: str | None,
    exit_code: int,
    output: str = "",
) -> Optional[dict[str, Any]]:
    """Record a foreground terminal result when it is verification evidence."""

    evidence = classify_verification_command(
        command,
        cwd=cwd,
        session_id=session_id,
        exit_code=exit_code,
        output=output,
    )
    if evidence is None:
        return None
    return _insert_evidence(evidence)


def record_verify_run(
    *,
    root: str | Path,
    session_id: str | None = None,
    ok: bool,
    command: str = "hermes verify",
    scope: str = "full",
    output: str = "",
) -> Optional[dict[str, Any]]:
    """Record a completed ``hermes verify`` run as verification evidence.

    Explicit CLI-side write: unlike :func:`record_terminal_result` there is
    nothing to classify — the caller (the ``hermes verify`` command) already
    knows the run was a verification pass and whether it succeeded. A passing
    run marks the workspace ``passed`` for the verify-on-stop guard exactly
    like a passing canonical test command would; a failing run records the
    failure so the guard keeps asking for a fix.

    ``root`` is re-resolved through :func:`agent.coding_context.project_facts_for`
    so the recorded workspace root matches what :func:`verification_status`
    derives when the stop guard later looks the evidence up.
    """
    try:
        from agent.coding_context import project_facts_for

        facts = project_facts_for(root)
    except Exception:
        facts = None

    resolved = str(Path(root).resolve())
    evidence = VerificationEvidence(
        command=command,
        canonical_command="hermes verify",
        kind="verify",
        scope=scope if scope in {"full", "targeted"} else "full",
        status="passed" if ok else "failed",
        exit_code=0 if ok else 1,
        cwd=resolved,
        root=str((facts or {}).get("root") or resolved),
        session_id=str(session_id or "default"),
        output_summary=_summarize_output(output),
    )
    return _insert_evidence(evidence)


def _insert_evidence(evidence: VerificationEvidence) -> dict[str, Any]:
    """Insert a classified evidence row and repoint the workspace state."""
    created_at = _utc_now()
    with _DB_LOCK:
        with _transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO verification_events(
                    created_at, session_id, cwd, root, command, canonical_command,
                    kind, scope, status, exit_code, output_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    evidence.session_id,
                    evidence.cwd,
                    evidence.root,
                    evidence.command,
                    evidence.canonical_command,
                    evidence.kind,
                    evidence.scope,
                    evidence.status,
                    evidence.exit_code,
                    evidence.output_summary,
                ),
            )
            if cur.lastrowid is None:
                raise RuntimeError("verification event insert did not return an id")
            event_id = int(cur.lastrowid)
            conn.execute(
                """
                INSERT INTO verification_state(
                    session_id, root, last_event_id, last_edit_at, changed_paths_json
                ) VALUES (?, ?, ?, NULL, '[]')
                ON CONFLICT(session_id, root) DO UPDATE SET
                    last_event_id = excluded.last_event_id,
                    last_edit_at = NULL,
                    changed_paths_json = '[]'
                """,
                (evidence.session_id, evidence.root, event_id),
            )
            _prune_old_events(conn, session_id=evidence.session_id, root=evidence.root)
            conn.commit()

    return {"id": event_id, **evidence.__dict__, "created_at": created_at}


def mark_workspace_edited(
    *,
    session_id: str | None,
    cwd: str | Path | None,
    paths: list[str] | tuple[str, ...] | None = None,
) -> Optional[dict[str, Any]]:
    """Mark verification evidence stale after a successful file edit."""

    try:
        from agent.coding_context import project_facts_for

        facts = project_facts_for(cwd)
    except Exception:
        facts = None
    if not facts:
        return None

    sid = str(session_id or "default")
    root = str(facts.get("root") or Path(cwd or ".").resolve())
    changed_paths = sorted({str(p) for p in (paths or []) if p})
    edited_at = _utc_now()

    with _DB_LOCK:
        with _transaction() as conn:
            row = conn.execute(
                """
                SELECT changed_paths_json FROM verification_state
                WHERE session_id = ? AND root = ?
                """,
                (sid, root),
            ).fetchone()
            existing: set[str] = set()
            if row is not None:
                try:
                    existing = set(json.loads(row["changed_paths_json"] or "[]"))
                except (TypeError, ValueError):
                    existing = set()
            merged = sorted((existing | set(changed_paths)))[-200:]
            conn.execute(
                """
                INSERT INTO verification_state(
                    session_id, root, last_event_id, last_edit_at, changed_paths_json
                ) VALUES (?, ?, NULL, ?, ?)
                ON CONFLICT(session_id, root) DO UPDATE SET
                    last_edit_at = excluded.last_edit_at,
                    changed_paths_json = excluded.changed_paths_json
                """,
                (sid, root, edited_at, json.dumps(merged)),
            )
            conn.commit()

    return {"session_id": sid, "root": root, "last_edit_at": edited_at, "changed_paths": changed_paths}


def verification_status(
    *,
    session_id: str | None,
    cwd: str | Path | None,
) -> dict[str, Any]:
    """Return the best known verification state for a session/workspace."""

    try:
        from agent.coding_context import project_facts_for

        facts = project_facts_for(cwd)
    except Exception:
        facts = None
    if not facts:
        return {"status": "not_applicable", "evidence": None}

    sid = str(session_id or "default")
    root = str(facts.get("root") or Path(cwd or ".").resolve())
    with _DB_LOCK:
        with _transaction() as conn:
            state = conn.execute(
                """
                SELECT last_event_id, last_edit_at, changed_paths_json
                FROM verification_state
                WHERE session_id = ? AND root = ?
                """,
                (sid, root),
            ).fetchone()
            if state is None:
                return {
                    "status": "unverified",
                    "evidence": None,
                    "root": root,
                    "session_id": sid,
                    "changed_paths": [],
                }
            event = None
            if state["last_event_id"] is not None:
                event = conn.execute(
                    "SELECT * FROM verification_events WHERE id = ?",
                    (state["last_event_id"],),
                ).fetchone()

    changed_paths: list[str] = []
    try:
        changed_paths = json.loads(state["changed_paths_json"] or "[]")
    except (TypeError, ValueError):
        changed_paths = []

    if event is None:
        return {
            "status": "unverified",
            "evidence": None,
            "root": root,
            "session_id": sid,
            "changed_paths": changed_paths,
        }

    evidence = dict(event)
    if state["last_edit_at"] and state["last_edit_at"] > evidence["created_at"]:
        status = "stale"
    else:
        status = evidence["status"]
    return {
        "status": status,
        "evidence": evidence,
        "root": root,
        "session_id": sid,
        "changed_paths": changed_paths,
    }
