"""Coding verification evidence ledger: records what the agent actually proved in
a code workspace. Deliberately passive — it never runs a suite, never blocks
completion, and never upgrades targeted checks into "repo green"."""

from __future__ import annotations

import json
import re
import shlex
import sqlite3
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, UTC
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home


_DB_LOCK = threading.Lock()
_MAX_OUTPUT_SUMMARY_CHARS = 2000
_MAX_EVIDENCE_AGE_DAYS = 30
_MAX_EVENTS_PER_SESSION_ROOT = 100
_MAX_TOTAL_UNREFERENCED_EVENTS = 10_000
_AD_HOC_SCRIPT_NAME_PREFIXES = ("hermes-verify-", "hermes-ad-hoc-")
_VERIFY_SCHEMA_VERSION = 1
# Latest verify-command run per (session, root) whose own exit status the shell hid, with the
# time it ran. Read by the stop nudge in this same process, so it stays in memory; bounded for
# long-lived gateways.
_UNATTRIBUTED_RUNS: dict[tuple[str, str], tuple[str, str]] = {}
_MAX_UNATTRIBUTED_RUNS = 256

_INTERPRETERS = {"python", "python3", "py", "node", "bash", "sh", "ruby", "perl"}
# Windows spells the same interpreters `python.exe` / `py.exe` (a venv's absolute Scripts path).
_WINDOWS_EXE_SUFFIX_RE = re.compile(r"\.(?:exe|bat|cmd)$", re.IGNORECASE)
# The same interpreter is reached as `python3`, `python3.12`, `/usr/bin/python3.12`, or
# `env python3`. Matching only the bare token recorded no evidence for the invocation shapes
# the verify-on-stop nudge itself hands the agent, so a passing run left the workspace
# "unverified" and the nudge returned every turn.
_INTERPRETER_NAME_RE = re.compile(
    r"^(?:" + "|".join(sorted(_INTERPRETERS, key=len, reverse=True)) + r")(?:[0-9]+(?:\.[0-9]+)*)?$"
)
_TARGET_EXTENSIONS = (".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go", ".java")
_TARGET_PREFIXES = ("test_", "tests", "spec", "__tests__")
# Ordered: first matching keyword group wins; "check" only counts when the
# command is not itself a test command; anything else is a test.
_KIND_KEYWORDS = (
    (("lint", "eslint", "ruff"), "lint"),
    (("typecheck", "tsc", "mypy", "pyright", "ty"), "typecheck"), (("build",), "build"),
    (("fmt", "format"), "format"),
)
_PYTEST_SPELLINGS = (
    ["python", "-m", "pytest"], ["python3", "-m", "pytest"],
    ["uv", "run", "pytest"], ["poetry", "run", "pytest"], ["pipenv", "run", "pytest"],
)
_SCHEMA_DDL = (
    """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
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
        """,
    """
        CREATE TABLE IF NOT EXISTS verification_state (
            session_id TEXT NOT NULL,
            root TEXT NOT NULL,
            last_event_id INTEGER,
            last_edit_at TEXT,
            changed_paths_json TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY (session_id, root)
        )
        """,
    """
        CREATE INDEX IF NOT EXISTS idx_verification_events_session_root
        ON verification_events(session_id, root, id DESC)
        """,
)


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
    return datetime.now(UTC).isoformat()


def _db_path() -> Path:
    return get_hermes_home() / "verification_evidence.db"


def _ledger_enabled() -> bool:
    """The ledger exists only to feed verify-on-stop; when that guard is off nothing may
    record, read, or even create the database (an unconsumed ledger is pure disk churn)."""
    from agent.verification_stop import verify_on_stop_enabled

    return verify_on_stop_enabled()


def _connect() -> sqlite3.Connection:
    from hermes_cli.sqlite_util import open_db

    return open_db(_db_path(), db_label="verification_evidence.db", initialize=_ensure_schema)


def _transaction():
    from hermes_cli.sqlite_util import transaction

    return transaction(_connect())


def _ensure_schema(conn: sqlite3.Connection) -> None:
    for ddl in _SCHEMA_DDL:
        conn.execute(ddl)
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(_VERIFY_SCHEMA_VERSION),),
    )
    conn.commit()


def _split_shell_segments(command: str, *, posix: bool = True) -> list[_ShellSegment]:
    """Tokenize top-level shell commands while preserving their control operators.

    Returns ``[]`` for anything unparseable (unbalanced quotes, empty segment,
    trailing operator other than ``;``) so callers never match a partial parse.
    """
    raw_segments: list[tuple[str, str | None]] = []
    start = 0
    quote: str | None = None
    escaped = False
    index = 0

    while index < len(command):
        char = command[index]
        if escaped or (char == "\\" and quote != "'"):
            escaped = not escaped  # consume the escaped char / start an escape
            index += 1
            continue
        if quote or char in "'\"":
            quote = None if char == quote else (quote or char)  # close / stay / open
            index += 1
            continue

        operator = None
        if command.startswith(("&&", "||", "|&"), index):
            operator = command[index:index + 2]
        elif char == "\n":
            operator = ";"
        elif char in ";|" or (
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
    elif raw_segments and raw_segments[-1][1] != ";":
        return []

    try:
        segments = [_ShellSegment(shlex.split(raw, posix=posix), operator) for raw, operator in raw_segments]
    except ValueError:
        return []
    return segments if all(s.tokens for s in segments) else []


def _segment_status(
    segments: list[_ShellSegment], match_index: int, exit_code: int,
    pipeline_status: Optional[list[int]] = None,
) -> Optional[int]:
    """The matched segment's own exit status when the shell proves it, else ``None``.

    Only the last ``;``-separated sequence reports its status; backgrounding and ``||``
    hide it; an ``&&`` chain proves each member only on success. A pipeline hides every
    status but the last segment's, unless ``pipeline_status`` (the shell's PIPESTATUS
    array) is supplied and the match leads the sequence's last pipeline. Behind ``&&``
    that pipeline may never have run, leaving PIPESTATUS from an earlier one
    (``true | false && pytest | tail``); a short-circuited chain exits non-zero, so
    there the array counts only when the shell's own code is 0, as for plain ``&&``.
    """
    if not segments or not 0 <= match_index < len(segments) or any(s.following_operator == "&" for s in segments):
        return None

    sequence_start = max(
        (i + 1 for i, s in enumerate(segments[:-1]) if s.following_operator == ";"), default=0
    )
    if match_index < sequence_start:
        return None

    operators = {segment.following_operator for segment in segments[sequence_start:-1]}
    if "||" in operators:
        return None
    if operators & {"|", "|&"}:
        pipeline_start = max(
            (
                sequence_start + index + 1
                for index, operator in enumerate(
                    segment.following_operator for segment in segments[sequence_start:-1]
                )
                if operator not in {"|", "|&"}
            ),
            default=sequence_start,
        )
        chain_ran = "&&" not in operators or int(exit_code) == 0
        if chain_ran and pipeline_status and len(pipeline_status) >= 2 and match_index == pipeline_start:
            return pipeline_status[0]
        return None
    if not operators:
        return int(exit_code)
    if operators == {"&&"}:
        return int(exit_code) if int(exit_code) == 0 else None
    return None


def _canonical_tokens(canonical: str) -> list[str]:
    """Tokenize a canonical command, stripping leading ``./`` from each token."""
    try:
        return [re.sub(r"^(?:\./)+", "", t.strip()) for t in shlex.split(canonical) if t]
    except ValueError:
        return []


# Status-preserving wrappers -> (options that still run the child, options taking a value).
# Anything else (``--help``, ``--version``) may exit without running the child: no match.
_WRAPPER_OPTIONS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "timeout": (frozenset({"-p", "--preserve-status", "-f", "--foreground", "-v", "--verbose"}),
                frozenset({"-s", "--signal", "-k", "--kill-after"})),
    "nice": (frozenset(), frozenset({"-n", "--adjustment"})),
    "nohup": (frozenset(), frozenset()),
    "stdbuf": (frozenset(), frozenset({"-i", "-o", "-e", "--input", "--output", "--error"})),
}
# A ``timeout`` duration: ``300``, ``5m``, ``1.5h``.
_TIMEOUT_DURATION_RE = re.compile(r"^\d+(?:\.\d+)?[smhd]?$")


def _wrapper_option_width(token: str, flags: frozenset[str], valued: frozenset[str]) -> int:
    """Tokens one wrapper option spans (``-n 5`` 2, ``-n5``/``--signal=KILL`` 1), 0 if unknown."""
    if token in flags:
        return 1
    if token in valued:
        return 2
    attached = any(token.startswith(f"{v}=") if v.startswith("--") else token.startswith(v) for v in valued)
    return 1 if attached else 0


def _skip_wrapper(tokens: list[str], i: int) -> Optional[int]:
    """Index of the child command after the wrapper at ``tokens[i]``, ``None`` unless the
    wrapper's options and arguments are ones that run the child."""
    flags, valued = _WRAPPER_OPTIONS[tokens[i]]
    timeout = tokens[i] == "timeout"
    i += 1
    while i < len(tokens) and tokens[i].startswith("-") and tokens[i] != "--":
        width = _wrapper_option_width(tokens[i], flags, valued)
        if not width:
            return None
        i += width
    i += int(i < len(tokens) and tokens[i] == "--")
    if timeout:
        if i >= len(tokens) or not _TIMEOUT_DURATION_RE.match(tokens[i]):
            return None
        i += 1
    return i


def _strip_command_prefix(tokens: list[str]) -> list[str]:
    """Remove harmless command prefixes: env assignments, command/time/noglob, and
    status-preserving wrappers with the options and duration that still run the child."""
    i = 1 if tokens and tokens[0] == "env" else 0
    while i < len(tokens) and "=" in tokens[i] and not tokens[i].startswith("-"):
        i += 1
    while i < len(tokens):
        if tokens[i] in {"command", "time", "noglob"}:
            i += 1
        elif tokens[i] in _WRAPPER_OPTIONS and (child := _skip_wrapper(tokens, i)) is not None:
            i = child
        else:
            break
    return list(tokens[i:])


def _interpreter_family(token: str) -> Optional[str]:
    """Family name for an interpreter token wherever it lives, else ``None``.

    ``./venv/bin/python3.12``, ``python3`` and ``/usr/bin/python`` are all ``python``:
    the detected command is ``pytest`` and the agent runs it through a venv interpreter.
    """
    name = token.replace("\\", "/").rsplit("/", 1)[-1]
    name = _WINDOWS_EXE_SUFFIX_RE.sub("", re.sub(r"^(?:\./)+", "", name))
    if not _INTERPRETER_NAME_RE.match(name):
        return None
    family = re.sub(r"[0-9.]+$", "", name)
    return "python" if family == "py" else family


_INTERPRETER_FLAG_RE = re.compile(r"^-(?:[IBuEsSOd]+|3(?:\.\d+)?|[WX].*)$")


def _is_harmless_interpreter_flag(token: str) -> bool:
    return bool(_INTERPRETER_FLAG_RE.match(token))


def _normalize_tokens(tokens: list[str]) -> list[str]:
    """Comparison form: no leading ``./``, interpreters collapsed, Python's harmless flags
    dropped. Only Python's: the same letters mean other things elsewhere (``bash -s``
    reads commands from stdin and never runs the script named after it)."""
    normalized: list[str] = []
    i = 0
    while i < len(tokens):
        family = _interpreter_family(tokens[i])
        normalized.append(family or re.sub(r"^(?:\./)+", "", tokens[i]))
        i += 1
        if family == "python":
            while i < len(tokens) and _is_harmless_interpreter_flag(tokens[i]):
                i += 2 if tokens[i] in {"-W", "-X"} else 1
    return normalized


def _equivalent_needles(needle: list[str]) -> list[list[str]]:
    """Return command spellings equivalent to the detected canonical command."""
    candidates = [needle]
    if len(needle) >= 3 and needle[1] == "run" and needle[0] in {"npm", "pnpm", "yarn", "bun"}:
        candidates.append([needle[0], needle[2]])
    if len(needle) == 1 and "/" in needle[0]:
        candidates += [["bash", needle[0]], ["sh", needle[0]]]
    if needle == ["pytest"]:
        candidates += _PYTEST_SPELLINGS
    return candidates


def _find_canonical_match(
    command: str, canonical_commands: list[str], exit_code: int,
    pipeline_status: Optional[list[int]] = None, *, attributed: bool = True,
) -> Optional[tuple[str, list[str], Optional[int]]]:
    """Return ``(canonical, trailing_args, status)`` for the first detected command.

    ``attributed=False`` also matches a run whose own status the shell hid (status ``None``).
    """
    segments = _split_shell_segments(command)
    for canonical in canonical_commands:
        needle = _canonical_tokens(canonical)
        if not needle:
            continue
        for index, segment in enumerate(segments):
            candidate_tokens = _normalize_tokens(_strip_command_prefix(segment.tokens))
            status = _segment_status(segments, index, exit_code, pipeline_status)
            if status is None and attributed:
                continue
            for candidate in _equivalent_needles(needle):
                candidate = _normalize_tokens(candidate)
                if candidate_tokens[:len(candidate)] == candidate:
                    return canonical, candidate_tokens[len(candidate):], status
    return None


def _kind_for_command(canonical: str) -> str:
    lowered = canonical.lower()
    kind = next((k for words, k in _KIND_KEYWORDS if any(w in lowered for w in words)), None)
    return kind or ("check" if "check" in lowered and "test" not in lowered else "test")


def _looks_like_target(arg: str) -> bool:
    return bool(arg) and not arg.startswith("-") and "=" not in arg and (
        any(m in arg for m in ("/", "\\", "::")) or arg.endswith(_TARGET_EXTENSIONS) or arg.startswith(_TARGET_PREFIXES)
    )


def _is_under(token: str, base: str | Path | None) -> bool:
    """Whether absolute path ``token`` is ``base`` or lies beneath it (resolved)."""
    if not base:
        return False
    try:
        path = Path(token).expanduser()
        if not path.is_absolute():
            return False
        resolved, base_path = path.resolve(), Path(base).expanduser().resolve()
        return resolved == base_path or base_path in resolved.parents
    except Exception:
        return False


def _is_temp_script_path(token: str, root: str | Path | None) -> bool:
    """An ad-hoc verify script: prefixed name, under the temp dir, outside the repo."""
    try:
        name = Path(token).expanduser().name
    except Exception:
        return False
    return name.startswith(_AD_HOC_SCRIPT_NAME_PREFIXES) and _is_under(token, tempfile.gettempdir()) and not _is_under(token, root)


def _is_interpreter_token(token: str) -> bool:
    """Whether a command word names a shell interpreter.

    ``python3``, ``python3.12`` and ``/usr/bin/python3.12`` are the same interpreter; matching
    only the bare token left the ad-hoc branch blind to the invocation shapes the nudge hands
    the agent, so a passing run recorded no evidence and the workspace stayed ``unverified``.
    """
    # Basename by hand: on POSIX ``Path`` treats a Windows backslash path as one component.
    name = token.replace("\\", "/").rsplit("/", 1)[-1]
    return bool(_INTERPRETER_NAME_RE.match(_WINDOWS_EXE_SUFFIX_RE.sub("", name)))


def _ad_hoc_script_args(tokens: list[str], root: str | Path | None) -> Optional[list[str]]:
    candidate_tokens = _strip_command_prefix(tokens)
    if not candidate_tokens:
        return None
    command = candidate_tokens[0]
    if _is_temp_script_path(command, root):
        return candidate_tokens[1:]
    if Path(command).name == "env" and len(candidate_tokens) > 1 and _is_interpreter_token(candidate_tokens[1]):
        # `/usr/bin/env python3 script` names the same interpreter as `python3 script`.
        candidate_tokens, command = candidate_tokens[1:], candidate_tokens[1]
    if _is_interpreter_token(command):
        # Skip interpreter flags; the first positional must be the script.
        for idx, token in enumerate(candidate_tokens[1:], start=1):
            if _is_temp_script_path(token, root):
                return candidate_tokens[idx + 1:]
            if token != "--" and not token.startswith("-"):
                return None
    return None


def _find_ad_hoc_match(command: str, root: str | Path | None, exit_code: int = 0) -> Optional[list[str]]:
    # posix=False is retried so Windows backslash script paths survive splitting.
    for posix in (True, False):
        segments = _split_shell_segments(command, posix=posix)
        for index, segment in enumerate(segments):
            trailing_args = _ad_hoc_script_args(segment.tokens, root)
            if trailing_args is not None and _segment_status(segments, index, exit_code) is not None:
                return trailing_args
    return None


def _summarize_output(output: str) -> str:
    text = (output or "").strip()
    if len(text) <= _MAX_OUTPUT_SUMMARY_CHARS:
        return text
    head = _MAX_OUTPUT_SUMMARY_CHARS // 3
    return (
        f"{text[:head]}\n... [{len(text) - _MAX_OUTPUT_SUMMARY_CHARS} chars omitted] ...\n"
        f"{text[head - _MAX_OUTPUT_SUMMARY_CHARS:]}"
    )


def _prune_old_events(conn: sqlite3.Connection, *, session_id: str, root: str) -> None:
    """Bound ledger growth without deleting the current state pointer.

    Order matters: per-(session, root) cap, expire stale state rows, then expire
    old events and cap the total — never dropping an event still referenced
    by a ``verification_state.last_event_id``.
    """
    cutoff = (datetime.now(UTC) - timedelta(days=_MAX_EVIDENCE_AGE_DAYS)).isoformat()
    conn.execute(
        "DELETE FROM verification_events WHERE session_id = ? AND root = ? AND id NOT IN ("
        " SELECT id FROM verification_events WHERE session_id = ? AND root = ?"
        " ORDER BY id DESC LIMIT ?)",
        (session_id, root, session_id, root, _MAX_EVENTS_PER_SESSION_ROOT),
    )
    conn.execute(
        "DELETE FROM verification_state"
        " WHERE (last_edit_at IS NOT NULL AND last_edit_at < ?)"
        " OR (last_edit_at IS NULL AND last_event_id IN ("
        " SELECT id FROM verification_events WHERE created_at < ?))",
        (cutoff, cutoff),
    )
    conn.execute(
        "DELETE FROM verification_events WHERE created_at < ? AND id NOT IN ("
        " SELECT last_event_id FROM verification_state WHERE last_event_id IS NOT NULL)",
        (cutoff,),
    )
    conn.execute(
        "DELETE FROM verification_events WHERE id NOT IN ("
        " SELECT id FROM verification_events ORDER BY id DESC LIMIT ?)"
        " AND id NOT IN ("
        " SELECT last_event_id FROM verification_state WHERE last_event_id IS NOT NULL)",
        (_MAX_TOTAL_UNREFERENCED_EVENTS,),
    )


def _project_facts(cwd: str | Path | None) -> Optional[dict[str, Any]]:
    """Workspace facts for ``cwd``; ``None`` when detection fails or finds nothing."""
    try:
        from agent.coding_context import project_facts_for

        return project_facts_for(cwd)
    except Exception:
        return None


def _root_for(facts: dict[str, Any] | None, cwd: str | Path | None) -> str:
    return str((facts or {}).get("root") or Path(cwd or ".").resolve())


def _load_changed_paths(raw: Any) -> list[Any]:
    try:
        return json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []


def classify_verification_command(
    command: str, *, cwd: str | Path | None = None, session_id: str | None = None, exit_code: int = 0,
    output: str = "", pipeline_status: Optional[list[int]] = None,
) -> Optional[VerificationEvidence]:
    """Classify a terminal command as verification evidence, if applicable.

    ``pipeline_status`` is the shell's ``PIPESTATUS`` array: when the verifier is the
    leading command of a pipeline, its own status comes from there rather than from the
    shell's code (which belongs to the last filter).

    Ad-hoc temp scripts only count when the project has no canonical verify
    commands at all, so they never shadow a real suite.
    """
    if not command or not isinstance(command, str):
        return None
    facts = _project_facts(cwd)
    if not facts:
        return None
    return _classify_in_workspace(
        command, facts, cwd=cwd, session_id=session_id, exit_code=exit_code, output=output,
        pipeline_status=pipeline_status,
    )


def _classify_in_workspace(
    command: str, facts: dict[str, Any], *, cwd: str | Path | None, session_id: str | None,
    exit_code: int, output: str, pipeline_status: Optional[list[int]],
) -> Optional[VerificationEvidence]:
    verify_commands = list(facts.get("verifyCommands") or [])
    match = _find_canonical_match(command, verify_commands, int(exit_code), pipeline_status)
    if match is not None and match[2] is not None:
        canonical, trailing_args, status = match
        kind = _kind_for_command(canonical)
        scope = "targeted" if any(map(_looks_like_target, trailing_args)) else "full"
    else:
        if verify_commands or _find_ad_hoc_match(command, facts.get("root"), int(exit_code)) is None:
            return None
        canonical, kind, scope = "ad-hoc verification script", "ad_hoc", "targeted"
        status = int(exit_code)
    return VerificationEvidence(
        command=command, canonical_command=canonical, kind=kind, scope=scope,
        status="passed" if int(status) == 0 else "failed", exit_code=int(exit_code),
        cwd=str(Path(cwd or ".").resolve()), root=_root_for(facts, cwd),
        session_id=str(session_id or "default"), output_summary=_summarize_output(output),
    )


def record_terminal_result(
    *, command: str, cwd: str | Path | None, session_id: str | None, exit_code: int, output: str = "",
    pipeline_status: Optional[list[int]] = None,
) -> Optional[dict[str, Any]]:
    """Record a foreground terminal result when it is verification evidence.

    A run of a detected verify command whose own status the shell hid (``pytest || true``,
    a pipe on a backend without ``PIPESTATUS``) is remembered so the stop nudge can say why
    it did not count.
    """
    if not _ledger_enabled() or not command or not isinstance(command, str):
        return None
    facts = _project_facts(cwd)
    if not facts:
        return None
    key = (str(session_id or "default"), _root_for(facts, cwd))
    evidence = _classify_in_workspace(
        command, facts, cwd=cwd, session_id=session_id, exit_code=exit_code, output=output,
        pipeline_status=pipeline_status,
    )
    if evidence is not None:
        return _insert_evidence(evidence)
    verify_commands = list(facts.get("verifyCommands") or [])
    if _find_canonical_match(command, verify_commands, int(exit_code), attributed=False):
        _UNATTRIBUTED_RUNS.pop(key, None)
        _UNATTRIBUTED_RUNS[key] = (command, _utc_now())
        while len(_UNATTRIBUTED_RUNS) > _MAX_UNATTRIBUTED_RUNS:
            _UNATTRIBUTED_RUNS.pop(next(iter(_UNATTRIBUTED_RUNS)))
    return None


def last_unattributed_run(
    *, session_id: str | None, root: str | None, since: str | None = None,
) -> Optional[str]:
    """The latest verify-command run in ``root`` whose own exit status could not be read,
    unless evidence recorded at ``since`` came after it. ``hermes verify`` usually records
    from its own process, so this process's memory alone cannot know about that run.
    """
    run = _UNATTRIBUTED_RUNS.get((str(session_id or "default"), str(root)))
    if run is None or (since and since >= run[1]):
        return None
    return run[0]


def record_verify_run(
    *, root: str | Path, session_id: str | None = None, ok: bool, command: str = "hermes verify",
    scope: str = "full", output: str = "",
) -> Optional[dict[str, Any]]:
    """Record a completed ``hermes verify`` run as verification evidence.

    A pass marks the workspace ``passed`` for the verify-on-stop guard like a
    canonical test command would. ``root`` is re-resolved through project facts
    so it matches what :func:`verification_status` derives later.
    """
    if not _ledger_enabled():
        return None
    resolved = str(Path(root).resolve())
    return _insert_evidence(VerificationEvidence(
        command=command, canonical_command="hermes verify", kind="verify",
        scope=scope if scope in {"full", "targeted"} else "full",
        status="passed" if ok else "failed", exit_code=0 if ok else 1, cwd=resolved,
        root=str((_project_facts(root) or {}).get("root") or resolved),
        session_id=str(session_id or "default"), output_summary=_summarize_output(output),
    ))


def _insert_evidence(evidence: VerificationEvidence) -> dict[str, Any]:
    """Insert a classified evidence row and repoint the workspace state."""
    created_at = _utc_now()
    e = evidence
    _UNATTRIBUTED_RUNS.pop((e.session_id, e.root), None)
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            "INSERT INTO verification_events("
            " created_at, session_id, cwd, root, command, canonical_command,"
            " kind, scope, status, exit_code, output_summary"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (created_at, e.session_id, e.cwd, e.root, e.command, e.canonical_command, e.kind, e.scope,
             e.status, e.exit_code, e.output_summary),
        )
        if cur.lastrowid is None:
            raise RuntimeError("verification event insert did not return an id")
        event_id = int(cur.lastrowid)
        conn.execute(
            "INSERT INTO verification_state("
            " session_id, root, last_event_id, last_edit_at, changed_paths_json"
            ") VALUES (?, ?, ?, NULL, '[]')"
            " ON CONFLICT(session_id, root) DO UPDATE SET"
            " last_event_id = excluded.last_event_id,"
            " last_edit_at = NULL,"
            " changed_paths_json = '[]'",
            (e.session_id, e.root, event_id),
        )
        # A full passing ``hermes verify`` proves the workspace. Partial or
        # failed runs must not wipe another session's edit record (#103650).
        if e.kind == "verify" and e.status == "passed" and e.scope == "full":
            conn.execute(
                "UPDATE verification_state"
                " SET last_event_id = ?, last_edit_at = NULL, changed_paths_json = '[]'"
                " WHERE root = ? AND session_id != ?",
                (event_id, e.root, e.session_id),
            )
        _prune_old_events(conn, session_id=e.session_id, root=e.root)
        conn.commit()

    return {"id": event_id, **e.__dict__, "created_at": created_at}


def mark_workspace_edited(
    *, session_id: str | None, cwd: str | Path | None, paths: list[str] | tuple[str, ...] | None = None
) -> Optional[dict[str, Any]]:
    """Mark verification evidence stale after a successful file edit."""
    if not _ledger_enabled():
        return None
    facts = _project_facts(cwd)
    if not facts:
        return None

    sid = str(session_id or "default")
    root = _root_for(facts, cwd)
    changed_paths = sorted({str(p) for p in (paths or []) if p})
    edited_at = _utc_now()

    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT changed_paths_json FROM verification_state WHERE session_id = ? AND root = ?",
            (sid, root),
        ).fetchone()
        # Merge with what was already recorded, bounded to the last 200 paths.
        existing = _load_changed_paths(row["changed_paths_json"]) if row is not None else []
        merged = sorted(set(existing) | set(changed_paths))[-200:]
        conn.execute(
            "INSERT INTO verification_state("
            " session_id, root, last_event_id, last_edit_at, changed_paths_json"
            ") VALUES (?, ?, NULL, ?, ?)"
            " ON CONFLICT(session_id, root) DO UPDATE SET"
            " last_edit_at = excluded.last_edit_at,"
            " changed_paths_json = excluded.changed_paths_json",
            (sid, root, edited_at, json.dumps(merged)),
        )
        conn.commit()

    return {"session_id": sid, "root": root, "last_edit_at": edited_at, "changed_paths": changed_paths}


def _newest_workspace_verify(
    conn: sqlite3.Connection, *, root: str, since: str | None
) -> Optional[sqlite3.Row]:
    """Latest full passing ``hermes verify`` for ``root``, optionally not older than ``since``."""
    if since:
        return conn.execute(
            "SELECT * FROM verification_events"
            " WHERE root = ? AND kind = 'verify' AND status = 'passed' AND scope = 'full'"
            " AND created_at >= ?"
            " ORDER BY id DESC LIMIT 1",
            (root, since),
        ).fetchone()
    return conn.execute(
        "SELECT * FROM verification_events"
        " WHERE root = ? AND kind = 'verify' AND status = 'passed' AND scope = 'full'"
        " ORDER BY id DESC LIMIT 1",
        (root,),
    ).fetchone()


def verification_status(*, session_id: str | None, cwd: str | Path | None) -> dict[str, Any]:
    """Return the best known verification state for a session/workspace.

    Evidence recorded before the latest edit is reported as ``stale``. A
    ``hermes verify`` event for the same ``root`` that is at least as new as
    that edit satisfies the guard even if another session recorded it.
    """
    if not _ledger_enabled():
        return {"status": "disabled", "evidence": None}
    facts = _project_facts(cwd)
    if not facts:
        return {"status": "not_applicable", "evidence": None}

    sid = str(session_id or "default")
    root = _root_for(facts, cwd)
    with _DB_LOCK, _transaction() as conn:
        state = conn.execute(
            "SELECT last_event_id, last_edit_at, changed_paths_json"
            " FROM verification_state WHERE session_id = ? AND root = ?",
            (sid, root),
        ).fetchone()
        if state is None:
            return {"status": "unverified", "evidence": None, "root": root, "session_id": sid, "changed_paths": []}
        event = None
        if state["last_event_id"] is not None:
            event = conn.execute("SELECT * FROM verification_events WHERE id = ?", (state["last_event_id"],)).fetchone()
        last_edit_at = state["last_edit_at"]
        stale = bool(event is not None and last_edit_at and last_edit_at > event["created_at"])
        if event is None or stale:
            shared = _newest_workspace_verify(conn, root=root, since=last_edit_at)
            if shared is not None:
                event = shared
                stale = False

        result = {
            "evidence": None, "root": root, "session_id": sid,
            "changed_paths": _load_changed_paths(state["changed_paths_json"]),
        }
        if event is None:
            return {"status": "unverified", **result}

        evidence = dict(event)
        return {"status": "stale" if stale else evidence["status"], **result, "evidence": evidence}
