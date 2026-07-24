"""Persistent Claude Code sessions for the ``claude-code-sdk`` Hermes skill.

The skill uses Anthropic's official ``claude-agent-sdk`` package. Hermes
terminal calls run in separate processes, so this dispatcher stores only the
Claude session ID and resumes it on the next invocation; no daemon remains
running between queries.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping

try:
    import fcntl  # POSIX

    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

try:
    import msvcrt  # Windows

    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False


def _get_hermes_home(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: Path | None = None,
) -> Path:
    """Return the active Hermes profile directory.

    This mirrors ``hermes_constants.get_hermes_home()`` without importing the
    Hermes package, because a skill script can run from an isolated Python
    environment containing only ``claude-agent-sdk``.
    """
    env = os.environ if environ is None else environ
    configured = env.get("HERMES_HOME")
    if configured:
        return Path(configured).expanduser()

    current_platform = sys.platform if platform is None else platform
    user_home = Path.home() if home is None else home
    if current_platform == "win32":
        local_app_data = env.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "hermes"
        return user_home / "AppData" / "Local" / "hermes"
    return user_home / ".hermes"


HERMES_HOME = _get_hermes_home()
STATE_DIR = HERMES_HOME / "skill-state" / "claude-code-sdk"
SESSIONS_FILE = STATE_DIR / "sessions.json"
LOCK_FILE = STATE_DIR / ".sessions.lock"
SESSION_LOCK_DIR = STATE_DIR / ".session-locks"
COST_LOG = STATE_DIR / "cost.log"
IDLE_TTL_SECONDS = 3600
QUERY_TIMEOUT_SECONDS = 300


class SessionBusyError(RuntimeError):
    """Raised when another process is already querying the same handle."""


@dataclass
class SessionRecord:
    handle: str
    project_path: str
    created_at: str
    last_activity: str
    message_count: int = 0
    total_cost_usd: float = 0.0
    claude_session_id: str | None = None


@dataclass
class SessionStore:
    sessions: dict[str, SessionRecord] = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_state_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def _file_lock(path: Path, *, blocking: bool = True) -> Iterator[None]:
    """Acquire an advisory cross-process lock for ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(b"\0")
    fh = path.open("r+b")
    try:
        if _HAS_FCNTL:
            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(fh.fileno(), flags)
            except BlockingIOError as exc:
                raise SessionBusyError("lock is already held") from exc
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        elif _HAS_MSVCRT:
            fh.seek(0)
            mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            try:
                msvcrt.locking(fh.fileno(), mode, 1)
            except OSError as exc:
                raise SessionBusyError("lock is already held") from exc
            try:
                yield
            finally:
                fh.seek(0)
                with contextlib.suppress(OSError):
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            sys.stderr.write(
                json.dumps({
                    "warning": "no file-locking primitive is available; concurrent writes may race"
                })
                + "\n"
            )
            yield
    finally:
        fh.close()


def _store_lock() -> contextlib.AbstractContextManager[None]:
    """Serialise load-modify-save operations on the shared session store."""
    return _file_lock(LOCK_FILE)


def _session_lock(handle: str) -> contextlib.AbstractContextManager[None]:
    """Reject overlapping operations on one Claude conversation."""
    digest = hashlib.sha256(handle.encode("utf-8")).hexdigest()
    return _file_lock(SESSION_LOCK_DIR / f"{digest}.lock", blocking=False)


def _load_store() -> SessionStore:
    _ensure_state_dir()
    if not SESSIONS_FILE.exists():
        return SessionStore()
    try:
        data = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Corrupt sessions file at {SESSIONS_FILE}: {exc}") from exc
    sessions = {
        handle: SessionRecord(**record)
        for handle, record in data.get("sessions", {}).items()
    }
    return SessionStore(sessions=sessions)


def _save_store(store: SessionStore) -> None:
    _ensure_state_dir()
    payload = {
        "sessions": {
            handle: asdict(record) for handle, record in store.sessions.items()
        }
    }
    tmp = SESSIONS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(SESSIONS_FILE)


def _reap(store: SessionStore, ttl: int = IDLE_TTL_SECONDS) -> list[str]:
    now = datetime.now(timezone.utc)
    expired: list[str] = []
    for handle, record in list(store.sessions.items()):
        try:
            last = datetime.strptime(
                record.last_activity, "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if (now - last).total_seconds() > ttl:
            expired.append(handle)
    for handle in expired:
        store.sessions.pop(handle, None)
    return expired


def _append_cost_log(handle: str, cost_usd: float | None) -> None:
    if cost_usd is None:
        return
    _ensure_state_dir()
    with COST_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{_now_iso()}\t{handle}\t{cost_usd:.6f}\n")


def _import_sdk():
    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            ResultMessage,
            TextBlock,
        )
    except ImportError as exc:
        _die(
            "claude-agent-sdk is not installed. Run: "
            "python3 -m pip install --upgrade claude-agent-sdk\n"
            f"(import failed with: {exc})",
            code=2,
        )
    return (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ResultMessage,
        TextBlock,
    )


async def _run_query(
    project_path: str,
    message: str,
    resume_id: str | None,
    *,
    max_turns: int | None = None,
    max_budget_usd: float | None = None,
) -> tuple[str, float | None, str | None]:
    """Run one query and return text, query cost, and Claude session ID."""
    (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ResultMessage,
        TextBlock,
    ) = _import_sdk()

    option_kwargs: dict[str, object] = {"cwd": project_path}
    if resume_id:
        option_kwargs["resume"] = resume_id
    if max_turns is not None:
        option_kwargs["max_turns"] = max_turns
    if max_budget_usd is not None:
        option_kwargs["max_budget_usd"] = max_budget_usd
    options = ClaudeAgentOptions(**option_kwargs)

    response_parts: list[str] = []
    cost_usd: float | None = None
    returned_session_id: str | None = None
    client = ClaudeSDKClient(options=options)
    await client.connect()
    try:
        await client.query(message)
        async for sdk_message in client.receive_response():
            if isinstance(sdk_message, AssistantMessage):
                for block in sdk_message.content:
                    if isinstance(block, TextBlock):
                        response_parts.append(block.text)
            elif isinstance(sdk_message, ResultMessage):
                if getattr(sdk_message, "total_cost_usd", None) is not None:
                    cost_usd = float(sdk_message.total_cost_usd)
                returned_session_id = getattr(sdk_message, "session_id", None)
                if getattr(sdk_message, "is_error", False):
                    detail = getattr(sdk_message, "result", None)
                    if not detail:
                        detail = "; ".join(getattr(sdk_message, "errors", None) or [])
                    if not detail:
                        detail = getattr(sdk_message, "subtype", "unknown SDK error")
                    api_status = getattr(sdk_message, "api_error_status", None)
                    status_note = f" (API status {api_status})" if api_status else ""
                    raise RuntimeError(f"Claude query failed{status_note}: {detail}")
    finally:
        try:
            await client.disconnect()
        except Exception as exc:
            sys.stderr.write(
                json.dumps({
                    "warning": "client.disconnect() raised; ignoring",
                    "type": type(exc).__name__,
                    "message": str(exc),
                })
                + "\n"
            )

    return "".join(response_parts), cost_usd, returned_session_id


def _die(message: str, code: int = 1) -> None:
    sys.stderr.write(json.dumps({"error": message}) + "\n")
    raise SystemExit(code)


def _print_json(payload: object) -> None:
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")


def cmd_open(args: argparse.Namespace) -> int:
    project_path = Path(args.project_path).expanduser().resolve()
    if not project_path.is_dir():
        _die(f"project_path is not a directory: {project_path}")

    handle = uuid.uuid4().hex[:12]
    now = _now_iso()
    with _store_lock():
        store = _load_store()
        _reap(store)
        store.sessions[handle] = SessionRecord(
            handle=handle,
            project_path=str(project_path),
            created_at=now,
            last_activity=now,
        )
        _save_store(store)
    _print_json({"session_id": handle, "project_path": str(project_path)})
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    handle = args.handle
    try:
        with _session_lock(handle):
            return _cmd_query_locked(args)
    except SessionBusyError:
        _die(
            f"session {handle} is busy with another query; wait for it or open a separate handle"
        )
    return 1


def _cmd_query_locked(args: argparse.Namespace) -> int:
    handle = args.handle
    with _store_lock():
        store = _load_store()
        _reap(store)
        record = store.sessions.get(handle)
        if record is None:
            _die(f"session handle not found: {handle}")
        record.last_activity = _now_iso()
        project_path = record.project_path
        resume_id = record.claude_session_id
        _save_store(store)

    try:
        text, cost_usd, claude_session_id = asyncio.run(
            asyncio.wait_for(
                _run_query(
                    project_path,
                    args.message,
                    resume_id,
                    max_turns=args.max_turns,
                    max_budget_usd=args.max_budget_usd,
                ),
                timeout=args.timeout,
            )
        )
    except asyncio.TimeoutError:
        _die(f"query timed out after {args.timeout}s for handle {handle}")
    except Exception as exc:
        _die(f"query failed: {type(exc).__name__}: {exc}")

    with _store_lock():
        store = _load_store()
        record = store.sessions.get(handle)
        if record is None:
            _append_cost_log(handle, cost_usd)
            _die(f"session {handle} no longer exists")
        if record.claude_session_id is None and claude_session_id:
            record.claude_session_id = claude_session_id
        record.message_count += 1
        record.last_activity = _now_iso()
        if cost_usd is not None:
            record.total_cost_usd = round(record.total_cost_usd + cost_usd, 6)
        _append_cost_log(handle, cost_usd)
        _save_store(store)
        total_cost = record.total_cost_usd
        message_count = record.message_count

    _print_json({
        "session_id": handle,
        "text": text,
        "cost_usd": cost_usd,
        "total_cost_usd": total_cost,
        "message_count": message_count,
    })
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    with _store_lock():
        store = _load_store()
        _reap(store)
        _save_store(store)
        payload = [asdict(record) for record in store.sessions.values()]
    _print_json(payload)
    return 0


def cmd_close(args: argparse.Namespace) -> int:
    handle = args.handle
    try:
        with _session_lock(handle):
            with _store_lock():
                store = _load_store()
                if handle not in store.sessions:
                    _print_json({"session_id": handle, "status": "not_found"})
                    return 0
                del store.sessions[handle]
                _save_store(store)
    except SessionBusyError:
        _die(f"session {handle} is busy and cannot be closed")
    _print_json({"session_id": handle, "status": "closed"})
    return 0


def cmd_costs(args: argparse.Namespace) -> int:
    with _store_lock():
        store = _load_store()
        record = store.sessions.get(args.handle)
        total = 0.0
        queries = 0
        if COST_LOG.exists():
            for line in COST_LOG.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                if len(parts) >= 3 and parts[1] == args.handle:
                    try:
                        total += float(parts[2])
                        queries += 1
                    except ValueError:
                        continue
    _print_json({
        "session_id": args.handle,
        "total_cost_usd": round(total, 6),
        "query_count": queries,
        "tracked_in_store": record.total_cost_usd if record else None,
    })
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    try:
        sdk_version = importlib.metadata.version("claude-agent-sdk")
        _import_sdk()
    except importlib.metadata.PackageNotFoundError:
        _die(
            "claude-agent-sdk is not installed. Run: "
            "python3 -m pip install --upgrade claude-agent-sdk",
            code=2,
        )
    _print_json({
        "status": "ok",
        "skill": "claude-code-sdk",
        "package": "claude-agent-sdk",
        "package_version": sdk_version,
        "state_dir": str(STATE_DIR),
    })
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="session_manager",
        description="Persistent Claude Code sessions through claude-agent-sdk.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    open_parser = subparsers.add_parser("open", help="Register a project session")
    open_parser.add_argument("project_path", help="Project directory")
    open_parser.set_defaults(func=cmd_open)

    query_parser = subparsers.add_parser("query", help="Query an existing session")
    query_parser.add_argument("handle", help="Session handle returned by open")
    query_parser.add_argument("message", help="Prompt to send to Claude Code")
    query_parser.add_argument(
        "--max-turns",
        type=_positive_int,
        help="Maximum Claude agent turns for this query",
    )
    query_parser.add_argument(
        "--max-budget-usd",
        type=_positive_float,
        help="Maximum SDK-reported spend for this query",
    )
    query_parser.add_argument(
        "--timeout",
        type=_positive_int,
        default=QUERY_TIMEOUT_SECONDS,
        help=f"Query timeout in seconds (default: {QUERY_TIMEOUT_SECONDS})",
    )
    query_parser.set_defaults(func=cmd_query)

    list_parser = subparsers.add_parser("list", help="List active sessions")
    list_parser.set_defaults(func=cmd_list)

    close_parser = subparsers.add_parser("close", help="Drop a session record")
    close_parser.add_argument("handle", help="Session handle to drop")
    close_parser.set_defaults(func=cmd_close)

    costs_parser = subparsers.add_parser("costs", help="Summarise session costs")
    costs_parser.add_argument("handle", help="Session handle to summarise")
    costs_parser.set_defaults(func=cmd_costs)

    doctor_parser = subparsers.add_parser("doctor", help="Verify the SDK install")
    doctor_parser.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
