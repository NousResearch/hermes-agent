"""CLI dispatcher for the claude-code-sdk Hermes skill.

Wraps the official ``claude-agent-sdk`` Python package in a path-based
session model that survives across separate ``terminal`` tool invocations.

Each Hermes terminal call is its own OS process, so we cannot keep a live
``ClaudeSDKClient`` connection in memory between calls. Instead we persist a
small JSON record per session at ``~/.hermes/skills/claude-code-sdk/sessions.json``
and use the SDK's native session-resume to restore conversation context on
each query.

Commands:
    open <project_path>       Register a new session for a project directory
    query <handle> <message>  Send a message; resumes prior context
    list                      List active sessions
    close <handle>            Drop a session record
    costs <handle>            Summarise total cost from cost.log

Run ``python session_manager.py --help`` for full usage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = Path(
    os.environ.get("HERMES_CLAUDE_SDK_STATE_DIR")
    or (Path.home() / ".hermes" / "skills" / "claude-code-sdk")
)
SESSIONS_FILE = STATE_DIR / "sessions.json"
COST_LOG = STATE_DIR / "cost.log"
IDLE_TTL_SECONDS = int(os.environ.get("HERMES_CLAUDE_SDK_IDLE_TTL", "3600"))
QUERY_TIMEOUT_SECONDS = int(os.environ.get("HERMES_CLAUDE_SDK_QUERY_TIMEOUT", "300"))


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


def _load_store() -> SessionStore:
    _ensure_state_dir()
    if not SESSIONS_FILE.exists():
        return SessionStore()
    try:
        data = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Corrupt sessions file at {SESSIONS_FILE}: {exc}") from exc
    sessions = {
        handle: SessionRecord(**rec) for handle, rec in data.get("sessions", {}).items()
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
            last = datetime.strptime(record.last_activity, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        if (now - last).total_seconds() > ttl:
            expired.append(handle)
    for handle in expired:
        store.sessions.pop(handle, None)
    return expired


def _append_cost_log(handle: str, cost_usd: float | None) -> None:
    _ensure_state_dir()
    if cost_usd is None:
        return
    line = f"{_now_iso()}\t{handle}\t{cost_usd:.6f}\n"
    with COST_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line)


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
            "claude-agent-sdk is not installed. Run: pip install claude-agent-sdk\n"
            f"(import failed with: {exc})",
            code=2,
        )
    return AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, TextBlock


async def _run_query(
    project_path: str,
    message: str,
    resume_id: str | None,
) -> tuple[str, float | None, str | None]:
    """Run one query against Claude Code via the SDK.

    Returns (response_text, total_cost_usd, returned_session_id).
    """
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
    options = ClaudeAgentOptions(**option_kwargs)

    response_parts: list[str] = []
    cost_usd: float | None = None
    returned_session_id: str | None = None

    client = ClaudeSDKClient(options=options)
    await client.connect()
    try:
        await client.query(message)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        response_parts.append(block.text)
            elif isinstance(msg, ResultMessage):
                if getattr(msg, "total_cost_usd", None) is not None:
                    cost_usd = float(msg.total_cost_usd)
                returned_session_id = getattr(msg, "session_id", None)
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass

    return "".join(response_parts), cost_usd, returned_session_id


def _die(msg: str, code: int = 1) -> None:
    sys.stderr.write(json.dumps({"error": msg}) + "\n")
    sys.exit(code)


def _print_json(payload: object) -> None:
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")


def cmd_open(args: argparse.Namespace) -> int:
    project_path = Path(args.project_path).expanduser().resolve()
    if not project_path.exists() or not project_path.is_dir():
        _die(f"project_path is not a directory: {project_path}")

    store = _load_store()
    _reap(store)

    handle = uuid.uuid4().hex[:12]
    now = _now_iso()
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
    message = args.message
    store = _load_store()
    _reap(store)

    record = store.sessions.get(handle)
    if record is None:
        _die(f"session handle not found: {handle}")
    assert record is not None

    try:
        text, cost_usd, claude_session_id = asyncio.run(
            asyncio.wait_for(
                _run_query(record.project_path, message, record.claude_session_id),
                timeout=QUERY_TIMEOUT_SECONDS,
            )
        )
    except asyncio.TimeoutError:
        _die(f"query timed out after {QUERY_TIMEOUT_SECONDS}s for handle {handle}")
    except Exception as exc:
        _die(f"query failed: {type(exc).__name__}: {exc}")

    if record.claude_session_id is None and claude_session_id:
        record.claude_session_id = claude_session_id
    record.message_count += 1
    record.last_activity = _now_iso()
    if cost_usd is not None:
        record.total_cost_usd = round(record.total_cost_usd + cost_usd, 6)
    _append_cost_log(handle, cost_usd)
    _save_store(store)

    _print_json(
        {
            "session_id": handle,
            "text": text,
            "cost_usd": cost_usd,
            "total_cost_usd": record.total_cost_usd,
            "message_count": record.message_count,
        }
    )
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    store = _load_store()
    _reap(store)
    _save_store(store)
    payload = [asdict(rec) for rec in store.sessions.values()]
    _print_json(payload)
    return 0


def cmd_close(args: argparse.Namespace) -> int:
    handle = args.handle
    store = _load_store()
    if handle not in store.sessions:
        _print_json({"session_id": handle, "status": "not_found"})
        return 0
    del store.sessions[handle]
    _save_store(store)
    _print_json({"session_id": handle, "status": "closed"})
    return 0


def cmd_costs(args: argparse.Namespace) -> int:
    handle = args.handle
    store = _load_store()
    record = store.sessions.get(handle)
    total = 0.0
    queries = 0
    if COST_LOG.exists():
        for line in COST_LOG.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 3 and parts[1] == handle:
                try:
                    total += float(parts[2])
                    queries += 1
                except ValueError:
                    continue
    _print_json(
        {
            "session_id": handle,
            "total_cost_usd": round(total, 6),
            "query_count": queries,
            "tracked_in_store": record.total_cost_usd if record else None,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="session_manager",
        description="Hermes claude-code-sdk skill: persistent Claude Code sessions via the Python SDK.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_open = sub.add_parser("open", help="Register a new session for a project directory")
    p_open.add_argument("project_path", help="Absolute path to the project directory")
    p_open.set_defaults(func=cmd_open)

    p_query = sub.add_parser("query", help="Send a message in an existing session")
    p_query.add_argument("handle", help="Session handle returned by 'open'")
    p_query.add_argument("message", help="The prompt to send to Claude Code")
    p_query.set_defaults(func=cmd_query)

    p_list = sub.add_parser("list", help="List active sessions")
    p_list.set_defaults(func=cmd_list)

    p_close = sub.add_parser("close", help="Drop a session record")
    p_close.add_argument("handle", help="Session handle to drop")
    p_close.set_defaults(func=cmd_close)

    p_costs = sub.add_parser("costs", help="Summarise cost for a session from cost.log")
    p_costs.add_argument("handle", help="Session handle to summarise")
    p_costs.set_defaults(func=cmd_costs)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
