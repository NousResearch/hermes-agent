"""Best-effort "last used" timestamps for the Context dashboard/mobile view.

Derives real usage recency for MCP servers, messaging channels, and API keys
from ``state.db`` — never invents a timestamp. Everything here is read-only
(a ``mode=ro`` SQLite URI via the tracked connection helper) and bounded (a
row-count ``LIMIT`` on the message scan) so it stays cheap enough for mobile
pull-to-refresh even against a large, long-lived database.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from tools.mcp_tool import (
    MCP_TOOL_NAME_PREFIX,
    _MCP_NAME_DELIM,
    sanitize_mcp_name_component,
)

DEFAULT_LOOKBACK_MESSAGES = 50_000


def _split_mcp_server(tool_name: str) -> Optional[str]:
    """Return the *sanitized* server component from ``mcp__<server>__<tool>``.

    Registration stores sanitized names (hyphens → underscores, see
    :func:`tools.mcp_tool.sanitize_mcp_name_component`), so the value returned
    here is the sanitized form, not the configured config key.
    """
    if not tool_name.startswith(MCP_TOOL_NAME_PREFIX):
        return None
    remainder = tool_name[len(MCP_TOOL_NAME_PREFIX) :]
    server, sep, _tool = remainder.partition(_MCP_NAME_DELIM)
    if not sep or not server:
        return None
    return server


def _configured_name_lookup(
    mcp_server_names: Optional[Iterable[str]],
) -> Dict[str, str]:
    """Map sanitized MCP server name → configured name.

    Config may use ``github-enterprise`` while tool_calls persist
    ``mcp__github_enterprise__…``. When two configured names sanitize to the
    same key, prefer an already-sanitized identity match, else first-seen.
    """
    if not mcp_server_names:
        return {}
    out: Dict[str, str] = {}
    for name in mcp_server_names:
        if not isinstance(name, str) or not name:
            continue
        safe = sanitize_mcp_name_component(name)
        if not safe:
            continue
        existing = out.get(safe)
        if existing is None or (existing != safe and name == safe):
            out[safe] = name
    return out


def _iter_tool_call_names(tool_calls_json: str) -> Iterable[str]:
    """Yield each function name in a stored ``tool_calls`` JSON blob."""
    try:
        calls = json.loads(tool_calls_json)
    except (TypeError, ValueError):
        return
    if not isinstance(calls, list):
        return
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        name = function.get("name") if isinstance(function, dict) else call.get("name")
        if isinstance(name, str) and name:
            yield name


def _tool_to_unique_env_key() -> Dict[str, str]:
    """Map tool name -> env key, for tools backed by exactly one key.

    Built from ``OPTIONAL_ENV_VARS``' existing ``tools`` lists so it tracks
    that catalog automatically. Many tools (``web_search``, ``browser_navigate``,
    ...) are served by whichever of several configured providers wins at
    runtime — attributing those to any single key would be a guess, so they
    are excluded here rather than guessed wrong.
    """
    from hermes_cli.config import OPTIONAL_ENV_VARS

    tool_keys: Dict[str, list] = {}
    for key, info in OPTIONAL_ENV_VARS.items():
        for tool_name in info.get("tools") or ():
            tool_keys.setdefault(tool_name, []).append(key)
    return {tool: keys[0] for tool, keys in tool_keys.items() if len(keys) == 1}


def _connect_state_db_ro(db_path: Path) -> sqlite3.Connection:
    """Open ``state.db`` read-only through the tracked-connection guard.

    Matches :class:`hermes_state.SessionDB` read-only attaches so concurrent
    byte-probes cannot cancel POSIX advisory locks held by other state.db
    connections in this process.
    """
    from hermes_cli.sqlite_safe_read import connect_tracked

    return connect_tracked(
        f"file:{db_path}?mode=ro",
        tracking_path=db_path,
        uri=True,
        timeout=1.0,
        isolation_level=None,
    )


def compute_context_last_used(
    *,
    home: Optional[Path] = None,
    lookback_messages: int = DEFAULT_LOOKBACK_MESSAGES,
    mcp_server_names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Compute real last-used timestamps from ``state.db``.

    ``mcp_server_names``, when given, lets the message scan stop early once
    every configured server already has a hit, and remaps sanitized tool-call
    server components back to the configured names used in API responses
    (e.g. ``github_enterprise`` → ``github-enterprise``).

    Returns:
        {
          "mcp": {"<server_name>": <unix_ts>, ...},
          "channels": {"<platform_id>": <unix_ts>, ...},
          "keys": {"<ENV_VAR>": <unix_ts>, ...},
          "computed_at": <unix_ts>,
        }
    """
    from hermes_constants import get_hermes_home

    resolved_home = Path(home) if home is not None else get_hermes_home()
    db_path = resolved_home / "state.db"
    result: Dict[str, Any] = {
        "mcp": {},
        "channels": {},
        "keys": {},
        "computed_at": time.time(),
    }
    if not db_path.exists():
        return result

    mcp_last: Dict[str, float] = {}
    tool_last: Dict[str, float] = {}
    channels: Dict[str, float] = {}
    sanitized_to_configured = _configured_name_lookup(mcp_server_names)
    # Early-exit tracks sanitized keys (what tool_calls actually store).
    pending_servers = (
        set(sanitized_to_configured.keys()) if sanitized_to_configured else None
    )

    conn = None
    try:
        conn = _connect_state_db_ro(db_path)
        conn.row_factory = sqlite3.Row

        # The inner LIMIT bounds rows actually scanned (not rows matched) to
        # `lookback_messages`, walking newest-first via the rowid/PK order so
        # a cold multi-GB history never triggers a full table scan.
        cursor = conn.execute(
            """
            SELECT tool_calls, timestamp FROM (
                SELECT tool_calls, timestamp, id FROM messages
                ORDER BY id DESC
                LIMIT ?
            )
            WHERE tool_calls IS NOT NULL
            """,
            (lookback_messages,),
        )
        for row in cursor:
            ts = row["timestamp"]
            if ts is None:
                continue
            for name in _iter_tool_call_names(row["tool_calls"]):
                server = _split_mcp_server(name)
                if server is not None:
                    display = sanitized_to_configured.get(server, server)
                    if ts > mcp_last.get(display, 0.0):
                        mcp_last[display] = ts
                    if pending_servers is not None:
                        pending_servers.discard(server)
                elif ts > tool_last.get(name, 0.0):
                    tool_last[name] = ts
            if pending_servers is not None and not pending_servers:
                # Every configured MCP server already has a newest-first hit;
                # older messages can only add earlier (non-newest) timestamps
                # for MCP, so there's nothing left worth the extra scan.
                break

        for row in conn.execute(
            "SELECT source, MAX(COALESCE(ended_at, started_at)) AS ts "
            "FROM sessions GROUP BY source"
        ):
            if row["source"] and row["ts"] is not None:
                channels[row["source"]] = row["ts"]
    finally:
        if conn is not None:
            conn.close()

    keys: Dict[str, float] = {}
    for tool_name, env_key in _tool_to_unique_env_key().items():
        ts = tool_last.get(tool_name)
        if ts is not None:
            # Keep the newest timestamp if several unique tools share one key.
            prev = keys.get(env_key)
            if prev is None or ts > prev:
                keys[env_key] = ts

    result["mcp"] = mcp_last
    result["channels"] = channels
    result["keys"] = keys
    return result
