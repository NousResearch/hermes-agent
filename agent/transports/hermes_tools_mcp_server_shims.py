"""Stateless agent-loop shims for the hermes-tools MCP server (#26567).

``memory`` and ``session_search`` are ``_AGENT_LOOP_TOOLS``: the generic
dispatcher refuses them because natively they receive live agent state (the
loop's MemoryStore / session-DB handle) from tool_executor. Both have workable
stateless equivalents, so a runtime that owns its own agent loop (codex
app-server, the Claude Agent SDK stdio profile) can regain them through the
server without touching that refusal:

  memory         → a fresh ``load_on_disk_store()`` per call. Char caps, config
                   overrides, external-drift guard, threat scan and file locking
                   live in MemoryStore/memory_tool and are inherited. The
                   consolidation-failure breaker is NOT: a fresh store per call
                   resets its counter every time, so it can never trip here (it
                   is reset natively at the turn boundary, and this subprocess
                   has no turns).
                   NOTE: a shim write cannot mirror through MemoryProvider hooks
                   (no MemoryManager in this subprocess), so when
                   ``memory.provider`` configures an external backend the shim
                   FAILS CLOSED — unregistered, and refused at dispatch — rather
                   than silently diverging the stores.
                   NOTE: with no foreground approver in a stdio subprocess the
                   native write-approval gate can return ``staged``; the call
                   reports success and the write does not land yet.
  session_search → ``SessionDB(read_only=True)`` over the state DB (never a
                   writable handle in a model-facing subprocess). NOT a faithful
                   equivalent: it adds a deterministic zero-hit OR-relaxation
                   the native tool does not have (see ``dispatch_session_search``).
                   The calling session's id arrives via the canonical
                   HERMES_SESSION_ID (see ``_SESSION_ID_ENV``); when that is
                   unset, own-lineage exclusion is simply INACTIVE — fail-open,
                   results just include the caller, no error. The DB path can be
                   pointed elsewhere with HERMES_MCP_STATE_DB (defaults to the
                   profile's state.db) — an internal mechanism bridge, not a
                   user-facing setting.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# The CANONICAL session-context variable — deliberately not a shim-specific name.
# ``set_current_session_id()`` writes it (gateway/session_context.py), ``_VAR_MAP``
# carries it, and ``_inject_session_context_env()`` inside ``hermes_subprocess_env()``
# bridges it into the HOST process's spawn env. That is hop 1 only: codex builds an
# MCP child's env from a fixed whitelist plus the names listed in the entry's
# ``env_vars`` (a spawn-time snapshot of the codex process env), and the migration
# entry names none — so under codex today the shim does not receive this var and
# own-lineage exclusion is INACTIVE (fail-open, documented above). A host delivers it
# by naming it in the entry's ``env_vars`` or by setting it in the server's spawn env
# directly. Reading a bespoke name instead would be strictly worse: it has no producer
# in any launch path, and the cross-session leak guard in
# ``_inject_session_context_env`` covers only ``_VAR_MAP`` keys, so a bespoke var could
# carry a SIBLING session's id under a concurrent host and exclude the wrong lineage.
_SESSION_ID_ENV = "HERMES_SESSION_ID"
_STATE_DB_ENV = "HERMES_MCP_STATE_DB"

ShimDefinition = tuple[str, str, dict[str, Any], Callable[[dict[str, Any]], str]]


def _external_memory_provider() -> str | None:
    """Name of the external memory provider configured via ``memory.provider``,
    or None for the builtin on-disk store. Config-read failure counts as
    builtin — the same fail-open posture as ``_memory_enabled_in_config()``."""
    try:
        from hermes_cli.config import load_config

        provider = str(
            (((load_config() or {}).get("memory", {}) or {}).get("provider") or "")
        ).strip().lower()
    except Exception:
        return None
    if provider in ("", "none", "builtin", "off", "disabled"):
        return None
    return provider


def _memory_enabled_in_config() -> bool:
    """Honor the operator's ``memory.memory_enabled`` config (default on)."""
    try:
        from hermes_cli.config import load_config

        return bool(
            ((load_config() or {}).get("memory", {}) or {}).get("memory_enabled", True)
        )
    except Exception:
        return True


def dispatch_memory(kwargs: dict[str, Any]) -> str:
    """Stateless ``memory`` dispatch: native handler + on-disk store."""
    from tools.memory_tool import load_on_disk_store, memory_tool
    from tools.registry import tool_error

    provider = _external_memory_provider()
    if provider is not None:
        # Every memory action mutates (add/replace/remove/batch), and a
        # mutation here can never reach the external backend — refuse with
        # the reason instead of letting the two stores drift apart.
        return tool_error(
            "memory is disabled in this MCP shim: external memory provider "
            f"'{provider}' is configured (memory.provider) and shim writes "
            "cannot mirror to it. Use the memory tool in the main agent loop "
            "instead.",
            success=False,
        )
    return memory_tool(
        action=kwargs.get("action", ""),
        target=kwargs.get("target", "memory"),
        content=kwargs.get("content"),
        old_text=kwargs.get("old_text"),
        operations=kwargs.get("operations"),
        store=load_on_disk_store(),
    )


def _session_search_error(message: str) -> str:
    return json.dumps({"success": False, "error": message})


def dispatch_session_search(kwargs: dict[str, Any]) -> str:
    """Stateless ``session_search`` dispatch: read-only DB + env session id."""
    import hermes_state
    from tools import session_search_tool

    db_path = Path(
        os.environ.get(_STATE_DB_ENV, "").strip() or hermes_state.DEFAULT_DB_PATH
    )
    if not db_path.exists():
        # Explicit degrade — a missing DB must never read as "no results".
        return _session_search_error(
            f"session_search unavailable: state DB not found at {db_path}"
        )
    try:
        db = hermes_state.SessionDB(db_path=db_path, read_only=True)
    except Exception as exc:
        return _session_search_error(
            f"session_search unavailable: cannot open state DB read-only: {exc}"
        )

    try:
        # A present-but-uninitialized DB (0-byte file from a crashed first
        # init) opens fine and would return a SILENT empty result — the
        # exact failure the missing-file guard above exists to prevent.
        # Probe the schema and degrade explicitly instead.
        db.get_session("__schema-probe__")
    except Exception as exc:
        try:
            db.close()
        except Exception:
            pass
        return _session_search_error(
            f"session_search unavailable: state DB not initialized: {exc}"
        )

    def _run(query: str) -> str:
        return session_search_tool.session_search(
            query=query,
            role_filter=kwargs.get("role_filter"),
            limit=kwargs.get("limit", 3),
            session_id=kwargs.get("session_id"),
            around_message_id=kwargs.get("around_message_id"),
            window=kwargs.get("window", 5),
            sort=kwargs.get("sort"),
            profile=kwargs.get("profile"),
            db=db,
            current_session_id=os.environ.get(_SESSION_ID_ENV, "").strip() or None,
        )

    try:
        query = kwargs.get("query") or ""
        result = _run(query)
        # Deterministic OR-relaxation: FTS5 ANDs terms, and models routinely
        # write "topic word word word" discovery queries that miss content
        # matching one distinctive term. On a ZERO-hit multi-term query with
        # no explicit FTS operators, retry ONCE with the terms OR-joined and
        # annotate the result — never silently, never for a query that
        # states its own operators, never on a single term.
        try:
            parsed = json.loads(result)
            terms = query.split()
            has_operators = any(
                operator in query
                for operator in ('"', "*", " OR ", " NOT ", " AND ")
            )
            if (
                isinstance(parsed, dict)
                and parsed.get("mode") == "discover"
                and parsed.get("count") == 0
                and len(terms) >= 2
                and not has_operators
            ):
                relaxed_query = " OR ".join(terms)
                relaxed = json.loads(_run(relaxed_query))
                if isinstance(relaxed, dict) and relaxed.get("count", 0) > 0:
                    relaxed["relaxed_query"] = relaxed_query
                    relaxed["note"] = (
                        "No result matched ALL terms (FTS ANDs them); showing "
                        "matches for ANY term instead."
                    )
                    return json.dumps(relaxed)
        except Exception:
            logger.debug("session_search relaxation skipped", exc_info=True)
        return result
    finally:
        try:
            db.close()
        except Exception:
            pass


def stateless_shim_definitions() -> list[ShimDefinition]:
    """(name, description, input_schema, handler) 4-tuples to register.

    session_search is always defined — a missing state DB degrades to an
    explicit error at call time, which is more diagnosable than an absent
    tool. memory respects the config kill-switch AND stays unregistered when
    an external memory provider is configured (shim writes cannot mirror
    through MemoryProvider hooks — see the module docstring; #26604).
    """
    definitions: list[ShimDefinition] = []
    if _memory_enabled_in_config() and _external_memory_provider() is None:
        from tools.memory_tool import MEMORY_SCHEMA

        definitions.append((
            "memory",
            MEMORY_SCHEMA.get("description", "Hermes memory tool"),
            MEMORY_SCHEMA.get("parameters") or {"type": "object", "properties": {}},
            dispatch_memory,
        ))

    from tools.session_search_tool import SESSION_SEARCH_SCHEMA

    definitions.append((
        "session_search",
        SESSION_SEARCH_SCHEMA.get("description", "Search past Hermes sessions"),
        SESSION_SEARCH_SCHEMA.get("parameters") or {"type": "object", "properties": {}},
        dispatch_session_search,
    ))
    return definitions
