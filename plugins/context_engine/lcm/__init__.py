"""Hermes LCM. Opt-in, DAG-based context engine (``context.engine: lcm``).

Layout

    plugins/context_engine/lcm/
    ├── __init__.py     entry point: ``LCMEngine`` + ``register(ctx)``
    ├── engine.py       the ContextEngine implementation
    ├── storage.py      SQLite raw-message store, summary nodes, lineage, backup
    ├── compaction.py   DAG compaction + deterministic digest rendering
    ├── recall.py       bounded recall (search / expand / page / status)
    ├── config.py       ``context.lcm`` settings with hard ceilings
    └── skills/hermes-lcm/   the recall skill and its policy references

Why the engine lives here rather than under ``plugins/hermes-lcm/``: the host
discovers always-available engines by scanning ``plugins/context_engine/<name>/``
(``plugins/context_engine/__init__.py``, and the developer-guide contract
"place your engine in ``plugins/context_engine/<name>/``").  The general plugin
tree is opt-in-by-default and would leave the engine unavailable until a user
enabled it, which is the opposite of the required behaviour.  This is the
plugin boundary either way. Nothing outside this directory changes.

Selecting the engine

    context:
      engine: "lcm"        # config.yaml: "compressor" (the built-in) is default

Unset, misspelled, or failing to load, the host stays on the built-in
``ContextCompressor``. Disabling the plugin is the rollback path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .engine import ENGINE_NAME, LCMEngine, build_engine

__all__ = ["ENGINE_NAME", "LCMEngine", "build_engine", "register"]

_SKILL_PATH = Path(__file__).resolve().parent / "skills" / "hermes-lcm" / "SKILL.md"

_COMMAND_HELP = (
    "LCM context engine. Usage: /lcm status | /lcm search <query> | "
    "/lcm expand <node_id|message_id> [page] | /lcm page [cursor]"
)

_COMMAND_OWNER = "context-engine:lcm"


def _command_is_ours(name: str) -> bool:
    """Whether */name* is already registered by this engine.

    Discovery and engine loading are independent host paths, so ``register``
    can legitimately run more than once per process.  Re-registering would log a
    spurious plugin-vs-plugin conflict, so we skip only when the existing entry
    is already ours. Another plugin's claim on the name is left to the host's
    own conflict policy.
    """
    try:
        from hermes_cli.plugins import get_plugin_manager

        entry = get_plugin_manager()._plugin_commands.get(name)
    except Exception:
        return False
    return bool(entry) and entry.get("plugin") == _COMMAND_OWNER


def register(ctx: Any) -> None:
    """Register the engine (and its companions) with the host.

    Called by ``plugins/context_engine/__init__.py`` for repo-shipped engines
    and by the general plugin loader for installed ones.  Every optional
    registration is feature-detected, because the repo-shipped collector
    implements fewer methods than a full plugin context.
    """
    engine = build_engine()
    ctx.register_context_engine(engine)

    register_command = getattr(ctx, "register_command", None)
    if callable(register_command) and not _command_is_ours("lcm"):
        register_command(
            "lcm",
            lambda raw_args: _handle_command(engine, raw_args),
            description="LCM context engine: status, bounded recall, and DAG expansion",
            args_hint="[status|search <query>|expand <id> [page]|page [cursor]]",
        )

    register_skill = getattr(ctx, "register_skill", None)
    if callable(register_skill) and _SKILL_PATH.exists():
        try:
            register_skill(
                "recall",
                _SKILL_PATH,
                description=(
                    "Use LCM bounded recall for the current session. Page summary "
                    "nodes with lcm_expand instead of loading whole transcripts."
                ),
            )
        except Exception:
            # A skill that will not register must not stop the engine loading.
            pass


def _handle_command(engine: LCMEngine, raw_args: str) -> str:
    """Handle ``/lcm ...``: diagnostics and bounded recall from the CLI."""
    parts = (raw_args or "").split()
    action = parts[0].lower() if parts else "status"

    if action in ("status", "diag", "diagnostics"):
        diagnostics = engine.diagnostics()
        store = diagnostics.get("store") or {}
        stats = store.get("stats") or {}
        lines = [
            f"LCM engine: {diagnostics.get('engine')}",
            f"session: {diagnostics.get('session_id')}",
            f"db: {store.get('db_path')}",
            f"schema: v{stats.get('schema_version')}  fts: {stats.get('fts')}",
            f"messages: {stats.get('messages')} "
            f"(compacted {stats.get('compacted_messages')})  "
            f"nodes: {stats.get('nodes')}  lineage edges: {stats.get('lineage_edges')}",
            f"last node: {diagnostics.get('last_node_id')}",
        ]
        return "\n".join(lines)

    if action == "search":
        query = " ".join(parts[1:]).strip()
        if not query:
            return f"usage: /lcm search <query>\n{_COMMAND_HELP}"
        payload = _tool(engine, "lcm_search", {"query": query})
        results = payload.get("results") or []
        if not results:
            return f"no LCM matches for {query!r} in this session"
        return "\n".join(
            f"{hit['message_id']}  {hit['body'][:160]}" for hit in results
        )

    if action == "expand":
        if len(parts) < 2:
            return f"usage: /lcm expand <node_id|message_id> [page]\n{_COMMAND_HELP}"
        ref = parts[1]
        page = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        key = "node_id" if ref.startswith("n") else "message_id"
        payload = _tool(engine, "lcm_expand", {key: ref, "page": page})
        if payload.get("error"):
            return f"LCM expand failed: {payload['error']}"
        if key == "message_id":
            message = payload.get("message") or {}
            return f"{message.get('message_id')} ({message.get('role')}):\n{message.get('body')}"
        node = payload.get("node") or {}
        lines = [
            f"{node.get('node_id')} (level {node.get('level')}, "
            f"{node.get('source_count')} sources) page {payload.get('page')}/"
            f"{'more' if payload.get('has_more') else 'end'}",
            node.get("summary", ""),
            "---- lineage ----",
        ]
        for item in payload.get("lineage") or []:
            if item.get("kind") == "message":
                lines.append(f"  {item['id']} ({item.get('role')}): {item.get('body', '')[:160]}")
            else:
                lines.append(f"  {item['id']} (superseded node, level {item.get('level')})")
        return "\n".join(lines)

    if action == "page":
        cursor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        payload = _tool(engine, "lcm_page", {"cursor": cursor})
        items = payload.get("items") or []
        lines = [f"page cursor={cursor} returned={payload.get('returned')} "
                 f"total={payload.get('total')} has_more={payload.get('has_more')}"]
        lines += [f"  {row['seq']} {row['role']}: {row['body'][:120]}" for row in items]
        return "\n".join(lines)

    return _COMMAND_HELP


def _tool(engine: LCMEngine, name: str, args: dict) -> dict:
    import json

    try:
        return json.loads(engine.handle_tool_call(name, args))
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": str(exc)}
