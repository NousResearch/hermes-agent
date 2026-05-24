#!/usr/bin/env python3
"""Hermes bridge to the original MemoryMunch implementation.

This intentionally imports the original MemoryMunch modules directly instead of
using memorymunch_mcp.cli_bridge because that bridge imports optional jCodeMunch
modules not installed in the Hermes runtime. This file preserves the original
MemoryMunch vault-first tools while keeping Hermes-side control/gating explicit.

Reads JSON from stdin: {"tool": "smart_search", "args": {...}}
Writes JSON to stdout. Logs/errors go to stderr.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

ORIGINAL_REPO = Path(os.environ.get("MEMORYMUNCH_ORIGINAL_REPO", "/mnt/c/Users/paulcooke1976/memorymunch-mcp"))
VAULT_PATH = Path(os.environ.get("MEMORYMUNCH_VAULT_PATH", "/mnt/c/Users/paulcooke1976/memorymunch-vault"))

os.environ.setdefault("MEMORYMUNCH_VAULT_PATH", str(VAULT_PATH))
if str(ORIGINAL_REPO) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_REPO))

from memorymunch_mcp.db import close_pool, ensure_schema, get_pool  # type: ignore  # noqa: E402
from memorymunch_mcp.tools.brain_health import brain_health  # type: ignore  # noqa: E402
from memorymunch_mcp.tools.get_memory import get_memory  # type: ignore  # noqa: E402
from memorymunch_mcp.tools.ingest_exchange import ingest_exchange  # type: ignore  # noqa: E402
from memorymunch_mcp.tools.store_memory import store_memory  # type: ignore  # noqa: E402
from memorymunch_mcp.tools.recall_memories import vault_search  # type: ignore  # noqa: E402
from memorymunch_mcp.tools.smart_cleanup import smart_cleanup  # type: ignore  # noqa: E402
from memorymunch_mcp.tools.smart_search import smart_search  # type: ignore  # noqa: E402
from memorymunch_mcp.tools.sync_vault import sync_vault  # type: ignore  # noqa: E402


async def edge_cleanup() -> dict[str, Any]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        r1 = await conn.execute(
            """
            DELETE FROM edges WHERE source_id IN (SELECT id FROM memories WHERE archived)
              OR target_id IN (SELECT id FROM memories WHERE archived)
            """
        )
        r2 = await conn.execute(
            """
            DELETE FROM edges WHERE source_id NOT IN (SELECT id FROM memories)
              OR target_id NOT IN (SELECT id FROM memories)
            """
        )
        r3 = await conn.execute("DELETE FROM edges WHERE weight < 0.05")
        total_edges_after = await conn.fetchval("SELECT COUNT(*) FROM edges")
    def _affected(result: str) -> int:
        try:
            return int(str(result).split()[-1])
        except Exception:
            return 0
    return {
        "orphan_edges_deleted": _affected(r1),
        "dangling_edges_deleted": _affected(r2),
        "below_floor_deleted": _affected(r3),
        "total_deleted": _affected(r1) + _affected(r2) + _affected(r3),
        "total_edges_after": total_edges_after,
    }


async def schema_counts() -> dict[str, Any]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        counts: dict[str, Any] = {}
        for table in ("memories", "edges", "activation_log", "embedding_cache", "system_metrics", "token_usage"):
            try:
                counts[table] = await conn.fetchval(f"SELECT count(*) FROM {table}")
            except Exception as exc:  # pragma: no cover - live schema dependent
                counts[table] = f"ERROR:{exc}"
        edge_columns = await conn.fetch(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name='edges'
            ORDER BY ordinal_position
            """
        )
        edge_pk = await conn.fetch(
            """
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=ANY(i.indkey)
            WHERE i.indrelid='edges'::regclass AND i.indisprimary
            ORDER BY array_position(i.indkey,a.attnum)
            """
        )
        orphan_edges = await conn.fetchval(
            """
            SELECT count(*) FROM edges e
            LEFT JOIN memories s ON s.id=e.source_id
            LEFT JOIN memories t ON t.id=e.target_id
            WHERE s.id IS NULL OR t.id IS NULL OR s.archived OR t.archived
            """
        )
    vault_count = 0
    for memory_type in ("semantic", "episodic", "procedural", "conversational", "preference"):
        type_dir = VAULT_PATH / memory_type
        if type_dir.is_dir():
            vault_count += len([p for p in type_dir.glob("*.md") if not p.name.startswith("_")])
    return {
        "counts": counts,
        "vault_count": vault_count,
        "edges_columns": [dict(row) for row in edge_columns],
        "edges_pk": [row["attname"] for row in edge_pk],
        "orphan_or_archived_edges": orphan_edges,
        "vault_path": str(VAULT_PATH),
        "original_repo": str(ORIGINAL_REPO),
    }


async def run_tool(tool: str, args: dict[str, Any]) -> Any:
    if tool == "smart_search":
        return await smart_search(
            query=args["query"],
            concepts=args.get("concepts"),
            entities=args.get("entities"),
            scope_entity=args.get("scope_entity"),
            max_results=args.get("max_results", 20),
            exclude_ids=args.get("exclude_ids"),
        )
    if tool == "vault_search":
        return vault_search(args.get("concepts", []), args.get("entities", []))
    if tool == "sync_vault":
        return await sync_vault(direction=args.get("direction", "vault_to_db"))
    if tool == "smart_cleanup":
        return await smart_cleanup(
            exchange_text=args.get("exchange_text", ""),
            max_candidates=args.get("max_candidates", 20),
        )
    if tool == "get_memory":
        return await get_memory(memory_id=args["memory_id"])
    if tool == "ingest_exchange":
        return await ingest_exchange(
            user_message=args["user_message"],
            bot_response=args["bot_response"],
            facts=args["facts"],
            entity=args.get("entity", "user"),
            previous_exchange_id=args.get("previous_exchange_id"),
            domain=args.get("domain", "general"),
        )
    if tool == "store_memory":
        return await store_memory(
            memory_id=args["memory_id"],
            memory_type=args["memory_type"],
            entity=args["entity"],
            content=args["content"],
            links=args.get("links"),
            decay_rate=args.get("decay_rate", 0.02),
            activation_weight=args.get("activation_weight", 0.5),
            domain=args.get("domain", "general"),
        )
    if tool == "edge_cleanup":
        return await edge_cleanup()
    if tool == "brain_health":
        return await brain_health()
    if tool == "schema_counts":
        return await schema_counts()
    raise ValueError(f"unknown MemoryMunch bridge tool: {tool}")


async def main() -> None:
    try:
        request = json.loads(sys.stdin.read() or "{}")
        tool = request["tool"]
        args = request.get("args") or {}
        if tool not in {"vault_search"}:
            await ensure_schema()
        result = await run_tool(tool, args)
        envelope = {
            "tool": tool,
            "mode": "original_memorymunch_bridge",
            "vault_path": str(VAULT_PATH),
            "original_repo": str(ORIGINAL_REPO),
            "result": result,
        }
        print(json.dumps(envelope, ensure_ascii=False, default=str))
    except Exception as exc:
        print(json.dumps({"error": str(exc), "mode": "original_memorymunch_bridge"}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
