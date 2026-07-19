#!/usr/bin/env python3
"""Safely backfill persistent headers for known Hermes Discord workspaces.

The default is a read-only dry run. ``--apply`` is required for message
creation/editing, and every planned candidate is re-fetched and revalidated
immediately before that write. This script never renames threads.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from gateway.config import Platform, load_gateway_config
from gateway.session import SessionSource
from hermes_constants import get_hermes_home
from plugins.platforms.discord.adapter import DiscordAdapter, discord
from plugins.platforms.discord.workspace_headers import (
    WorkspaceHeaderStore,
    collect_workspace_header_candidates,
    revalidate_workspace_header_candidate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or apply Discord workspace-header backfill."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create/edit headers after live revalidation (default: dry run).",
    )
    parser.add_argument(
        "--guild-id",
        action="append",
        default=[],
        help="Limit reads/writes to this Discord guild id (repeatable).",
    )
    return parser


def _load_participated_thread_ids() -> set[str]:
    path = get_hermes_home() / "discord_threads.json"
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if not isinstance(payload, list):
        return set()
    return {str(value) for value in payload if str(value).strip()}


def _active_threads(client: Any, allowed_guild_ids: set[str]) -> list[Any]:
    threads: list[Any] = []
    for guild in list(getattr(client, "guilds", []) or []):
        guild_id = str(getattr(guild, "id", "") or "")
        if allowed_guild_ids and guild_id not in allowed_guild_ids:
            continue
        threads.extend(list(getattr(guild, "threads", []) or []))
    return threads


async def _run(args: argparse.Namespace) -> int:
    config = load_gateway_config()
    discord_config = config.platforms.get(Platform.DISCORD)
    token = str(getattr(discord_config, "token", "") or "").strip()
    if discord_config is None or not token:
        print(json.dumps({"ok": False, "error": "Discord is not configured"}))
        return 2

    intents = getattr(discord, "Intents").none()
    intents.guilds = True
    client = getattr(discord, "Client")(intents=intents)
    store = WorkspaceHeaderStore()
    adapter = DiscordAdapter(discord_config)
    adapter._client = client
    allowed_guild_ids = {str(value) for value in args.guild_id}
    outcome: dict[str, Any] = {
        "mode": "apply" if args.apply else "dry-run",
        "candidates": [],
        "results": [],
    }

    @client.event
    async def on_ready() -> None:
        participated = _load_participated_thread_ids()
        candidates = collect_workspace_header_candidates(
            _active_threads(client, allowed_guild_ids),
            participated_thread_ids=participated,
            store=store,
        )
        outcome["candidates"] = [
            {
                "scope_id": candidate.scope_id,
                "thread_id": candidate.thread_id,
                "observed_title": candidate.observed_title,
                "reasons": list(candidate.reasons),
                "proposed_title": candidate.proposed_title,
            }
            for candidate in candidates
        ]

        if args.apply:
            for candidate in candidates:
                try:
                    live_thread = client.get_channel(int(candidate.thread_id))
                    if live_thread is None:
                        live_thread = await client.fetch_channel(
                            int(candidate.thread_id)
                        )
                except Exception as error:
                    outcome["results"].append(
                        {
                            "thread_id": candidate.thread_id,
                            "action": "skipped",
                            "error": f"live fetch failed: {type(error).__name__}",
                        }
                    )
                    continue

                participated_now = _load_participated_thread_ids()
                if not revalidate_workspace_header_candidate(
                    candidate,
                    live_thread=live_thread,
                    participated_thread_ids=participated_now,
                    store=store,
                ):
                    outcome["results"].append(
                        {
                            "thread_id": candidate.thread_id,
                            "action": "skipped",
                            "error": "live state changed after planning",
                        }
                    )
                    continue

                source = SessionSource(
                    platform=Platform.DISCORD,
                    chat_id=candidate.thread_id,
                    chat_name=candidate.observed_title,
                    chat_type="thread",
                    thread_id=candidate.thread_id,
                    scope_id=candidate.scope_id,
                    parent_chat_id=(
                        str(getattr(live_thread, "parent_id", "") or "") or None
                    ),
                )
                result = await adapter.ensure_workspace_header(source)
                outcome["results"].append(
                    {
                        "thread_id": candidate.thread_id,
                        "action": result.action,
                        "success": result.success,
                        "message_id": result.message_id,
                        "error": result.error,
                    }
                )

        await client.close()

    try:
        await client.start(token, reconnect=False)
    finally:
        if not client.is_closed():
            await client.close()

    outcome["ok"] = True
    outcome["candidate_count"] = len(outcome["candidates"])
    print(json.dumps(outcome, indent=2, sort_keys=True))
    return 0


def main() -> int:
    return asyncio.run(_run(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
