"""CLI commands for Mem0 integration — hermes mem0 <subcommand>."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from mem0_integration.client import (
    Mem0ClientConfig,
    get_mem0_client,
    reset_mem0_client,
    resolve_config_path,
)


def _read_config_path() -> Path:
    return resolve_config_path()


def _get_client_and_config():
    """Helper: load config and create client. Raises on failure."""
    cfg = Mem0ClientConfig.from_global_config()
    if not cfg.api_key:
        print("Error: No Mem0 API key configured. Run 'hermes mem0 setup'.")
        sys.exit(1)
    reset_mem0_client()
    client = get_mem0_client(cfg)
    return client, cfg


def _prompt(text: str, default: str = "") -> str:
    """Prompt user for input with optional default."""
    suffix = f" [{default}]" if default else ""
    val = input(f"  {text}{suffix}: ").strip()
    return val or default


def _build_user_filters(user_id: str) -> dict:
    """Build v2 filters that find all memories for a user.

    Records stored with a run_id won't match a bare user_id filter
    (Mem0 treats missing fields as "must be null"), so we OR both cases.
    """
    return {
        "OR": [
            {"user_id": user_id},
            {"AND": [{"user_id": user_id}, {"run_id": "*"}]},
        ]
    }


def _write_config(data: dict, path: Path | None = None) -> Path:
    """Write config JSON to disk. Returns the path written to."""
    target = path or resolve_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return target


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------

def cmd_setup(args: Any) -> None:
    """Interactive setup wizard for Mem0 integration."""
    from hermes_cli.colors import Colors, color

    print(f"\n{color('Welcome to Mem0 setup for Hermes!', Colors.CYAN, Colors.BOLD)}\n")

    # 1. API Key
    print("1. API Key")
    api_key = _prompt("Enter your Mem0 API key (from app.mem0.ai)")
    if not api_key:
        print("  No API key provided. Aborting.")
        return
    print("  \u2713 API key saved\n")

    # 2. Identity
    print("2. Identity")
    user_id = _prompt("Your name (used as user_id for memory scoping)")
    agent_id = _prompt("Agent name", "hermes")
    print()

    # 3. Memory Mode
    print("3. Memory Mode")
    print("    hybrid  \u2014 write to both Mem0 and local MEMORY.md (default)")
    print("    mem0    \u2014 Mem0 only, skip MEMORY.md writes")
    memory_mode = _prompt("Choice", "hybrid")
    if memory_mode not in ("hybrid", "mem0"):
        memory_mode = "hybrid"
    print()

    # 4. Recall Mode
    print("4. Recall Mode")
    print("    hybrid  \u2014 inject context + expose tools (default)")
    print("    context \u2014 inject into prompt only, hide tools")
    print("    tools   \u2014 tools only, no prompt injection")
    recall_mode = _prompt("Choice", "hybrid")
    if recall_mode not in ("hybrid", "context", "tools"):
        recall_mode = "hybrid"
    print()

    # 5. Retrieval Quality
    print("5. Retrieval Quality")
    rerank = _prompt("Enable reranking for better search results? (+150ms) [Y]", "y").lower() in ("y", "yes")
    keyword_search = _prompt("Enable keyword search alongside semantic? [Y]", "y").lower() in ("y", "yes")
    print()

    # 6. Custom Instructions
    print("6. Custom Instructions (optional)")
    print("    Natural language guidelines for what Mem0 should extract.")
    custom_instructions = _prompt("Instructions (leave blank for default)", "") or None
    print()

    # 7. Session Strategy
    print("7. Session Strategy")
    print("    per-directory \u2014 one session per project directory (default)")
    print("    per-session   \u2014 fresh session each time")
    print("    global        \u2014 one session across everything")
    session_strategy = _prompt("Choice", "per-directory")
    if session_strategy not in ("per-directory", "per-session", "global"):
        session_strategy = "per-directory"
    print()

    config = {
        "apiKey": api_key,
        "hosts": {
            "hermes": {
                "enabled": True,
                "userId": user_id or None,
                "agentId": agent_id,
                "memoryMode": memory_mode,
                "recallMode": recall_mode,
                "rerank": rerank,
                "keywordSearch": keyword_search,
                "customInstructions": custom_instructions,
                "sessionStrategy": session_strategy,
            }
        }
    }

    written_path = _write_config(config)
    print(f"{color('\u2713', Colors.GREEN)} Configuration saved to {color(str(written_path), Colors.CYAN)}")
    print(f"  Run {color('hermes mem0 status', Colors.CYAN)} to verify connection.")


def cmd_status(args: Any) -> None:
    """Show current Mem0 config and connection status."""
    from hermes_cli.colors import Colors, color

    path = _read_config_path()
    if not path.exists():
        print(f"  {color('\u2717', Colors.RED)} Mem0 not configured. Run {color('hermes mem0 setup', Colors.CYAN)}.")
        return

    cfg = Mem0ClientConfig.from_global_config(config_path=path)
    print(f"\n{color('\u25c6 Mem0 Integration Status', Colors.CYAN, Colors.BOLD)}")
    _masked = f"{cfg.api_key[:8]}...****" if cfg.api_key and len(cfg.api_key) > 8 else color("(not set)", Colors.RED)
    _enabled = color("true", Colors.GREEN) if cfg.enabled else color("false", Colors.DIM)
    _user_id = cfg.user_id or color("(not set)", Colors.YELLOW)

    def _on_off(val: bool) -> str:
        return color("enabled", Colors.GREEN) if val else color("disabled", Colors.DIM)

    print(f"  Enabled:          {_enabled}")
    print(f"  API Key:          {_masked}")
    print(f"  User ID:          {_user_id}")
    print(f"  Agent ID:         {cfg.agent_id}")
    print(f"  Memory Mode:      {cfg.memory_mode}")
    print(f"  Recall Mode:      {cfg.recall_mode}")
    print(f"  Rerank:           {_on_off(cfg.rerank)}")
    print(f"  Keyword Search:   {_on_off(cfg.keyword_search)}")
    print(f"  Session Strategy: {cfg.session_strategy}")

    if cfg.api_key and cfg.enabled:
        try:
            reset_mem0_client()
            client = get_mem0_client(cfg)
            client.search(
                "connection-test",
                version="v2",
                filters={"OR": [{"user_id": "health-check"}]},
            )
            print(f"\n  Connection:       {color('\u2713 OK', Colors.GREEN)}")
        except Exception as e:
            print(f"\n  Connection:       {color('\u2717 FAILED', Colors.RED)} {color(f'({e})', Colors.DIM)}")
    elif not cfg.api_key:
        print(f"\n  Connection:       {color('\u2717 No API key', Colors.RED)}")
    elif not cfg.enabled:
        print(f"\n  Connection:       {color('\u2014 Disabled', Colors.DIM)}")
    print()


def cmd_search(args: Any) -> None:
    """Search memories from the terminal."""
    from hermes_cli.colors import Colors, color

    client, cfg = _get_client_and_config()
    query = getattr(args, "query", None)
    if not query:
        print(f"Usage: {color('hermes mem0 search <query>', Colors.CYAN)}")
        return

    try:
        results = client.search(
            query,
            version="v2",
            filters=_build_user_filters(cfg.user_id),
            keyword_search=cfg.keyword_search,
            rerank=True,
        )
        memories = results if isinstance(results, list) else results.get("results", results.get("memories", []))
        if not memories:
            print(f"  {color('No memories found.', Colors.DIM)}")
            return
        print(f"\n{color(f'Found {len(memories)} memories:', Colors.GREEN)}\n")
        for i, m in enumerate(memories, 1):
            score = m.get("score", 0)
            text = m.get("memory", "")
            cats = ", ".join(m.get("categories", []))
            created = (m.get("created_at") or "")[:10]
            score_color = Colors.GREEN if score >= 0.7 else Colors.YELLOW if score >= 0.4 else Colors.DIM
            print(f"  {i}. {color(f'[{score:.2f}]', score_color)} {text}")
            if cats:
                print(f"     {color('Categories:', Colors.DIM)} {cats}")
            if created:
                print(f"     {color('Created:', Colors.DIM)} {created}")
            print()
    except Exception as e:
        print(f"  {color('\u2717', Colors.RED)} Search failed: {e}")


def cmd_memories(args: Any) -> None:
    """List all memories for the current user."""
    from hermes_cli.colors import Colors, color

    client, cfg = _get_client_and_config()
    try:
        result = client.get_all(
            version="v2",
            filters=_build_user_filters(cfg.user_id),
            page_size=50,
        )
        memories = result if isinstance(result, list) else result.get("results", result.get("memories", []))
        if not memories:
            print(f"  {color('No memories stored yet.', Colors.DIM)}")
            return
        print(f"\n{color(f'Memories for user', Colors.CYAN)} {color(cfg.user_id, Colors.GREEN)} {color(f'({len(memories)})', Colors.DIM)}:\n")
        for m in memories:
            text = m.get("memory", "")
            cats = ", ".join(m.get("categories", []))
            print(f"  {color('\u2022', Colors.CYAN)} {text}")
            if cats:
                print(f"    {color(f'[{cats}]', Colors.DIM)}")
        print()
    except Exception as e:
        print(f"  {color('\u2717', Colors.RED)} Failed to list memories: {e}")


def cmd_clear(args: Any) -> None:
    """Delete all memories for the current user."""
    from hermes_cli.colors import Colors, color

    client, cfg = _get_client_and_config()
    print(f"\n  {color('\u26a0', Colors.YELLOW)} This will delete {color('ALL', Colors.RED, Colors.BOLD)} memories for user {color(cfg.user_id, Colors.CYAN)}.")
    confirm = input(f"  Type {color('yes', Colors.RED)} to confirm: ")
    if confirm.lower() not in ("y", "yes"):
        print(f"  {color('Cancelled.', Colors.DIM)}")
        return
    try:
        client.delete_all(user_id=cfg.user_id)
        print(f"  {color('\u2713', Colors.GREEN)} All memories for '{cfg.user_id}' deleted.")
    except Exception as e:
        print(f"  {color('\u2717', Colors.RED)} Failed to clear memories: {e}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def mem0_command(args: Any) -> None:
    """Route hermes mem0 <subcommand> to the right handler."""
    cmd = getattr(args, "mem0_command", None)
    handlers = {
        "setup": cmd_setup,
        "status": cmd_status,
        "search": cmd_search,
        "memories": cmd_memories,
        "clear": cmd_clear,
    }
    handler = handlers.get(cmd)
    if handler:
        handler(args)
    else:
        print("Usage: hermes mem0 {setup|status|search|memories|clear}")
        print("Run 'hermes mem0 setup' to get started.")
