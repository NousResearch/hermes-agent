"""Context Resolver for Task Runtime.

Resolves local context (HERMES_HOME, config, etc.) without executing any
sync / import / embed / HTTP. Returns a plain dict of metadata that
TaskContractBuilder can serialize into TaskContract.context.

This is intentionally minimal: the MVP does NOT pre-fetch GBrain / Obsidian
content. It only reports WHERE the content COULD be fetched from, leaving
that to the ExecutionPipeline if/when the mode permits.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def resolve(resolved_intent, hermes_home: Path | None = None) -> dict[str, Any]:
    """Return a context dict (pure metadata, no I/O sync).

    Args:
        resolved_intent: the output of IntentResolver.resolve(...).
        hermes_home: optional override; defaults to env HERMES_HOME or ~/.hermes.

    Returns:
        A dict with keys: hermes_home, hermes_home_display, vault_root,
        skills_dir, sessions_dir, kanban_board, knowledge_refs (empty list),
        memory_keys (empty list), config_version (str|None).
    """
    home = hermes_home or Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    home = Path(home)

    skills_dir = home / "skills"
    sessions_dir = home / "sessions"
    vault_root = home / "vault"
    kanban_dir = home / "kanban"
    kanban_board = None
    try:
        env_board = os.environ.get("HERMES_KANBAN_BOARD")
        if env_board:
            kanban_board = env_board
        elif (kanban_dir / "current_board").exists():
            kanban_board = (kanban_dir / "current_board").read_text().strip() or None
    except Exception:
        kanban_board = None

    config_version = None
    try:
        cfg_path = home / "config.yaml"
        if cfg_path.exists():
            text = cfg_path.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                if line.strip().startswith("_config_version:"):
                    config_version = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    return {
        "hermes_home": str(home),
        "hermes_home_display": "~/.hermes" if str(home) == str(Path.home() / ".hermes") else str(home),
        "vault_root": str(vault_root),
        "skills_dir": str(skills_dir),
        "sessions_dir": str(sessions_dir),
        "kanban_board": kanban_board,
        "knowledge_refs": [],   # populated by ExecutionPipeline if mode permits
        "memory_keys": [],      # populated by ExecutionPipeline if mode permits
        "config_version": config_version,
        "hermes_disable_self_improvement": bool(int(os.environ.get("HERMES_DISABLE_SELF_IMPROVEMENT", "0") or 0)),
    }