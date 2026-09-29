"""Which of the desktop's connections are actually in place — from local state only.

Powers the rail's "connect more" card and the badges on the connections page. Answers come from
files and configured credentials, never from the network, so it is instant and cannot leak anything:
it reports *presence*, never a value. A connection whose state cannot be told from local state
reports ``unknown`` so the UI shows no badge instead of a wrong one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from hermes_constants import get_hermes_home

CONNECTED = "connected"
MISSING = "missing"
UNKNOWN = "unknown"

_MESSAGING_KEYS = ("TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "SLACK_BOT_TOKEN")


def _any_env(get: Callable[[str], Optional[str]], *keys: str) -> str:
    return CONNECTED if any((get(k) or "").strip() for k in keys) else MISSING


def snapshot(get_env: Optional[Callable[[str], Optional[str]]] = None, config: Optional[dict] = None) -> dict[str, str]:
    """``{connection id: connected | missing | unknown}`` for the ids in the desktop catalog."""
    if get_env is None:
        from hermes_cli.config import get_env_value as get_env
    if config is None:
        try:
            from hermes_cli.config import load_config
            config = load_config() or {}
        except Exception:
            config = {}
    home = get_hermes_home()
    himalaya = Path.home() / ".config" / "himalaya" / "config.toml"
    servers = config.get("mcp_servers") if isinstance(config, dict) else None
    return {
        "google": CONNECTED if (home / "google_token.json").is_file() else MISSING,
        "email": CONNECTED if himalaya.is_file() else MISSING,
        "messaging": _any_env(get_env, *_MESSAGING_KEYS),
        "notion": _any_env(get_env, "NOTION_API_KEY"),
        "github": _any_env(get_env, "GITHUB_TOKEN", "GH_TOKEN"),
        "smartHome": _any_env(get_env, "HASS_TOKEN"),
        "mcp": CONNECTED if isinstance(servers, dict) and servers else MISSING,
        "phone": UNKNOWN,
    }
