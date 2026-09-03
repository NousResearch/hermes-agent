"""
Prefill messages: ephemeral few-shot priming for every API call.

Ported out of ``cli.py`` (``_load_prefill_messages`` and
``_resolve_prefill_messages_file``) so the Desktop app and any other
``hermes_cli`` consumer can load prefill messages without depending on
``cli.py`` module-level globals.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def load_prefill_messages(file_path: str) -> List[Dict[str, Any]]:
    """Load ephemeral prefill messages from a JSON file.

    The file should contain a JSON array of ``{role, content}`` dicts, e.g.::

        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]

    Relative paths are resolved from ``~/.hermes/``.

    Returns an empty list if the path is empty or the file doesn't exist.
    """
    if not file_path:
        return []
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = get_hermes_home() / path
    if not path.exists():
        logger.warning("Prefill messages file not found: %s", path)
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("Prefill messages file must contain a JSON array: %s", path)
            return []
        return data
    except Exception as e:
        logger.warning("Failed to load prefill messages from %s: %s", path, e)
        return []


def resolve_prefill_messages_file(config: Dict[str, Any]) -> str:
    """Resolve the prefill-file path from env var / config.

    Checks ``HERMES_PREFILL_MESSAGES_FILE`` env var first, then the top-level
    ``prefill_messages_file`` config key, and finally the legacy
    ``agent.prefill_messages_file`` key.

    Returns the resolved path string (possibly empty).
    """
    env_path = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "").strip()
    if env_path:
        return env_path
    top_level = str(config.get("prefill_messages_file", "") or "").strip()
    if top_level:
        return top_level
    agent_cfg = config.get("agent", {})
    if isinstance(agent_cfg, dict):
        return str(agent_cfg.get("prefill_messages_file", "") or "").strip()
    return ""
