"""Shared prefill-messages resolution for every agent factory.

Ephemeral prefill messages are injected into the API call and never persisted to
session history. Three surfaces need the same two answers — *which* file, and
*what is in it*:

* the classic CLI, in ``cli._init_prompt_and_reasoning``,
* the gateway, in ``GatewayRunner._load_prefill_messages``,
* the TUI / Desktop backend, in ``tui_gateway.server._make_agent`` (the Agent
  factory the Desktop app boots through ``hermes serve``).

After the Phase-4 extraction of the CLI mixins, each surface grew its own copy
and the Desktop path ended up with none, so a configured
``prefill_messages_file`` was silently ignored there. One resolver here keeps
the three in step, and — because the home directory is resolved per call rather
than at import time — it stays correct when ``HERMES_HOME`` points at a
non-default profile.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

__all__ = [
    "resolve_prefill_messages_file",
    "load_prefill_messages",
    "load_configured_prefill_messages",
]


def resolve_prefill_messages_file(config: Optional[Dict[str, Any]] = None) -> str:
    """Path of the configured prefill file, or ``""``.

    Order: ``HERMES_PREFILL_MESSAGES_FILE`` env var, then the top-level
    ``prefill_messages_file`` key, then the legacy ``agent.prefill_messages_file``
    key. Callers that already hold a config dict pass it in; otherwise the active
    profile's config is read here, so nobody has to import the CLI config globals
    to ask the question.
    """
    file_path = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "").strip()
    if file_path:
        return file_path

    from hermes_cli.config import cfg_get, load_config_readonly

    if config is None:
        config = load_config_readonly() or {}
    return (
        str(config.get("prefill_messages_file", "") or "").strip()
        or str(cfg_get(config, "agent", "prefill_messages_file", default="") or "").strip()
    )


def load_prefill_messages(
    file_path: str, base_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Load the JSON array of prefill messages at *file_path*.

    Relative paths resolve against *base_dir*, or the current hermes home when
    that is omitted — resolved per call, so a profile switched at runtime reads
    that profile's file. Callers whose home is not the profile home (the gateway
    can run against an overridden config home) pass their own base directory.
    Every failure mode (unset, missing, unreadable, not a JSON array) returns
    ``[]``: a broken prefill file must not stop an agent from being built.
    """
    if not file_path:
        return []

    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = (Path(base_dir) if base_dir is not None else get_hermes_home()) / path
    if not path.exists():
        logger.warning("Prefill messages file not found: %s", path)
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Failed to load prefill messages from %s: %s", path, e)
        return []

    if not isinstance(data, list):
        logger.warning("Prefill messages file must contain a JSON array: %s", path)
        return []
    return data


def load_configured_prefill_messages(
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Resolve + load in one call: what an agent factory should invoke."""
    return load_prefill_messages(resolve_prefill_messages_file(config))
