"""
Agent Orchestration Configuration
===================================

Reads ``orchestration:`` top-level key from Hermes config.yaml.
Hermes deep-merges unknown keys transparently, so no core changes needed.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "max_agents": 5,
    "default_max_iterations": 50,
    "default_toolsets": ["terminal", "file", "web"],
    "permissions": {
        "mode": "inherit",
        "allowlist": [],
        "blocklist": [],
    },
    "context_sharing": {
        "enabled": True,
        "max_shared_context_tokens": 4000,
    },
    "mailbox_dir": "",  # default: ~/.hermes/orchestration/mailboxes/
}


def _load_hermes_config() -> Dict[str, Any]:
    """Load the Hermes main config.yaml."""
    try:
        from hermes_cli.config import load_config
        return load_config()
    except Exception:
        return {}


def load_orchestration_config() -> Dict[str, Any]:
    """Load and merge orchestration config with defaults."""
    cfg = _load_hermes_config()
    orch = cfg.get("orchestration", {})
    if not isinstance(orch, dict):
        orch = {}

    # Deep merge with defaults
    merged = dict(_DEFAULTS)
    for key, val in orch.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val

    # Resolve mailbox_dir
    if not merged.get("mailbox_dir"):
        try:
            from hermes_constants import get_hermes_home
            merged["mailbox_dir"] = str(
                get_hermes_home() / "orchestration" / "mailboxes"
            )
        except Exception:
            merged["mailbox_dir"] = os.path.expanduser(
                "~/.hermes/orchestration/mailboxes"
            )

    return merged


def get_max_agents() -> int:
    return int(load_orchestration_config().get("max_agents", 5))


def get_default_toolsets() -> List[str]:
    return list(load_orchestration_config().get("default_toolsets", ["terminal", "file", "web"]))
