"""Release gate for the "Claude subscription via Agent SDK" runtime.

The runtime ships default-off because Anthropic's Agent SDK overview states
that third party developers may not offer claude.ai login or rate limits for
their products without prior approval, while their Help Center simultaneously
says SDK usage still draws from subscription limits — an unresolved tension.
Until Anthropic confirms in writing, only a developer running private
validation on their own consenting account may flip this on; see
`docs/design/claude-subscription-via-agent-sdk.md` for the full record.

This module is imported by the provider catalog and the dashboard web server,
so it must stay dependency-light and must never raise at import time.
"""

from __future__ import annotations

import functools
import importlib.util
from typing import Any, Optional

# Pinned floors, recorded here so the runtime, the `hermes doctor` probe, and
# the packaging extra all read the same numbers. `claude-agent-sdk` ships the
# Claude executable inside its platform wheels, so the SDK floor also fixes a
# CLI floor; a separately installed newer CLI is fine, an older one is not.
CLAUDE_AGENT_SDK_MIN_VERSION = "0.2.128"
CLAUDE_CLI_MIN_VERSION = "2.1.220"

_CONFIG_SECTION = "claude_subscription"


def claude_subscription_enabled(config: Optional[dict] = None) -> bool:
    """True when `claude_subscription.enabled` is explicitly set in config.

    Tolerates a missing, empty, or partially-shaped config dict: anything
    that isn't an explicit truthy `enabled` reads as off. Callers pass the
    already-loaded config rather than loading one, so the gate is usable from
    all three config loaders (CLI, `load_config()`, raw gateway YAML).
    """
    if not isinstance(config, dict):
        return False
    section: Any = config.get(_CONFIG_SECTION)
    if not isinstance(section, dict):
        return False
    return bool(section.get("enabled", False))


@functools.lru_cache(maxsize=1)
def claude_agent_sdk_available() -> bool:
    """True when `claude_agent_sdk` is importable in this interpreter.

    Uses `find_spec` rather than a real import: the package is an ~80 MB
    wheel that bundles the Claude executable, and every caller only needs to
    know whether the optional `claude-code` extra is installed.
    """
    try:
        return importlib.util.find_spec("claude_agent_sdk") is not None
    except (ImportError, ValueError):
        return False
