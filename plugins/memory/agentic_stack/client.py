"""Dynamic import of agentic-stack harness hooks.

The harness lives outside hermes-agent (typical location
``~/.agent/harness/``). We load it on demand so the plugin doesn't ship
duplicate code. Resolution order for the brain path:

1. ``brain_path`` from config - absolute paths preferred; tilde
   expansion uses the *real* user home (from ``pwd``) rather than
   ``$HOME``. Hermes's terminal tool overrides ``$HOME`` to a
   per-profile sandbox on subprocess calls, which would otherwise
   misdirect ``~/.agent`` to a nonexistent nested path when a specialist
   is spawned by pepessimo's shell-out.
2. Env var ``AGENTIC_STACK_BRAIN`` (also real-home-aware).
3. Fall back to ``<real-home>/.agent``.

All imports are lazy so a missing brain only breaks the specific call,
not plugin load.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import pwd
import sys
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _real_home() -> str:
    """Return the owning user's real home directory, ignoring $HOME.

    Hermes's terminal backend overrides ``$HOME`` for subprocess
    isolation; ``os.path.expanduser`` honors that override. For the
    agentic-stack brain we want the *actual* user home so that every
    subprocess resolves the same canonical location.
    """
    try:
        return pwd.getpwuid(os.getuid()).pw_dir
    except Exception:
        return os.environ.get("HOME", "")


def _expand_with_real_home(path: str) -> str:
    """Tilde-expand ``path`` using the real user home, not ``$HOME``."""
    if not path:
        return path
    if path.startswith("~/") or path == "~":
        home = _real_home()
        if home:
            return os.path.join(home, path[2:]) if path.startswith("~/") else home
    return path


def resolve_brain_path(configured: Optional[str]) -> Path:
    if configured:
        return Path(_expand_with_real_home(configured)).resolve()
    env = os.environ.get("AGENTIC_STACK_BRAIN")
    if env:
        return Path(_expand_with_real_home(env)).resolve()
    home = _real_home()
    fallback = f"{home}/.agent" if home else "~/.agent"
    return Path(_expand_with_real_home(fallback)).resolve()


def _ensure_harness_importable(brain_path: Path) -> None:
    """Prepend the harness directory to sys.path so its hooks import."""
    harness = brain_path / "harness"
    if not harness.is_dir():
        raise FileNotFoundError(f"agentic-stack harness not found at {harness}")
    harness_str = str(harness)
    if harness_str not in sys.path:
        sys.path.insert(0, harness_str)


def get_log_execution(brain_path: Path) -> Optional[Callable[..., Any]]:
    """Lazy-load post_execution.log_execution from the harness."""
    try:
        _ensure_harness_importable(brain_path)
        # Re-import each time in case the brain path changed
        if "hooks.post_execution" in sys.modules:
            mod = importlib.reload(sys.modules["hooks.post_execution"])
        else:
            mod = importlib.import_module("hooks.post_execution")
        return getattr(mod, "log_execution", None)
    except Exception as e:
        logger.warning("agentic_stack: log_execution unavailable: %s", e)
        return None


def get_on_failure(brain_path: Path) -> Optional[Callable[..., Any]]:
    try:
        _ensure_harness_importable(brain_path)
        if "hooks.on_failure" in sys.modules:
            mod = importlib.reload(sys.modules["hooks.on_failure"])
        else:
            mod = importlib.import_module("hooks.on_failure")
        return getattr(mod, "on_failure", None)
    except Exception as e:
        logger.warning("agentic_stack: on_failure unavailable: %s", e)
        return None
