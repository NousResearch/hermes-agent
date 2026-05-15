"""
Multi-agent gateway: registry of available agents.

In a multi-agent gateway, every Hermes profile is exposed as a distinct
"agent" that gateway sessions can switch to (``/agent <name>``) or
target inline (``@<name> <message>``).  Each agent is backed by an
isolated ``HERMES_HOME`` directory — its own memory, skills, soul,
sessions, and config.

This module is the gateway-facing thin wrapper around the existing
``hermes_cli.profiles`` machinery.  It hides the filesystem detail and
exposes a small, side-effect-free API:

* ``AgentProfile`` — gateway-relevant metadata for one agent
* ``AgentRegistry`` — list / get / default / refresh

Kept deliberately small to avoid pulling the CLI's ``ProfileInfo``
(which scans gateway PIDs, alias scripts, distribution manifests, …)
into hot gateway code paths.  Refreshing is cheap and lazy.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional

from hermes_constants import get_default_hermes_root

logger = logging.getLogger(__name__)


_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# Default agent name when no per-session override is set and no profile
# is explicitly bound.  Maps to the default ``~/.hermes`` directory.
DEFAULT_AGENT_NAME = "default"


@dataclass(frozen=True)
class AgentProfile:
    """Gateway-relevant view of one Hermes profile.

    Frozen for use in cache keys and dict identity comparisons.
    """

    name: str
    home: Path
    display_name: str
    description: str = ""
    is_default: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "home": str(self.home),
            "display_name": self.display_name,
            "description": self.description,
            "is_default": self.is_default,
        }


class AgentRegistry:
    """Thread-safe enumeration of agent profiles available to the gateway.

    Reads the on-disk profile layout (``<root>``  + ``<root>/profiles/*``)
    and caches the result.  Call :meth:`refresh` after creating /
    deleting profiles so the gateway picks up the change without a
    restart.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._cache: Optional[Dict[str, AgentProfile]] = None

    # -- Public API ------------------------------------------------------

    def list(self) -> List[AgentProfile]:
        """Return all available agents, default first then alphabetical."""
        cache = self._ensure_cache()
        rest = sorted(
            (p for p in cache.values() if not p.is_default),
            key=lambda p: p.name,
        )
        default = [p for p in cache.values() if p.is_default]
        return default + rest

    def get(self, name: str) -> Optional[AgentProfile]:
        """Look up an agent by name; case-insensitive for ``default``."""
        if not isinstance(name, str) or not name.strip():
            return None
        canon = self._canonicalize(name)
        if canon is None:
            return None
        cache = self._ensure_cache()
        return cache.get(canon)

    def default(self) -> AgentProfile:
        """Return the default agent (always present — backed by ``~/.hermes``)."""
        cache = self._ensure_cache()
        default = cache.get(DEFAULT_AGENT_NAME)
        if default is None:
            # Should never happen — the default profile always exists on
            # disk by virtue of being the hermes root.  If it doesn't,
            # synthesize an entry pointing at the configured root so
            # callers don't crash.
            root = get_default_hermes_root()
            return AgentProfile(
                name=DEFAULT_AGENT_NAME,
                home=root,
                display_name=DEFAULT_AGENT_NAME,
                is_default=True,
            )
        return default

    def names(self) -> List[str]:
        """Return all agent names (for tab-completion / help)."""
        return [p.name for p in self.list()]

    def refresh(self) -> None:
        """Force a rescan on the next read."""
        with self._lock:
            self._cache = None

    # -- Internal --------------------------------------------------------

    @staticmethod
    def _canonicalize(name: str) -> Optional[str]:
        """Normalize a user-typed name to the on-disk profile id."""
        stripped = name.strip()
        if not stripped:
            return None
        if stripped.casefold() == DEFAULT_AGENT_NAME:
            return DEFAULT_AGENT_NAME
        canon = stripped.lower()
        if not _PROFILE_ID_RE.match(canon):
            return None
        return canon

    def _ensure_cache(self) -> Dict[str, AgentProfile]:
        with self._lock:
            if self._cache is not None:
                return self._cache
            cache: Dict[str, AgentProfile] = {}

            root = get_default_hermes_root()

            # Default profile — root directory itself.  Always present
            # in the registry, even when the directory hasn't been
            # initialised yet (a fresh install in profile mode).
            cache[DEFAULT_AGENT_NAME] = AgentProfile(
                name=DEFAULT_AGENT_NAME,
                home=root,
                display_name=_read_display_name(root, DEFAULT_AGENT_NAME),
                description=_read_description(root),
                is_default=True,
            )

            # Named profiles.
            profiles_dir = root / "profiles"
            if profiles_dir.is_dir():
                try:
                    entries = sorted(profiles_dir.iterdir())
                except OSError as exc:
                    logger.warning(
                        "AgentRegistry: cannot scan %s: %s", profiles_dir, exc
                    )
                    entries = []
                for entry in entries:
                    if not entry.is_dir():
                        continue
                    name = entry.name
                    if not _PROFILE_ID_RE.match(name):
                        continue
                    cache[name] = AgentProfile(
                        name=name,
                        home=entry,
                        display_name=_read_display_name(entry, name),
                        description=_read_description(entry),
                        is_default=False,
                    )

            self._cache = cache
            return cache


# ---------------------------------------------------------------------------
# Disk readers — kept module-level so they can be unit-tested independently.
# ---------------------------------------------------------------------------


def _read_display_name(profile_home: Path, fallback: str) -> str:
    """Return the agent's human-readable name.

    Resolution order:
    1. ``gateway.agent_display_name`` in the profile's ``config.yaml``
    2. ``branding.agent_name`` in the skin (if configured)
    3. ``fallback`` (the canonical profile id)

    Reading is best-effort — any failure falls back so a corrupt
    profile config can't take the entire registry down.
    """
    config_path = profile_home / "config.yaml"
    if config_path.is_file():
        try:
            import yaml  # imported here so the registry has no import-time cost when unused

            with config_path.open("r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            gateway_cfg = cfg.get("gateway") or {}
            display = gateway_cfg.get("agent_display_name")
            if isinstance(display, str) and display.strip():
                return display.strip()
            branding = (cfg.get("display") or {}).get("branding") or {}
            agent_name = branding.get("agent_name")
            if isinstance(agent_name, str) and agent_name.strip():
                return agent_name.strip()
        except Exception as exc:  # noqa: BLE001 — config errors must not break registry
            logger.debug(
                "AgentRegistry: failed to read display name from %s: %s",
                config_path,
                exc,
            )
    return fallback


def _read_description(profile_home: Path) -> str:
    """Return a one-line description for the agent, or ''.

    Source of truth: the first non-empty, non-heading line of SOUL.md.
    Limited to 240 chars to keep ``/agent`` listings compact.
    """
    soul = profile_home / "SOUL.md"
    if not soul.is_file():
        return ""
    try:
        text = soul.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        return line[:240]
    return ""


# Module-level singleton — gateway code grabs this rather than passing
# a registry instance around.  Lazy: nothing happens until the first
# call to a public method.
_default_registry: Optional[AgentRegistry] = None
_default_registry_lock = RLock()


def default_registry() -> AgentRegistry:
    """Return the gateway-wide ``AgentRegistry`` singleton."""
    global _default_registry
    if _default_registry is None:
        with _default_registry_lock:
            if _default_registry is None:
                _default_registry = AgentRegistry()
    return _default_registry


def reset_default_registry() -> None:
    """Drop the cached singleton (for tests)."""
    global _default_registry
    with _default_registry_lock:
        _default_registry = None
