"""Singleton state for the CLI memory-provider health indicator.

One instance per process (module-level ``_STATE``).  Writers (agent_init,
MemoryManager, CLI after-turn probe) mutate it; the CLI status-bar renderer
reads it — never probes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Cooldown (seconds) after a failed probe before retrying.  Prevents a
# persistently-down backend from adding 2 s of latency to every turn.
PROBE_COOLDOWN_S = 30.0

# AIAgent spawn contexts that are NOT the foreground user-facing agent and
# therefore never own the process-wide MemoryHealthState.  Audited 2026-09-28
# over every in-process ``AIAgent()`` construction site:
#   cron/scheduler.py       platform="cron"           (background)
#   tools/delegate_tool.py  platform="subagent", side_agent=True, parent_session_id=…
#   agent/curator.py        platform="curator"        (background review)
#   agent/background_review.py  parent_session_id=…   (platform inherited from parent!)
#   hermes_cli/cli_commands_mixin.py bg_agent  side_agent=True
# A NEW background spawn site must join this list or pass side_agent /
# parent_session_id — otherwise it re-opens singleton pollution (the foreground
# indicator is wiped and, active_provider being empty, never probed again).
BACKGROUND_AGENT_PLATFORMS = frozenset({"cron", "subagent", "curator"})

# Positive identification of the owner (frozen §8: the singleton represents the
# foreground CLI main agent's user-visible memory state).  Ownership is
# fail-closed: a surface that is not the CLI foreground never writes, even
# though it is not "background" either — gateway ``api_server`` spawns an
# independent agent per session in ONE process, and default-allow let those
# agents overwrite each other's (and the foreground's) state.  The classic CLI
# REPL (``platform="cli"``) is currently the only reader of this singleton
# (cli_status_bar_mixin), so the allowlist loses nothing user-visible.
FOREGROUND_HEALTH_PLATFORMS = frozenset({"cli"})


def is_background_agent(agent, platform: str = "") -> bool:
    """True when ``agent`` is not the foreground user-facing agent.

    Three structural signals, strongest first: the agent is a child/fork of
    another agent (``parent_session_id`` — covers review forks whose platform is
    inherited from the parent), it declares itself a side agent, or its platform
    names a known background context.
    """
    return (
        bool(getattr(agent, "_parent_session_id", None))
        or bool(getattr(agent, "side_agent", False))
        or (platform or "") in BACKGROUND_AGENT_PLATFORMS
    )


def is_foreground_health_owner(agent, platform: str = "") -> bool:
    """True when ``agent`` owns the process-wide ``MemoryHealthState``.

    Positive identification, never a "not background" default: the agent's
    platform must name the CLI foreground surface AND the agent must carry no
    child/side-agent markers.  Unknown surfaces (api_server, acp, gateway,
    batch, diagnostic agents, a bare ``AIAgent()`` with no platform) fail
    closed — they may read the singleton but must not write it.
    """
    return (platform or "") in FOREGROUND_HEALTH_PLATFORMS and not is_background_agent(
        agent, platform
    )


@dataclass
class MemoryHealthState:
    """Process-wide memory-provider health state."""

    # From config.yaml ``memory.provider``; "" when no external provider
    # is configured.
    configured_provider: str = ""

    # Name of the actually-registered external provider for the foreground
    # agent. "" when no external provider is loaded.
    active_provider: str = ""

    # ``"healthy"`` | ``"unavailable"`` | ``"unknown"``.
    # Starts as ``"unknown"``; first successful probe → ``"healthy"``;
    # exception / timeout / failed probe → ``"unavailable"``.
    health: str = "unknown"

    # Optional reason for the current ``"unavailable"`` state.
    reason: str = ""

    # Monotonic timestamp of the last failed runtime probe.
    _last_probe_failure_at: float = field(default=0.0, repr=False)

    # -- writers ---------------------------------------------------------------
    def set_active_provider(self, provider: str) -> None:
        """Set the active provider and reset health when the provider changes."""
        if self.active_provider == provider:
            return

        self.active_provider = provider
        self.health = "unknown"
        self.reason = ""
        self._last_probe_failure_at = 0.0

    def mark_healthy(self) -> None:
        was = self.health
        self.health = "healthy"
        self.reason = ""
        self._last_probe_failure_at = 0.0
        if was != "healthy":
            logger.info("MemoryHealth: %s → healthy", was)

    def mark_unavailable(self, reason: str = "") -> None:
        was = self.health
        self.health = "unavailable"
        self.reason = reason
        if was != "unavailable":
            logger.warning("MemoryHealth: %s → unavailable (%s)", was, reason)

    def record_probe_failure(self) -> None:
        """Record a failed runtime probe for cooldown purposes."""
        self._last_probe_failure_at = time.monotonic()

    def probe_cooldown_active(self) -> bool:
        """True when the provider is unavailable and we should skip another
        probe to avoid repeated 2 s blocks."""
        if self.health != "unavailable":
            return False
        if self._last_probe_failure_at <= 0:
            # No probe failure recorded ("0.0 = never").  An unavailable state
            # from an ordinary operation failure must not read as cooldown —
            # §5: only probe failure arms the cooldown, even when the monotonic
            # clock itself is small (fresh boot, uptime < PROBE_COOLDOWN_S).
            return False
        return (time.monotonic() - self._last_probe_failure_at) < PROBE_COOLDOWN_S

    # -- read helpers ----------------------------------------------------------

    def indicator_text(self) -> str:
        """``"provider ● Healthy"`` / ``"provider ○ Connecting"`` / ``""``."""
        provider = self.active_provider or self.configured_provider
        if not provider:
            return ""
        glyph = {"healthy": "●", "unavailable": "✖", "unknown": "○"}.get(
            self.health, "○"
        )
        label = self.health.capitalize() if self.health != "unknown" else "Connecting"
        return f"{provider} {glyph} {label}"


# -- singleton ----------------------------------------------------------------

_STATE = MemoryHealthState()


def get_health_state() -> MemoryHealthState:
    return _STATE


def reset_health_state() -> None:
    """Test isolation / explicit state reset helper — resets to defaults."""
    _STATE.configured_provider = ""
    _STATE.active_provider = ""
    _STATE.health = "unknown"
    _STATE.reason = ""
    _STATE._last_probe_failure_at = 0.0
