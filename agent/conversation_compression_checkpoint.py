"""Fail-closed pre-compress checkpoint gate for ``agent.conversation_compression``.

The refusal type, its remediation text and the startup warning for
``compression.checkpoint_required``; ``conversation_compression`` re-exports every name.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.memory_provider import PRE_COMPRESS_CHECKPOINT_API_VERSION

# Log-record parity with the origin module.
logger = logging.getLogger("agent.conversation_compression")


class CompressionCheckpointUnavailable(RuntimeError):
    """Raised when required durable pre-compress checkpointing is unavailable."""


# Shared by the startup warning and compress-time block so operators see the
# same recovery path: disable the fail-closed flag, or switch providers.
_CHECKPOINT_REQUIRED_REMEDIATION = (
    "set compression.checkpoint_required: false, or switch to a memory "
    "provider that implements checkpoint API v2"
)


def _active_memory_provider_label(agent: Any) -> str:
    """Human-readable active provider name, or an explicit none sentinel."""
    memory_manager = getattr(agent, "_memory_manager", None)
    if memory_manager is None:
        return "no active provider"
    providers = getattr(memory_manager, "providers", None)
    names: list[str] = []
    if providers is not None:
        try:
            iterable = list(providers)
        except TypeError:
            iterable = []
        for provider in iterable:
            name = getattr(provider, "name", None)
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return ", ".join(names) if names else "no active provider"


def _checkpoint_blocked(reason: str) -> CompressionCheckpointUnavailable:
    return CompressionCheckpointUnavailable(
        f"BLOCKED_MISSING_PREREQUISITE: required pre-compress checkpoint unavailable: {reason}"
    )


def _checkpoint_incapable(reason: str) -> CompressionCheckpointUnavailable:
    """Capability refusal: the gate can never pass with this provider set, so the
    message carries the config-level way out. Transient checkpoint failures keep
    the plain form — there the provider is capable and the right move is a retry."""
    return _checkpoint_blocked(f"{reason}. Recover by {_CHECKPOINT_REQUIRED_REMEDIATION}")


def _warn_checkpoint_required_without_capable_provider(agent: Any) -> None:
    """Startup warning when fail-closed compress will block (init itself must not refuse).

    Capability-probe exceptions are suppressed so a broken provider cannot crash
    agent construction — same fail-open style as the micro-compact warning.
    Probe failures log at DEBUG so a flaky probe can be diagnosed without
    refusing init or emitting the incapable-provider WARNING.
    """
    if getattr(agent, "compression_checkpoint_required", False) is not True:
        return
    memory_manager = getattr(agent, "_memory_manager", None)
    if memory_manager is not None:
        supports_checkpoint = getattr(memory_manager, "supports_pre_compress_checkpoint", None)
        if callable(supports_checkpoint):
            try:
                if bool(supports_checkpoint(PRE_COMPRESS_CHECKPOINT_API_VERSION)):
                    return
            except Exception as exc:
                logger.debug(
                    "checkpoint-required capability probe failed; init continues: %s",
                    exc,
                    exc_info=True,
                )
                return
    logger.warning(
        "compression.checkpoint_required is enabled but the active memory "
        "provider (%s) does not implement checkpoint API v%s. Compress will "
        "fail closed. Recover by %s.",
        _active_memory_provider_label(agent),
        PRE_COMPRESS_CHECKPOINT_API_VERSION,
        _CHECKPOINT_REQUIRED_REMEDIATION,
    )
