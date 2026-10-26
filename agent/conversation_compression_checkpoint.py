"""Fail-closed pre-compress checkpoint gate for ``agent.conversation_compression``.

The refusal type, its remediation text and the startup warning for
``compression.checkpoint_required`` (``conversation_compression`` re-exports the refusal
helpers), the init refusals for compaction authorities no checkpoint can precede, and the
record for lossy authorities outside compression the armed gate withholds.
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


def _refuse_checkpoint_required_on_codex_app_server(checkpoint_required: bool, api_mode: str | None) -> None:
    """Fail closed at init: the codex app-server compacts its own thread without a truthful
    pre-compaction boundary (default "native" mode), so a required checkpoint can't be
    guaranteed — the compress_context() guard alone cannot cover native turns."""
    if checkpoint_required and api_mode == "codex_app_server":
        raise RuntimeError(
            "BLOCKED_MISSING_PREREQUISITE: compression.checkpoint_required "
            "is incompatible with the codex_app_server API mode: the codex "
            "agent compacts its own thread without a truthful pre-compaction "
            "transcript boundary, so a required pre-compress checkpoint "
            "cannot be guaranteed. Disable compression.checkpoint_required "
            "or use a non-app-server API mode."
        )


def _refuse_checkpoint_required_on_plugin_context_engine(checkpoint_required: bool, engine: Any) -> None:
    """Fail closed at init when a plugin context engine may compact where no checkpoint runs.

    ``compress()`` is dispatched right after ``on_pre_compress()``; an engine compacting from
    ``on_turn_complete()`` or its own scheduler never crosses that seam. Refused before the
    engine slot is assigned, so the operator sees the incompatibility before a turn exists.
    """
    if checkpoint_required is not True:
        return
    from agent.context_engine import engine_compacts_outside_compress

    unsafe, reason = engine_compacts_outside_compress(engine)
    if not unsafe:
        return
    # Class name, not ``engine.name``: an abstract property on an object the resolver just
    # declined to trust; a raising getter would replace the diagnosis.
    raise RuntimeError(
        "BLOCKED_MISSING_PREREQUISITE: compression.checkpoint_required is incompatible with "
        f"context engine '{type(engine).__name__}': {reason}. An engine whose only lossy "
        "authority is compress() may declare compacts_outside_compress = False; otherwise "
        "disable compression.checkpoint_required or use the built-in compressor."
    )


_checkpoint_gate_warned: set[str] = set()


def _record_checkpoint_gate_suppression(agent: Any, authority: str) -> None:
    """Count a lossy authority the armed gate withheld; warn once per process per authority.

    The suppression itself is re-evaluated on every call, only the log is deduplicated (per
    authority, so an operator sees which path was lost). The line names the availability
    trade: with every other reclamation path withheld, a checkpoint outage halts the session.
    """
    agent._checkpoint_gate_suppression_count = getattr(agent, "_checkpoint_gate_suppression_count", 0) + 1
    if authority in _checkpoint_gate_warned:
        return
    _checkpoint_gate_warned.add(authority)
    logger.warning(
        "compression.checkpoint_required is enabled: %s is withheld so the checkpoint-gated "
        "compressor stays the only lossy authority. If the checkpoint provider is unavailable, "
        "compaction fails closed and the session halts instead of compacting unarchived context.",
        authority,
    )


def _prune_would_have_run(compressor: Any, current_tokens: int) -> bool:
    """Only a prune that would have run is a withheld authority.

    The built-in answers via ``would_proactively_prune()``. An engine that overrides
    ``prune_tool_results_only`` owns its trigger policy, so occupying the hook is the
    precondition; the inherited ABC no-op is not an authority. A raising predicate reads as
    "would have run": the call was withheld either way.
    """
    from agent.context_engine import ContextEngine

    prune = getattr(compressor, "prune_tool_results_only", None)
    if not callable(prune) or getattr(prune, "__func__", None) is ContextEngine.prune_tool_results_only:
        return False
    predicate = getattr(compressor, "would_proactively_prune", None)
    if not callable(predicate):
        return True
    try:
        return bool(predicate(current_tokens))
    except Exception:
        logger.debug("would_proactively_prune() failed; counting the withheld prune", exc_info=True)
        return True


def _checkpoint_gate_withholds_turn_hook(agent: Any, engine: Any) -> bool:
    """Whether the armed gate withholds ``on_turn_complete`` from ``engine`` this turn.

    The init refuse fires once, but the engine slot and the hook are plain attributes a later code
    path can rebind (on_session_start, a class monkeypatch, a rebound context_compressor), so the
    verdict is re-resolved per turn. Withheld rather than raised: the dispatch swallows exceptions.
    An engine declaring ``compacts_outside_compress = False`` keeps observing turns.
    """
    try:
        from agent.context_engine import engine_compacts_outside_compress

        unsafe = engine_compacts_outside_compress(engine)[0]
    except Exception:
        # Fail closed: a failed resolution (the lazy import, a raising descriptor on the engine)
        # must not read as "safe".
        logger.debug("compaction-authority resolution failed; withholding on_turn_complete", exc_info=True)
        unsafe = True
    if unsafe:
        _record_checkpoint_gate_suppression(
            agent, "the context engine's on_turn_complete hook (an engine that only observes may "
            "declare compacts_outside_compress = False)",
        )
    return unsafe
