"""Native-call engine selector (WP11).

Deterministically picks the call engine from config so the turn-based path
remains the default and future streaming is opt-in via a single config key.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from gateway.calls.native.streaming.pipecat_runtime import pipecat_available

logger = logging.getLogger(__name__)

TURN_BASED = "turn_based"
STREAMING = "streaming"
_VALID = {TURN_BASED, STREAMING}


class StreamingExtraNotInstalled(RuntimeError):
    """Raised when engine=streaming but the simplex-streaming extra is absent."""


def select_call_engine(config: Any) -> str:
    """Return the native-call engine: 'turn_based' (default) or 'streaming'.

    Reads config['calls']['native']['engine']. Missing/unknown -> 'turn_based'
    (with a warning for unknown non-empty values). Pure; safe on any mapping-ish
    config or None.
    """
    value = ""
    try:
        calls = config.get("calls") if hasattr(config, "get") else None
        native = calls.get("native") if hasattr(calls, "get") else None
        raw = native.get("engine") if hasattr(native, "get") else None
        value = str(raw or "").strip().lower()
    except Exception:
        value = ""
    if value in _VALID:
        return value
    if value:
        logger.warning("Unknown calls.native.engine %r; falling back to turn_based", value)
    return TURN_BASED


def build_native_pipeline(
    config: Any,
    *,
    turn_based_factory: Callable[[], Any],
) -> Any:
    """Select the call engine and construct the pipeline accordingly.

    When engine is 'turn_based' (default), delegates to *turn_based_factory*
    unchanged — zero behavior change to existing native calls.

    When engine is 'streaming', routes to the deferred Pipecat production seam,
    which raises ``PipecatIntegrationDeferred`` — failing loudly with a clear
    message rather than silently doing nothing.

    Args:
        config: Hermes config mapping (or None).
        turn_based_factory: Callable (or class) that builds the existing
            ``HermesVoiceTurnPipeline`` — called with no arguments.

    Returns:
        A pipeline object produced by the chosen factory.

    Raises:
        StreamingExtraNotInstalled: When engine='streaming' but the optional
            simplex-streaming (Pipecat) extra is not installed.
        PipecatIntegrationDeferred: When engine='streaming' and the extra is
            present (the real process_pcm16 aiortc bridge lands in a later slice).
    """
    engine = select_call_engine(config)
    logger.debug("native call engine selected: %s", engine)
    if engine == STREAMING:
        if not pipecat_available():
            raise StreamingExtraNotInstalled(
                "calls.native.engine='streaming' requires the optional Pipecat "
                "dependency. Install it with: pip install 'hermes-agent[simplex-streaming]'"
            )
        # Pipecat present; the real process_pcm16 aiortc bridge lands in Slice 6.
        from gateway.calls.native.streaming.pipecat_transport import build_pipeline

        return build_pipeline(config=config)  # raises PipecatIntegrationDeferred
    # Default: turn_based — delegate to the existing factory, unchanged.
    return turn_based_factory()
