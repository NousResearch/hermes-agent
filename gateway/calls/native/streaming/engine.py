"""Native-call engine selector (WP11).

Deterministically picks the call engine from config so the turn-based path
remains the default and future streaming is opt-in via a single config key.
"""
from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

TURN_BASED = "turn_based"
STREAMING = "streaming"
_VALID = {TURN_BASED, STREAMING}


class StreamingExtraNotInstalled(RuntimeError):
    """Raised when engine=streaming but the simplex-native-calls extra is absent."""


def _aiortc_available() -> bool:
    """True when the optional aiortc dependency is importable (no import side-effect)."""
    return importlib.util.find_spec("aiortc") is not None


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
    cognitive: str = "fake",
) -> Any:
    """Select the call engine and construct the pipeline accordingly.

    When engine is 'turn_based' (default), delegates to *turn_based_factory*
    unchanged — zero behavior change to existing native calls.

    When engine is 'streaming', constructs the pure-asyncio ``StreamingPipeline``
    (Slice 6 core) guarded by the optional aiortc dependency. The streaming path
    no longer depends on Pipecat — the Slice 6 reflex core + Fake cognitive ports
    run without it. If aiortc is absent, raises ``StreamingExtraNotInstalled``
    with a clear install hint rather than silently doing nothing.

    Args:
        config: Hermes config mapping (or None).
        turn_based_factory: Callable (or class) that builds the existing
            ``HermesVoiceTurnPipeline`` — called with no arguments.
        cognitive: Cognitive-port selector for the streaming path. ``"fake"``
            (default, Slice 6) wires deterministic fakes; ``"real"`` lands in
            Slice 7 and currently raises ``NotImplementedError``.

    Returns:
        A pipeline object produced by the chosen factory.

    Raises:
        StreamingExtraNotInstalled: When engine='streaming' but the optional
            ``simplex-native-calls`` (aiortc) extra is not installed.
    """
    engine = select_call_engine(config)
    logger.debug("native call engine selected: %s", engine)
    if engine == STREAMING:
        if not _aiortc_available():
            raise StreamingExtraNotInstalled(
                "calls.native.engine='streaming' requires the optional aiortc "
                "dependency. Install it with: "
                "pip install 'hermes-agent[simplex-native-calls]'"
            )
        from gateway.calls.native.streaming.aiortc_transport import (
            build_streaming_pipeline,
        )

        return build_streaming_pipeline(config, cognitive=cognitive)
    # Default: turn_based — delegate to the existing factory, unchanged.
    return turn_based_factory()
