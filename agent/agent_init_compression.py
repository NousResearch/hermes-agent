"""Compression-config helpers moved out of the ``agent_init`` facade (provider-native compaction toggles)."""

from __future__ import annotations

from typing import Any, Optional

from agent.agent_runtime_helpers import _ra
from utils import is_truthy_value


def _compression_codex_settings(cfg: dict[str, Any]) -> tuple[str, bool, Optional[int]]:
    """``codex_app_server_auto`` / ``codex_responses_native`` / ``codex_responses_compact_threshold``."""
    from agent.agent_init import _positive_int
    app_server_auto = str(cfg.get("codex_app_server_auto", "native") or "native").lower()
    if app_server_auto not in {"native", "hermes", "off"}:
        _ra().logger.warning(
            "Invalid compression.codex_app_server_auto=%r; using 'native'. "
            "Valid values are: native, hermes, off.",
            app_server_auto,
        )
        app_server_auto = "native"
    # Native Responses server-side compaction (opt-in; gate in agent/native_compaction.py).
    # Truthy coercion so "false"/"off" strings stay disabled.
    responses_native = is_truthy_value(cfg.get("codex_responses_native", False))
    _raw = cfg.get("codex_responses_compact_threshold")
    compact_threshold = None
    if _raw is not None:
        compact_threshold = _positive_int(_raw, reject=(bool, float))
        if compact_threshold is None:
            _ra().logger.warning(
                "Invalid compression.codex_responses_compact_threshold=%r; "
                "using the automatic threshold derived from local compression.",
                _raw,
            )
    return app_server_auto, responses_native, compact_threshold
