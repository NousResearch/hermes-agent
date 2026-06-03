"""Helpers for resolving compression settings from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from agent.auxiliary_client import _compression_threshold_for_model, _is_codex_gpt55


@dataclass(frozen=True)
class CompressionSettings:
    threshold: float
    threshold_autoraised: Optional[Dict[str, float]]
    target_ratio: float


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float_or_default(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _model_compression_cfg(model_cfg: Any) -> Mapping[str, Any]:
    cfg = _mapping(model_cfg)
    return _mapping(cfg.get("compression"))


def resolve_compression_settings(
    *,
    model: Optional[str],
    provider: Optional[str] = None,
    compression_cfg: Any,
    model_cfg: Any,
) -> CompressionSettings:
    """Resolve effective compression threshold and summary target ratio.

    Precedence for ``threshold``:
    1. ``model.compression.threshold`` when explicitly configured.
    2. Built-in model-specific default, if any.
    3. Global ``compression.threshold``.

    ``target_ratio`` has no built-in model defaults, so
    ``model.compression.target_ratio`` overrides only the global
    ``compression.target_ratio``.
    """
    global_cfg = _mapping(compression_cfg)
    model_comp_cfg = _model_compression_cfg(model_cfg)

    global_threshold = _float_or_default(global_cfg.get("threshold"), 0.50)
    threshold_autoraised = None
    if model_comp_cfg.get("threshold") is not None:
        threshold = float(model_comp_cfg["threshold"])
    else:
        allow_codex_gpt55_autoraise = str(
            global_cfg.get("codex_gpt55_autoraise", True)
        ).strip().lower() in {"true", "1", "yes"}
        threshold = _compression_threshold_for_model(
            model,
            provider,
            allow_codex_gpt55_autoraise=allow_codex_gpt55_autoraise,
        )
        if threshold is None:
            threshold = global_threshold
        elif _is_codex_gpt55(model, provider) and threshold > global_threshold + 1e-9:
            threshold_autoraised = {
                "from": global_threshold,
                "to": threshold,
            }

    target_ratio = _float_or_default(global_cfg.get("target_ratio"), 0.20)
    if model_comp_cfg.get("target_ratio") is not None:
        target_ratio = float(model_comp_cfg["target_ratio"])

    return CompressionSettings(
        threshold=threshold,
        threshold_autoraised=threshold_autoraised,
        target_ratio=target_ratio,
    )
