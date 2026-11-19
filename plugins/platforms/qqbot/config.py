"""QQ Bot platform-specific configuration parsing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

DEFAULT_UPLOAD_PART_TIMEOUT_SECONDS = 300.0
_UPLOAD_PART_TIMEOUT_KEY = "upload_part_timeout_seconds"
_UPLOAD_PART_TIMEOUT_CONFIG_PATH = f"platforms.qqbot.extra.{_UPLOAD_PART_TIMEOUT_KEY}"


def _validate_upload_part_timeout_seconds(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{_UPLOAD_PART_TIMEOUT_CONFIG_PATH} must be a positive finite number of seconds")
    try:
        timeout = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{_UPLOAD_PART_TIMEOUT_CONFIG_PATH} must be a positive finite number of seconds"
        ) from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{_UPLOAD_PART_TIMEOUT_CONFIG_PATH} must be a positive finite number of seconds")
    return timeout


def resolve_upload_part_timeout_seconds(extra: Mapping[str, Any] | None) -> float:
    """Read the per-COS-part timeout from QQBot's ``extra`` config block."""
    if extra is None:
        return DEFAULT_UPLOAD_PART_TIMEOUT_SECONDS
    if not isinstance(extra, Mapping):
        raise ValueError("platforms.qqbot.extra must be a mapping")
    if _UPLOAD_PART_TIMEOUT_KEY not in extra:
        return DEFAULT_UPLOAD_PART_TIMEOUT_SECONDS
    return _validate_upload_part_timeout_seconds(extra[_UPLOAD_PART_TIMEOUT_KEY])
