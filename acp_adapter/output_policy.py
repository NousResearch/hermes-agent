"""ACP output-detail policy helpers.

The ACP adapter keeps UI-facing transcript rendering separate from the
model-facing tool result. The default stays compact and human-readable, while
clients can explicitly request fuller visible transcript content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import os

ACPOutputDetail = Literal["condensed", "full"]

VALID_OUTPUT_DETAILS: tuple[ACPOutputDetail, ...] = ("condensed", "full")
DEFAULT_OUTPUT_DETAIL: ACPOutputDetail = "condensed"
DEFAULT_RESOURCE_MAX_BYTES = 512 * 1024


@dataclass(frozen=True)
class ACPOutputPolicy:
    """Resolved ACP output policy for one session/render operation."""

    detail: ACPOutputDetail = DEFAULT_OUTPUT_DETAIL
    resource_max_bytes: int = DEFAULT_RESOURCE_MAX_BYTES

    @property
    def full_visible_output(self) -> bool:
        return self.detail == "full"


def parse_output_detail(value: Any) -> ACPOutputDetail | None:
    """Return a normalized output detail, or None when the value is invalid."""

    raw = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "compact": "condensed",
        "summary": "condensed",
        "summarized": "condensed",
        "summarised": "condensed",
        "verbose": "full",
        "expanded": "full",
    }
    raw = aliases.get(raw, raw)
    if raw in VALID_OUTPUT_DETAILS:
        return raw  # type: ignore[return-value]
    return None


def normalize_output_detail(value: Any, *, default: ACPOutputDetail = DEFAULT_OUTPUT_DETAIL) -> ACPOutputDetail:
    """Normalize a user/client supplied output detail value."""

    parsed = parse_output_detail(value)
    if parsed is not None:
        return parsed
    return default


def _coerce_int(value: Any, default: int | None, *, minimum: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return default
    if minimum is not None and parsed < minimum:
        return minimum
    return parsed


def _config_acp_output(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    acp_cfg = config.get("acp")
    if not isinstance(acp_cfg, dict):
        return {}
    output_cfg = acp_cfg.get("output")
    return output_cfg if isinstance(output_cfg, dict) else {}


def resolve_acp_output_policy(
    *,
    session_detail: Any = None,
    config: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> ACPOutputPolicy:
    """Resolve ACP output policy with session > env > config > default precedence."""

    env_map = os.environ if env is None else env
    cfg = _config_acp_output(config)

    detail = normalize_output_detail(cfg.get("detail"), default=DEFAULT_OUTPUT_DETAIL)

    env_detail = env_map.get("HERMES_ACP_TOOL_OUTPUT_DETAIL") or env_map.get("HERMES_ACP_OUTPUT_DETAIL")
    if env_detail:
        detail = normalize_output_detail(env_detail, default=detail)
    if session_detail:
        detail = normalize_output_detail(session_detail, default=detail)

    resource_max = _coerce_int(cfg.get("resource_max_bytes"), DEFAULT_RESOURCE_MAX_BYTES, minimum=1)
    if env_map.get("HERMES_ACP_RESOURCE_MAX_BYTES") is not None:
        resource_max = _coerce_int(env_map.get("HERMES_ACP_RESOURCE_MAX_BYTES"), resource_max, minimum=1)
    if resource_max is None or resource_max <= 0:
        resource_max = DEFAULT_RESOURCE_MAX_BYTES

    return ACPOutputPolicy(
        detail=detail,
        resource_max_bytes=resource_max,
    )


def load_acp_output_policy(*, session_detail: Any = None) -> ACPOutputPolicy:
    """Resolve policy from the live Hermes config plus env/session overrides."""

    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception:
        config = None
    return resolve_acp_output_policy(session_detail=session_detail, config=config)


def should_advertise_output_config(config: dict[str, Any] | None = None) -> bool:
    """Return whether ACP responses should advertise configOptions.

    Default is false for compatibility with clients where configOptions compete
    with the model picker. The server still accepts `session/set_config_option`
    for advanced clients even when this is false.
    """

    cfg = _config_acp_output(config)
    value = cfg.get("advertise_config_option", False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
