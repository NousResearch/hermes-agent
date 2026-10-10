"""Bounded, explicit opt-in content previews for gateway completion logs."""

import json

from agent.redact import redact_for_egress, REDACTION_UNAVAILABLE

PREVIEW_CHARS = 500


def completion_preview(value, *, json_value=False):
    """Redact the complete value before clipping, so partial secrets cannot escape."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str) if json_value else str(value)
        safe = redact_for_egress(text)
        # Use JSON quoting to keep one log record on one physical line.
        if safe == REDACTION_UNAVAILABLE:
            return json.dumps(safe)
        if len(safe) > PREVIEW_CHARS:
            safe = safe[:PREVIEW_CHARS] + "..."
        return json.dumps(safe, ensure_ascii=False)
    except Exception:
        return json.dumps(REDACTION_UNAVAILABLE)


def gateway_previews_enabled(config):
    logging_config = config.get("logging") if isinstance(config, dict) else None
    return isinstance(logging_config, dict) and logging_config.get("tool_previews") is True
