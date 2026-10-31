"""Content-free duration fields shared by turn finalization and telemetry."""
import math


def valid_seconds(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def add_turn_seconds(agent, attribute: str, seconds) -> None:
    if valid_seconds(seconds):
        setattr(agent, attribute, getattr(agent, attribute, 0.0) + seconds)


def turn_timing_fields(agent) -> dict:
    # Parallel tool durations are summed work, and can exceed the turn's wall time.
    fields = {"tools_s": getattr(agent, "_turn_tool_seconds", None),
              "compaction_s": getattr(agent, "_turn_compaction_seconds", None),
              "model_s": getattr(agent, "_turn_model_seconds", None)}
    first = getattr(agent, "_turn_first_token_at", None)
    start = getattr(agent, "_turn_first_api_request_at", None)
    # Only a matching first-request stamp can define TTFB; turn admission cannot.
    if valid_seconds(first) and valid_seconds(start):
        fields["ttfb_s"] = first - start
    return {key: value for key, value in fields.items() if valid_seconds(value)}
