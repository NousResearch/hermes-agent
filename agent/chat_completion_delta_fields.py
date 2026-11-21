"""Message fields a chat-completions stream chunk carries beside its text, reasoning and tool calls."""

from typing import Any

# Fields an adapter relays in its chunk unchanged (the MoA aggregator's one-chunk stream) and the
# stream consumer sets on the assembled message, where ``normalize_response`` reads them.
RELAYED_MESSAGE_FIELDS = ("stop_details",)


def collect_delta_extras(delta: Any, refusal_parts: list[str], relayed: dict[str, Any]) -> None:
    """Accumulate a delta's refusal text and relayed fields.

    Neither is routed to the live display: the transport promotes a sole-payload refusal to content +
    ``content_filter`` and the loop surfaces it terminally.
    """
    refusal = getattr(delta, "refusal", None)
    if refusal is None and isinstance(getattr(delta, "model_extra", None), dict):
        refusal = delta.model_extra.get("refusal")
    if isinstance(refusal, str) and refusal:
        refusal_parts.append(refusal)
    for key in RELAYED_MESSAGE_FIELDS:
        if isinstance(value := getattr(delta, key, None), (dict, list)):
            relayed[key] = value
