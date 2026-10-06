"""Destination-only role compatibility; never mutate canonical conversation history."""

from typing import Any

from utils import base_url_host_matches

# Conservative compatibility for Mistral-family deployments reported in #20154;
# newer routes may accept the adjacency, but a model name alone cannot distinguish
# their validators. The provider/host also cover aliases on the native endpoint.
# Adapted from Abhirama Sonny's provider policy in NousResearch/hermes-agent#84944.
_MISTRAL_FAMILIES = ("mistral", "mixtral", "magistral", "ministral", "codestral", "devstral", "pixtral")


def project_tool_user_boundaries(
    messages: list[dict[str, Any]], *, provider: str | None = None,
    model: str | None = None, base_url: str | None = None,
) -> list[dict[str, Any]]:
    """Bridge strict endpoints' tool→user boundary on a fresh request list (#20154).

    The same canonical list may feed retries and different destinations. Keeping
    inserted rows local to this conversion needs no persistent marker or inverse
    repair. Fixed text keeps the projected prefix stable on a given destination.
    """
    provider = (provider or "").strip().lower()
    if provider == "moa":
        return messages  # The auxiliary SDK boundary knows the real aggregator.
    strict = (
        provider == "mistral"
        or any(family in (model or "").lower() for family in _MISTRAL_FAMILIES)
        or base_url_host_matches(base_url, "api.mistral.ai")
    )
    if not strict:
        return messages
    projected = []
    for message in messages:
        if projected and projected[-1].get("role") == "tool" and message.get("role") == "user":
            # Retain the previous interrupted-turn wire wording on strict routes;
            # this row is provider scaffolding, never a durable assistant answer.
            projected.append({"role": "assistant", "content": "Operation interrupted."})
        projected.append(message)
    return projected if len(projected) != len(messages) else messages
