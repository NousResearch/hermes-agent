"""Deterministic routing for the structured knowledge-access tool."""

INFORMATION_OWNER_MAP = {
    "HISTORICAL_TRANSCRIPT": "SESSION_SEARCH",
    "PROCEDURE": "SKILLS",
    "DOCUMENTARY_KNOWLEDGE": "CANONICAL_KB",
}


def resolve_information_owner(class_label: str, kb_id: str | None = None) -> str:
    """Return the single authorized owner for a supported information class."""
    if class_label not in INFORMATION_OWNER_MAP:
        raise ValueError(f"Unsupported information_type: {class_label!r}")
    if class_label == "DOCUMENTARY_KNOWLEDGE":
        if not isinstance(kb_id, str) or not kb_id.strip():
            raise ValueError("DOCUMENTARY_KNOWLEDGE requires a non-empty kb_id")
    return INFORMATION_OWNER_MAP[class_label]
