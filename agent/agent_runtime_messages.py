"""Response-state predicates shared by runtime message normalization."""


def is_codex_interim(m: dict) -> bool:
    """Codex Responses interim turn: carries its own continuation state, replayed verbatim."""
    return bool(
        m.get("codex_reasoning_items")
        or m.get("codex_message_items")
        or m.get("finish_reason") == "incomplete"
    )
