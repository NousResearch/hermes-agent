"""Shared Telegram standard-reaction emoji canonicalisation."""


def canonical_standard_emoji(emoji: str) -> str | None:
    """Return Telegram's canonical spelling for a standard reaction emoji.

    Telegram's API enum may omit optional Unicode presentation selectors while
    user/model-facing text includes them. Match only against Telegram's known
    standard reaction set so valid multi-codepoint and ZWJ reactions remain
    intact and unsupported values fail closed.
    """
    raw = str(emoji or "").strip()
    if not raw:
        return None

    try:
        from telegram.constants import ReactionEmoji

        allowed = {
            str(getattr(value, "value", value))
            for value in ReactionEmoji
            if str(getattr(value, "value", value))
        }
    except (ImportError, TypeError):
        allowed = set()

    if not allowed:
        # Compatibility fallback for older/minimal PTB installations whose API
        # still validates the value. Avoid mutating ZWJ sequences blindly.
        if "\u200d" not in raw:
            return raw.rstrip("\ufe0e\ufe0f") or raw
        return raw

    if raw in allowed:
        return raw

    candidates = []
    if "\u200d" not in raw:
        candidates.append(raw.rstrip("\ufe0e\ufe0f"))
    else:
        candidates.extend(
            candidate
            for candidate in (
                raw.replace("\ufe0e", "\ufe0f"),
                raw.replace("\ufe0f", ""),
            )
            if candidate
        )

    for candidate in candidates:
        if candidate in allowed:
            return candidate
        for allowed_emoji in allowed:
            if allowed_emoji.replace("\ufe0f", "") == candidate.replace("\ufe0f", ""):
                return allowed_emoji
    return None
