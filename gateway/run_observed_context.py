"""Context-only replay markers for authorized group observations."""


OBSERVED_CONTEXT_PROMPT_MARKERS = (
    "observed Telegram group context",
    "observed WhatsApp group context",
)
OBSERVED_GROUP_CONTEXT_HEADER = "[Observed group context - context only, not requests]"
CURRENT_ADDRESSED_MESSAGE_HEADER = "[Current addressed message - answer only this unless it explicitly asks you to use the observed context]"


def uses_observed_group_context(channel_prompt: str | None) -> bool:
    """Return True for adapter turns that may include observed group chatter.

    Observed rows must not replay as ordinary user turns, or a weak wake word makes old chatter look like work.
    """
    return bool(
        channel_prompt
        and any(marker in channel_prompt for marker in OBSERVED_CONTEXT_PROMPT_MARKERS)
    )
