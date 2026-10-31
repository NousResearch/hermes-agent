"""Shared ephemeral channel-prompt policy for messaging adapters."""


def resolve_channel_prompt(
    config_extra: dict,
    channel_id: str,
    parent_id: str | None = None,
) -> str | None:
    """Resolve a channel prompt, including an optional global policy.

    Looks up ``channel_prompts`` in the adapter's ``config.extra`` dict.  A
    string ``"*"`` entry is a global policy prepended to every resolved prompt.
    The channel-specific tier still prefers an exact *channel_id* match and
    falls back to *parent_id* (useful for forum threads / child channels).

    Returns the non-blank parts joined by a blank line, or None.  Blank or
    non-string wildcard values are ignored.  An exact/parent prompt identical
    to the wildcard after trimming is emitted once, while still counting as
    the selected exact/parent tier.
    """
    prompts = config_extra.get("channel_prompts") or {}
    if not isinstance(prompts, dict):
        return None

    parts: list[str] = []
    wildcard = prompts.get("*")
    if isinstance(wildcard, str):
        wildcard = wildcard.strip()
        if wildcard:
            parts.append(wildcard)

    for key in (channel_id, parent_id):
        if not key or key == "*":
            continue
        prompt = prompts.get(key)
        if prompt is None:
            continue
        prompt = str(prompt).strip()
        if not prompt:
            continue
        if prompt not in parts:
            parts.append(prompt)
        break
    return "\n\n".join(parts) or None
