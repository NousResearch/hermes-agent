"""Platform-agnostic reply-policy classification for gateway turns.

A single enum-valued policy, resolved once per inbound event from the same
config knobs each adapter's ``_should_process_message`` ingestion gate already
consults (``require_mention`` / ``free_response_*``), and carried on the
event's :class:`~gateway.session.SessionSource` metadata into the post-turn
delivery block.

This is a READ-ONLY classification for the *delivery* layer — it does not
touch any adapter's ingestion gate. The default (no group signal) is
:attr:`ReplyPolicy.DM`, the safe value that makes partial platform rollout
non-breaking: an unstamped event never triggers the tool-gated delivery
inversion.
"""

from enum import Enum


class ReplyPolicy(str, Enum):
    """How the delivery layer should treat an inbound turn.

    - ``DM``            — a direct message (or any non-group turn). Deliver by
                          default; never gated.
    - ``MENTION_GATED`` — a group that only processes messages that mention the
                          bot. Ingestion already gated it, so there is no
                          reply-ambiguity to solve; deliver by default.
    - ``FREE_RESPONSE`` — a group the bot listens to freely (``require_mention``
                          off, or the chat is in the platform's
                          ``free_response_*`` set). The ambiguous class the
                          reply gate targets.
    """

    DM = "dm"
    MENTION_GATED = "mention_gated"
    FREE_RESPONSE = "free_response"


def resolve_reply_policy(
    *,
    is_group: bool,
    in_free_response_set: bool,
    require_mention: bool,
) -> ReplyPolicy:
    """Classify a turn from the three inputs every mention/free-response gate reads.

    Mirrors the group branch of the adapters' ``_should_process_message`` as a
    value instead of a control-flow decision:

    - not a group                         -> ``DM``
    - group, in the free-response set      -> ``FREE_RESPONSE``
    - group, ``require_mention`` is false  -> ``FREE_RESPONSE``
    - group, ``require_mention`` is true   -> ``MENTION_GATED``
    """
    if not is_group:
        return ReplyPolicy.DM
    if in_free_response_set:
        return ReplyPolicy.FREE_RESPONSE
    if not require_mention:
        return ReplyPolicy.FREE_RESPONSE
    return ReplyPolicy.MENTION_GATED
