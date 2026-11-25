"""Feishu interactive-card topics used by ``feishu/adapter.py``.

``authz``: who may answer an approval / update-prompt / clarify / choice card (the operator and
chat re-checks every click lane runs) and the chat_mode -> dm/group/forum mapping the chat-info
cache and card clicks share.
"""

from .authz import (
    cached_chat_type,
    is_operator_authorized,
    map_chat_type,
    pop_validated_prompt_state,
    resolve_source_chat_type,
    validate_card_action,
)

__all__ = [
    "cached_chat_type",
    "is_operator_authorized",
    "map_chat_type",
    "pop_validated_prompt_state",
    "resolve_source_chat_type",
    "validate_card_action",
]
