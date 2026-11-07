"""In-memory reference implementation of the message (transcript) capability (test fixture).

Registered onto every assembled ``InMemoryBackend`` as the ``messages`` attribute. The storage layout and the row codecs live in ``transcript_store``; the verbs are split by topic over the mixins below, all reading the same scope namespace, so cross-scope leakage is structurally impossible. The turn and projection policy comes from ``_turn_policy`` only.
"""

from __future__ import annotations

from ..backend import register_repository
from .messages_compact import MessageCompactMixin
from .messages_reads import MessageReadsMixin
from .messages_turns import MessageTurnsMixin
from .messages_writes import MessageWritesMixin

__all__ = ["InMemoryMessageRepository"]


class InMemoryMessageRepository(MessageWritesMixin, MessageCompactMixin, MessageReadsMixin, MessageTurnsMixin):
    """See the ``persistence.conversation.messages`` protocol for the contract each method implements."""


register_repository("messages", InMemoryMessageRepository)
