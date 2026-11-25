"""Record-only speech inputs and durable append receipts for the canonical owner."""
from dataclasses import dataclass
from typing import Literal

from persistence.core.errors import AuthorityRefusal


@dataclass(frozen=True, slots=True)
class ExternalMessage:
    role: Literal['user', 'assistant']
    content: str
    client_id: str | None = None
    observed: bool = False


@dataclass(frozen=True, slots=True)
class ExternalMessageReceipt:
    id: int
    message_uid: str
    session_id: str
    role: Literal['user', 'assistant']
    client_id: str | None


@dataclass(frozen=True, slots=True)
class ExternalAppendOutcome:
    session_id: str
    messages: tuple[ExternalMessageReceipt, ...]
    inserted_ids: tuple[int, ...]
    review_due: bool


def validate_messages(messages):
    if not isinstance(messages, list) or not 1 <= len(messages) <= 100:
        raise AuthorityRefusal('invalid_params', field='messages')
    result = []
    for message in messages:
        if not isinstance(message, dict) or set(message) - {'role', 'content', 'client_id', 'observed'}:
            raise AuthorityRefusal('invalid_params', field='messages')
        role, content, client_id = (message.get(key) for key in ('role', 'content', 'client_id'))
        if role not in ('user', 'assistant'):
            error = AuthorityRefusal('invalid_params', field='role')
            error.wire_code, error.wire_message = 4000, "role must be 'user' or 'assistant'"
            raise error
        if not isinstance(content, str) or not content.strip() or len(content) > 1_000_000:
            raise AuthorityRefusal('invalid_params', field='content')
        if client_id is not None and (not isinstance(client_id, str) or not client_id.strip() or len(client_id) > 1024):
            raise AuthorityRefusal('invalid_params', field='client_id')
        observed = message.get('observed', False)
        if type(observed) is not bool:
            raise AuthorityRefusal('invalid_params', field='observed')
        result.append(ExternalMessage(role=role, content=content, client_id=client_id, observed=observed))
    return tuple(result)
