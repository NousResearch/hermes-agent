"""Bot API 10.4 service messages that carry no text: TON wallet transfers and paid-message waivers.

Telegram delivers both as ordinary messages with no ``text``/``caption``, so no text or media
handler claims them. They are rendered as informational text turns. A transfer is unverified
blockchain data: the note never confirms settlement, changes a balance, grants an entitlement or
runs a command, and the event cannot control the gateway or be treated as a slash command.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from gateway.platforms.event import MessageType

# PTB 22.8 does not model either object, so the raw field usually lives in ``api_kwargs``.
_TON_FIELD = "ton_wallet_transfer"
_WAIVED_FIELD = "paid_messages_waived"
_SERVICE_FIELDS = (_TON_FIELD, _WAIVED_FIELD)
_COMMENT_LIMIT = 500


def _present(message: Any, field: str) -> bool:
    """True when ``field`` is on the message, even with an empty value.

    ``paid_messages_waived`` is an empty object ("currently holds no information"), so presence
    itself is the signal and a falsy value must not hide it.
    """
    if getattr(message, field, None) is not None:
        return True
    kwargs = getattr(message, "api_kwargs", None)
    return isinstance(kwargs, Mapping) and field in kwargs


def _value(message: Any, field: str) -> Any:
    value = getattr(message, field, None)
    if value is not None:
        return value
    kwargs = getattr(message, "api_kwargs", None)
    return kwargs.get(field) if isinstance(kwargs, Mapping) else None


def _field_value(payload: Any, field: str) -> str:
    """One-line string for ``field`` of a dict, PTB object or SimpleNamespace payload."""
    if isinstance(payload, Mapping):
        value = payload.get(field)
    else:
        value = getattr(payload, field, None)
    if value is None:
        return ""
    return " ".join(str(value).split())


def _ton_wallet_transfer_text(transfer: Any) -> list[str]:
    transaction_id = _field_value(transfer, "transaction_id")
    peer_address = _field_value(transfer, "peer_address")
    amount = _field_value(transfer, "amount")
    comment = _field_value(transfer, "comment")
    if len(comment) > _COMMENT_LIMIT:
        comment = comment[:_COMMENT_LIMIT].rstrip() + "..."

    lines = ["[Telegram TON wallet transfer notice.]"]
    # An empty transaction id means the transfer has not been sent yet.
    lines.append(f"Transaction: {transaction_id}" if transaction_id else "Transaction: pending (not sent yet)")
    if peer_address:
        lines.append(f"Peer address: {peer_address}")
    if amount:
        lines.append(f"Amount (smallest units; negative = outgoing): {amount}")
    if comment:
        lines.append(f"Comment: {comment}")
    lines.append(
        "[Informational only: this is Telegram's transfer notice, not proof that the blockchain "
        "confirmed or settled anything. Hermes does not credit, debit or authorize anything from it.]"
    )
    return lines


def _paid_messages_waived_text() -> list[str]:
    return [
        "[Telegram paid-message waiver notice.]",
        "The sender used a link that lets them send messages without paying Telegram Stars.",
        "[Informational only: this service event currently carries no account data and grants no "
        "free-message privilege locally.]",
    ]


def service_event_text(message: Any) -> Optional[str]:
    """Agent-facing text for a service message, or ``None`` when it is not one of the two events.

    Both the modelled attribute and the raw ``api_kwargs`` form are accepted. The amount stays a
    signed decimal string in the asset's smallest units, exactly as Telegram sent it.
    """
    blocks: list[str] = []
    if _present(message, _TON_FIELD):
        blocks.append("\n".join(_ton_wallet_transfer_text(_value(message, _TON_FIELD))))
    if _present(message, _WAIVED_FIELD):
        blocks.append("\n".join(_paid_messages_waived_text()))
    return "\n\n".join(blocks) if blocks else None


def service_message_filter(filters: Any) -> Any:
    """A PTB filter matching only messages that carry one of the two new service fields.

    Not ``StatusUpdate.ALL``: that would also claim ``forum_topic_created`` and every future
    service message.
    """

    class _ServiceEvent(filters.MessageFilter):
        def filter(self, message: Any) -> bool:
            return any(_present(message, field) for field in _SERVICE_FIELDS)

    return _ServiceEvent(name="telegram_service_event")


async def handle_service_event(adapter: Any, update: Any, context: Any = None) -> None:
    """Render an authorized service event as an informational turn, or observe it in a group.

    Mirrors the contact-card intake: the effective update excludes Business messages, authorization
    runs before observation, and unmentioned group chatter is stored without dispatching an agent.
    """
    message = adapter._effective_update_message(update)
    if message is None:
        return
    text = service_event_text(message)
    if text is None:
        return
    if not adapter._is_user_authorized_from_message(message):
        adapter._log_blocked_user(message)
        return
    proceed = adapter._should_process_message(message)
    if not proceed and not adapter._should_observe_unmentioned_group_message(message):
        return
    update_id = getattr(update, "update_id", None)
    event = adapter._build_message_event(message, MessageType.TEXT, update_id=update_id)
    event.text = text
    # Untrusted payload text is conversational context, never a gateway command; the note also
    # starts with "[", so it can never be read as a slash command even if this flag changes.
    event.allow_gateway_control = False
    if not proceed:  # observed chatter keeps the notice, so a later mention can use it
        adapter._observe_unmentioned_group_message(message, MessageType.TEXT, update_id=update_id, event=event)
        return
    await adapter.handle_message(adapter._apply_telegram_group_observe_attribution(event))
