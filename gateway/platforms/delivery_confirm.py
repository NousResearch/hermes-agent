"""Delivery confirmation for final sends (split from :mod:`gateway.platforms.base`)."""
import logging
from typing import TYPE_CHECKING, Any, Optional

from gateway.platforms.event import MessageEvent

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter, SendResult

logger = logging.getLogger(__name__)


def confirm_delivery_result(
    delivery_adapter: "BasePlatformAdapter", text_content: str, event: MessageEvent, result: "SendResult",
) -> "SendResult":
    """Downgrade a success that carries no message id to the ledger's unconfirmed state.

    An adapter can report ``success=True`` with ``message_id=None`` for a send that never landed.
    Two live shapes produce it: the Telegram rich fast-path returns that when the API result has no
    ``message_id``, and ``_send_chunks`` returns it when the formatter reduces non-empty content to
    zero chunks (it also stringifies a missing id, so the literal ``"None"`` arrives here). A final
    finalized as ``delivered`` is pruned, so a genuine loss was neither recovered nor logged.

    A delivery the platform acknowledged always carries an id, so this cannot fire for a landed
    message. Empty or whitespace-only text is excluded on purpose: nothing was owed, and the
    whitespace guard in every adapter legitimately reports that as a success.
    """
    from gateway.platforms.base import SendResult

    message_id = result.message_id
    if message_id is not None and str(message_id).strip().lower() in ("", "none"):
        message_id = None  # ``str(None)`` reaches here from the delivered list in ``_send_chunks``
    if not result.success or message_id or not text_content or not text_content.strip():
        return result
    logger.warning(
        "[%s] Final response reported success with no message id (chat %s, %d chars); "
        "treating it as unconfirmed so the delivery ledger can recover it",
        delivery_adapter.name, event.source.chat_id, len(text_content))
    return SendResult(success=False, error="unconfirmed_delivery_no_message_id", retryable=True)


def _split_post_delivery_entry(entry: Any) -> tuple[Optional[int], Any]:
    """``(generation, callback)`` from a post-delivery slot; legacy bare callbacks have no
    generation."""
    return entry if isinstance(entry, tuple) and len(entry) == 2 else (None, entry)
