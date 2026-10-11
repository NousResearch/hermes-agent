"""Recipient-scoped continuable cron handoffs."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def create_cron_handoff_thread(
    job: dict, adapter, chat_id: str, name: str, platform_name: Optional[str],
) -> Optional[str]:
    """Only persisted scheduling identity, never an authorization allowlist, identifies a recipient."""
    kwargs = {}
    if str(platform_name or getattr(adapter, "name", "")).lower() == "discord":
        from cron.scheduler_delivery import _resolve_origin

        origin = _resolve_origin(job) or {}
        recipient = origin.get("user_id")
        if (str(origin.get("platform", "")).lower() != "discord"
                or not isinstance(recipient, (str, int))
                or not str(recipient).isascii() or not str(recipient).isdigit()
                or int(recipient) <= 0):
            logger.warning(
                "Job '%s': Discord cron thread has no explicit recipient; "
                "falling back to the configured channel", job.get("id", "?"))
            return None
        kwargs["recipient_user_id"] = str(recipient)
    return await adapter.create_handoff_thread(chat_id, name, **kwargs)
