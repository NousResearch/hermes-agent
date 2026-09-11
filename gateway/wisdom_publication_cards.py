"""Edit moderation cards on the gateway loop, without starting an agent turn."""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def refresh(adapters):
    from hermes_wisdom.publication_cards import PublicationCards
    from hermes_wisdom.service import WisdomService

    for adapter in adapters.values():
        edit = getattr(adapter, "edit_wisdom_publication", None)
        scoped = getattr(adapter, "_run_wisdom_profile_operation", None)
        if not callable(edit) or not callable(scoped):
            continue
        platform = str(getattr(adapter.platform, "value", adapter.platform))
        try:
            jobs = await scoped(
                lambda: PublicationCards(WisdomService()).claim(platform)
            )
            for job in jobs:
                success = False
                try:
                    await asyncio.wait_for(
                        edit(job["receipt"], job["view"]), timeout=30
                    )
                    success = True
                    logger.info(
                        "Wisdom publication card updated: platform=%s message=%s state=%s",
                        platform,
                        job["receipt"]["message_id"],
                        job["state"],
                    )
                except Exception as exc:
                    logger.warning(
                        "Wisdom publication card edit deferred (%s)", type(exc).__name__
                    )
                finally:
                    await scoped(
                        lambda: PublicationCards(WisdomService()).finish(
                            job, success=success
                        )
                    )
        except Exception as exc:
            logger.debug(
                "Wisdom publication card refresh deferred (%s)", type(exc).__name__
            )
