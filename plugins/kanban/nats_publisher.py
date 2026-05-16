"""NATS JetStream publisher for kanban promote/ready events.

Best-effort publish: logs failures silently, never raises.
The gateway dispatcher must NEVER block or break because of NATS.

Subject format:  ``kanban.dispatch.<assignee>``
Payload format:  ``{"task_id": "...", "assignee": "...", "timestamp": N}``
"""

import json
import logging
import time

logger = logging.getLogger(__name__)


def publish_promoted(task_id: str, assignee: str, nats_url: str) -> None:
    """Publish a single JetStream message for a promoted kanban task.

    Args:
        task_id:  The promoted task's id (e.g. ``t_a1b2c3d4``).
        assignee: The task's assignee profile name.
        nats_url: NATS server URL from ``kanban.nats_server_url`` config.

    Best-effort: if ``nats-py`` is not installed or the connection/publish
    fails, the error is logged at DEBUG/WARNING level and swallowed.
    """
    if not task_id or not assignee or not nats_url:
        return

    try:
        import nats  # type: ignore[import-untyped]
    except ImportError:
        logger.debug(
            "NATS: nats-py not installed; skipping publish for task %s", task_id
        )
        return

    payload = json.dumps({
        "task_id": task_id,
        "assignee": assignee,
        "timestamp": int(time.time()),
    })
    subject = f"kanban.dispatch.{assignee}"
    try:
        # dispatch_once runs inside asyncio.to_thread (a thread pool), so
        # no event loop is active here and asyncio.run() is safe.
        async def _publish() -> None:
            nc = await nats.connect(nats_url)
            try:
                await nc.publish(subject, payload.encode())
            finally:
                await nc.close()

        import asyncio
        asyncio.run(_publish())

        logger.info("NATS: published to %s for task %s", subject, task_id)
    except Exception as exc:
        logger.warning(
            "NATS: publish failed for task %s (subject=%s): %s",
            task_id, subject, exc,
        )


def publish_batch(
    promoted: list[tuple[str, str]],
    nats_url: str,
) -> None:
    """Publish NATS messages for a batch of promoted tasks.

    Args:
        promoted:  List of ``(task_id, assignee)`` tuples.
        nats_url:  NATS server URL.

    Each task is published independently so a single failure does not
    silence the rest of the batch.
    """
    if not promoted or not nats_url:
        return
    for task_id, assignee in promoted:
        if assignee:
            publish_promoted(task_id, assignee, nats_url)
