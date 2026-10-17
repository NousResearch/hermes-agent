"""A message a direct platform bot routes to a parked profile is refused out loud, never dropped.

A parked profile comes back on a rescan, so its traffic is temporarily unavailable. Relay frames and
webhooks are left unacknowledged for the platform to redeliver (``webhook_ingress``). A direct bot
(Telegram's PTB ``Updater``, Slack Socket Mode, Discord's gateway, Google Chat Pub/Sub) has already
confirmed the update to its platform before the gateway sees it, so nothing redelivers it. Holding it
here would be a second inbound queue outside the one durable queue, kept for a profile whose store is
the one that cannot be trusted. So the sender is told, in-channel, to resend, and nothing is admitted.
"""
import asyncio
import logging

logger = logging.getLogger(__name__)

# Generic on purpose: the chat is not told which profile the route targets or why it is parked.
PARKED_ROUTE_NOTICE = 'This assistant is temporarily unavailable; please resend your message in a moment.'


def drop_rejected_route(adapter, source):
    """Log an inbound event dropped because its explicit profile route targets a profile this
    gateway does not serve; when that profile is parked, also refuse it to the sender in-channel."""
    from gateway.run_runtime import profile_parked
    name = getattr(adapter, 'name', None) or getattr(getattr(source, 'platform', None), 'value', '?')
    chat_id = getattr(source, 'chat_id', '?')
    profile = getattr(source, '_rejected_route_profile', None)
    if adapter is None or not profile_parked(getattr(adapter, 'gateway_runner', None), profile):
        logger.warning(
            '[%s] Dropping inbound event for %s: explicit profile route targets an unserved profile', name, chat_id)
        return
    platform = getattr(getattr(source, 'platform', None), 'value', None) or '?'
    logger.warning(
        '[%s] Refusing inbound event for %s: routed profile %r on %s is parked (reason=profile_parked); '
        'nothing was admitted, the sender was asked to resend', name, chat_id, profile, platform)
    try:
        task = asyncio.get_running_loop().create_task(_send_refusal(adapter, source))
    except RuntimeError:
        logger.warning('[%s] Could not refuse the parked-profile event for %s: no running loop', name, chat_id)
        return
    tasks = getattr(adapter, '_background_tasks', None)
    if isinstance(tasks, set):
        tasks.add(task)
        task.add_done_callback(tasks.discard)


async def _send_refusal(adapter, source):
    """Reply in the sender's chat and thread, as the runner's own notices do."""
    try:
        metadata = adapter.gateway_runner._thread_metadata_for_source(source)
        await adapter.send(source.chat_id, PARKED_ROUTE_NOTICE, reply_to=getattr(source, 'message_id', None),
                           metadata=metadata)
    except Exception:
        logger.warning('[%s] Could not send the parked-profile refusal to %s',
                       getattr(adapter, 'name', '?'), getattr(source, 'chat_id', '?'), exc_info=True)
