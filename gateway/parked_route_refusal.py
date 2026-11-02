"""A message a direct platform bot routes to a parked profile is refused out loud, never dropped.

A parked profile comes back on a rescan, so its traffic is temporarily unavailable. Webhooks are
answered 503 for the provider to redeliver (``webhook_ingress``). A direct bot (Telegram's PTB
``Updater``, Slack Socket Mode, Discord's gateway, Google Chat Pub/Sub) has already confirmed the
update to its platform before the gateway sees it, so the platform never redelivers it (Discord's own
missed-message backfill, when enabled, can still pick it up after recovery). Holding it here would be
a second inbound queue outside the one durable queue, kept for a profile whose store is the one that
cannot be trusted. So the sender is told, in-channel, to resend, and nothing is admitted.

A relay frame is refused the same way (:class:`ParkedProfileRefusal`), and acknowledged: the park
lasts until a rescan or a restart, and the connector drains its per-instance buffer in order,
ack-gated, so a withheld frame would hold back every later frame for every profile on that instance;
a live frame has no redelivery at all. Only transient refusals (a busy store, a drain) withhold it.

A transport that answers only after an awaited dispatch (Google Chat HTTP events) has not acknowledged
the delivery yet: it runs that dispatch in :func:`retryable_ingress` and answers the named, retryable
:class:`ParkedRouteRefused` instead, so the platform redelivers it.
"""
import asyncio
import contextvars
import logging
import time
from contextlib import contextmanager

from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)


class ParkedRouteRefused(Exception):
    """An unacknowledged delivery routed to a parked profile: nothing was admitted, answer ``status``."""
    code, status, headers = 'profile_parked', 503, {'Retry-After': '30'}

    def __init__(self):
        super().__init__('The routed profile is not being served right now; retry this delivery')


class ParkedProfileRefusal(RuntimeStoreError):
    """Producer admission refused because the delivery routes to the parked *profile*: retryable
    (``runtime_draining``), but for as long as the park lasts, so an ordered transport acknowledges it
    and refuses it to the sender (:func:`refuse_parked_route`) instead of redelivering it."""

    def __init__(self, profile):
        super().__init__('runtime_draining', field='profile', detail='the routed profile is parked')
        self.profile = profile


class _RetryableIngress:
    def __init__(self):
        self.open, self.refused = True, False


_retryable_ingress = contextvars.ContextVar('parked_route_retryable_ingress', default=None)


@contextmanager
def retryable_ingress():
    """Scope an awaited dispatch its transport has not acknowledged. A parked route inside it is only
    recorded (``scope.refused``) for the caller to raise :class:`ParkedRouteRefused`; no in-chat notice.
    Closed on exit, so a task that outlives the dispatch still refuses in-channel."""
    scope = _RetryableIngress()
    token = _retryable_ingress.set(scope)
    try:
        yield scope
    finally:
        scope.open = False
        _retryable_ingress.reset(token)

# Generic on purpose: the chat is not told which profile the route targets or why it is parked.
# Worded as an outage notice Discord's backfill recognises (``_is_down_notice_content``): the reply
# must not mark the never-admitted message answered, so backfill picks it up once it is served again.
PARKED_ROUTE_NOTICE = 'The agent is currently unavailable; please resend your message in a moment.'
# One notice per (profile, platform, chat) per window: a relay reconnect drains its whole backlog
# for a parked profile at once, and N frames must not post N notices in the same chat.
PARKED_NOTICE_WINDOW_S = 60.0
_monotonic = time.monotonic


def _notice_due(adapter, profile, platform, chat_id):
    """True (and the window restarts) when this chat was not told about *profile* in the window."""
    sent = getattr(adapter, '_parked_notice_sent_at', None)
    if not isinstance(sent, dict):
        sent = {}
        try:
            adapter._parked_notice_sent_at = sent
        except AttributeError:
            return True
    now, key = _monotonic(), (profile, platform, chat_id)
    for stale in [k for k, at in sent.items() if now - at >= PARKED_NOTICE_WINDOW_S]:
        del sent[stale]
    if key in sent:
        return False
    sent[key] = now
    return True


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
    refuse_parked_route(adapter, source, profile)


def refuse_parked_route(adapter, source, profile, reply_to=None):
    """Refuse an event routed to the parked *profile*, nothing admitted: answered retryably inside
    :func:`retryable_ingress`, else told to the sender in-channel, in reply to *reply_to* (default:
    the source's message), off the caller, which may be the transport reader a relay send waits on."""
    name = getattr(adapter, 'name', None) or getattr(getattr(source, 'platform', None), 'value', '?')
    chat_id = getattr(source, 'chat_id', '?')
    platform = getattr(getattr(source, 'platform', None), 'value', None) or '?'
    scope = _retryable_ingress.get()
    if scope is not None and scope.open:
        scope.refused = True
        logger.warning(
            '[%s] Refusing inbound event for %s: routed profile %r on %s is parked (reason=profile_parked); '
            'nothing was admitted, the delivery is answered retryably', name, chat_id, profile, platform)
        return
    if not _notice_due(adapter, profile, platform, chat_id):
        logger.info(
            '[%s] Refusing inbound event for %s: routed profile %r on %s is parked (reason=profile_parked); '
            'nothing was admitted, the chat was already told within %.0fs', name, chat_id, profile, platform,
            PARKED_NOTICE_WINDOW_S)
        return
    logger.warning(
        '[%s] Refusing inbound event for %s: routed profile %r on %s is parked (reason=profile_parked); '
        'nothing was admitted, a resend notice to the sender is scheduled', name, chat_id, profile, platform)
    try:
        task = asyncio.get_running_loop().create_task(
            _send_refusal(adapter, source, reply_to or getattr(source, 'message_id', None)))
    except RuntimeError:
        logger.warning('[%s] Could not refuse the parked-profile event for %s: no running loop', name, chat_id)
        return
    tasks = getattr(adapter, '_background_tasks', None)
    if isinstance(tasks, set):
        tasks.add(task)
        task.add_done_callback(tasks.discard)


async def _send_refusal(adapter, source, reply_to):
    """Reply in the sender's chat and thread, as the runner's own notices do. Marked interim: it is no
    turn's answer, so it never seals another turn's open stream (relay Slack) or fills a pending
    answer slot (LINE's postback cache)."""
    from gateway.run import _interim_metadata
    try:
        metadata = _interim_metadata(adapter.gateway_runner._thread_metadata_for_source(source))
        result = await adapter.send(source.chat_id, PARKED_ROUTE_NOTICE, reply_to=reply_to, metadata=metadata)
        if getattr(result, 'success', True) is False:
            # Safe fields only: the raw ``error`` can echo tokens or message content.
            logger.warning('[%s] The parked-profile refusal to %s was not delivered (error_kind=%s, retryable=%s)',
                           getattr(adapter, 'name', '?'), getattr(source, 'chat_id', '?'),
                           getattr(result, 'error_kind', None), getattr(result, 'retryable', None))
    except Exception:
        logger.warning('[%s] Could not send the parked-profile refusal to %s',
                       getattr(adapter, 'name', '?'), getattr(source, 'chat_id', '?'), exc_info=True)
