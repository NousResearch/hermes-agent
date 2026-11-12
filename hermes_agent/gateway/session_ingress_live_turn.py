"""A producer delivery (Hermes Cloud relay) that answers or redirects the chat's running turn.

A messaging adapter's inbound passes ``GatewayInboundMixin._handle_message`` before the durable FIFO:
a typed reply to a pending clarify / slash-confirm / ``/update`` prompt resolves it there, a bare
"yes" answers a blocking command approval, and ``busy_input_mode`` steers or interrupts the running
turn. A relay delivery is committed straight to the FIFO by ``admit_producer``, so it skipped all
of that: the reply queued behind the very turn waiting for it, which sat out the whole clarify
timeout before the reply ran as a new turn. ``admit_producer`` offers it here once the delivery is
prepared (authorized, routed), its replay check passed and ``pre_gateway_dispatch`` ran.
"""
from collections import OrderedDict

# Deliveries consumed here are never committed, so the FIFO's replay check cannot recognize a
# relay re-send of one (an ACK lost in flight): (profile, platform, chat, provider message id) of
# each; the profile keeps one channel message delivered to two served profiles apart.
_consumed: OrderedDict = OrderedDict()
_CONSUMED_MAX = 512


def may_answer_live_turn(runner, event) -> bool:
    """Whether the chat's in-flight work could consume *event* right now: a running turn (its
    clarify or approval prompt, busy steer/interrupt), a pending clarify (an idle plugin command's
    ``ask_choice`` registers one before any turn slot is claimed) or an idle-time prompt
    (slash-confirm, ``/update``). A read only: nothing is routed before the delivery is prepared,
    and a miss only means it queues like any follow-up."""
    from hermes_agent.tools import clarify_gateway, slash_confirm
    key = runner._session_key_for_source(event.source)
    if runner._is_session_running(key) or clarify_gateway.has_pending(key) or slash_confirm.get_pending(key):
        return True
    state = runner._peek_session_state(key)
    return bool(state is not None and state.persistent.update_prompt_pending)


async def answers_live_turn(runner, event) -> bool:
    """True when *event* was consumed by the chat's in-flight work (never admitted); a reply to send
    rides ``event._consumer_reply``. False: admit it as usual (an interrupt-mode follow-up too: the
    interrupted turn ends and the FIFO runs it next). A bot peer's message is always admitted, so
    the bot loop guard charges it at execution as before. Runs inside ``producer_scope``."""
    if event.get_command() or getattr(event, 'internal', False) or getattr(event.source, 'is_bot', False):
        return False
    replay = _replay_key(event)
    consumed = await _route(runner, event)
    if consumed and replay is not None:
        _consumed[replay] = None
        while len(_consumed) > _CONSUMED_MAX:
            _consumed.popitem(last=False)
    return consumed


def _replay_key(event):
    source = event.source
    if not event.message_id:
        return None
    return (getattr(source, 'profile', None), source.platform.value, source.chat_id, event.message_id)


def was_consumed(event) -> bool:
    """A re-send of a delivery this module already consumed: nothing left to do, whether or not
    the work that consumed it is still running."""
    replay = _replay_key(event)
    return replay is not None and replay in _consumed


async def _route(runner, event) -> bool:
    source = event.source
    key = runner._session_key_for_source(source)
    reply = await runner._hm_pending_reply_intercepts(event, source, key)
    if reply is not None:
        event._consumer_reply = reply or None
        return True
    if not runner._is_session_running(key):
        return False
    return bool(await runner._route_plaintext_approval_while_busy(event, key)
                or await runner._hm_busy_preempt(event, source, key))
