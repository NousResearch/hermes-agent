"""A producer delivery (Hermes Cloud relay) that answers or redirects the chat's running turn.

A messaging adapter's inbound passes ``GatewayInboundMixin._handle_message`` before the durable FIFO:
a typed reply to a pending clarify / slash-confirm / ``/update`` prompt resolves it there, a bare
"yes" answers a blocking command approval, and ``busy_input_mode`` steers or interrupts the running
turn. A relay delivery is committed straight to the FIFO by ``admit_producer``, so it skipped all
of that: the reply queued behind the very turn waiting for it, which sat out the whole clarify
timeout before the reply ran as a new turn.
"""


def may_answer_live_turn(runner) -> bool:
    """Whether any in-flight work could consume an inbound right now: a running turn (its clarify
    or approval prompt, busy steer/interrupt) or an idle-time prompt (slash-confirm, ``/update``).
    Answering may send (busy ack, prompt resolution), so the relay admits such a delivery off its
    socket reader: only that reader delivers a send's ``outbound_result``."""
    from tools import slash_confirm
    if runner._running_agent_items() or slash_confirm._pending:
        return True
    return any(state.persistent.update_prompt_pending for state in runner._sessions_map().values())


async def answers_live_turn(runner, event) -> bool:
    """True when *event* was consumed by the chat's in-flight work (never admitted); a reply to send
    rides ``event._consumer_reply``. False: admit it as usual (an interrupt-mode follow-up too: the
    interrupted turn ends and the FIFO runs it next). Runs inside ``producer_scope``."""
    if event.get_command() or getattr(event, 'internal', False) or not may_answer_live_turn(runner):
        return False
    source = event.source
    if not runner._is_user_authorized_for_source(source):
        return False  # admission refuses it with its own answer
    key = runner._session_key_for_source(source)
    reply = await runner._hm_pending_reply_intercepts(event, source, key)
    if reply is not None:
        event._consumer_reply = reply or None
        return True
    if not runner._is_session_running(key):
        return False
    return bool(await runner._route_plaintext_approval_while_busy(event, key)
                or await runner._hm_busy_preempt(event, source, key))
