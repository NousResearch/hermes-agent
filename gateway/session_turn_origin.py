"""Where a gateway turn came from: the ``turn_origin`` that ``pre_llm_call`` / ``post_llm_call`` receive.

Attribution, never authorization. ``native`` means Hermes passed a platform message through as its
adapter attributed it: ``sender_id`` is the account the adapter names. Hermes never labels native a turn
whose author it replaced (a viewer's, API or CLI submit, an automation, text re-sent on a borrowed
source), and adds nothing to the adapter's attribution: a bot reads ``peer`` only where the adapter flags
it, messages batched into one turn carry one of their senders, and text rendered for the account's own
command or action counts as that account's. Cron and managed-worker admissions leave
``execute_admission`` before it derives anything, so their turns (and every non-gateway caller) get None.
"""
from gateway.session_display import HIDDEN

_PRODUCER_AUTHORS = ('local_automation_v1', 'api_turn_v1', 'cli_turn_v1')


def row_turn_origin(row, live, policy, event):
    """``turn_origin`` of an admitted row, first match wins. ``policy`` is the session's own (a2a)
    policy; ``event`` is the restored event before an API execution forces it internal."""
    from gateway.config import Platform
    from gateway.session_a2a import forward_author
    from gateway.session_operator import is_viewer_input
    payload = row['payload']
    authors = [(payload.get(key) or {}).get('turn_author') for key in _PRODUCER_AUTHORS]
    # Another agent: an a2a forward, a Bot Chat delivery (the only automation that stamps an
    # author: a relayed or peer bot DM) or any producer author flagged a bot (peer run, relay -q).
    if (forward_author(policy) is not None or getattr(policy, 'source', None) == 'a2a'
            or authors[0] is not None or any(author and author.get('is_bot') for author in authors)):
        return 'peer'
    if live.source.platform == Platform.API_SERVER:
        return 'api'
    native = payload.get('native_text_v1')
    # ``event.internal``: a producer's wake committed on a chat route (an ``automation`` envelope).
    # ``inherit_channel_pin``: a goal continuation or /goal resume committed as plain native text.
    # ``kanban_card_v1`` passes ``is_viewer_input``; card sessions run in a managed worker today.
    if (event.internal or 'local_automation_v1' in payload or 'kanban_card_v1' in payload
            or (native is not None and native['event'].get('inherit_channel_pin') is True)):
        return 'automation'
    if native is not None:
        return _platform_sender(event)
    if 'cli_turn_v1' in payload:
        return 'cli'
    if is_viewer_input(row):
        # Hidden rows are model-facing client text (the onboarding opening prompt, widget intents).
        return 'automation' if (payload.get('display_v1') or {}).get('kind') == HIDDEN else 'viewer'
    return None


def direct_turn_origin(event):
    """``turn_origin`` of a turn the gateway runs without an admission (``/moa``, ``/model --once``,
    or any turn with no session authority). The no-authority heartbeat and goal continuation are
    synthetic but stay non-internal so authorization still applies: never native."""
    if (event.internal or getattr(event, '_heartbeat_session_id', None)
            or getattr(event, 'inherit_channel_pin', False)):
        return 'automation'
    return _platform_sender(event)


def _platform_sender(event):
    """A platform message as its adapter attributed it: None when it names no author (a /retry or
    leftover /steer on a borrowed source, a webhook route, an ntfy topic, no sender id), ``peer`` when
    another agent sent it (a bot account, the A2A platform's caller), else ``native``."""
    source = event.source
    platform = getattr(getattr(source, 'platform', None), 'value', None)
    if (getattr(event, 'borrowed_source', False) or platform in ('webhook', 'ntfy')
            or not getattr(source, 'user_id', None)):
        return None
    return 'peer' if getattr(source, 'is_bot', False) or platform == 'a2a' else 'native'
