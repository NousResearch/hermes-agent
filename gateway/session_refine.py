"""``/refine`` for a local session whose cached agent is gone (T7).

``/moa`` arms a one-shot model swap that evicts the session's cached agent, and its settle evicts
the MoA-built one, so the cached agent is not where a local session's conversation lives. The
session's stored transcript is: the review runs over it on a side agent built from the session's
frozen launch policy (the route ``/bg`` and ``/btw`` use, ``gateway/session_tui_side.py``), and its
summary lands on the session's event stream as ``review.summary``, as a turn-time review does.
Messaging sessions keep the cached-agent path unchanged.
"""
import asyncio
import contextlib
import uuid

from agent.i18n import t


def _local_session(runner, source):
    from gateway.config import Platform
    from gateway.session_setting_commands import local_authority_live
    if source is None or getattr(source, 'platform', None) != Platform.LOCAL:
        return None, None
    return local_authority_live(runner, source)


async def refine_local_session(runner, source, focus):
    """The ``/refine`` reply for a local session, or None when ``source`` is not one."""
    authority, live = _local_session(runner, source)
    if live is None:
        return None
    from gateway.session_contract import SessionRef
    from gateway.session_local_recovery import transcript_target
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(runner, live.source)
    if policy is None:
        return None
    ref = SessionRef(authority.profile_id, source.chat_id)

    def load():
        target = transcript_target(authority, ref)
        return target, authority.db.get_messages_as_conversation(target, include_ancestors=True)

    target, history = await asyncio.to_thread(load)
    if not history:
        return t('gateway.refine.nothing_yet')
    from agent.background_review import finish_background_review_run, prepare_background_review_run
    from gateway.session_tui_side import _side_agent, _side_scope, _spawn
    # The guard is the session's (its side agent is per call): a /refine while one runs joins it, as
    # the resident agent's does, and the session's next turn cancels it (``_execute_claimed``).
    run = prepare_background_review_run(live.review)
    tail = t('gateway.refine.focus_suffix', focus=focus) if focus else ''
    if run is None:
        return t('gateway.refine.started', focus_suffix=tail)
    task_id = f'refine_{uuid.uuid4().hex[:6]}'
    holder = [None]

    def publish(message):
        current = authority.sessions.get(ref.session_id)
        if current is not None:
            current.event_stream.publish(ref.session_id, {'text': str(message)}, event_type='review.summary')

    def body():
        try:
            review_body()
        finally:  # whatever raised first, the session's guard is released
            finish_background_review_run(live.review, run)
        return ''  # the summary (if any) was published above; an empty review.summary draws nothing

    def review_body():
        from agent.background_review import spawn_background_review_thread
        from agent.turn_finalizer import _clone_background_review_messages
        with _side_scope(authority, policy, task_id):
            # The session's own id: the review's usage is recorded against it, and the agent never
            # runs a turn of its own, so it writes no row. close() would tear down that id's terminal,
            # browser and memory provider (the session's), so only the agent's own clients are released.
            # No memory provider of its own either (release_clients would not stop it): the review
            # reads the transcript, as the background review fork does (skip_memory=True).
            agent = holder[0] = _side_agent(authority, policy, live, task_id, session_id=target,
                                            skip_memory=True)
            agent._end_session_on_close = False
            agent.background_review_callback = publish
            agent._print_fn = lambda *_a, **_k: None
            try:
                review, _prompt = spawn_background_review_thread(
                    agent, _clone_background_review_messages(history), review_memory=True,
                    review_skills='skill_manage' in getattr(agent, 'valid_tool_names', set()),
                    focus=focus or None, review_run=run, explicit=True)
                review()
            finally:
                with contextlib.suppress(Exception):
                    agent.release_clients()

    try:
        _spawn(authority, ref, task_id, 'review.summary', body, holder=holder)
    except RuntimeError as exc:  # the review thread could not start
        finish_background_review_run(live.review, run)
        return t('gateway.refine.start_failed', error=exc)
    return t('gateway.refine.started', focus_suffix=tail)
