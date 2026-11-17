"""Automatic memory review of recorded speech, outside the transcript transaction."""
import asyncio
import logging
import threading

logger = logging.getLogger(__name__)


def memory_interval(authority, live):
    from gateway.session_policy import policy_for_source
    from gateway.run import _load_gateway_config
    from tools.memory_tool import get_builtin_memory_config, get_builtin_memory_store_flags
    if live.source is None:
        return 0
    policy = policy_for_source(authority.runner, live.source)
    config = policy.config(authority, secrets=False) if policy else _load_gateway_config()
    if (policy is not None and policy.ignore_rules) or not any(get_builtin_memory_store_flags(config)):
        return 0
    return max(0, int(get_builtin_memory_config(config).get('nudge_interval', 10)))


def schedule_review(authority, ref, live):
    from gateway.session_runtime_workers import track_mutation
    # Coalesce while a task runs, but retain a crossing that arrives during review.
    live.external_review_pending = True
    pending = getattr(live, 'external_review_task', None)
    if pending is None or pending.done():
        live.external_review_task = track_mutation(authority, _review_when_idle(authority, ref, live))


async def _review_when_idle(authority, ref, live):
    try:
        while live.external_review_pending:
            while live.task is not None and not live.task.done():
                await asyncio.shield(live.task)
            if authority.retiring or authority.sessions.get(ref.session_id) is not live:
                return
            live.external_review_pending = False
            stop = _ReviewStop()
            await asyncio.to_thread(_review, authority, ref, live, stop)
    except Exception:  # health: allow BLE001 -- best-effort after durable append, like turn finalization
        logger.warning('External conversation memory review failed', exc_info=True)


class _ReviewStop:
    """The owner's side-work shutdown path must interrupt the review fork, not its cold parent."""
    def __init__(self):
        self.stopped = threading.Event()
        self.parent = None

    def hard_interrupt(self, message=None):
        from agent.background_review import cancel_background_review_for_live_turn
        self.stopped.set()
        if self.parent is not None:
            cancel_background_review_for_live_turn(self.parent, message=message or 'Session owner is stopping')


def _review(authority, ref, live, stop):
    from gateway.session_authorities import owner_scope
    from gateway.session_cold_review import cold_review_parent
    from gateway.session_local_recovery import transcript_target
    ledger = getattr(authority, '_side_agents', None)
    if ledger is None:
        ledger = authority._side_agents = {}
    thread = threading.current_thread()
    ledger[thread] = [stop]
    try:
        with owner_scope(authority):
            target = transcript_target(authority, ref)
            history = authority.db.get_messages_as_conversation(target, include_ancestors=True)
            with cold_review_parent(authority.runner, live.source, live.route, target) as parent:
                if 'memory' not in parent.valid_tool_names or stop.stopped.is_set() or authority.retiring:
                    return
                current = getattr(live.review, '_background_review_run', None)
                if current is not None and not current.request_done.is_set():
                    return  # an existing native/manual review already owns this session
                stop.parent = parent
                live.review = parent  # next-turn preemption uses the same guard as cold /refine
                # Preserve automatic enabled/tool policy, auxiliary routing and idle deferral.
                parent._spawn_background_review(messages_snapshot=history, review_memory=True)
                # A new turn or shutdown may have won while the cold parent was constructed.
                if stop.stopped.is_set() or authority.retiring or (live.task is not None and not live.task.done()):
                    stop.hard_interrupt()
                run = getattr(parent, '_background_review_run', None)
                if run is not None:
                    run.request_done.wait()
    finally:
        ledger.pop(thread, None)
