"""Automatic memory review of recorded speech, outside the transcript transaction."""
import asyncio
from contextlib import contextmanager
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
    from gateway.session_persistence import authority_persistence
    from persistence.conversation.messages import ConversationRequest
    ledger = getattr(authority, '_side_agents', None)
    if ledger is None:
        ledger = authority._side_agents = {}
    thread = threading.current_thread()
    ledger[thread] = [stop]
    try:
        with owner_scope(authority), _api_review_scope(authority, ref, live):
            target = transcript_target(authority, ref)
            history = [row.as_openai_message() for row in authority_persistence(authority).messages.read_conversation(
                target, ConversationRequest(include_lineage=True))]
            agent_type = _owned_review_agent(authority, ref, live, stop)
            with cold_review_parent(authority.runner, live.source, live.route, target, agent_type=agent_type) as parent:
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


@contextmanager
def _api_review_scope(authority, ref, live):
    from gateway.config import Platform
    from gateway.session_api_turn import api_execution, api_policy_scope, prepare_api_execution
    prepared = (prepare_api_execution(authority, ref, {'text': ''})
                if live.source.platform == Platform.API_SERVER else None)
    token = api_execution.set(prepared)
    try:
        with api_policy_scope():
            yield
    finally:
        api_execution.reset(token)


def _review_allowed(authority, ref, live, stop, parent):
    return (not stop.stopped.is_set() and not authority.retiring
            and authority.sessions.get(ref.session_id) is live and live.review is parent
            and (live.task is None or live.task.done()))


def _owned_review_agent(authority, ref, live, stop):
    """Keep the native enqueue/requeue policy; fence every eventual dispatch against this owner."""
    from run_agent import AIAgent

    class OwnedReviewAgent(AIAgent):
        def _spawn_background_review_now(self, **kwargs):
            from agent.memory_provider import spawn_context_thread
            launch = super()._spawn_background_review_now
            ledger = authority._side_agents

            def work():
                try:
                    if not _review_allowed(authority, ref, live, stop, self):
                        return  # a deferred snapshot may outlive its session or profile owner
                    launch(**kwargs)
                    # Stop can win after the first guard but before the run token exists.
                    if not _review_allowed(authority, ref, live, stop, self):
                        stop.hard_interrupt()
                    run = getattr(self, '_background_review_run', None)
                    if run is not None:
                        run.request_done.wait()
                except Exception:  # health: allow BLE001 -- best-effort fork, durable speech already committed
                    logger.warning('External conversation memory review dispatch failed', exc_info=True)
                finally:
                    ledger.pop(thread, None)

            thread = spawn_context_thread(work, name='external-memory-review')
            ledger[thread] = [stop]  # visible to shutdown before dispatch can create its run token
            try:
                thread.start()
            except BaseException:
                ledger.pop(thread, None)
                raise

    return OwnedReviewAgent
