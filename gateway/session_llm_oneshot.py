"""``llm.oneshot`` on the shared gateway: a session-bound one-shot borrows that session's model.

Main's sidecar looked the session up in its own table and lent its live agent's route
(``tui_gateway/session_registry._main_runtime_from_agent``); a session-less call (project ideas)
ran on the ``task`` auxiliary backend. The sidecar's table never holds an owner session, so on the
shared gateway every Desktop one-shot fell through to the task backend. The owner answers it here:
the resident agent's route, else the session's own frozen launch route.

The model call never runs on the connection's serial dispatch: like the sidecar's long-handler
pool, the reply is written by a detached task, so a slow commit message does not hold up the next
request on that socket. Nor does it run on the loop's default executor, which session resume,
socket setup and disconnect cleanup share across every connection and profile: each owner has its
own small pool, and a call past its bound is refused by name instead of queueing without limit.
"""
import asyncio
from concurrent.futures import Future
from contextvars import copy_context
from functools import partial
import logging
import threading

from gateway.session_busy_controls import authorize
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)

_FIELDS = frozenset({'session_id', 'profile', 'template', 'instructions', 'input', 'variables', 'task',
                     'temperature', 'max_tokens'})
_TEXT = ('template', 'instructions', 'input', 'task')
# Per owner: model calls running at once, and calls admitted in all (running + queued). A call holds
# its slot until its thread returns, even after its socket closed.
_WORKERS = 4
_ADMITTED = 16


def handlers(connection):
    return {'llm.oneshot': partial(oneshot, connection)}


class _Pool:
    """One owner's one-shot threads, its admitted calls, and the answer task of each per connection.

    Its own FIFO, not a ThreadPoolExecutor: an executor keeps a cancelled queued call (its params and
    copied context) until a worker dequeues it, so with every worker parked on the model a reconnect
    loop would pile up payloads past the bound. Here a cancelled call leaves the queue at once."""

    def __init__(self):
        self.lock = threading.Lock()  # ``calls``, ``queue`` and ``threads`` are shared with the workers
        self.calls = set()  # admitted calls, queued or running: at most _ADMITTED
        self.queue = {}  # queued call -> its work, oldest first
        self.threads = 0  # worker threads alive: at most _WORKERS
        self.answers = {}  # answer task -> connection
        self.closed = False

    def submit(self, work):
        """Admit ``work`` (or refuse it: ``runtime_draining``, ``oneshot_busy``) and return its Future."""
        call = Future()
        with self.lock:
            if self.closed:
                raise RuntimeStoreError('runtime_draining')
            if len(self.calls) >= _ADMITTED:
                raise RuntimeStoreError('oneshot_busy', detail=f'{_ADMITTED} one-shot generations already in flight')
            self.calls.add(call)
            self.queue[call] = work
            spawn = self.threads < _WORKERS
            self.threads += spawn
        call.add_done_callback(self.release)
        if spawn:
            # Daemon: a call stuck on its model must not hold up interpreter exit (it writes no state).
            from agent.memory_provider import spawn_context_thread
            try:
                spawn_context_thread(self.drain, name='hermes-oneshot').start()
            except RuntimeError:
                # No new worker (thread exhaustion). A live worker drains this call; with none it
                # would never run, so it leaves the queue with its slot and the error goes out.
                with self.lock:
                    self.threads -= 1
                    orphaned = not self.threads
                if orphaned and call.cancel():
                    raise
        return call

    def release(self, call):
        """The call finished, or was cancelled while queued: its slot and its queued work go now."""
        with self.lock:
            self.calls.discard(call)
            self.queue.pop(call, None)

    def drain(self):
        while True:
            with self.lock:
                if not self.queue:
                    self.threads -= 1
                    return
                call = next(iter(self.queue))
                work = self.queue.pop(call)
            if call.set_running_or_notify_cancel():
                try:
                    call.set_result(work())
                except BaseException as exc:  # health: allow BLE001 -- the Future hands every failure to _answer, which logs it with exc_info
                    call.set_exception(exc)
            del call, work  # a finished call's payload is not kept while this thread waits for more


def _pool(authority) -> _Pool:
    pool = getattr(authority, '_oneshot_pool', None)
    if not isinstance(pool, _Pool):
        pool = authority._oneshot_pool = _Pool()
    return pool


def forget(connection):
    """The socket closed: drop its queued calls and answers. A call already running in its thread
    cannot be interrupted; it finishes, keeps its slot until then, and its reply is discarded."""
    pool = getattr(connection.authority, '_oneshot_pool', None)
    for task, owner in list(pool.answers.items()) if isinstance(pool, _Pool) else ():
        if owner is connection:
            task.cancel()


def stop(authority):
    """Owner shutdown (whole runtime or one profile's unserve): refuse new calls, cancel every answer
    and queued call. Running calls finish on their own (no SessionDB writes, only the model reply)."""
    pool = _pool(authority)  # an owner that never served one still refuses from now on
    with pool.lock:
        pool.closed = True
        queued = list(pool.queue)
    for task in list(pool.answers):
        task.cancel()
    for call in queued:
        call.cancel()


class DeferredReply:
    """A handler result answered later: ``dispatch`` hands over the request id and sends no frame."""

    def __init__(self, work):
        self.work = work  # blocking callable -> result dict; runs on the owner's one-shot pool

    def start(self, connection, rid, check=None):
        """Admit the call (or refuse it: ``oneshot_busy``, ``runtime_draining``) and answer it later;
        ``check`` is the contract's result check, run before the reply is written."""
        pool = _pool(connection.authority)
        # The handler's context (profile scope, secret scope) carries into the thread.
        call = pool.submit(partial(copy_context().run, self.work))
        task = asyncio.get_running_loop().create_task(_answer(connection.transport, rid, call, check))
        pool.answers[task] = connection

        def settled(done):
            pool.answers.pop(done, None)
            if done.cancelled():
                call.cancel()  # still queued: leaves the queue, never runs (also before its first step)
        task.add_done_callback(settled)


async def _answer(transport, rid, call, check):
    try:
        result = await asyncio.wrap_future(call)
        if check is not None:
            check(result)
        frame = {'jsonrpc': '2.0', 'id': rid, 'result': result}
    except Exception as exc:  # the request's only answer: every failure must reach the client
        # The sidecar's own failure code and text (tui_gateway/methods_session.py).
        logger.warning('llm.oneshot failed: %s', exc, exc_info=True)
        frame = {'jsonrpc': '2.0', 'id': rid,
                 'error': {'code': 5030, 'message': f'one-shot generation failed: {exc}'}}
    await transport.write_async(frame)


def _validate(params):
    from agent.oneshot import PROMPT_TEMPLATES
    if (set(params) - _FIELDS or any(key in params and params[key] is not None and not isinstance(params[key], str)
                                     for key in (*_TEXT, 'session_id'))
            or not isinstance(params.get('variables') or {}, dict)
            or any(key in params and params[key] is not None
                   and (isinstance(params[key], bool) or not isinstance(params[key], (int, float)))
                   for key in ('temperature', 'max_tokens'))):
        raise RuntimeStoreError('invalid_params')
    template = (params.get('template') or '').strip()
    if template and template not in PROMPT_TEMPLATES:
        raise RuntimeStoreError('invalid_params', field='template', detail='unknown one-shot template')
    if not template and not (params.get('instructions') or '').strip() and not (params.get('input') or '').strip():
        raise RuntimeStoreError('invalid_params', field='input', detail='needs a template or instructions/input')


def _session_runtime(authority, ref):
    """``main_runtime`` for the auxiliary client: the resident agent's live route (main's sidecar),
    snapshotted here on the loop, else the session's own frozen launch route as its next turn
    resolves it (read from config in the worker)."""
    from tui_gateway.session_registry import _main_runtime_from_agent
    agent = authority.agent(ref)
    if agent is not None:
        runtime = _main_runtime_from_agent(agent)
        return lambda: runtime
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    if policy is None:
        return lambda: None

    def resolve():
        from gateway.run_turn_prepare import _resolve_policy_agent_runtime
        # fallback=False: only the session's primary route. If it cannot resolve (an AuthError), the
        # one-shot fails with 5030 instead of walking the route's fallback chain, because that walk
        # sets the runner-wide fallback notice the next turn would report.
        model, kwargs = _resolve_policy_agent_runtime(authority.runner, policy, fallback=False)
        runtime = {key: kwargs[key] for key in ('provider', 'base_url', 'api_key', 'api_mode') if kwargs.get(key)}
        return {**runtime, 'model': model} if model else runtime or None
    return resolve


async def oneshot(connection, ref, params):
    _validate(params)
    authority = connection.authority
    if ref.session_id:
        authorize(connection, ref, params, 'session:read')
        runtime = _session_runtime(authority, ref)
    else:
        from gateway.session_authorities import require_serving
        require_serving(authority)
        if 'session:read' not in connection.actor.capabilities:
            raise RuntimeStoreError('permission_denied')
        if params.get('profile'):
            from pathlib import Path
            from hermes_cli.profiles import profile_matches_home
            if not profile_matches_home(params['profile'], Path(authority.profile_id)):
                raise RuntimeStoreError('profile_mismatch')
        runtime = lambda: None  # the task backend

    def work():
        from agent.oneshot import run_oneshot
        from gateway.session_authorities import owner_scope
        with owner_scope(authority):
            return {'text': run_oneshot(
                instructions=params.get('instructions') or '', user_input=params.get('input') or '',
                template=(params.get('template') or '').strip() or None, variables=params.get('variables') or {},
                task=(params.get('task') or '').strip() or 'title_generation',
                max_tokens=int(params.get('max_tokens') or 1024),
                temperature=0.3 if params.get('temperature') is None else float(params['temperature']),
                main_runtime=runtime())}
    return DeferredReply(work)
