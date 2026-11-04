"""Physical worker liveness and cooperative Stop across every served authority."""
import asyncio
import os
import time

from gateway.session_authorities import all_authorities


def executing_sessions(authority):
    for sid, live in getattr(authority, 'sessions', {}).items():
        execution = getattr(getattr(live, 'event_stream', None), 'execution', None) or {}
        task = live.task
        if execution.get('execution_generation') is not None and task is not None and not task.done():
            yield sid, live, execution['execution_generation']


def uncounted_runtime_work(runner):
    """Count claims absent from the ordinary agent map, including pre-bootstrap managed turns."""
    running = getattr(runner, '_running_agents', {})
    authorities = all_authorities(runner)
    managed = sum(not worker.closed.is_set() for authority in authorities
                  for worker in getattr(authority, '_managed_workers', {}).values())
    return (managed + sum(live.route not in running and sid not in getattr(authority, '_managed_workers', {})
                for authority in authorities for sid, live, _ in executing_sessions(authority))
            + sum(len(mutation_tasks(authority)) + len(side_agents(authority)) for authority in authorities))


def managed_units(runner, *, authority=None, now=None):
    """One dict per LIVE managed worker, naming the turn a stop would interrupt.

    A managed turn runs in a child process the owner spawns, so the owner's own pid says nothing
    about it: this reports the SESSION, its ADMISSION, the child's pid (the verified interpreter
    once reserved, else the spawn handle), the served profile and how long it has run. The restart
    guard prints these, which is what lets an operator tell real in-flight work from a leak.
    Best-effort per worker: a source that cannot be read is omitted, never raised.
    """
    now = time.time() if now is None else now
    units = []
    for served in ([authority] if authority is not None else all_authorities(runner)):
        for session_id, worker in list(getattr(served, '_managed_workers', {}).items()):
            closed = getattr(worker, 'closed', None)
            if closed is not None and closed.is_set():
                continue
            units.append(managed_unit(session_id, worker, now))
    return units


def managed_unit(session_id, worker, now):
    """The named identity of one live managed worker: session, admission, child pid, profile."""
    handle = getattr(worker, 'process', None)
    bound = getattr(worker, 'worker', None)
    spawn_pid = getattr(handle, 'pid', None)
    unit = {'kind': 'managed', 'session': getattr(worker, 'session_id', None) or session_id,
            'pid': bound[0] if bound else spawn_pid}
    if getattr(worker, 'admission_id', None):
        unit['admission_id'] = worker.admission_id
    if getattr(worker, 'route', None):
        unit['route'] = worker.route
    if getattr(worker, 'profile', None):
        unit['profile'] = worker.profile
    if spawn_pid and spawn_pid != unit['pid']:
        unit['spawn_pid'] = spawn_pid
    started = getattr(worker, 'started_at', None)
    if started:
        unit['elapsed_s'] = round(now - started, 1)
    if getattr(worker, 'stop', None) is not None and worker.stop.is_set():
        unit['stopping'] = True
    return unit


def unmapped_session_units(runner, *, authority=None):
    """One dict per executing session the ordinary agent map does not hold (a managed turn whose
    worker is not registered yet, or an in-process claim outside ``_running_agents``)."""
    units = []
    running = getattr(runner, '_running_agents', {})
    for served in ([authority] if authority is not None else all_authorities(runner)):
        workers = getattr(served, '_managed_workers', {})
        for sid, live, execution in executing_sessions(served):
            if live.route in running or sid in workers:
                continue
            unit = {'kind': 'session', 'session': sid}
            if isinstance(execution, dict) and execution.get('admission_id'):
                unit['admission_id'] = execution['admission_id']
            units.append(unit)
    return units


def counted_work_units(runner, now=None):
    """The count-based work units the restart report prints, NAMED wherever the source allows it.

    A managed turn is a child process, so it is reported by session, admission and its OWN pid:
    the gateway's pid names nothing an operator can act on, and repeating it once per turn is what
    made a blind ``--yes`` restart possible. Unmapped executing sessions are named by session.
    ``owner_work`` is the remainder of the managed-turn total this reader could not itemise (owner
    mutations, side agents, a source that raised) and ``api`` / ``deferred`` runs expose no
    identity at all: they stay count-only but countED, so a refusal never undercounts live work.

    Best-effort per source (``suppress``, as the caller had it): one unreadable source never
    silences the units the others name.
    """
    from contextlib import suppress
    units: list = []
    with suppress(Exception):
        units.extend(managed_units(runner, now=now))
        units.extend(unmapped_session_units(runner))
        from gateway.run_runtime import managed_turn_count
        units.extend({'kind': 'owner_work', 'pid': os.getpid()}
                     for _ in range(max(0, managed_turn_count(runner) - len(units))))
    for kind, name in (('api', '_active_api_run_count'), ('deferred', '_active_deferred_agent_worker_count')):
        counter = getattr(runner, name, None)
        count = 0
        with suppress(Exception):
            count = counter() if callable(counter) else 0
        units.extend({'kind': kind, 'pid': os.getpid()} for _ in range(count if isinstance(count, int) else 0))
    return units


def side_agents(authority):
    """``(thread, holder)`` for each detached side agent (/bg, /btw, preview restart) whose thread
    has not unwound (``gateway/session_tui_side.py`` ``_spawn``). A stopped ``/btw`` whose fork
    is gone (``_SideQuestionStop.abandoned``) is no longer owner work."""
    from gateway.session_tui_side import _abandoned
    return [(thread, holder) for thread, holder in list(getattr(authority, '_side_agents', {}).items())
            if not _abandoned(holder)]


def interrupt_side_agents(runner, reason):
    """Hard-interrupt every counted side agent's holder across served authorities (the stop
    drain waits for them, so its timed-out interrupt must reach them). Returns how many."""
    from agent.interrupt_compat import request_hard_interrupt
    count = 0
    for authority in all_authorities(runner):
        for _, holder in side_agents(authority):
            if holder and holder[0] is not None:
                count += bool(request_hard_interrupt(holder[0], reason))
    return count


def persist_work_count(authority, _task=None):
    """Re-publish the runner's in-flight count when owner work it counts (a drain, a tracked
    mutation, a mailbox operation) ends: the agent-slot release that last wrote
    ``gateway_state.json`` ran while that work was still live, so without this write the
    persisted count never returns to zero."""
    persist = getattr(getattr(authority, 'runner', None), '_persist_active_agents', None)
    if persist is not None:
        persist()


def mutation_tasks(authority):
    tasks = {*getattr(authority, '_mutation_tasks', ()), *getattr(authority, '_bot_mailbox_operations', ())}
    return [task for task in tasks if not task.done()]


def track_mutation(authority, operation):
    tasks = getattr(authority, '_mutation_tasks', None)
    if tasks is None:
        tasks = authority._mutation_tasks = set()
    task = asyncio.create_task(operation)
    tasks.add(task)
    def finished(done):
        tasks.discard(done)
        if not done.cancelled():
            done.exception()  # the observer may already have disconnected
        persist_work_count(authority)
    task.add_done_callback(finished)
    return task


async def tracked_write(authority, write, then=None, *, ordered=False, after=None):
    """Run a synchronous ledger write (``write()``) in a worker thread, off the owner loop.

    A held SQLite writer (``_WRITE_PATIENCE_S``) must not freeze every session, timer, socket,
    Stop and approval. The write and its loop-side follow-up (``then(result)``, sync or a
    coroutine: publication, scheduling, the claim's execution stamp) are one task tracked like ``track_mutation``, so
    retirement joins it, and shielded, so a cancelled caller cannot separate a commit from it.
    ``ordered`` writes (admissions) commit in call order, as they did on the loop: worker threads
    alone would let a later input take an earlier FIFO position.

    ``after(result)`` (an admission's drain scheduling) runs in the CALLER's own step once the
    write returns, as it did when admission was synchronous: the caller registers its delivery
    waiter before any drain it scheduled can claim, settle or pause that admission. A caller
    cancelled while the commit is in flight hands ``after`` to the commit's completion instead,
    so a committed row is never stranded without a drain."""
    order = getattr(authority, '_ordered_writes', None)
    if ordered and order is None:
        order = authority._ordered_writes = asyncio.Lock()

    async def run():
        if ordered:
            async with order:
                result = await asyncio.to_thread(write)
        else:
            result = await asyncio.to_thread(write)
        if then is None:
            return result
        followed = then(result)
        return await followed if asyncio.iscoroutine(followed) else followed
    task = track_mutation(authority, run())
    if after is None:
        return await asyncio.shield(task)
    try:
        result = await asyncio.shield(task)
    except asyncio.CancelledError:
        def late(done):
            if not done.cancelled() and done.exception() is None:
                after(done.result())
        task.add_done_callback(late)
        raise
    after(result)
    return result


def start_turn_worker(runner, worker, agent_holder, run_sync):
    authority = track_turn_worker(runner, worker, agent_holder)
    worker.executor_task = asyncio.ensure_future(runner._run_in_executor_with_context(run_sync))
    if authority is not None:
        # Normal settlement releases the retained agent promptly. A cancelled asyncio waiter is
        # not physical completion; leave that entry until worker_done proves the thread exited.
        worker.executor_task.add_done_callback(lambda _: list(physical_workers(authority)))


def track_turn_worker(runner, worker, agent_holder):
    from gateway.session_authorities import active_authority
    authority = active_authority(runner)
    if authority is not None:
        workers = getattr(authority, '_turn_workers', None)
        if workers is None:
            workers = authority._turn_workers = {}
        list(physical_workers(authority))
        workers[id(worker)] = (worker, agent_holder)
    return authority


def physical_workers(authority):
    workers = getattr(authority, '_turn_workers', {})
    for key, (worker, holder) in list(workers.items()):
        if worker.worker_done.is_set():
            del workers[key]
        else:
            yield worker, holder


def stop_authority_work(authority):
    from gateway.run_runtime import stop_authority_turns
    from agent.interrupt_compat import request_hard_interrupt
    stop_authority_turns(authority, in_process=True)
    for _, holder in [*physical_workers(authority), *side_agents(authority)]:
        if holder and holder[0] is not None:
            request_hard_interrupt(holder[0], 'Profile runtime is stopping')
    for cancel in getattr(authority, '_cron_cancellations', {}).values():
        cancel.set()


async def join_authority_work(authority, timeout):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        stop_authority_work(authority)
        if (not list(physical_workers(authority)) and not list(executing_sessions(authority))
                and not mutation_tasks(authority) and not side_agents(authority)):
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError('Profile workers did not stop; ownership retained')
        await asyncio.sleep(.05)
