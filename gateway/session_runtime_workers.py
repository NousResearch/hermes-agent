"""Physical worker liveness and cooperative Stop across every served authority."""
import asyncio

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
            + sum(len(mutation_tasks(authority)) for authority in authorities))


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
    task.add_done_callback(finished)
    return task


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
    for _, holder in physical_workers(authority):
        if holder and holder[0] is not None:
            request_hard_interrupt(holder[0], 'Profile runtime is stopping')
    for cancel in getattr(authority, '_cron_cancellations', {}).values():
        cancel.set()


async def join_authority_work(authority, timeout):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        stop_authority_work(authority)
        if not list(physical_workers(authority)) and not list(executing_sessions(authority)) and not mutation_tasks(authority):
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError('Profile workers did not stop; ownership retained')
        await asyncio.sleep(.05)
