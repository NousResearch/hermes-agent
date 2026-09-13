"""Transport-only cron execution; output/delivery remain with the firing scheduler."""
import asyncio
import hashlib
import json
import uuid


class CronExecutionUnknown(RuntimeError):
    """The owner may have accepted this fire; never book it as failed or re-fire."""


def journal_path(job_id, request_id):
    from hermes_constants import get_hermes_home
    key = hashlib.sha256(json.dumps([job_id, request_id]).encode()).hexdigest()
    return get_hermes_home() / 'cron' / 'admissions' / (key + '.json')


def run_canonical_job(job, *, extra_prompt=None, cancel_event=None, execution_id=None):
    from gateway.session_cron import owner_for_home, operation
    from hermes_cli.gateway_client import connect_gateway
    from hermes_constants import get_hermes_home
    from utils import atomic_json_write

    params = {'job_id': job['id'], 'request_id': execution_id or job.get('execution_id') or uuid.uuid4().hex,
              'extra_prompt': extra_prompt}
    root = get_hermes_home() / 'cron' / 'admissions'
    journal = journal_path(params['job_id'], params['request_id'])
    record = {'params': params, 'receipt': None}
    if journal.exists():
        record = json.loads(journal.read_text(encoding='utf-8'))
        if record['params'] != params:
            raise CronExecutionUnknown('cron admission identity conflict')
    attempted = journal.exists()
    owner = owner_for_home(get_hermes_home())

    def save():
        root.mkdir(parents=True, exist_ok=True)
        atomic_json_write(journal, record, mode=0o600)

    async def observe(call):
        nonlocal attempted
        attempted = True
        save()
        receipt = await call('submit', params)
        record['receipt'] = receipt
        save()
        while True:
            if cancel_event is not None and cancel_event.is_set():
                await call('cancel', receipt)
            state = await call('status', receipt)
            if state['status'] == 'terminal':
                return tuple(state['result'])
            if state['status'] == 'unknown':
                raise CronExecutionUnknown('unknown_execution: cron admission was not replayed')
            await asyncio.sleep(.1)

    async def remote():
        async with connect_gateway() as client:
            return await observe(lambda op, data: client.rpc('cron.' + op, **data))

    try:
        if owner is not None:
            authority, loop = owner
            try:
                current = asyncio.get_running_loop()
            except RuntimeError:
                current = None
            if current is loop:
                raise RuntimeError('cron synchronous execution must run off the owner event loop')
            return asyncio.run_coroutine_threadsafe(
                observe(lambda op, data: operation(authority, op, data)), loop).result()
        return asyncio.run(remote())
    except Exception as exc:
        if attempted:
            raise CronExecutionUnknown(f'Cron execution unverified; reconcile {journal}: {exc}') from exc
        error = f'{type(exc).__name__}: {exc}'
        return False, f'# Cron Job: {job["id"]} (FAILED)\n\n{error}\n', '', error

def reconcile_pending():
    """Observe prepared fires; never submit missing or interrupted work."""
    from gateway.session_cron import owner_for_home, operation
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_client import connect_gateway
    from cron.jobs import pause_job, mark_job_run, save_job_output
    from cron.scheduler import _compose_run_delivery, _is_cron_silence_response
    from cron.delivery_queue import enqueue
    import logging

    root = get_hermes_home() / 'cron' / 'admissions'
    for journal in sorted(root.glob('*.json')):
        try:
            record = json.loads(journal.read_text(encoding='utf-8'))
            params = record['params']
            if journal != journal_path(params['job_id'], params['request_id']):
                raise ValueError('cron journal identity conflict')
            owner = owner_for_home(get_hermes_home())
            async def observe():
                if owner is not None:
                    return await operation(owner[0], 'recover', params)
                async with connect_gateway() as client:
                    return await client.rpc('cron.recover', **params)
            if owner is not None:
                state = asyncio.run_coroutine_threadsafe(observe(), owner[1]).result(timeout=20)
            else:
                state = asyncio.run(observe())
            if state['status'] != 'terminal':
                if state['status'] in {'unknown', 'missing'}:
                    pause_job(params['job_id'], reason='Canonical cron ' + state['status'] + '; no automatic re-execution')
                continue
            success, output, answer, error = state['result']
            job = state['job']
            job['execution_id'] = params['request_id']
            output_file = save_job_output(job['id'], output)
            content, _, silent, _, _ = _compose_run_delivery(
                job, success=success, error=error, final_response=answer, output_file=output_file)
            deliver = bool(content.strip()) and not silent and not (success and _is_cron_silence_response(content))
            if deliver:
                enqueue(params['request_id'], job, content, for_failure=not success)
            mark_job_run(job['id'], success, error, status='delivery_queued' if deliver else None,
                         execution_id=params['request_id'])
            journal.unlink(missing_ok=True)
        except Exception:
            logging.getLogger(__name__).warning('Cron receipt recovery deferred: %s', journal, exc_info=True)
