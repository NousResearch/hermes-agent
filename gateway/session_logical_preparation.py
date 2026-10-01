"""Bounded logical-attempt preparation during the owning authority's lifetime."""
import asyncio
import logging

from gateway.session_authorities import owner_scope
from hermes_state_errors import is_transient_sqlite_error
from hermes_state_logical_attempts import prepare_logical_attempt_index, record_logical_preparation_worker

logger = logging.getLogger(__name__)
_BATCH_SIZE = 128
_BATCH_PAUSE = 0.05
_HELD_PAUSE = 1.0


async def _drain(authority, stopped):
    delay = _BATCH_PAUSE
    recorded = None
    while not stopped.is_set():
        try:
            with owner_scope(authority, hydrate_secrets=False):
                if recorded != 'running':
                    await asyncio.to_thread(record_logical_preparation_worker, authority.db,
                                           epoch=authority.epoch, state='running')
                    recorded = 'running'
                authority._logical_preparation_state = 'running'
                step = await asyncio.to_thread(prepare_logical_attempt_index, authority.db,
                                               batch_size=_BATCH_SIZE, epoch=authority.epoch)
        except Exception as exc:
            if is_transient_sqlite_error(exc):
                authority._logical_preparation_state = 'retrying'
                delay = min(max(delay * 2, _HELD_PAUSE), 5)
                logger.warning('Logical-attempt preparation waits for the database writer in profile %s',
                               authority.profile_id)
                try:
                    await asyncio.wait_for(stopped.wait(), timeout=delay)
                except TimeoutError:
                    pass
                continue
            authority._logical_preparation_state = 'failed'
            try:
                await asyncio.to_thread(record_logical_preparation_worker, authority.db,
                                       epoch=authority.epoch, state='failed')
            except Exception:
                pass  # a closed/replaced/stale store cannot accept this owner's verdict
            logger.exception('Logical-attempt preparation stopped for profile %s', authority.profile_id)
            return
        if step['complete']:
            authority._logical_preparation_state = 'ready'
            return
        # Every batch yields. A store containing only unresolved evidence does
        # not spend a writer transaction continuously; its durable cursor still
        # rotates past those holds when later evidence is repairable.
        delay = _HELD_PAUSE if step['phase'] == 'covered' else _BATCH_PAUSE
        authority._logical_preparation_state = 'held' if step['phase'] == 'covered' else 'running'
        try:
            await asyncio.wait_for(stopped.wait(), timeout=delay)
        except TimeoutError:
            pass


def start_logical_preparation(authority):
    authority._logical_preparation_state = 'waiting'
    stopped = authority._logical_preparation_stopped = asyncio.Event()
    authority._logical_preparation_task = asyncio.create_task(_drain(authority, stopped))


async def stop_logical_preparation(authority):
    task = getattr(authority, '_logical_preparation_task', None)
    if task is not None:
        authority._logical_preparation_stopped.set()
        # Do not cancel to_thread: cancellation would leave its bounded writer
        # running after the authority's database is closed or its home unserved.
        await task
        if authority._logical_preparation_state not in {'ready', 'failed'}:
            authority._logical_preparation_state = 'stopped'
            try:
                await asyncio.to_thread(record_logical_preparation_worker, authority.db,
                                       epoch=authority.epoch, state='stopped')
            except Exception:
                logger.debug('Could not record the stopped preparation owner', exc_info=True)
