"""A finite local preservation worker around the existing primary-turn route."""
import asyncio
import inspect
import json
import logging
from pathlib import Path
import threading
import time

from hermes_cli.conversation_closeout import CloseoutStore, public_receipt
from hermes_state import SessionDB

logger = logging.getLogger(__name__)
_ACTIVE = set()
_LOCK = threading.RLock()
PRESERVATION_WAIT_SECONDS = 30.0


def old_identity(db, stored_id, *, runtime_id=None, profile):
    from hermes_state_local import local_lineage_owner
    if db.get_session(stored_id) is None:
        raise ValueError('closeout_session_not_found')
    with db._read_ctx() as conn:
        logical = local_lineage_owner(conn, stored_id)
    if logical == stored_id:
        lineage = db.get_compression_lineage(stored_id)
        logical = lineage[0] if lineage else stored_id
    return {'logical_session_id': logical, 'stored_session_id': stored_id,
            'runtime_id': runtime_id, 'profile': profile}


def _segments(conn, db, logical):
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_local_lineage import validate_local_lineage
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + logical,)).fetchone()
    if row:
        receipt = json.loads(row[0])
        validate_local_lineage(conn, receipt)
        return list(dict.fromkeys([logical, *receipt.get('lineage', [logical])]))
    return db.get_compression_lineage(logical) or [logical]


def history_cutoff(db, logical):
    with db._read_ctx() as conn:
        segments = _segments(conn, db, logical)
        return '|'.join(f'{sid}:{conn.execute("SELECT COALESCE(MAX(id),0) FROM messages WHERE session_id=?", (sid,)).fetchone()[0]}'
                        for sid in segments)


def archive_conversation_in_transaction(db, conn, logical):
    """Only the closeout receipt's owning transaction may change these archive flags."""
    affected = set()
    for sid in _segments(conn, db, logical):
        affected.update(db._set_lineage_column_in_transaction(conn, 'archived', sid, 1))
    return sorted(affected)


def review_prompt(operation):
    return ('archive\n\n[Hermes closeout: ' + operation + ']\n'
            'Run the installed conversation-archive preservation workflow for this accessible conversation. '
            'Review the existing canonical task ledger through its normal tools; preserve newer receipts and owners. '
            'Curate eligible memory using only the installed summary provider final-response markers. '
            'This is preservation only: the host owns archiving, detaching, and fresh-chat creation. '
            'Do not call session.closeout, archive/reset commands, or a second memory writer. '
            'Report pending/unknown destinations honestly. Follow the skill task-review marker contract.')


def _refresh(store, record, settled):
    if record.get('_fence_released'):
        return record
    if record['status'] == 'failed' and (record.get('error') or {}).get('code') == 'closeout_local_failure':
        def retry_local(current):
            current['status'] = 'preserved'
            current.pop('error', None)
        record = store._update(record['closeout_id'], retry_local)
    if record['status'] == 'preserved':
        return store.archive(record['closeout_id'], settled=settled())
    return record


async def _submit_review(submit, operation):
    if inspect.iscoroutinefunction(submit):
        return await submit(review_prompt(operation), operation)
    result = await asyncio.to_thread(submit, review_prompt(operation), operation)
    return await result if inspect.isawaitable(result) else result


def begin_closeout(db, old, request_id, intent, *, submit, settled, launch, publish=lambda receipt: None,
                   epoch=None, wait_seconds=PRESERVATION_WAIT_SECONDS):
    store = CloseoutStore(db, epoch=epoch)
    key_base = str(Path(db.db_path).resolve())
    with _LOCK:
        record, _ = store.prepare(old, request_id, intent, history_cutoff(db, old['logical_session_id']))
        operation = record['closeout_id']
        key = key_base, operation
        if record['status'] == 'reviewing' and key not in _ACTIVE:
            if record['_review_started']:
                record = store.recover_uncertain(operation)
            elif store.claim_review(operation):
                _ACTIVE.add(key)
                coroutine = _progress(key, epoch, submit, settled, publish, wait_seconds)
                try:
                    launch(coroutine)
                except BaseException:
                    coroutine.close()
                    _ACTIVE.discard(key)
                    store.recover_uncertain(operation)
                    raise
        record = _refresh(store, store.read(operation), settled)
    return public_receipt(record)


async def _progress(key, epoch, submit, settled, publish, wait_seconds):
    try:
        await _run_progress(key, epoch, submit, settled, publish, wait_seconds)
    except Exception:  # health: allow BLE001 -- plugin/local preservation boundary; persist failure without private exception payloads
        logger.warning('Closeout local preservation could not finish; original history remains recoverable', exc_info=False)
        path = Path(key[0])
        if await asyncio.to_thread(path.is_file):
            try:
                with SessionDB(db_path=path) as db:
                    record = CloseoutStore(db, epoch=epoch)._update(key[1], lambda current: current.update(
                        status='failed', error={'code': 'closeout_local_failure',
                                               'message': 'Local closeout is incomplete; the existing history is retained.'}))
                    result = publish(public_receipt(record))
                    if inspect.isawaitable(result):
                        await result
            except Exception:  # health: allow BLE001 -- failure-receipt fallback must not log stored/plugin exception payloads
                logger.warning('Closeout failure receipt is unavailable; no archival success claimed', exc_info=False)
    finally:
        with _LOCK:
            _ACTIVE.discard(key)


async def _run_progress(key, epoch, submit, settled, publish, wait_seconds):
    operation = key[1]
    with SessionDB(db_path=Path(key[0])) as db:
        store = CloseoutStore(db, epoch=epoch)
        uncertain = False
        deadline = time.monotonic() + PRESERVATION_WAIT_SECONDS
        try:
            from hermes_cli.conversation_closeout_fence import review_admission
            with review_admission(db, operation):
                receipt = await asyncio.wait_for(_submit_review(submit, operation), PRESERVATION_WAIT_SECONDS)
            receipt = receipt.get('result', receipt) if isinstance(receipt, dict) else {}
            uncertain = receipt.get('status') in {'started', 'streaming'}
            unavailable = not receipt or receipt.get('error') or receipt.get('status') in {'failed', 'rejected'}
            deadline = min(deadline, time.monotonic() + max(0.0, wait_seconds))
            while not unavailable:
                record = store.read(operation)
                if record.get('_capture_received') or record.get('_fence_released') or time.monotonic() >= deadline:
                    break
                await asyncio.sleep(min(0.25, max(0, deadline - time.monotonic())))
        except asyncio.CancelledError:
            uncertain = True
            raise
        except Exception:  # health: allow BLE001 -- normal submit may fail after remote delivery; record unknown without leaking provider payloads
            uncertain = True
            logger.warning('Closeout review outcome is unconfirmed; retaining durable receipt', exc_info=False)
        finally:
            record = store.read(operation)
            if not record.get('_fence_released'):
                pending = 'unknown' if uncertain else 'pending'
                destinations = record['destinations'] if record.get('_capture_received') else {'memory': pending, 'tasks': pending}
                record = store.record_review(operation, destinations,
                    history_cutoff(db, record['old']['logical_session_id']),
                    memory_source_id=record.get('memory_source_id'), task_ids=record['task_ids'])
                record = _refresh(store, record, settled)
                published = publish(public_receipt(record))
                if inspect.isawaitable(published):
                    await published
