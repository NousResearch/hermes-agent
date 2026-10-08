"""Authority-local async serialization; file locks cover only short synchronous mailbox I/O."""
import asyncio
from contextlib import asynccontextmanager

from tools.bot_live_delivery import _locked, _read, _write


@asynccontextmanager
async def mailbox_lock(authority):
    lock = getattr(authority, '_bot_mailbox_lock', None)
    if lock is None:
        lock = authority._bot_mailbox_lock = asyncio.Lock()
    async with lock:
        active = getattr(authority, '_bot_mailbox_operations', None)
        if active is None:
            active = authority._bot_mailbox_operations = set()
        task = asyncio.current_task()
        active.add(task)
        try:
            yield
        finally:
            active.discard(task)


async def _mailbox_io(operation, *args):
    """Keep the caller's mailbox ownership until its physical I/O ends, even after cancellation."""
    pending = asyncio.create_task(asyncio.to_thread(operation, *args))
    cancelled = None
    while True:
        try:
            result = await asyncio.shield(pending)
        except asyncio.CancelledError as exc:
            if pending.cancelled():
                raise
            cancelled = exc
        except Exception:
            if cancelled is not None:
                raise cancelled
            raise
        else:
            if cancelled is not None:
                raise cancelled
            return result


def _read_receipt(home, key):
    with _locked(home) as root:
        return root, _read(root / f'{key}.json')


async def read_receipt(home, key):
    return await _mailbox_io(_read_receipt, home, key)


_UNCHECKED = object()


def _write_receipt(home, record, expect=_UNCHECKED):
    with _locked(home) as root:
        path = root / f"{record['delivery_id']}.json"
        current = _read(path)
        if expect is not _UNCHECKED and (current or {}).get('status') != expect:
            return False
        if current is not None and current.get('principal_id'):
            from hermes_state_runtime import RuntimeStoreError
            fields = ('profile_home', 'session_id', 'principal_id', 'message', 'author')
            if (any(current.get(key) != record.get(key) for key in fields)
                    or current.get('notification_category', 'result') != record.get('notification_category', 'result')
                    or (current.get('admission_id') and current['admission_id'] != record.get('admission_id'))):
                raise RuntimeStoreError('admission_conflict')
        merged = {**(current or {}), **record}
        if merged != current:
            _write(path, merged)
        return True


async def write_receipt(home, record, *, expect=_UNCHECKED):
    return await _mailbox_io(_write_receipt, home, record, expect)


def _scan_receipts(home):
    from gateway.session_bot import _scan_records
    with _locked(home) as root:
        return root, list(_scan_records(root))


async def scan_receipts(home):
    return await _mailbox_io(_scan_receipts, home)


def remember_receipt(authority, home, record):
    index = getattr(authority, '_bot_receipt_index', None)
    if index is None:
        index = authority._bot_receipt_index = {}
    if record['status'] in {'settled', 'failed', 'cancelled'}:
        index.pop(record['delivery_id'], None)
    else:
        index[record['delivery_id']] = (home, record['session_id'])


def track_receipt_task(authority, task):
    tasks = getattr(authority, '_bot_receipt_tasks', None)
    if tasks is None:
        tasks = authority._bot_receipt_tasks = set()
    tasks.add(task)
    def done(finished):
        tasks.discard(finished)
        if not finished.cancelled():
            error = finished.exception()
            if error is not None:
                import logging
                logging.getLogger(__name__).warning('Bot receipt projection failed: %s', type(error).__name__)
    task.add_done_callback(done)


def wake_bot_receipts(authority, session_id):
    """Re-arm paused receipts from owner publication, without making external pollers dial RPC."""
    pending = getattr(authority, '_bot_receipt_refreshes', None)
    if pending is None:
        pending = authority._bot_receipt_refreshes = {}
        authority._bot_receipt_dirty = set()
    for key, (home, owner) in list(getattr(authority, '_bot_receipt_index', {}).items()):
        if owner != session_id:
            continue
        if key in pending and not pending[key].done():
            authority._bot_receipt_dirty.add(key)
        else:
            _start_refresh(authority, home, key)


def _start_refresh(authority, home, key):
    from gateway.session_bot import refresh_receipt
    async def refresh_until_current():
        while True:
            authority._bot_receipt_dirty.discard(key)
            await refresh_receipt(authority, home, key)
            if key not in authority._bot_receipt_dirty or key not in authority._bot_receipt_index:
                return
    task = asyncio.create_task(refresh_until_current())
    authority._bot_receipt_refreshes[key] = task
    def done(finished):
        if authority._bot_receipt_refreshes.get(key) is finished:
            authority._bot_receipt_refreshes.pop(key, None)
            authority._bot_receipt_dirty.discard(key)
    task.add_done_callback(done)
    track_receipt_task(authority, task)
