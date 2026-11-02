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
            from gateway.session_runtime_workers import persist_work_count
            persist_work_count(authority)


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


def _scan_receipts(home, pins=None):
    from gateway.session_bot import _scan_records
    with _locked(home) as root:
        return root, list(_scan_records(root, pins))


def _withdraw_pin(home, key):
    with _locked(home) as root:
        path = root / f'{key}.json'
        current = _read(path)
        if current is not None and current.get('status') == 'ambiguous' and not current.get('admission_id'):
            path.unlink()
            return True
        return False


async def withdraw_pin(home, key):
    """Drop a pre-admission pin the FIFO definitely refused (nothing committed), only while it is
    still that pin, so the same-id retry admits instead of answering ``unknown_execution``."""
    return await _mailbox_io(_withdraw_pin, home, key)


# What ``_admit`` lays over a legacy ticket when it pins it (``legacy=``); the ticket never had them.
_PIN_FIELDS = ('profile_home', 'session_id', 'principal_id', 'operator')


def _requeue_pin(home, key):
    with _locked(home) as root:
        path = root / f'{key}.json'
        current = _read(path)
        if (current is None or current.get('status') != 'ambiguous' or current.get('admission_id')
                or 'owner' not in current):
            return None
        ticket = {k: v for k, v in current.items() if k not in _PIN_FIELDS}
        ticket['status'] = 'queued'
        _write(path, ticket)
        return ticket


async def requeue_pin(home, key):
    """Restore the queued legacy ticket a pre-admission pin was laid over (nothing committed), only
    while it is still that pin: its id, message, sequence and created_at are kept, so ``_migrate``
    admits it once in its FIFO place. Returns the ticket, or None when the pin is gone or changed."""
    return await _mailbox_io(_requeue_pin, home, key)


def _refuse_pin(home, key):
    from tools.bot_failure_reasons import AGENT_BLOCKED
    with _locked(home) as root:
        path = root / f'{key}.json'
        current = _read(path)
        if current is not None and current.get('status') == 'ambiguous' and not current.get('admission_id'):
            current.update(status='failed', refused=True, reply='', error='Bot Chat delivery was refused',
                           reason=AGENT_BLOCKED)
            _write(path, current)


async def refuse_pin(home, key):
    """Settle a pre-admission pin the FIFO definitely refused for good (nothing committed) as a
    terminal ``failed`` receipt, only while it is still that pin: never re-sent, nothing unknown."""
    return await _mailbox_io(_refuse_pin, home, key)


async def scan_receipts(home, pins=None):
    """Bulk scan; an owner scan passes ``pins`` to collect pre-admission pins instead of warning."""
    return await _mailbox_io(_scan_receipts, home, pins)


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
