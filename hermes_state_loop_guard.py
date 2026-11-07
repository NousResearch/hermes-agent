"""Debug guard: flag ``SessionDB`` storage calls made on a thread that is running an asyncio loop.

A synchronous SQLite read or write inside a coroutine blocks its whole event loop for as long as the
storage takes: on slow storage (a network filesystem holding WAL ``-shm`` locks) one call stops every
socket, timer, Stop and liveness probe the gateway serves. Async code reaches the store through a
worker (``asyncio.to_thread``, ``gateway.session_runtime_workers.tracked_write``) instead.

Off by default and free when off: nothing is wrapped until :func:`install` runs. The gateway installs
it at startup when ``HERMES_SESSIONDB_LOOP_GUARD`` is set (``warn`` logs each call site once with its
stack; ``raise`` raises :class:`LoopSessionDBCall` at the call). Tests call ``install('record')`` and
read :func:`recorded`. Only ``SessionDB``'s two storage choke points are wrapped (``_read_ctx`` and
``_execute_write``); every public read and write goes through one of them.
"""
import asyncio
import logging
import os
import sys
import threading
import traceback

logger = logging.getLogger(__name__)

ENV = 'HERMES_SESSIONDB_LOOP_GUARD'
_MODES = ('warn', 'raise', 'record')
_RECORD_MAX = 2000

_state = {'mode': None}
_records: list = []
_warned: set = set()
_lock = threading.Lock()


class LoopSessionDBCall(RuntimeError):
    """A SessionDB storage call ran on an event-loop thread (``HERMES_SESSIONDB_LOOP_GUARD=raise``)."""


def _on_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _caller(stack):
    """The innermost frame outside the store and this guard: the code that made the call."""
    here = os.path.dirname(os.path.abspath(__file__))
    for frame in reversed(stack):
        name = os.path.basename(frame.filename)
        if os.path.dirname(os.path.abspath(frame.filename)) == here and (
                name.startswith('hermes_state') or name == 'contextlib.py'):
            continue
        if name == 'contextlib.py':
            continue
        return frame
    return stack[-1]


def _check(kind):
    mode = _state['mode']
    if mode is None or not _on_event_loop():
        return
    stack = traceback.extract_stack()[:-2]
    caller = _caller(stack)
    site = f'{caller.filename}:{caller.lineno}'
    if mode == 'raise':
        raise LoopSessionDBCall(f'SessionDB {kind} on the event-loop thread from {site}')
    with _lock:
        if mode == 'record':
            if len(_records) < _RECORD_MAX:
                _records.append({'kind': kind, 'site': site, 'function': caller.name,
                                 'thread': threading.current_thread().name, 'stack': stack})
            return
        if site in _warned:
            return
        _warned.add(site)
    logger.warning('SessionDB %s on the event-loop thread from %s (%s):\n%s', kind, site, caller.name,
                   ''.join(traceback.format_list(stack[-12:])))


def install(mode=None):
    """Wrap ``SessionDB``'s storage choke points; ``mode`` defaults to the env value. Idempotent:
    a second call only changes the mode. Returns the active mode (None: not installed)."""
    mode = mode if mode is not None else (os.environ.get(ENV) or '').strip().lower() or None
    if mode is None:
        return _state['mode']
    if mode not in _MODES:
        logger.warning('%s=%r is not one of %s; guard not installed', ENV, mode, '/'.join(_MODES))
        return _state['mode']
    from hermes_state import SessionDB
    if not getattr(SessionDB, '_loop_guard_installed', False):
        read, write = SessionDB._read_ctx, SessionDB._execute_write

        def _read_ctx(self, *args, **kwargs):
            _check('read')
            return read(self, *args, **kwargs)

        def _execute_write(self, *args, **kwargs):
            _check('write')
            return write(self, *args, **kwargs)
        _read_ctx.__wrapped__, _execute_write.__wrapped__ = read, write
        SessionDB._read_ctx, SessionDB._execute_write = _read_ctx, _execute_write
        SessionDB._loop_guard_installed = True
    _state['mode'] = mode
    return mode


def uninstall():
    """Restore the unwrapped methods (tests)."""
    from hermes_state import SessionDB
    if getattr(SessionDB, '_loop_guard_installed', False):
        SessionDB._read_ctx = SessionDB._read_ctx.__wrapped__
        SessionDB._execute_write = SessionDB._execute_write.__wrapped__
        SessionDB._loop_guard_installed = False
    _state['mode'] = None
    clear()


def recorded():
    with _lock:
        return list(_records)


def clear():
    with _lock:
        _records.clear()
        _warned.clear()


def install_from_env():
    """Gateway startup hook: install when ``HERMES_SESSIONDB_LOOP_GUARD`` is set, else do nothing."""
    if os.environ.get(ENV):
        mode = install()
        if mode is not None:
            print(f'[gateway] {ENV}={mode}: SessionDB calls on the event loop are flagged', file=sys.stderr)
