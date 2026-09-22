"""Finite activity leases for turns and tracked jobs on every Hermes surface."""
from contextlib import contextmanager
import logging
from pathlib import Path
import threading
import uuid

from hermes_cli import sprites_api

log = logging.getLogger(__name__)


def available():
    return Path('/opt/hermes/.sprites-ready').is_file() and Path(sprites_api.SOCKET_PATH).exists()


@contextmanager
def active_work(*, admitted=False):
    """Protect actual work, independent of gateway connections or dashboard lifetime.

    New turns require an acknowledged hold. Already spawned processes must still
    be drained/reaped when the API is unavailable, so their reader retries. A
    crashed owner leaves at most 90 seconds of protection, never a permanent hold.
    """
    if not available():
        yield
        return
    from agent.memory_provider import spawn_context_thread

    path = '/tasks/hermes-work-' + uuid.uuid4().hex
    stopped = threading.Event()

    def renew():
        sprites_api.request('PUT', path, {'expire': '90s'})

    def watch():
        while not stopped.wait(10):
            try:
                renew()
            except (OSError, RuntimeError):
                log.warning('Sprites work lease renewal failed; retrying before expiry')

    try:
        renew()
    except (OSError, RuntimeError):
        if not admitted:
            raise RuntimeError('Cannot protect this turn from suspension; retry shortly') from None
        log.warning('Sprites work lease unavailable for an admitted job; retrying')
    thread = spawn_context_thread(watch, name='sprites-work-lease')
    try:
        thread.start()
        yield
    finally:
        stopped.set()
        if thread.ident is not None:
            thread.join(timeout=16)
        # A slow in-flight renewal can outlive join's bound. Let that finite lease
        # expire rather than deleting it only for the late response to renew it.
        if not thread.is_alive():
            try:
                sprites_api.request('DELETE', path)
            except (OSError, RuntimeError):
                log.warning('Sprites work lease release failed; it will expire')


def run_tracked(reader, session, *args):
    """Keep reading and reaping an already spawned job even during API outages."""
    with active_work(admitted=True):
        return reader(session, *args)
