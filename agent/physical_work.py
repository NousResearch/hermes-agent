"""Optional physical-work accounting inherited across executor submissions.

The owner supplies a claim returning an idempotent release callback. A cancelled
waiter is never completion; only a cancelled-before-start future or actual return
releases ownership. Without an owner this is the ordinary executor API.
"""
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
import threading

_claim = ContextVar('physical_work_claim', default=None)


@contextmanager
def bind_work(claim):
    token = _claim.set(claim)
    try:
        yield
    finally:
        _claim.reset(token)


def submit_work(executor, target, *args, **kwargs):
    claim = _claim.get()
    if claim is None:
        return executor.submit(target, *args, **kwargs)
    release = claim()
    context = copy_context()
    lock = threading.Lock()
    released = False

    def finish():
        nonlocal released
        with lock:
            if not released:
                release()
                released = True

    def run():
        try:
            return context.run(target, *args, **kwargs)
        finally:
            finish()

    try:
        future = executor.submit(run)
    except BaseException:
        finish()
        raise
    future.add_done_callback(lambda done: finish() if done.cancelled() else None)
    return future
