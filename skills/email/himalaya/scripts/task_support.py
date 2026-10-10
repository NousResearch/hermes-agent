"""Shared local task state: atomic commits, unique attempts, live-process locks."""
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode()
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def attempt_path(parent, purpose):
    """Allocate a name, never reuse a counter or delete an earlier capture."""
    if purpose not in ('page', 'get', 'body', 'move', 'version', 'target'):
        raise ValueError('Unknown attempt purpose')
    return Path(parent)/(purpose+'-'+uuid.uuid4().hex)


def stop_check(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('Use an absolute task stop-file path')
    return path.exists


@contextmanager
def task_lock(path):
    """Hold an OS lock in THIS process across the coordinated operation.

    The persistent file is not an ownership marker: never unlink it. OS unlock
    follows descriptor close/process exit. All cooperating workers use this path.
    Local filesystem only; network filesystem locking is outside this contract.
    """
    path = Path(path)
    stream = path.open('a+b')
    acquired = False
    try:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        yield
    except OSError as exc:
        if not acquired:
            raise ValueError('Task lock unavailable; do not clear its file') from exc
        raise
    finally:
        if acquired:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()
