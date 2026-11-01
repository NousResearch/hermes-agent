"""Opt-in original owner-journal launch hold; no new journal or lifecycle control."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import stat

GUARD_OWNER = 'hermes-s6-launch-guard/v1'


def reference(value, home):
    if (not isinstance(value, dict) or set(value) != {'path', 'owner', 'lock'}
            or not isinstance(value['path'], str) or not isinstance(value['owner'], str)
            or not 1 <= len(value['owner']) <= 128):
        raise ValueError('invalid journal guard')
    if not isinstance(value['lock'], str):
        raise ValueError('invalid guard lock')
    path = Path(value['path'])
    lock = Path(value['lock'])
    if lock.parent != home or lock != lock.resolve() or lock.is_symlink() or lock == path:
        raise ValueError('noncanonical guard lock')
    if path.parent != home or path != path.resolve() or path.is_symlink():
        raise ValueError('noncanonical journal guard')
    return path


def read_journal(value, home):
    from hermes_cli.service_manager_admission import _uid
    path = reference(value, home)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'r', encoding='utf-8-sig') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != _uid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1 or info.st_size > 524288):
            raise ValueError('unsafe journal guard')
        record = json.loads(stream.read(524289))
    if not isinstance(record, dict) or record.get('owner') != value['owner']:
        raise ValueError('journal guard owner changed')
    return record


def held(value, home, slot):
    """Only a recognized original-journal hold can block/release this enrollment."""
    record = read_journal(value, home)
    if 'launch_hold' not in record:
        return False  # Explicitly enrolled existing owner journal, before its first reservation.
    hold = record['launch_hold']
    fd = os.open(slot / 'launch-admission.json', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        if os.fstat(stream.fileno()).st_size > 8192:
            raise ValueError('enrollment bound exceeded')
        digest = hashlib.sha256(stream.read(8193)).hexdigest()
    if (not isinstance(hold, dict) or hold.get('owner') != GUARD_OWNER
            or hold.get('slot') != str(slot) or hold.get('enrollment_sha256') != digest
            or hold.get('state') not in {'held', 'released'}):
        raise ValueError('journal hold unavailable')
    return hold['state'] == 'held'


@contextlib.contextmanager
def guard_lock(value, home):
    """Serialize the final read and exec with the original owner's journal writer.

    Python marks this descriptor non-inheritable: exec atomically closes it.
    A held journal drops the lock before sleeping so release can acquire it.
    """
    import fcntl
    from hermes_cli.service_manager_admission import _uid
    reference(value, home)
    fd = os.open(value['lock'], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != _uid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
            raise ValueError('unsafe journal lock')
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({'native_launch_guard': 'journal_lock_busy', 'pid': os.getpid()}), flush=True)
            fcntl.flock(stream, fcntl.LOCK_SH)
        yield
