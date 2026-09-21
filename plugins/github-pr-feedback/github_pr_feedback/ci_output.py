"""Private, content-addressed CI diagnostics; never included in GitHub comments."""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from hermes_constants import get_hermes_home


def retain_output(output: str) -> Path:
    payload = output.encode('utf-8', errors='replace')
    digest = hashlib.sha256(payload).hexdigest()
    root = get_hermes_home() / 'github-pr-feedback' / 'ci-output'
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = root / (digest + '.log')
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=root)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise RuntimeError('CI diagnostic artifact identity mismatch')
    finally:
        Path(temporary).unlink()
    return destination
