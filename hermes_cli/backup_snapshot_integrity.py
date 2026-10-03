"""Content verification for versioned quick-snapshot manifests.

Digests detect accidental payload alteration, not an attacker rewriting both
payload and manifest. Legacy size-only snapshots remain readable.
"""

import hashlib
from pathlib import Path


def payload_digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def payload_matches(directory: Path, rel: str, size: object, meta: dict) -> bool:
    relative = Path(rel)
    if relative.is_absolute() or '..' in relative.parts or type(size) is not int or size < 0:
        return False
    path = directory / relative
    try:
        current = path
        while current != directory:
            if current.is_symlink():
                return False
            current = current.parent
        if directory.is_symlink():
            return False
        if not path.is_file() or path.stat().st_size != size:
            return False
        if meta.get('version', 1) == 1 and 'sha256' not in meta:
            return True
        if meta.get('version') != 2 or not isinstance(meta.get('sha256'), dict):
            return False
        expected = meta['sha256'].get(rel)
        return isinstance(expected, str) and payload_digest(path) == expected
    except OSError:
        return False
