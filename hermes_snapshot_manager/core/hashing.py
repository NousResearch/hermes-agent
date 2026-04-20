from __future__ import annotations

import hashlib
from pathlib import Path


BUF_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(BUF_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
