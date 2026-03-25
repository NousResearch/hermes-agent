from __future__ import annotations

import hashlib
from pathlib import Path

from .models import IntegrityState


class IntegrityVerifier:
    def __init__(self, algorithm: str = "sha256") -> None:
        self.algorithm = algorithm

    def hash_path(self, path: Path) -> str:
        digest = hashlib.new(self.algorithm)
        target = Path(path)
        if target.is_dir():
            for child in sorted(p for p in target.rglob("*") if p.is_file()):
                digest.update(child.relative_to(target).as_posix().encode("utf-8"))
                digest.update(child.read_bytes())
        else:
            digest.update(target.read_bytes())
        return digest.hexdigest()

    def capture(self, path: Path) -> IntegrityState:
        return IntegrityState(algorithm=self.algorithm, digest=self.hash_path(path))

    def verify(self, path: Path, expected: IntegrityState) -> bool:
        if expected.algorithm != self.algorithm:
            verifier = IntegrityVerifier(expected.algorithm)
            return verifier.hash_path(path) == expected.digest
        return self.hash_path(path) == expected.digest
