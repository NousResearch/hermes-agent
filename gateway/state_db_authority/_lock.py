from __future__ import annotations

import os
from pathlib import Path

from ._model import StateDBAdmissionError


class CrossProcessAdmissionLock:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.path = db_path.with_name(db_path.name + ".gateway-admission.lock")
        self._handle = None

    def __enter__(self) -> "CrossProcessAdmissionLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except Exception as exc:
            self._handle.close()
            self._handle = None
            raise StateDBAdmissionError(
                f"state.db admission lock unavailable for {self.path}: {exc}",
                path=self.db_path,
            ) from exc
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        finally:
            handle.close()
