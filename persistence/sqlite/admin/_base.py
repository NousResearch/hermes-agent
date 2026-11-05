"""The file one admin handle is bound to, and the holders its operations run on."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from persistence import PermanentStoreError

from .. import _registry
from .._backend import translated
from .._holder import StoreHolder


class AdminFile:
    """One ``state.db`` path, opened for reading only unless ``read_only`` is False.

    Operations that run on the store's own connection take the registry's single writer for the file and release it
    when they return, so they share its write lock with every other writer in the process; a read-only handle reads
    through an owned ``mode=ro`` holder instead, which never creates, migrates or repairs the file."""

    def __init__(self, db_path: Path, *, read_only: bool = True) -> None:
        self.db_path = Path(db_path)
        self.read_only = read_only

    def _require_writable(self, operation: str) -> None:
        if self.read_only:
            raise PermanentStoreError(f"{operation} needs a writable store admin for {self.db_path}")

    @contextmanager
    def _writer(self, operation: str) -> Iterator[StoreHolder]:
        """The registry's writer for the file, with store failures raised as the contract's errors."""
        self._require_writable(operation)
        with translated():
            holder = _registry.acquire(self.db_path)
            try:
                yield holder
            finally:
                _registry.release_or_close(holder)

    @contextmanager
    def _reader(self) -> Iterator[StoreHolder]:
        """The writer on a writable handle, else an owned ``mode=ro`` holder closed on exit."""
        if not self.read_only:
            with self._writer("read") as holder:
                yield holder
            return
        with translated():
            holder = StoreHolder(self.db_path, read_only=True)
            try:
                yield holder
            finally:
                holder.close()
