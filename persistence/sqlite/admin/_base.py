"""The file one admin handle is bound to, and the holders its operations run on."""

from __future__ import annotations

import functools
from contextlib import _GeneratorContextManager, contextmanager  # the type ``@contextmanager`` operations return
from pathlib import Path
from typing import Any, Callable, Iterator

from persistence import PermanentStoreError

from .. import _holder as _holder_module
from .. import _registry
from .._backend import translated
from .._holder import StoreHolder


class _LocalOnlyContext:
    """A context manager an admin operation returned: kept and entered later, it refuses while a hosted scope is bound
    before the context runs, as the operation that made it does."""

    __slots__ = ("_target",)

    def __init__(self, target: Any) -> None:
        self._target = target

    def __enter__(self) -> Any:
        _holder_module.refuse_hosted_activity()
        return self._target.__enter__()

    def __exit__(self, *exc: object) -> Any:
        return self._target.__exit__(*exc)


def local_only_methods(cls: type) -> type:
    """Make every public method of ``cls`` refuse while a hosted scope is bound: for operation objects an admin hands
    out (a repair guard), which a caller may keep and use after a hosted scope was bound."""
    for name, member in list(vars(cls).items()):
        if not name.startswith("_") and callable(member) and not isinstance(member, (type, staticmethod, classmethod)):
            setattr(cls, name, _local_only(member))
    return cls


def _local_only(operation: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(operation)
    def run(*args: Any, **kwargs: Any) -> Any:
        _holder_module.refuse_hosted_activity()
        result = operation(*args, **kwargs)
        return _LocalOnlyContext(result) if isinstance(result, _GeneratorContextManager) else result
    return run


class AdminFile:
    """One ``state.db`` path, opened for reading only unless ``read_only`` is False.

    Operations that run on the store's own connection take the registry's single writer for the file and release it
    when they return, so they share its write lock with every other writer in the process; a read-only handle reads
    through an owned ``mode=ro`` holder instead, which never creates, migrates or repairs the file.

    Physical admin serves local stores only: every public operation of every subclass (static ones too) refuses
    (``ScopePinRefused``) while a hosted scope is bound, before touching a file, however the admin was kept."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name, member in list(vars(cls).items()):
            if name.startswith("_"):
                continue
            if isinstance(member, staticmethod):
                setattr(cls, name, staticmethod(_local_only(member.__func__)))
            elif callable(member) and not isinstance(member, type):
                setattr(cls, name, _local_only(member))

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
