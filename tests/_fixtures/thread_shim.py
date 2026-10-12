"""A synchronous ``threading.Thread`` stand-in for tests (#137194).

Tests that need thread targets to run inline (no real concurrency) patch
``threading.Thread`` with this class. The contract is the one ``subprocess``
relies on when the patch leaks into pipe handling: honor ``target``/``args``/
``kwargs``, accept ``daemon``/``name`` and any other thread option, run the
target synchronously in ``start()``, and report finished work through
``join()`` + ``is_alive()``.

The historical per-file ``_ImmediateThread`` copies called ``target()`` with no
arguments and had no ``join``/``is_alive``. On POSIX that never mattered —
``Popen._communicate`` reads pipes with selectors — but on win32 it builds
reader threads as ``Thread(target=self._readerthread, args=(fh, buffer))``,
crashing every test that captured subprocess output. One shared implementation
keeps the contract from drifting again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from typing import Any


class ImmediateThread:
    """Run the target synchronously in ``start()``; the work is never alive after."""

    def __init__(
        self,
        target: Callable[..., Any] | None = None,
        args: Iterable[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
        daemon: bool | None = None,
        **_thread_options: Any,
    ) -> None:
        self._target = target
        self._args = tuple(args)
        self._kwargs = dict(kwargs or {})
        self.daemon = bool(daemon)

    def start(self) -> None:
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def join(self, timeout: float | None = None) -> None:
        return None

    def is_alive(self) -> bool:
        return False
