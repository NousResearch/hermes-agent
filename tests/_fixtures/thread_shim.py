"""A synchronous ``threading.Thread`` stand-in for tests (#137194).

Tests that need thread targets to run inline (no real concurrency) patch
``threading.Thread`` with this class. Because ``server.threading`` IS the
shared ``threading`` module, the patch also catches stdlib callers: on win32
``Popen._communicate()`` builds its pipe readers as
``Thread(target=self._readerthread, args=(fh, buffer))`` and then joins them.
Subclassing the real class inherits the constructor shape, ``name``/``daemon``
handling and every thread option, so the contract cannot drift from the
stdlib; only the concurrency moves: ``start()`` runs the target inline, and
``join()`` — which refuses never-started real threads — is a no-op.

The historical per-file ``_ImmediateThread`` copies called ``target()`` with
no arguments and had no ``join``. On POSIX that never mattered (selectors
read pipes), but on win32 every test that captured subprocess output died
with TypeError. One shared implementation keeps the copies from coming back.
"""

from __future__ import annotations

import threading


class ImmediateThread(threading.Thread):
    """``threading.Thread`` that runs its target synchronously inside ``start()``."""

    def start(self) -> None:
        self.run()

    def join(self, timeout: float | None = None) -> None:
        return None
