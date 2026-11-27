"""Hold a module's import open on another thread, so a test can act while that import runs.

Python lists a module in ``sys.modules`` as soon as its import starts, before its body has defined
anything. Code that looks a module up there instead of importing it gets a module with none of its
names yet when two threads reach a cold import together, and Python reports the miss as "partially
initialized module ... (most likely due to a circular import)". ``held_import`` makes that window
deterministic:

    with held_import("pkg.mod", answer=lambda: 42) as held:
        assert held.run_during_import(late("answer", "pkg.mod")) == 42

On entry a helper thread starts importing ``pkg.mod`` (served by a stand-in loader, never a real
module) and parks before the body runs. ``run_during_import(fn)`` calls ``fn`` on another thread and
lets the import finish once ``fn`` has returned or some thread is waiting for that import, so
neither path sits out a timeout. It returns ``fn``'s result or raises its exception. The finished
module holds the given attributes. ``sys.modules`` and the parent package are restored on exit.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any

from tests.waiting import DEFAULT_TIMEOUT, wait_until

_MISSING = object()
_IMPORT_SYSTEM = "<frozen importlib._bootstrap"


class HeldImport:
    """Meta path finder and loader for one module name; the module body waits for the release."""

    def __init__(self, name: str, attrs: dict[str, Any]) -> None:
        self.name = name
        self._attrs = attrs
        self._started = threading.Event()
        self._release = threading.Event()
        self._importer: threading.Thread | None = None
        self._import_error: BaseException | None = None

    def find_spec(self, fullname, path=None, target=None):
        return importlib.util.spec_from_loader(fullname, self) if fullname == self.name else None

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        self._started.set()
        if not self._release.wait(DEFAULT_TIMEOUT):
            raise TimeoutError(f"the test never released the import of {self.name}")
        vars(module).update(self._attrs)

    def run_during_import(self, fn: Callable[[], Any]) -> Any:
        """``fn()`` on another thread while the import is held; the import then completes."""
        outcome: dict[str, Any] = {}

        def call() -> None:
            try:
                outcome["value"] = fn()
            except BaseException as exc:  # handed to the test below
                outcome["error"] = exc

        caller = threading.Thread(target=call, name="caller during import", daemon=True)
        caller.start()
        wait_until(lambda: not caller.is_alive() or self._someone_waits_on_import(),
                   what=f"the caller to return or to wait for the import of {self.name}")
        self._finish()
        caller.join(DEFAULT_TIMEOUT)
        assert not caller.is_alive(), "the caller never returned after the import finished"
        if self._import_error is not None:
            raise self._import_error
        if "error" in outcome:
            raise outcome["error"]
        return outcome["value"]

    def _start(self) -> None:
        def run() -> None:
            try:
                importlib.import_module(self.name)
            except BaseException as exc:  # reported by run_during_import
                self._import_error = exc

        self._importer = threading.Thread(target=run, name=f"import {self.name}", daemon=True)
        self._importer.start()
        assert self._started.wait(DEFAULT_TIMEOUT), f"the import of {self.name} never started"

    def _finish(self) -> None:
        self._release.set()
        if self._importer is not None:
            self._importer.join(DEFAULT_TIMEOUT)
            assert not self._importer.is_alive(), f"the import of {self.name} never finished"

    def _someone_waits_on_import(self) -> bool:
        """A thread other than the importer is inside the import system for this module name (an
        unrelated import on the way does not count): it is waiting for the held import."""
        importer = self._importer.ident if self._importer is not None else None
        for ident, frame in sys._current_frames().items():
            if ident == importer:
                continue
            while frame is not None:
                if (frame.f_code.co_filename.startswith(_IMPORT_SYSTEM)
                        and frame.f_locals.get("name") == self.name):
                    return True
                frame = frame.f_back
        return False


@contextlib.contextmanager
def held_import(name: str, **attrs: Any) -> Iterator[HeldImport]:
    """Start importing ``name`` on a helper thread and hold it before the module body runs."""
    parent, _, child = name.rpartition(".")
    package = importlib.import_module(parent) if parent else None
    saved_module = sys.modules.pop(name, _MISSING)
    saved_attr = package.__dict__.pop(child, _MISSING) if package is not None else _MISSING
    held = HeldImport(name, attrs)
    sys.meta_path.insert(0, held)
    try:
        held._start()
        yield held
    finally:
        held._finish()
        sys.meta_path.remove(held)
        sys.modules.pop(name, None)
        if saved_module is not _MISSING:
            sys.modules[name] = saved_module
        if package is not None:
            package.__dict__.pop(child, None)
            if saved_attr is not _MISSING:
                setattr(package, child, saved_attr)
