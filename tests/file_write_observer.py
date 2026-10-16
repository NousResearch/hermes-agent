"""Run a "concurrent reader" at the exact instant a writer emits bytes (torn-read regression tests).

A file rewritten in place (``open(path, "w")`` / ``Path.write_text``) is empty from the truncating
open until the payload lands; another process that reads it then sees ``""``. Racing real threads
only hits that window probabilistically. This helper makes it deterministic: ``callback`` runs at
every Python-level ``write()`` on a file object in the current thread, i.e. while the writer is
mid-publication, so the callback can read the target exactly as a concurrent process would.

    from tests.file_write_observer import observe_file_writes

    seen = []
    with observe_file_writes(lambda: seen.append(target.read_text() if target.exists() else None)):
        writer()
    assert "" not in seen

An atomic writer (temp file + rename) only writes to its temp file, so the target still holds the
old bytes (or does not exist yet) at every observation.
"""

from __future__ import annotations

import io
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager


@contextmanager
def observe_file_writes(callback: Callable[[], None]) -> Iterator[None]:
    """Call ``callback()`` before every ``write``/``writelines`` on a file object in this thread."""

    def _profile(frame, event, arg):
        if event != "c_call":
            return
        if getattr(arg, "__name__", "") not in ("write", "writelines"):
            return
        if isinstance(getattr(arg, "__self__", None), io.IOBase):
            callback()

    previous = sys.getprofile()
    sys.setprofile(_profile)
    try:
        yield
    finally:
        sys.setprofile(previous)
