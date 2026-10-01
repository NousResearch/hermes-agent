"""Merge-order-safe expected failures for live gaps whose fix is an open PR.

A plain ``xfail(strict=True)`` turns main red the moment its fix merges (XPASS), and a
non-strict one guards nothing. ``known_failure`` is a run-time xfail keyed on the gap's own
failure message: the cell XFAILs only while it fails exactly that way, fails loudly on any other
failure (a timeout, a lost reply, a boot failure), and simply passes once the fix lands, whichever
merges first. Wrap only the final assertions. When a fix has landed, delete its ``known_failure``.
Probe-gated gaps live in ``tests/e2e/core/delivery/_pending_fixes.py``.
"""

from __future__ import annotations

import contextlib
import re
from typing import Iterator, Tuple, Type

import pytest


@contextlib.contextmanager
def known_failure(pattern: str, reason: str,
                  raises: Type[BaseException] | Tuple[Type[BaseException], ...] = AssertionError) -> Iterator[None]:
    """Run-time xfail for a live gap: an exception of type ``raises`` raised inside the block whose
    message matches ``pattern`` (``re.search``) XFAILs the cell; any other failure propagates, and a
    clean pass stays a pass. Wrap only the final assertions, after every wait has settled, so a lost
    reply, a failed boot or a timeout can never be mistaken for the gap."""
    try:
        yield
    except raises as exc:
        if not re.search(pattern, str(exc)):
            raise
        pytest.xfail(f"{reason} [observed: {str(exc).splitlines()[0][:240]}]")
