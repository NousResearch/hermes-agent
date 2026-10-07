"""Merge-order-safe expected failures for live gaps whose fix is an open PR.

A plain ``xfail(strict=True)`` turns main red the moment its fix merges (XPASS), and a
non-strict one guards nothing. ``known_failure`` is a run-time xfail keyed on the gap's own
failure message: the cell XFAILs only while it fails exactly that way, fails loudly on any other
failure (a timeout, a lost reply, a boot failure), and simply passes once the fix lands, whichever
merges first. Wrap only the final assertions. When a fix has landed, delete its ``known_failure``.
Probe-gated gaps live in ``tests/e2e/core/delivery/_pending_fixes.py``.

Strict acceptance: ``HERMES_E2E_STRICT_ACCEPTANCE`` turns ``known_failure`` blocks into hard
failures (the gap's own assertion, annotated): ``1`` every block, or a comma list of owners
(``upd-txn``) for the blocks whose reason starts with ``<owner>:``. Only an acceptance run of an
integrated batch sets it (the ``strict_acceptance`` dispatch input of ci.yaml /
windows-install-update-e2e.yml), where an expected failure would accept exactly the gap the
batch claims to close; gaps other work owns keep their xfail. Never a user setting.
"""

from __future__ import annotations

import contextlib
import os
import re
from typing import ContextManager, Iterator, Mapping, Tuple, Type

import pytest

STRICT_ACCEPTANCE_ENV = "HERMES_E2E_STRICT_ACCEPTANCE"


def strict_acceptance(reason: str) -> bool:
    """True when this acceptance run refuses to excuse the gap ``reason`` describes."""
    value = os.environ.get(STRICT_ACCEPTANCE_ENV, "").strip()
    if value == "1":
        return True
    owners = [o.strip() for o in value.split(",") if o.strip() and o.strip() != "0"]
    return any(reason.startswith(f"{owner}:") for owner in owners)


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
        if strict_acceptance(reason):
            exc.add_note(f"{STRICT_ACCEPTANCE_ENV}={os.environ.get(STRICT_ACCEPTANCE_ENV, '').strip()}: "
                         f"the known gap is not excused in this acceptance run ({reason})")
            raise
        pytest.xfail(f"{reason} [observed: {str(exc).splitlines()[0][:240]}]")


def known_gate(known: Mapping[str, Tuple[str, str]], key: str,
               raises: Type[BaseException] | Tuple[Type[BaseException], ...] = AssertionError) -> ContextManager[None]:
    """:func:`known_failure` for a cell a file's ``KNOWN`` table (key -> ``(pattern, reason)``)
    names, a no-op for every other cell, so one wrapped block serves gated and plain cells."""
    entry = known.get(key)
    return known_failure(*entry, raises=raises) if entry else contextlib.nullcontext()
