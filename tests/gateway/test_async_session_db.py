"""AsyncSessionDB offload facade.

The gateway runs one asyncio loop for every session; SessionDB is synchronous,
so a raw call on the loop freezes every conversation until it returns.
AsyncSessionDB offloads each call via asyncio.to_thread. These tests pin the
facade's contract.
"""

import asyncio
import threading

import pytest

import hermes_state
from hermes_state import AsyncSessionDB


class _SpyDB:
    """SessionDB stand-in recording the thread each call ran on."""

    def __init__(self):
        self.calls = []
        self.attr = "plain-value"

    def _ran_on(self, name):
        self.calls.append((name, threading.get_ident()))

    def returns_none(self):
        self._ran_on("returns_none")
        return None

    def returns_bool(self):
        self._ran_on("returns_bool")
        return True

    def returns_str(self):
        self._ran_on("returns_str")
        return "title"

    def returns_dict(self):
        self._ran_on("returns_dict")
        return {"id": "s1"}

    def returns_list(self):
        self._ran_on("returns_list")
        return [{"id": "s1"}, {"id": "s2"}]

    def raises(self):
        self._ran_on("raises")
        raise ValueError("boom")


# --------------------------------------------------------------------------
# Facade behaviour
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_offloads_off_calling_thread():
    """A call must execute on a worker thread, not the caller's loop thread."""
    db = _SpyDB()
    facade = AsyncSessionDB(db)
    caller_ident = threading.get_ident()

    await facade.returns_none()

    ran_idents = [ident for _name, ident in db.calls]
    assert ran_idents and all(i != caller_ident for i in ran_idents)


# --------------------------------------------------------------------------
# Interleaving safety: offloading opens await points where coroutines can
# interleave against the same session rows. The gateway relies on SessionDB's
# atomic operations (compare-and-set, INSERT OR IGNORE) to stay single-winner.
# These pin that the defenses hold when driven concurrently through the facade.
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_claim_handoff_single_winner(tmp_path):
    db = AsyncSessionDB(hermes_state.SessionDB(db_path=tmp_path / "state.db"))
    sid = "s-handoff"
    await db.create_session(sid, "test")
    await db.request_handoff(sid, "telegram")

    results = await asyncio.gather(*(db.claim_handoff(sid) for _ in range(20)))

    assert sum(results) == 1, f"exactly one claim must win, got {sum(results)}"


