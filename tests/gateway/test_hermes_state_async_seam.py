"""Re-export seam: hermes_state.AsyncSessionDB is hermes_state_async.AsyncSessionDB.

Slice R5-C9 (epic #78647) moved the AsyncSessionDB facade verbatim from
hermes_state.py into hermes_state_async.py. hermes_state re-exports the
name so ``from hermes_state import AsyncSessionDB, SessionDB`` (gateway,
cli, 16 test files) keeps working with zero consumer edits.

These tests pin the seam: the re-export must resolve to the SAME object,
and the facade must still forward calls off the event loop.
"""

import asyncio
import threading

import pytest

import hermes_state
import hermes_state_async
from hermes_state import AsyncSessionDB
from hermes_state_async import AsyncSessionDB as AsyncSessionDBAsync


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

    def returns_str(self):
        self._ran_on("returns_str")
        return "title"


def test_reexport_is_same_object():
    """hermes_state.AsyncSessionDB must BE hermes_state_async.AsyncSessionDB."""
    assert AsyncSessionDB is AsyncSessionDBAsync
    assert AsyncSessionDB is hermes_state.AsyncSessionDB
    assert hermes_state.AsyncSessionDB is hermes_state_async.AsyncSessionDB


def test_module_has_no_hermes_state_import():
    """hermes_state_async must not import hermes_state (no cycle)."""
    src = open(hermes_state_async.__file__, "r", encoding="utf-8").read()
    imports = [
        line.strip()
        for line in src.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    assert imports == ["import asyncio"], imports


@pytest.mark.asyncio
async def test_facade_still_forwards_off_thread():
    """The re-exported facade must still offload calls via asyncio.to_thread."""
    db = _SpyDB()
    facade = AsyncSessionDB(db)
    caller_ident = threading.get_ident()

    result = await facade.returns_str()

    assert result == "title"
    ran_idents = [ident for _name, ident in db.calls]
    assert ran_idents and all(i != caller_ident for i in ran_idents)


@pytest.mark.asyncio
async def test_facade_routes_through_to_thread(monkeypatch):
    """The offload must route through asyncio.to_thread (where the facade lives)."""
    db = _SpyDB()
    facade = AsyncSessionDB(db)

    seen = []
    real = asyncio.to_thread

    async def _spy(func, *args, **kwargs):
        seen.append(getattr(func, "__name__", repr(func)))
        return await real(func, *args, **kwargs)

    monkeypatch.setattr(hermes_state_async.asyncio, "to_thread", _spy)
    await facade.returns_str()
    assert "returns_str" in seen


@pytest.mark.asyncio
async def test_non_callable_attributes_pass_through():
    """Plain (non-callable) attributes must pass through unchanged, not wrapped."""
    db = _SpyDB()
    facade = AsyncSessionDB(db)

    assert facade.attr == "plain-value"
    assert not callable(facade.attr)
