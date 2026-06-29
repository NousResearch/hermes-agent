"""AsyncSessionDB offload facade + gateway raw-call guard.

The gateway runs one asyncio loop for every session; SessionDB is synchronous,
so a raw call on the loop freezes every conversation until it returns.
AsyncSessionDB offloads each call via asyncio.to_thread. These tests pin the
facade's contract and lock the gateway boundary so a 39th raw call can't regress.
"""

import ast
import asyncio
import threading
from pathlib import Path

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


@pytest.mark.asyncio
async def test_offload_goes_through_to_thread(monkeypatch):
    """The offload must route through asyncio.to_thread (where the facade lives)."""
    db = _SpyDB()
    facade = AsyncSessionDB(db)

    seen = []
    real = asyncio.to_thread

    async def _spy(func, *args, **kwargs):
        seen.append(getattr(func, "__name__", repr(func)))
        return await real(func, *args, **kwargs)

    monkeypatch.setattr(hermes_state.asyncio, "to_thread", _spy)
    await facade.returns_str()
    assert "returns_str" in seen


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,expected",
    [
        ("returns_none", None),
        ("returns_bool", True),
        ("returns_str", "title"),
        ("returns_dict", {"id": "s1"}),
        ("returns_list", [{"id": "s1"}, {"id": "s2"}]),
    ],
)
async def test_returns_underlying_value_unchanged(method, expected):
    facade = AsyncSessionDB(_SpyDB())
    assert await getattr(facade, method)() == expected


@pytest.mark.asyncio
async def test_propagates_exception():
    facade = AsyncSessionDB(_SpyDB())
    with pytest.raises(ValueError, match="boom"):
        await facade.raises()


def test_non_callable_attribute_passes_through():
    facade = AsyncSessionDB(_SpyDB())
    assert facade.attr == "plain-value"


# --------------------------------------------------------------------------
# Guard: no raw self._session_db.<method>( on the gateway loop
# --------------------------------------------------------------------------

_GATEWAY_FILES = ("gateway/run.py", "gateway/slash_commands.py")
# The only legitimate non-loop paths:
#   - SessionDB.sanitize_title: pure @staticmethod string cleaning, no DB.
#   - self._session_db._db.<x>: the sync escape, allowed ONLY where the call is
#     provably off the event loop — construction (__init__, before the loop
#     serves) and the run_sync closure (executed in a thread-pool executor).
#     Three such sites today; a fourth must be justified and this count bumped.
_ALLOWED_SYNC_DB_ESCAPES = 3


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class _RawCallVisitor:
    """Collect non-awaited self._session_db.<method>(...) calls in a module.

    An ``await x.y()`` parses as Await(value=Call(...)); those Call nodes are
    exempt — they're the migrated path. We flag only Calls that are NOT directly
    awaited, and separately count the self._session_db._db.<x> sync escape. The
    sanitize_title staticmethod is called on the class (SessionDB.sanitize_title),
    so it never matches the self._session_db.<method> shape.
    """

    def __init__(self, tree: ast.AST):
        self.raw_calls = []  # (method, lineno) — non-awaited
        self.db_escapes = []  # self._session_db._db.<x> sites (lineno)

        awaited = {id(n.value) for n in ast.walk(tree)
                   if isinstance(n, ast.Await) and isinstance(n.value, ast.Call)}

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute)):
                continue
            inner = func.value
            # self._session_db._db.<method>(...)  -> sync escape
            if (
                inner.attr == "_db"
                and isinstance(inner.value, ast.Attribute)
                and inner.value.attr == "_session_db"
                and isinstance(inner.value.value, ast.Name)
                and inner.value.value.id == "self"
            ):
                self.db_escapes.append(inner.lineno)
            # self._session_db.<method>(...) not wrapped in await -> raw loop call
            elif (
                inner.attr == "_session_db"
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "self"
                and id(node) not in awaited
            ):
                self.raw_calls.append((func.attr, node.lineno))


def _scan(rel_path: str) -> _RawCallVisitor:
    source = (_repo_root() / rel_path).read_text(encoding="utf-8")
    return _RawCallVisitor(ast.parse(source))


def test_no_raw_session_db_calls_on_gateway_loop():
    """Fail if any raw self._session_db.<method>( appears in gateway files.

    Every loop-reachable DB call must go through AsyncSessionDB (await). The
    sanitize_title staticmethod is called on the class, not self, so it is not
    matched here; the _db. construction escape is checked separately below.
    """
    violations = []
    for rel in _GATEWAY_FILES:
        v = _scan(rel)
        violations.extend(f"{rel}:{ln} self._session_db.{m}(" for m, ln in v.raw_calls)
    assert not violations, (
        "Raw SessionDB calls on the gateway loop — route through AsyncSessionDB "
        "(await self._session_db.<method>(...)):\n  " + "\n  ".join(violations)
    )


def test_sync_db_escape_confined_to_off_loop_sites():
    """The self._session_db._db. sync escape must stay confined to known sites.

    It is legitimate only where the call is provably off the loop: construction
    (before the loop serves) and the run_sync executor closure. More occurrences
    than the reviewed count means a blocking call may have leaked back onto the
    loop through the escape hatch.
    """
    total = sum(len(_scan(rel).db_escapes) for rel in _GATEWAY_FILES)
    assert total <= _ALLOWED_SYNC_DB_ESCAPES, (
        f"self._session_db._db. sync escape used {total} times; "
        f"at most {_ALLOWED_SYNC_DB_ESCAPES} (construction + run_sync) is allowed."
    )
