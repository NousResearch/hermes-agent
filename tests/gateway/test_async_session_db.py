"""AsyncSessionDB offload facade + gateway raw-call guard.

The gateway runs one asyncio loop for every session; SessionDB is synchronous,
so a raw call on the loop freezes every conversation until it returns.
AsyncSessionDB offloads each call via asyncio.to_thread. These tests pin the
facade's contract and lock the gateway boundary so a 39th raw call can't regress.
"""

import ast
import asyncio
import threading
import time
import types
from pathlib import Path

import pytest

import hermes_state
from hermes_state import AsyncSessionDB
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


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

# Sync helpers that touch SessionDB but are NEVER invoked bare on the loop:
# every loop-side call wraps them in ``asyncio.to_thread(...)`` and the only
# bare calls live in the run_sync thread-pool closure. Their DB calls therefore
# run off-loop. The guard exempts their bodies AND enforces the contract — see
# test_offloaded_helpers_never_called_bare_on_loop. Adding a helper here without
# wrapping its loop call sites makes that test fail.
_OFFLOADED_SYNC_HELPERS = frozenset({
    "_telegram_topic_mode_enabled",
    "_is_telegram_topic_lane",
    "_is_telegram_topic_root_lobby",
    "_recover_telegram_topic_thread_id",
    "_normalize_source_for_session_key",
    "_record_telegram_topic_binding",
    "_sync_telegram_topic_binding",
    "_telegram_topic_new_header",
    "_schedule_telegram_topic_title_rename",
    "_apply_topic_recovery",
})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class _RawCallVisitor:
    """Collect non-awaited SessionDB calls reachable on the gateway loop.

    Catches both shapes:
      * direct:  self._session_db.<method>(...)
      * aliased: db = getattr(self, "_session_db", None)  /  db = self._session_db
                 then db.<method>(...)
    An ``await x.y()`` is Await(value=Call(...)); those Calls are exempt (the
    migrated path). The self._session_db._db.<x> sync escape is counted
    separately. SessionDB.sanitize_title is a staticmethod called on the class,
    so it never matches either shape.

    Alias detection scans, per function scope, for locals bound to the gateway's
    _session_db (incl. closures that bind it off a captured ``self``-like param),
    then flags non-awaited calls on those names. The literal-grep blind spot that
    let six loop-reachable calls hide behind ``getattr(self, "_session_db")`` is
    exactly what this closes.
    """

    def __init__(self, tree: ast.AST):
        self.raw_calls = []  # (method, lineno) — direct, non-awaited, on-loop
        self.alias_calls = []  # (method, lineno) — via a _session_db-bound local, on-loop
        self.db_escapes = []  # self._session_db._db.<x> sites (lineno)
        # BARE self.<helper>(...) call sites of offloaded helpers — i.e. the
        # helper is actually *called*, not passed to asyncio.to_thread (which
        # references it as an attribute, producing no Call node here). Each is
        # (helper, lineno, enclosing_fn) for the contract test.
        self.bare_helper_calls = []

        awaited = {id(n.value) for n in ast.walk(tree)
                   if isinstance(n, ast.Await) and isinstance(n.value, ast.Call)}
        alias_names = self._collect_alias_names(tree)
        # Map each node to the name of the function whose body lexically encloses
        # it, so DB calls inside an offloaded helper (which runs off-loop) are
        # exempt while bare on-loop calls are not.
        enclosing = self._enclosing_fn_map(tree)
        ancestry = self._ancestor_fns(tree)  # id(node) -> frozenset of enclosing fn names

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            encl_fn = enclosing.get(id(node))
            in_offloaded_helper = encl_fn in _OFFLOADED_SYNC_HELPERS
            # Bare call of an offloaded helper (self._helper(...)). A to_thread
            # offload passes the helper as an attribute arg, not a Call, so it
            # never lands here — exactly the distinction the contract test needs.
            if (
                isinstance(func.value, ast.Name) and func.value.id == "self"
                and func.attr in _OFFLOADED_SYNC_HELPERS
            ):
                self.bare_helper_calls.append(
                    (func.attr, node.lineno, ancestry.get(id(node), frozenset()))
                )
            # alias.<method>(...)  -> aliased loop call (var bound to _session_db)
            if (
                isinstance(func.value, ast.Name)
                and func.value.id in alias_names
                and func.attr not in ("_db",)
                and id(node) not in awaited
                and not in_offloaded_helper
            ):
                self.alias_calls.append((func.attr, node.lineno))
                continue
            if not isinstance(func.value, ast.Attribute):
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
                and not in_offloaded_helper
            ):
                self.raw_calls.append((func.attr, node.lineno))

    @staticmethod
    def _enclosing_fn_map(tree: ast.AST) -> dict:
        """Map id(node) -> name of the nearest lexically-enclosing function."""
        out = {}

        def walk(node, fn_name):
            this_fn = fn_name
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                this_fn = node.name
            for child in ast.iter_child_nodes(node):
                out[id(child)] = this_fn
                walk(child, this_fn)

        walk(tree, None)
        return out

    @staticmethod
    def _ancestor_fns(tree: ast.AST) -> dict:
        """Map id(node) -> frozenset of ALL enclosing function names (any depth)."""
        out = {}

        def walk(node, stack):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stack = stack + (node.name,)
            for child in ast.iter_child_nodes(node):
                out[id(child)] = frozenset(stack)
                walk(child, stack)

        walk(tree, ())
        return out

    @staticmethod
    def _is_session_db_source(value: ast.AST) -> bool:
        """True if an assignment RHS resolves to <obj>._session_db.

        Matches both ``<obj>._session_db`` and ``getattr(<obj>, "_session_db", ...)``
        where <obj> is any Name (covers ``self`` and captured closure params like
        ``_self``). Excludes the ``._db`` sync handle.
        """
        if isinstance(value, ast.Attribute):
            return value.attr == "_session_db" and isinstance(value.value, ast.Name)
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "getattr"
            and len(value.args) >= 2
            and isinstance(value.args[1], ast.Constant)
            and value.args[1].value == "_session_db"
        ):
            return True
        return False

    @classmethod
    def _collect_alias_names(cls, tree: ast.AST) -> set:
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and cls._is_session_db_source(node.value):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        names.add(tgt.id)
            elif isinstance(node, ast.AnnAssign) and node.value is not None \
                    and cls._is_session_db_source(node.value) \
                    and isinstance(node.target, ast.Name):
                names.add(node.target.id)
        return names


def _scan(rel_path: str) -> _RawCallVisitor:
    source = (_repo_root() / rel_path).read_text(encoding="utf-8")
    return _RawCallVisitor(ast.parse(source))


def test_no_raw_session_db_calls_on_gateway_loop():
    """Fail if any non-awaited SessionDB call appears in gateway files.

    Every loop-reachable DB call must go through AsyncSessionDB (await), whether
    spelled directly (self._session_db.<method>(...)) or via a local alias
    (db = getattr(self, "_session_db", None); db.<method>(...)). The
    sanitize_title staticmethod is called on the class, not self/an alias, so it
    is not matched; the _db. sync escape is checked separately below.
    """
    violations = []
    for rel in _GATEWAY_FILES:
        v = _scan(rel)
        violations.extend(f"{rel}:{ln} self._session_db.{m}(" for m, ln in v.raw_calls)
        violations.extend(f"{rel}:{ln} <alias>.{m}( (binds _session_db)" for m, ln in v.alias_calls)
    assert not violations, (
        "Non-awaited SessionDB calls on the gateway loop — route through "
        "AsyncSessionDB (await ...):\n  " + "\n  ".join(violations)
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


def test_offloaded_helpers_never_called_bare_on_loop():
    """The offloaded sync helpers must never be called bare on the event loop.

    They touch SessionDB synchronously, so a bare ``self._helper(...)`` on the
    loop would freeze it. The contract: loop-side callers wrap them in
    ``await asyncio.to_thread(self._helper, ...)`` (which references the helper
    as an attribute — no Call node — so it never appears here). A bare call is
    only legitimate when it runs off-loop: inside the ``run_sync`` thread-pool
    closure, or inside another offloaded helper (sync->sync, same thread). Any
    other bare call means a helper whose body the guard exempts is being invoked
    on the loop anyway — re-freezing the loop through the exemption.
    """
    off_loop_ok = _OFFLOADED_SYNC_HELPERS | {"run_sync"}
    violations = []
    for rel in _GATEWAY_FILES:
        v = _scan(rel)
        for helper, ln, ancestors in v.bare_helper_calls:
            if not (ancestors & off_loop_ok):
                violations.append(f"{rel}:{ln} bare self.{helper}( on the loop")
    assert not violations, (
        "Offloaded sync helper called bare on the gateway loop — wrap in "
        "await asyncio.to_thread(self.<helper>, ...):\n  " + "\n  ".join(violations)
    )


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


@pytest.mark.asyncio
async def test_concurrent_create_session_idempotent(tmp_path):
    db = AsyncSessionDB(hermes_state.SessionDB(db_path=tmp_path / "state.db"))
    sid = "s-create"

    await asyncio.gather(*(db.create_session(sid, "test") for _ in range(20)))

    rows = await db.list_sessions_rich(limit=100)
    assert sum(1 for r in rows if r["id"] == sid) == 1


# --------------------------------------------------------------------------
# API server ownership: aiohttp handlers must use a per-home async facade.
# --------------------------------------------------------------------------

def test_api_session_handlers_do_not_use_raw_store_helper():
    """The request path must not construct/access SessionDB synchronously."""
    source = (_repo_root() / "gateway/platforms/api_server.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    violations = []
    state_methods = {
        "create_session",
        "create_api_session_with_title",
        "delete_session",
        "end_session",
        "fork_api_session",
        "get_messages",
        "get_messages_as_conversation",
        "get_next_title_in_lineage",
        "get_session",
        "list_sessions_rich",
        "replace_messages",
        "resolve_resume_session_id",
        "set_session_title",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef) or not node.name.startswith("_handle_"):
            continue
        awaited = {
            id(child.value)
            for child in ast.walk(node)
            if isinstance(child, ast.Await) and isinstance(child.value, ast.Call)
        }
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "self"
                and child.func.attr == "_ensure_session_db"
            ):
                violations.append(f"{node.name}:{child.lineno}")
            if (
                isinstance(child, ast.Call)
                and (
                    (
                        isinstance(child.func, ast.Name)
                        and child.func.id == "SessionDB"
                    )
                    or (
                        isinstance(child.func, ast.Attribute)
                        and isinstance(child.func.value, ast.Name)
                        and child.func.value.id == "SessionDB"
                    )
                )
            ):
                violations.append(f"{node.name}:{child.lineno} direct SessionDB call")
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "db"
                and child.func.attr in state_methods
                and id(child) not in awaited
            ):
                violations.append(f"{node.name}:{child.lineno} db.{child.func.attr}")
    assert not violations, "aiohttp handlers must use _ensure_async_session_db: " + ", ".join(violations)


@pytest.mark.asyncio
async def test_api_session_store_cache_tracks_resolved_store_and_config(monkeypatch, tmp_path):
    """Resolved store/config identity scopes async facades and replaces stale raw stores."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    first_home = tmp_path / "first"
    second_home = tmp_path / "second"
    first_home.mkdir()
    second_home.mkdir()
    configs = {
        first_home: {"model": {"default": "first"}},
        second_home: {"model": {"default": "second"}},
    }
    stores = []
    calls = []
    loop_thread = threading.get_ident()

    class Store:
        def __init__(self, home):
            self.home = home
            self.closed = 0

        def close(self):
            self.closed += 1

    def load_config(home):
        return configs[home]

    def open_for_home(home, *, config=None):
        calls.append((home, config, threading.get_ident()))
        store = Store(home)
        stores.append(store)
        return store

    monkeypatch.setattr("hermes_cli.config.load_config_for_home", load_config)
    monkeypatch.setattr(hermes_state.SessionDB, "for_home", open_for_home)
    monkeypatch.setattr(adapter, "_session_db_home", lambda: first_home)

    first = await adapter._ensure_async_session_db()
    again = await adapter._ensure_async_session_db()
    configs[first_home] = {"model": {"default": "first-reconfigured"}}
    refreshed = await adapter._ensure_async_session_db()
    monkeypatch.setattr(adapter, "_session_db_home", lambda: second_home)
    second = await adapter._ensure_async_session_db()

    assert first is again
    assert refreshed is not first
    assert first is not second
    assert first._db.home == first_home
    assert refreshed._db.home == first_home
    assert second._db.home == second_home
    assert first._db.closed == 1
    assert len(adapter._session_dbs) == 2
    assert len(adapter._async_session_dbs) == 2
    assert all(key.startswith("sqlite:") for key in adapter._session_dbs)
    assert [home for home, _config, _thread in calls] == [
        first_home,
        first_home,
        second_home,
    ]
    assert all(worker_thread != loop_thread for _home, _config, worker_thread in calls)


@pytest.mark.asyncio
async def test_api_session_store_never_falls_back_from_configured_postgres(monkeypatch, tmp_path):
    """A selected non-SQLite backend must fail closed without creating state.db."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    home = tmp_path / "postgres-profile"
    home.mkdir()
    config = {
        "sessions": {
            "state": {
                "backend": "postgres",
                "postgres": {"schema": "profile_state"},
            },
        },
    }
    monkeypatch.setattr("hermes_cli.config.load_config_for_home", lambda _home: config)
    monkeypatch.setattr(adapter, "_session_db_home", lambda: home)

    assert await adapter._ensure_async_session_db() is None
    assert adapter._session_dbs == {}
    assert adapter._async_session_dbs == {}
    assert not (home / "state.db").exists()


@pytest.mark.asyncio
async def test_api_disconnect_closes_all_cached_raw_session_stores_off_loop():
    """Shutdown releases every cached raw store without closing manual overrides."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    loop_thread = threading.get_ident()

    class Store:
        def __init__(self):
            self.close_threads = []

        def close(self):
            self.close_threads.append(threading.get_ident())

    first = Store()
    second = Store()
    adapter._session_dbs = {"one": first, "two": second}
    adapter._session_db_home_keys = {"one-home": "one", "two-home": "two"}
    adapter._async_session_dbs = {"one": object(), "two": object()}

    await adapter.disconnect()

    assert len(first.close_threads) == len(second.close_threads) == 1
    assert first.close_threads[0] != loop_thread
    assert second.close_threads[0] != loop_thread
    assert adapter._session_dbs == {}
    assert adapter._session_db_home_keys == {}
    assert adapter._async_session_dbs == {}


@pytest.mark.asyncio
async def test_api_disconnect_waits_for_inflight_store_acquisition(monkeypatch, tmp_path):
    """Shutdown drains an opening store instead of caching it after close begins."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    home = tmp_path / "profile"
    home.mkdir()
    opening = threading.Event()
    release = threading.Event()

    class Store:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    store = Store()

    def open_for_home(_home, *, config=None):
        opening.set()
        assert release.wait(timeout=3), "test did not release store opening"
        return store

    monkeypatch.setattr("hermes_cli.config.load_config_for_home", lambda _home: {})
    monkeypatch.setattr(hermes_state.SessionDB, "for_home", open_for_home)
    monkeypatch.setattr(adapter, "_session_db_home", lambda: home)

    acquire = asyncio.create_task(adapter._ensure_async_session_db())
    assert await asyncio.to_thread(opening.wait, 3)
    shutdown = asyncio.create_task(adapter.disconnect())
    await asyncio.sleep(0.02)
    assert not shutdown.done(), "disconnect must wait for the active open"

    release.set()
    assert await acquire is None
    await shutdown

    assert store.closed == 1
    assert adapter._session_dbs == {}
    assert adapter._async_session_dbs == {}


@pytest.mark.asyncio
async def test_api_disconnect_waits_for_active_session_request_before_close(monkeypatch, tmp_path):
    """A handler's off-loop DB call completes before disconnect closes its store."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    home = tmp_path / "profile"
    home.mkdir()
    started = threading.Event()
    release = threading.Event()

    class Store:
        def __init__(self):
            self.closed = 0

        def list_sessions_rich(self, **_kwargs):
            started.set()
            assert release.wait(timeout=3), "test did not release DB call"
            return []

        def close(self):
            self.closed += 1

    store = Store()
    monkeypatch.setattr("hermes_cli.config.load_config_for_home", lambda _home: {})
    monkeypatch.setattr(hermes_state.SessionDB, "for_home", lambda _home, *, config=None: store)
    monkeypatch.setattr(adapter, "_session_db_home", lambda: home)
    await adapter._ensure_async_session_db()

    request = types.SimpleNamespace(query={})
    request_task = asyncio.create_task(adapter._handle_list_sessions(request))
    assert await asyncio.to_thread(started.wait, 3)
    shutdown = asyncio.create_task(adapter.disconnect())
    await asyncio.sleep(0.02)
    assert not shutdown.done(), "disconnect closed a store while a request used it"

    release.set()
    response = await request_task
    await shutdown

    assert response.status == 200
    assert store.closed == 1


@pytest.mark.asyncio
async def test_api_session_history_does_not_stall_event_loop():
    """A blocking store fake must run on a worker while the loop keeps ticking."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    started = threading.Event()
    finished = threading.Event()
    ticked = asyncio.Event()
    worker_threads = []
    loop_thread = threading.get_ident()

    class BlockingStore:
        def get_messages_as_conversation(self, session_id):
            worker_threads.append(threading.get_ident())
            started.set()
            time.sleep(0.15)
            finished.set()
            return [{"role": "user", "content": session_id}]

    async def ticker():
        await asyncio.sleep(0.02)
        ticked.set()

    adapter._session_db = BlockingStore()
    history_task = asyncio.create_task(adapter._conversation_history_for_session("session-1"))
    ticker_task = asyncio.create_task(ticker())

    await asyncio.sleep(0.05)

    assert started.is_set()
    assert ticked.is_set(), "blocking SessionDB work stalled the event loop"
    assert not finished.is_set(), "store call unexpectedly ran on the event loop"
    assert await history_task == [{"role": "user", "content": "session-1"}]
    await ticker_task
    assert worker_threads and all(thread_id != loop_thread for thread_id in worker_threads)
