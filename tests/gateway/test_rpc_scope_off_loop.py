"""RPC dispatch enters its owner's profile scope without reading profile files on the event loop.

Every client RPC (and the classic CLI / Ink / Desktop poll ``subagent.list`` and ``process.list``
every 1-5 s per open session) runs under ``owner_scope``. On a multiplexed gateway that scope reads
the profile's ``.env``, ``config.yaml`` and terminal policy; on the loop it was the largest single
cost of dispatch with many sessions open (gateway disconnects under load).
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest


def _authority(home, multiplex=True):
    return SimpleNamespace(profile_id=str(home),
                           runner=SimpleNamespace(config=SimpleNamespace(multiplex_profiles=multiplex)))


def _bound():
    from agent.secret_scope import current_secret_scope, current_secret_scope_home
    from hermes_constants import get_hermes_home_override
    from tools.terminal_scope import get_terminal_scope
    secrets = current_secret_scope()
    terminal = get_terminal_scope()
    return {"home": get_hermes_home_override(),
            "secrets": dict(secrets) if secrets is not None else None,
            "secret_home": current_secret_scope_home(),
            "terminal": dict(terminal) if terminal is not None else None,
            "terminal_type": type(terminal).__name__}


@pytest.fixture
def profile(tmp_path):
    home = tmp_path / "profiles" / "worker"
    home.mkdir(parents=True)
    (home / ".env").write_text("LOOPSTALL_PROBE_KEY=from-profile\nTERMINAL_TIMEOUT=77\n", encoding="utf-8")
    (home / "config.yaml").write_text("terminal:\n  backend: local\n  cwd: /srv/work\n", encoding="utf-8")
    return home


@pytest.mark.asyncio
async def test_async_owner_scope_binds_what_owner_scope_binds_with_the_reads_off_the_loop(profile, monkeypatch):
    from gateway import session_authorities
    from tools import terminal_scope

    authority = _authority(profile)
    with session_authorities.owner_scope(authority):
        expected = _bound()
    assert expected["home"] == str(profile) and expected["secrets"]["LOOPSTALL_PROBE_KEY"] == "from-profile"
    assert expected["terminal"]["TERMINAL_TIMEOUT"] == "77"

    loop_thread = threading.get_ident()
    build_threads = []
    real_build = terminal_scope.build_profile_terminal_scope

    def build(*args, **kwargs):
        build_threads.append(threading.get_ident())
        return real_build(*args, **kwargs)

    monkeypatch.setattr(terminal_scope, "build_profile_terminal_scope", build)
    before = _bound()
    async with session_authorities.async_owner_scope(authority):
        got = _bound()
        await asyncio.sleep(0)  # the scope spans awaits, as a dispatched handler's does
        assert _bound() == got
    assert got == expected
    assert build_threads and loop_thread not in build_threads, "the terminal policy was built on the event loop"
    assert _bound() == before, "the scope leaked past the async with"


@pytest.mark.asyncio
async def test_async_owner_scope_keeps_owner_scope_for_unrouted_owners(tmp_path):
    """A single-profile gateway, a symbolic test owner and a vanished home bind exactly what
    ``owner_scope`` binds (nothing, or the launch scope), without a worker hop."""
    from gateway.session_authorities import async_owner_scope, owner_scope

    for authority in (_authority(tmp_path, multiplex=False), _authority("fixture"),
                      _authority(tmp_path / "gone")):
        with owner_scope(authority):
            expected = _bound()
        async with async_owner_scope(authority):
            assert _bound() == expected


class _RecordingLock:
    """``event_stream.lock`` stand-in: records the thread that takes it, optionally runs a hook first."""

    def __init__(self, on_enter=None):
        self.threads = []
        self._on_enter = on_enter

    def __enter__(self):
        self.threads.append(threading.get_ident())
        if self._on_enter:
            self._on_enter()
        return self

    def __exit__(self, *exc):
        return False


def _ancillary_fixture(monkeypatch, *, on_enter=None, snapshot=None):
    from gateway import session_ancillary
    lock = _RecordingLock(on_enter)
    live = SimpleNamespace(event_stream=SimpleNamespace(lock=lock))
    authority = SimpleNamespace(sessions={"s1": live})
    monkeypatch.setattr(session_ancillary, "authorize", lambda *a, **k: None)
    monkeypatch.setattr(session_ancillary, "control_snapshot", snapshot or (lambda authority, ref: {"revision": "r1"}))
    ref = SimpleNamespace(session_id="s1", profile_id="fixture")
    return session_ancillary, SimpleNamespace(authority=authority), ref, lock


@pytest.mark.asyncio
async def test_ancillary_polls_take_the_event_lock_off_the_loop(monkeypatch):
    """The projection takes the session's event lock (held by every streamed token) and reads state.db:
    it must do both in a worker, not on the event loop."""
    session_ancillary, connection, ref, lock = _ancillary_fixture(monkeypatch)
    result = await session_ancillary.read(connection, ref, {"session_id": "s1"}, kind="control")
    assert result == {"control": {"revision": "r1"}}
    assert lock.threads and threading.get_ident() not in lock.threads, "the event lock was taken on the loop"


@pytest.mark.asyncio
async def test_ancillary_poll_of_a_session_retired_mid_read_is_not_found(monkeypatch):
    """A chat deleted or pruned while its poll waits in the worker answers not_found (what the loop-side
    check answers), not an internal error from the projection re-indexing ``authority.sessions``."""
    from hermes_state_runtime import RuntimeStoreError

    def retire_then_index(authority, ref):
        authority.sessions.pop("s1")
        return authority.sessions["s1"]  # physical_target / _require_current_route re-index like this

    session_ancillary, connection, ref, _lock = _ancillary_fixture(monkeypatch, snapshot=retire_then_index)
    with pytest.raises(RuntimeStoreError) as exc:
        await session_ancillary.read(connection, ref, {"session_id": "s1"}, kind="control")
    assert str(exc.value) == "not_found"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind, params", [("subagents", {}), ("processes", {}), ("tail", {"subagent_id": "sa-1"})])
async def test_real_projection_paused_while_the_session_is_retired_is_not_found(monkeypatch, kind, params):
    """The real projection (no stubs past the lock) paused on the event lock while delete/prune retires the
    session: ``retire_live_sessions`` pops ``authority.sessions`` on the loop without that lock, and the
    projection's own re-index (``_require_current_route``) must answer not_found, not a dispatch crash."""
    from hermes_state_runtime import RuntimeStoreError

    holder = {}
    session_ancillary, connection, ref, lock = _ancillary_fixture(
        monkeypatch, on_enter=lambda: holder["authority"].sessions.pop("s1"))
    holder["authority"] = connection.authority
    with pytest.raises(RuntimeStoreError) as exc:
        await session_ancillary.read(connection, ref, {"session_id": "s1", **params}, kind=kind)
    assert str(exc.value) == "not_found"
    assert lock.threads and threading.get_ident() not in lock.threads


@pytest.mark.asyncio
async def test_ancillary_projection_keyerror_of_a_live_session_still_raises(monkeypatch):
    """Only a retired session maps to not_found: a KeyError from a bug in a live session's projection
    still surfaces."""
    def broken(authority, ref):
        return {}["missing"]

    session_ancillary, connection, ref, _lock = _ancillary_fixture(monkeypatch, snapshot=broken)
    with pytest.raises(KeyError):
        await session_ancillary.read(connection, ref, {"session_id": "s1"}, kind="control")
