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
    return (get_hermes_home_override(), dict(secrets) if secrets is not None else None,
            current_secret_scope_home(), dict(terminal) if terminal is not None else None, type(terminal).__name__)


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
    assert expected[0] == str(profile) and expected[1]["LOOPSTALL_PROBE_KEY"] == "from-profile"
    assert expected[3]["TERMINAL_TIMEOUT"] == "77"

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


@pytest.mark.asyncio
async def test_ancillary_polls_project_off_the_loop(monkeypatch):
    """``subagent.list`` / ``process.list`` / ``session.control.read`` build their projection in a worker:
    it takes the session's event lock (held by every streamed token) and reads state.db."""
    from gateway import session_ancillary

    loop_thread = threading.get_ident()
    seen = []

    def projection(authority, live, ref, params, kind):
        seen.append((threading.get_ident(), kind))
        return {'subagents': [], 'delegations': []}

    monkeypatch.setattr(session_ancillary, "_read_projection", projection)
    monkeypatch.setattr(session_ancillary, "authorize", lambda *a, **k: None)
    ref = SimpleNamespace(session_id="s1", profile_id="fixture")
    connection = SimpleNamespace(authority=SimpleNamespace(sessions={"s1": object()}))
    result = await session_ancillary.read(connection, ref, {"session_id": "s1"}, kind="subagents")
    assert result == {'subagents': [], 'delegations': []}
    assert seen and seen[0][0] != loop_thread, "the projection ran on the event loop"
