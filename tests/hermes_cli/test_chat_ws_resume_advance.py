"""A reconnect whose ``?resume=`` advanced along the same session chain must
reattach the running PTY instead of closing it mid-turn (#133052).

The SPA follows a live turn's session forward (compaction handoffs, /model
child sessions) via latest-descendant and rebuilds the socket with the new
``?resume=``. The rebuilt registry key differs from the running PTY's key only
in that resume component, so ``close_other_sessions`` used to close the PTY —
interrupting the in-flight turn as a bogus "explicit stop requested".
"""

import pytest

import hermes_cli.web_server_chat as _web_server_chat
import hermes_cli.web_server_sessions as _web_server_sessions
import hermes_cli.web_routers.sessions as _sessions_router
from hermes_cli import web_server


pytestmark = pytest.mark.platforms("posix")  # PTY bridge is POSIX-only


class FakeBridge:
    def __init__(self):
        self.alive = True
        self.accept_input = True
        self.written = bytearray()

    def read(self, timeout):
        return b""  # idle forever

    async def write(self, data):
        if not self.accept_input:
            return False
        self.written.extend(data)
        return True

    def resize(self, *, cols, rows):
        pass

    def is_alive(self):
        return self.alive

    def close(self):
        self.alive = False


# Session chain of the fake store: s1 spawned child s2. Everything else is a
# disconnected (or unknown) session head.
CHAIN = {"s1": "s2", "s2": None}


def _fake_latest_descendant(session_id, db):
    if session_id not in CHAIN:
        return None, []
    child = CHAIN[session_id]
    return (child if child else session_id), (
        [session_id, child] if child else [session_id]
    )


@pytest.fixture
def resume_advance_harness(monkeypatch):
    bridges = []

    def fake_spawn(argv, cwd=None, env=None):
        b = FakeBridge()
        bridges.append(b)
        return b

    monkeypatch.setattr(_web_server_chat.PtyBridge, "spawn", staticmethod(fake_spawn))
    monkeypatch.setattr(_web_server_chat, "_ws_auth_reason", lambda ws: (None, "test"))
    monkeypatch.setattr(_web_server_chat, "_ws_host_origin_reason", lambda ws: None)
    monkeypatch.setattr(_web_server_chat, "_ws_client_reason", lambda ws: None)

    async def fake_argv(**kw):
        resume = kw.get("resume")
        env = {"HERMES_TUI_RESUME": resume} if resume else {}
        return (["x", resume or "fresh"], "/tmp", env)

    monkeypatch.setattr(_web_server_chat, "_resolve_chat_argv_async", fake_argv)
    monkeypatch.setattr(
        _sessions_router, "_with_db", lambda profile, fn, *, read_only: fn(object())
    )
    monkeypatch.setattr(
        _web_server_sessions, "_session_latest_descendant", _fake_latest_descendant
    )

    try:
        yield bridges
    finally:
        _web_server_chat.PTY_REGISTRY._sessions.clear()


def _connect(client, url):
    return client.websocket_connect(url)


@pytest.mark.asyncio
async def test_resume_advance_same_chain_reattaches_running_pty(resume_advance_harness):
    """s1 → s2 along one chain is the same chat: the running PTY survives."""
    from starlette.testclient import TestClient

    client = TestClient(web_server.app)
    with _connect(client, "/api/pty?attach=TOKR&resume=s1") as ws1:
        ws1.send_bytes(b"hi")
    # The turn advanced the session; the SPA rewrote ?resume= and rebuilt.
    with _connect(client, "/api/pty?attach=TOKR&resume=s2") as ws2:
        ws2.send_bytes(b"again")

    assert len(resume_advance_harness) == 1  # no respawn
    assert resume_advance_harness[0].alive is True  # not closed mid-turn
    assert (
        bytes(resume_advance_harness[0].written) == b"hi\x0cagain"
    )  # one PTY saw both


@pytest.mark.asyncio
async def test_resume_advance_backward_same_chain_reattaches_running_pty(
    resume_advance_harness,
):
    """A URL rewind to the parent head is still the same chain."""
    from starlette.testclient import TestClient

    client = TestClient(web_server.app)
    with _connect(client, "/api/pty?attach=TOKR&resume=s2") as ws1:
        ws1.send_bytes(b"hi")
    with _connect(client, "/api/pty?attach=TOKR&resume=s1") as ws2:
        ws2.send_bytes(b"again")

    assert len(resume_advance_harness) == 1
    assert resume_advance_harness[0].alive is True
    assert bytes(resume_advance_harness[0].written) == b"hi\x0cagain"


@pytest.mark.asyncio
async def test_resume_switch_to_other_chat_still_closes_old_pty(resume_advance_harness):
    """A resume on another chain is a chat switch: the old PTY is closed (4409 design)."""
    from starlette.testclient import TestClient

    client = TestClient(web_server.app)
    with _connect(client, "/api/pty?attach=TOKR&resume=s1") as ws1:
        ws1.send_bytes(b"hi")
    with _connect(client, "/api/pty?attach=TOKR&resume=other") as ws2:
        ws2.send_bytes(b"again")

    assert len(resume_advance_harness) == 2  # respawned for the new chat
    assert resume_advance_harness[0].alive is False  # old PTY closed by design


class _StubRegistry:
    """Only the read side _same_chain_attach_key needs."""

    def __init__(self, *keys):
        self.keys = list(keys)

    def live_sibling_keys(self, prefix, *, keep_key):
        return [
            k
            for k in self.keys
            if k != keep_key and (k == prefix or k.startswith(prefix + "\0"))
        ]


@pytest.mark.asyncio
async def test_same_chain_attach_key_bare_sibling_is_ignored(monkeypatch):
    """A bare key has no resume to compare: keep the default (close) behavior."""
    from hermes_cli.web_routers.chat_ws import _same_chain_attach_key

    reg = _StubRegistry("TOK")
    key = await _same_chain_attach_key(reg, "TOK", "TOK\0\0s2", None, "s2")
    assert key == "TOK\0\0s2"


@pytest.mark.asyncio
async def test_same_chain_attach_key_other_profile_is_ignored():
    """A sibling keyed under another profile is a profile switch, not a drift."""
    from hermes_cli.web_routers.chat_ws import _same_chain_attach_key

    reg = _StubRegistry("TOK\0beta\0s1")
    key = await _same_chain_attach_key(reg, "TOK", "TOK\0\0s2", None, "s2")
    assert key == "TOK\0\0s2"


@pytest.mark.asyncio
async def test_same_chain_attach_key_session_db_failure_falls_back(monkeypatch):
    """An unreadable session store must degrade to the default key, never block chat."""
    from hermes_cli.web_routers.chat_ws import _same_chain_attach_key

    def boom(profile, fn, *, read_only):
        raise RuntimeError("session store unreadable")

    monkeypatch.setattr(_sessions_router, "_with_db", boom)
    reg = _StubRegistry("TOK\0\0s1")
    key = await _same_chain_attach_key(reg, "TOK", "TOK\0\0s2", None, "s2")
    assert key == "TOK\0\0s2"


@pytest.mark.asyncio
async def test_same_chain_attach_key_without_resume_returns_wanted():
    from hermes_cli.web_routers.chat_ws import _same_chain_attach_key

    reg = _StubRegistry("TOK\0\0s1")
    key = await _same_chain_attach_key(reg, "TOK", "TOK\0p\0", "p", None)
    assert key == "TOK\0p\0"


@pytest.mark.asyncio
async def test_live_sibling_keys_filters_prefix_and_liveness():
    from hermes_cli.pty_session import PtySessionRegistry

    reg = PtySessionRegistry(
        ttl=1800.0, max_sessions=16, buffer_cap=1024, read_timeout=0.01
    )
    live = FakeBridge()
    dead = FakeBridge()
    dead.alive = False
    for key, bridge in {
        "TOK\0\0s1": live,
        "TOK\0beta\0s1": FakeBridge(),
        "TOK": dead,  # bare but dead: filtered by liveness
        "OTHER\0\0s1": FakeBridge(),  # different attach token: filtered by prefix
    }.items():
        session = type("S", (), {})()  # only .alive is read by the predicate
        session.alive = bridge.alive
        reg._sessions[key] = session

    keys = reg.live_sibling_keys("TOK", keep_key="TOK\0beta\0s1")
    assert keys == ["TOK\0\0s1"]
