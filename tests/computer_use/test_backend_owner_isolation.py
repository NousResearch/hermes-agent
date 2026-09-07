"""Backend and approval ownership must isolate profile/session pairs without
serializing unrelated owners behind startup or resurrecting released authority.

Collision and empty-session probes use real context-local homes; the collision
pair names need no on-disk profiles because backend construction is substituted.
"""

import json
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@contextmanager
def _profile(home):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("grant", ["approve_session", "always_approve"])
@pytest.mark.parametrize("release_index", [0, 1])
def test_collision_pair_grants_do_not_cross(monkeypatch, grant, release_index):
    from tools.computer_use import tool as cu
    from tools.computer_use_tool import registry

    owners = [("/tmp/astra-owner", "segment:session"),
              ("/tmp/astra-owner:segment", "session")]
    prompts = []
    monkeypatch.setattr(cu, "_new_backend", lambda mode: cu._NoopBackend())

    def approve(*args):
        prompts.append(args[0])
        return grant

    monkeypatch.setattr(cu, "_approval_callback", approve)

    def click(index):
        home, sid = owners[index]
        with _profile(home):
            result = registry.dispatch("computer_use", {"action": "click", "x": 1, "y": 1}, session_id=sid)
            assert "error" not in json.loads(result)

    click(0)
    click(0)
    click(1)
    assert prompts == ["click", "click"]
    with _profile(owners[release_index][0]):
        assert cu.release_computer_use_session(owners[release_index][1])
    click(1 - release_index)
    assert prompts == ["click", "click"]
    click(release_index)
    assert prompts == ["click", "click", "click"]


@pytest.mark.parametrize("release_index", [0, 1])
def test_collision_pair_backends_do_not_cross(monkeypatch, release_index):
    from tools.computer_use import tool as cu

    owners = [("/tmp/astra-owner", "segment:session"),
              ("/tmp/astra-owner:segment", "session")]
    monkeypatch.setattr(cu, "_new_backend", _InstantBackend)
    backends = []
    for home, sid in owners:
        with _profile(home):
            backends.append(cu._get_backend(sid))
    assert backends[0] is not backends[1]
    with _profile(owners[release_index][0]):
        assert cu.release_computer_use_session(owners[release_index][1])
    assert backends[release_index].stopped
    assert not backends[1 - release_index].stopped
    home, sid = owners[1 - release_index]
    with _profile(home):
        assert cu._get_backend(sid) is backends[1 - release_index]


def test_trailing_colon_release_does_not_enter_empty_session(monkeypatch, tmp_path):
    from tools.computer_use import tool as cu

    monkeypatch.setattr(cu, "_new_backend", _InstantBackend)
    with _profile(tmp_path / "a"):
        backend = cu._get_backend("named-session:")
    with _profile(tmp_path / "b"):
        assert cu.release_computer_use_session("x:") is False
    assert not backend.stopped
    with _profile(tmp_path / "a"):
        assert cu._get_backend("named-session:") is backend


def test_empty_session_acquisition_is_profile_local(monkeypatch, tmp_path):
    from tools.computer_use import tool as cu

    monkeypatch.setattr(cu, "_new_backend", _InstantBackend)
    with _profile(tmp_path / "a"):
        backend_a = cu._get_backend("")
    with _profile(tmp_path / "b"):
        backend_b = cu._get_backend("")
        assert backend_b is not backend_a
        assert cu.release_computer_use_session("")
    assert backend_b.stopped
    assert not backend_a.stopped
    with _profile(tmp_path / "a"):
        assert cu._get_backend("") is backend_a


def test_empty_session_injection_is_profile_local(monkeypatch, tmp_path):
    from tools.computer_use import tool as cu

    injected = _InstantBackend()
    monkeypatch.setattr(cu, "_new_backend", _InstantBackend)
    with _profile(tmp_path / "a"):
        cu._backend[cu.hermes_home_key()] = injected
    with _profile(tmp_path / "b"):
        assert cu.release_computer_use_session("") is False
        other = cu._get_backend("")
        assert other is not injected
        assert cu.release_computer_use_session("")
    assert not injected.stopped
    with _profile(tmp_path / "a"):
        assert cu._get_backend("") is injected
        assert cu.release_computer_use_session("")
    assert injected.stopped


def test_release_fences_mode_replacement_teardown(monkeypatch):
    from tools.computer_use import tool as cu

    stopping, finish_stop = threading.Event(), threading.Event()

    class BlockingStop(_InstantBackend):
        def stop(self):
            stopping.set()
            assert finish_stop.wait(5)
            super().stop()

    created = []

    def create(mode):
        backend = BlockingStop(mode) if not created else _InstantBackend(mode)
        created.append(backend)
        return backend

    mode = ["standard"]
    monkeypatch.setattr(cu, "_cua_permission_mode", lambda sid: mode[0])
    monkeypatch.setattr(cu, "_new_backend", create)
    cu._get_backend("mode-owner")
    mode[0] = "unrestricted"
    with ThreadPoolExecutor(max_workers=1) as pool:
        lookup = pool.submit(cu._get_backend, "mode-owner")
        try:
            assert stopping.wait(5)
            released = cu.release_computer_use_session("mode-owner")
        finally:
            finish_stop.set()
        with pytest.raises(RuntimeError, match="released"):
            lookup.result(timeout=5)
    assert released is True
    assert len(created) == 1
    assert cu._backend_owner_key("mode-owner") not in cu._backends


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("release_during_start", [False, True])
def test_failed_start_cleans_generation_and_candidate(monkeypatch, error_type, release_during_start):
    from tools.computer_use import tool as cu

    started, finish = threading.Event(), threading.Event()

    class FailingStart(_InstantBackend):
        def start(self):
            started.set()
            assert finish.wait(5)
            raise error_type("startup failed")

    candidate = FailingStart()
    monkeypatch.setattr(cu, "_new_backend", lambda mode: candidate)
    with ThreadPoolExecutor(max_workers=1) as pool:
        lookup = pool.submit(cu._get_backend, "failed-owner")
        try:
            assert started.wait(5)
            if release_during_start:
                cu.release_computer_use_session("failed-owner")
        finally:
            finish.set()
        with pytest.raises(error_type, match="startup failed"):
            lookup.result(timeout=5)
    owner = cu._backend_owner_key("failed-owner")
    assert candidate.stopped
    assert owner not in cu._backend_start_locks
    assert owner not in cu._backends


def test_queued_waiter_keeps_revoked_generation(monkeypatch):
    from tools.computer_use import tool as cu

    waiting = threading.Event()

    class ObservedLock:
        def __init__(self):
            self.lock = threading.Lock()
            self.entries = 0

        def __enter__(self):
            self.entries += 1
            if self.entries == 2:
                waiting.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    stale, replacement = _SlowStartBackend(), _InstantBackend()
    backends = iter((stale, replacement))
    monkeypatch.setattr(cu, "_new_backend", lambda mode: next(backends))
    cu._backend_start_locks[cu._backend_owner_key("queued-owner")] = ObservedLock()
    with ThreadPoolExecutor(max_workers=2) as pool:
        starter = pool.submit(cu._get_backend, "queued-owner")
        try:
            assert stale.started.wait(5)
            waiter = pool.submit(cu._get_backend, "queued-owner")
            assert waiting.wait(5)
            cu.release_computer_use_session("queued-owner")
            assert cu._get_backend("queued-owner") is replacement
        finally:
            stale.release.set()
        for lookup in (starter, waiter):
            with pytest.raises(RuntimeError, match="released"):
                lookup.result(timeout=5)
    assert stale.stopped
    assert not replacement.stopped


@pytest.fixture(autouse=True)
def _reset_computer_use_state():
    from hermes_constants import reset_hermes_home_key_cache
    from tools.computer_use.tool import reset_backend_for_tests

    reset_backend_for_tests()
    reset_hermes_home_key_cache()
    _SlowStartBackend.started.clear()
    _SlowStartBackend.release.clear()
    yield
    _SlowStartBackend.release.set()
    reset_backend_for_tests()
    reset_hermes_home_key_cache()


class _InstantBackend:
    """No-op backend that records stop() so release isolation is observable."""

    def __init__(self, permission_mode="standard"):
        self.permission_mode = permission_mode
        self.stopped = False
        self.started_flag = False

    def start(self):
        self.started_flag = True

    def stop(self):
        self.stopped = True


class _SlowStartBackend:
    """Blocks inside start() until the test releases it — a slow remote handshake."""

    started = threading.Event()
    release = threading.Event()

    def __init__(self, permission_mode="standard"):
        self.permission_mode = permission_mode
        self.stopped = False

    def start(self):
        self.started.set()
        self.release.wait(timeout=10)

    def stop(self):
        self.stopped = True


@pytest.fixture
def lifecycle_session(monkeypatch, tmp_path):
    from agent import context_compressor
    from run_agent import AIAgent
    from tui_gateway import server

    # Construction may probe model metadata; this lifecycle test never needs I/O.
    monkeypatch.setattr(context_compressor, "get_model_context_length", lambda *args, **kwargs: 128000)
    launch_home = tmp_path / "profile-a"
    launch_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setattr(server, "_hermes_home", launch_home)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_broadcast_global_event", lambda *args: None)
    agents = []

    def create(home, *, record_home=True):
        with _profile(home):
            agent = AIAgent(
                api_key="test", base_url="http://127.0.0.1:9/v1",
                provider="openai-compat", model="test", enabled_toolsets=[],
                quiet_mode=True, skip_context_files=True, skip_memory=True,
                session_id="same-durable-session",
            )
        agents.append(agent)
        session = {
            "agent": agent, "session_key": agent.session_id, "source": "tui",
            "history": [], "history_lock": threading.Lock(), "running": False,
            "transport": server._detached_ws_transport,
        }
        if record_home:
            session["profile_home"] = str(home)
        server._sessions["ui-b"] = session
        return agent

    yield server, launch_home, create
    for timer in server._pending_ws_reaps.values():
        timer.cancel()
        timer.join(timeout=5)
    for agent in agents:
        agent.close()


@pytest.mark.parametrize("profile", ["named", "default", "missing-record"])
def test_session_close_releases_its_profile_owner(monkeypatch, tmp_path, lifecycle_session, profile):
    from tools.computer_use import tool as cu

    server, launch_home, create = lifecycle_session
    home = tmp_path / ".hermes" if profile == "default" else tmp_path / "profile-b"
    agent = create(home, record_home=profile != "missing-record")
    monkeypatch.setattr(cu, "_new_backend", _InstantBackend)
    prompts = []
    monkeypatch.setattr(cu, "_approval_callback", lambda *args: prompts.append(args[0]) or "approve_session")
    args = {"action": "click", "x": 1, "y": 1}
    with _profile(launch_home):
        backend_a = cu._get_backend(agent.session_id)
        cu._request_approval("click", args, agent.session_id)
    with _profile(home):
        backend_b = cu._get_backend(agent.session_id)
        cu._request_approval("click", args, agent.session_id)

    # The reaper has no B scope; HERMES_HOME still belongs to launch profile A.
    assert server._close_session_by_id("ui-b", end_reason="idle_timeout")
    assert backend_b.stopped, "closing B must stop B's backend"
    assert not backend_a.stopped, "closing B must preserve A's same-ID backend"
    with _profile(launch_home):
        assert cu._get_backend(agent.session_id) is backend_a
        cu._request_approval("click", args, agent.session_id)
    assert prompts == ["click", "click"]
    with _profile(home):
        cu._request_approval("click", args, agent.session_id)
    assert prompts == ["click", "click", "click"]


def test_session_timer_close_fences_its_profile_blocked_start(monkeypatch, tmp_path, lifecycle_session):
    from tools.computer_use import tool as cu

    server, launch_home, create = lifecycle_session
    home = tmp_path / "profile-b"
    agent = create(home)
    monkeypatch.setattr(cu, "_new_backend", _InstantBackend)
    with _profile(launch_home):
        backend_a = cu._get_backend(agent.session_id)
    stale = _SlowStartBackend()
    monkeypatch.setattr(cu, "_new_backend", lambda mode: stale)
    closed = threading.Event()
    close = agent.close

    def observed_close():
        try:
            close()
        finally:
            closed.set()

    monkeypatch.setattr(agent, "close", observed_close)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 1)

    def acquire():
        with _profile(home):
            return cu._get_backend(agent.session_id)

    with ThreadPoolExecutor(max_workers=1) as pool:
        lookup = pool.submit(acquire)
        try:
            assert stale.started.wait(5)
            server._schedule_ws_orphan_reap("ui-b", delay_s=0)
            assert closed.wait(5), "real orphan timer did not finish session teardown"
        finally:
            stale.release.set()
        with pytest.raises(RuntimeError, match="released"):
            lookup.result(timeout=5)
    assert "ui-b" not in server._sessions
    assert stale.stopped, "revoked late startup must be stopped rather than published"
    assert not backend_a.stopped
    with _profile(home):
        owner = cu._backend_owner_key(agent.session_id)
        assert owner not in cu._backends
        assert owner not in cu._backend_start_locks


def _permission_mode(*args):
    return args[-1]


def test_backend_cache_keys_are_profile_qualified(monkeypatch):
    """Finding 1 (key shape): the same session id under two profile homes
    must resolve to two different cache entries."""
    from hermes_constants import hermes_home_key
    from tools.computer_use import tool as computer_use

    monkeypatch.setattr(
        "hermes_constants.get_hermes_home",
        lambda: Path("/home/fake/profile-a"),
    )
    key_a = computer_use._backend_owner_key("session-1")
    assert key_a == (hermes_home_key('/home/fake/profile-a'), "session-1")

    monkeypatch.setattr(
        "hermes_constants.get_hermes_home",
        lambda: Path("/home/fake/profile-b"),
    )
    key_b = computer_use._backend_owner_key("session-1")

    assert key_a != key_b, "same session id must not share a cache key across profiles"

    backend_a, backend_b = _InstantBackend(), _InstantBackend()
    computer_use._backends[key_a] = backend_a
    computer_use._backends[key_b] = backend_b
    assert computer_use._backends[key_a] is backend_a
    assert computer_use._backends[key_b] is backend_b


def test_get_backend_does_not_reuse_or_release_across_profiles(monkeypatch):
    """Finding 1 (behavior): two profile homes, same session id, different
    live backends. Reaching session-1 as profile B must not return profile
    A's backend, and releasing B must not stop A."""
    from tools.computer_use import tool as computer_use

    created = []

    def _fake_new_backend(*args):
        backend = _InstantBackend(_permission_mode(*args))
        created.append(backend)
        return backend

    monkeypatch.setattr(computer_use, "_new_backend", _fake_new_backend)

    monkeypatch.setattr(
        "hermes_constants.get_hermes_home",
        lambda: Path("/home/fake/profile-a"),
    )
    backend_a = computer_use._get_backend("session-1")

    monkeypatch.setattr(
        "hermes_constants.get_hermes_home",
        lambda: Path("/home/fake/profile-b"),
    )
    backend_b = computer_use._get_backend("session-1")

    assert backend_a is not backend_b
    assert len(created) == 2
    assert backend_a.started_flag and backend_b.started_flag

    # Release under profile B — only B's backend must stop.
    assert computer_use.release_computer_use_session("session-1") is True
    assert backend_b.stopped is True
    assert backend_a.stopped is False

    monkeypatch.setattr(
        "hermes_constants.get_hermes_home",
        lambda: Path("/home/fake/profile-a"),
    )
    assert computer_use._get_backend("session-1") is backend_a
    assert computer_use.release_computer_use_session("session-1") is True
    assert backend_a.stopped is True


def test_release_fences_inflight_start_before_same_owner_reacquires(monkeypatch):
    from tools.computer_use import tool as computer_use

    stale, replacement = _SlowStartBackend(), _InstantBackend()
    backends = iter((stale, replacement))
    monkeypatch.setattr(computer_use, "_new_backend", lambda *args: next(backends))

    with ThreadPoolExecutor(max_workers=2) as pool:
        original = pool.submit(computer_use._get_backend, "session-a")
        try:
            assert stale.started.wait(timeout=5)
            pool.submit(computer_use.release_computer_use_session, "session-a").result(timeout=5)
            acquired = pool.submit(computer_use._get_backend, "session-a").result(timeout=5)
            assert acquired is replacement
            assert not replacement.stopped
        finally:
            stale.release.set()
        with pytest.raises(RuntimeError, match="released"):
            original.result(timeout=5)
        assert stale.stopped
        assert computer_use._get_backend("session-a") is replacement
        assert computer_use.release_computer_use_session("session-a")
        assert replacement.stopped


@pytest.mark.parametrize("grant", ["approve_session", "always_approve"])
def test_approval_grants_and_release_are_profile_qualified(monkeypatch, tmp_path, grant):
    from tools.computer_use import tool as computer_use

    profile_a, profile_b = tmp_path / "profile-a", tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()
    session_id = "same-session"
    args = {"action": "click", "x": 1, "y": 1}
    prompts = []

    def approve(action, args, summary):
        prompts.append(action)
        return grant

    monkeypatch.setattr(computer_use, "_approval_callback", approve)
    monkeypatch.setenv("HERMES_HOME", str(profile_a))
    assert computer_use._request_approval("click", args, session_id) is None
    assert computer_use._request_approval("click", args, session_id) is None
    assert prompts == ["click"]

    monkeypatch.setenv("HERMES_HOME", str(profile_b))
    assert computer_use._request_approval("click", args, session_id) is None
    assert prompts == ["click", "click"]
    computer_use.release_computer_use_session(session_id)  # no backend is required to clear a grant
    assert computer_use._request_approval("click", args, session_id) is None
    assert prompts == ["click", "click", "click"]

    monkeypatch.setenv("HERMES_HOME", str(profile_a))
    assert computer_use._request_approval("click", args, session_id) is None
    assert prompts == ["click", "click", "click"]  # B's release preserved A's grant
    computer_use.release_computer_use_session(session_id)
    assert computer_use._request_approval("click", args, session_id) is None
    assert prompts == ["click", "click", "click", "click"]


def test_slow_start_for_one_owner_does_not_pin_unrelated_owner(monkeypatch):
    """Finding 3: while owner A's backend.start() is blocked (slow remote
    handshake), owner B must still be able to create, look up, and release
    a backend. Before the fix, start() ran inside _backend_lock and B would
    have blocked for the whole handshake."""
    from tools.computer_use import tool as computer_use

    created = []

    def _fake_new_backend(*args):
        # First create is the slow owner-A handshake; everything after is instant.
        backend = (
            _SlowStartBackend(_permission_mode(*args))
            if not created
            else _InstantBackend(_permission_mode(*args))
        )
        created.append(backend)
        return backend

    monkeypatch.setattr(
        "hermes_constants.get_hermes_home",
        lambda: Path("/home/fake/profile-a"),
    )
    monkeypatch.setattr(computer_use, "_new_backend", _fake_new_backend)

    pool = ThreadPoolExecutor(max_workers=3)
    try:
        future_a = pool.submit(computer_use._get_backend, "session-a")
        assert _SlowStartBackend.started.wait(timeout=5), "owner A's backend never started"

        # Create: B must finish while A is still inside start().
        future_b = pool.submit(computer_use._get_backend, "session-b")
        backend_b = future_b.result(timeout=2)
        assert isinstance(backend_b, _InstantBackend), "owner B blocked on owner A's slow start"
        assert backend_b.started_flag is True

        # Cached lookup of B must also complete without waiting on A.
        assert computer_use._get_backend("session-b") is backend_b

        # Release of B must complete without waiting on A.
        assert computer_use.release_computer_use_session("session-b") is True
        assert backend_b.stopped is True
        assert future_a.done() is False
    finally:
        _SlowStartBackend.release.set()
        pool.shutdown(wait=True)
