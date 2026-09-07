"""Regression tests for the profile-qualified backend cache and the
start()-outside-the-lock dispatcher (#104080 review findings 1 and 3).

Finding 1: backend caches keyed by bare session_id let profile B reuse
profile A's live backend under ``gateway.multiplex_profiles``. Keys are now
``hermes_home_key():session_id`` — a different profile home must get a
different backend for the same session id, and releasing one must not stop
the other.

Finding 3: ``backend.start()`` ran inside the process-global ``_backend_lock``,
so one slow remote handshake (up to ~35 s) pinned every other session's
lifecycle operations. start() now runs under a per-owner single-flight lock
outside the cache lock — an in-flight start for one owner must not block
another owner's create, lookup, or release.

Both tests use ``monkeypatch``-ed home keys (the same primitive
``hermes_constants.reset_hermes_home_key_cache`` clears) rather than moving
real directories on disk. ``_new_backend`` is patched via ``*args`` so the
same tests work on main (``permission_mode``) and the provider-seam branch
(``sid, permission_mode``).
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


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
    assert key_a == f"{hermes_home_key('/home/fake/profile-a')}:session-1"

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
