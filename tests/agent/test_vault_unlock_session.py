"""External-manager tokens belong to one context-bound session, never the process."""

import contextvars
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


@pytest.fixture
def manager(tmp_path, monkeypatch, request):
    from agent.vault_backends import bitwarden, onepassword, unlock
    from agent import secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home_token = set_hermes_home_override(tmp_path / "profile")
    monkeypatch.setattr(secret_scope, "get_secret", lambda _key, default="": default)
    backend = (bitwarden.BitwardenLoginBackend if request.param == "bitwarden" else onepassword.OnePasswordLoginBackend)(
        {"binary_path": "fake-manager"})
    monkeypatch.setattr(onepassword, "find_op", lambda _path: Path("fake-manager"))
    calls = []

    def run(argv, **kwargs):
        env = kwargs["env"]
        if "unlock" in argv or "signin" in argv:
            label = env.get("HERMES_BW_MASTER") or kwargs["input"].strip()
            return subprocess.CompletedProcess(argv, 0, f"fake-token-{label}", "")
        token = env.get("BW_SESSION") or env.get("OP_SESSION") or env.get("OP_SERVICE_ACCOUNT_TOKEN")
        calls.append(token)
        return subprocess.CompletedProcess(argv, 0, "fake-resolved-password", "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(unlock, "_released_sessions", set(), raising=False)
    unlock.lock_all_profiles()
    yield backend, calls, unlock
    unlock.lock_all_profiles()
    unlock.set_current_session_id(None)
    reset_hermes_home_override(home_token)


@pytest.mark.parametrize("manager", ["bitwarden", "onepassword"], indirect=True)
@pytest.mark.parametrize("binding", ["vault", "cli"])
def test_concurrent_sessions_keep_independent_tokens_and_propagate_only_the_owner(manager, binding, monkeypatch):
    from agent.vault_backends.base import UnlockRequired
    from agent.delegation_context import delegated_child_context
    from tools.approval_context import set_current_session_key
    from tools.thread_context import propagate_context_to_thread

    backend, calls, unlock = manager
    # copy_context retains the isolated profile, while independent contexts model concurrent turns.
    a, b = contextvars.copy_context(), contextvars.copy_context()
    bind = unlock.set_current_session_id if binding == "vault" else set_current_session_key
    a.run(bind, "session-A")
    b.run(bind, "session-B")
    a.run(backend.unlock, "A")
    assert not b.run(backend.is_unlocked)
    with pytest.raises(UnlockRequired):
        b.run(backend.resolve_password, backend.prefix + "fake-item")
    assert calls == [], "A sibling session cannot invoke the manager with the owner's token"
    b.run(backend.unlock, "B")
    assert a.run(backend.is_unlocked)
    assert b.run(backend.is_unlocked)

    worker = a.run(propagate_context_to_thread, lambda: backend.resolve_password(backend.prefix + "fake-item"))
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(worker).result(timeout=10) == "fake-resolved-password"
    b.run(backend.resolve_password, backend.prefix + "fake-item")
    assert calls == ["fake-token-A", "fake-token-B"]

    service_backend = None
    if backend.name == "onepassword":
        from agent import secret_scope
        from agent.vault_backends.onepassword import OnePasswordLoginBackend
        monkeypatch.setattr(secret_scope, "get_secret",
                            lambda key, default="": "fake-service-token" if key == "OP_SERVICE_ACCOUNT_TOKEN" else default)
        service_backend = OnePasswordLoginBackend({"binary_path": "fake-manager"})

    def child_probe():
        with delegated_child_context("child-session"):
            assert not backend.is_unlocked(), "A delegate must not borrow the parent session's token"
            # Delegates have no human-owned unlock surface or vault teardown hook. Do not mint
            # a second interactive token that could survive child completion in the parent process.
            with pytest.raises(RuntimeError, match="session"):
                backend.unlock("child")
            assert not unlock.store_session_token(backend.name, "fake-child-token")
            if service_backend is not None:
                assert service_backend.is_unlocked()
                assert service_backend.resolve_password("op:fake-item") == "fake-resolved-password"
                assert calls[-1] == "fake-service-token", "Configured service-account authority is independent"
    a.run(child_probe)
    assert a.run(backend.is_unlocked)
    unlock.release_session("session-A")
    assert not a.run(backend.is_unlocked)
    assert b.run(backend.is_unlocked)


@pytest.mark.parametrize("manager", ["bitwarden", "onepassword"], indirect=True)
@pytest.mark.parametrize("revoke", ["backend", "profile", "session", "shutdown"])
def test_pending_unlocks_and_missing_identity_fail_closed_without_revoking_other_profiles(manager, monkeypatch, tmp_path, revoke):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    backend, _calls, unlock = manager
    unlock.set_current_session_id("session-A")
    attempt = unlock.begin_unlock(backend.name)
    other = set_hermes_home_override(tmp_path / "other-profile")
    try:
        unlock.lock()
    finally:
        reset_hermes_home_override(other)
    assert unlock.store_session_token(backend.name, "fake-token-A", attempt)
    unlock.lock()
    attempt = unlock.begin_unlock(backend.name)
    if revoke == "backend":
        unlock.lock(backend.name)
    elif revoke == "profile":
        unlock.lock()
    elif revoke == "session":
        unlock.release_session("session-A")
    else:
        unlock.lock_all_profiles()
    assert not unlock.store_session_token(backend.name, "fake-late-token", attempt)
    assert not backend.is_unlocked()
    if revoke == "session":
        with pytest.raises(RuntimeError, match="session"):
            unlock.begin_unlock(backend.name)

    unlock.set_current_session_id("session-fresh")
    attempt = unlock.begin_unlock(backend.name)
    unlock.set_current_session_id("session-B")
    assert not unlock.store_session_token(backend.name, "fake-wrong-owner-token", attempt)
    unlock.set_current_session_id(None)
    monkeypatch.setenv("HERMES_SESSION_KEY", "session-A")
    monkeypatch.setenv("HERMES_SESSION_ID", "session-A")
    assert not backend.is_unlocked()
    assert not unlock.store_session_token(backend.name, "fake-unowned-token")
    with pytest.raises(RuntimeError, match="session"):
        unlock.begin_unlock(backend.name)

    # Status does not prolong an unlock, while a real manager call does.
    unlock.set_current_session_id("session-ttl")
    clock = [0.0]
    monkeypatch.setattr(unlock.time, "monotonic", lambda: clock[0])
    assert unlock.store_session_token(backend.name, "fake-idle-token")
    clock[0] = unlock._IDLE_TTL_S - 1
    assert backend.is_unlocked()
    clock[0] = unlock._IDLE_TTL_S + 1
    assert not backend.is_unlocked()
    assert unlock.store_session_token(backend.name, "fake-used-token")
    clock[0] += unlock._IDLE_TTL_S - 1
    backend.resolve_password(backend.prefix + "fake-item")
    clock[0] += 2
    assert backend.is_unlocked()
