"""Real subprocess contracts for manual and tokenless 1Password authorization.

The Python interpreter stands in for op; command-named scripts make the same
argv/stdin/environment boundary executable on Windows as well as POSIX.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from agent.secret_scope import is_multiplex_active, reset_secret_scope, set_multiplex_active, set_secret_scope
from agent.vault_backends import unlock
from agent.vault_backends.base import UnlockRequired
from agent.vault_backends.onepassword import OnePasswordLoginBackend
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


_FAKE_OP = r'''
import json, os, sys
from pathlib import Path
root = Path(__file__).parent
state = json.loads((root / "state.json").read_text())
command = Path(sys.argv[0]).name
stdin = sys.stdin.read()
with (root / "calls.jsonl").open("a") as stream:
    stream.write(json.dumps({"command": command, "args": sys.argv[1:],
                            "stdin": stdin, "account": os.environ.get("OP_ACCOUNT"),
                            "session": os.environ.get("OP_SESSION_example"),
                            "service": os.environ.get("OP_SERVICE_ACCOUNT_TOKEN")}) + "\n")
if command == "signin":
    if state.get("signin_failure"):
        sys.stderr.write("authorization denied")
        sys.exit(1)
    if state["mode"] == "manual":
        if stdin.strip() != "synthetic master":
            sys.exit(1)
        print("synthetic-session-token")
    sys.exit(0)
if command == "vault":
    if state.get("probe_failure"):
        sys.stderr.write("authorization denied")
        sys.exit(1)
    print("not json" if state.get("invalid_json") else "[]")
    sys.exit(0)
if state.get("expired"):
    sys.stderr.write("session expired")
    sys.exit(1)
if state.get("dismissed"):
    sys.stderr.write("authorization prompt dismissed, please try again")
    sys.exit(1)
if state["mode"] == "manual" and os.environ.get("OP_SESSION_example") != "synthetic-session-token":
    sys.exit(1)
if state["mode"] == "service" and os.environ.get("OP_SERVICE_ACCOUNT_TOKEN") != "synthetic-service-token":
    sys.exit(1)
if command == "item" and sys.argv[1] == "list":
    print(json.dumps([{"id": "demo", "title": "Example", "urls": [{"href": "https://example.com"}]}]))
elif command == "item" and sys.argv[1] == "get":
    print("synthetic-login-password")
else:
    sys.exit(2)
'''


@contextmanager
def _profile(path, secrets=None):
    home = set_hermes_home_override(path)
    scope = set_secret_scope(secrets or {}, profile_home=str(path))
    try:
        yield
    finally:
        reset_secret_scope(scope)
        reset_hermes_home_override(home)


@pytest.fixture
def fake_op(tmp_path, monkeypatch):
    for command in ("signin", "vault", "item"):
        (tmp_path / command).write_text(_FAKE_OP, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OP_CONFIG_DIR", str(tmp_path / "op-config"))
    was_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    unlock.lock_all_profiles()
    yield tmp_path
    unlock.lock_all_profiles()
    unlock.set_current_session_id(None)
    set_multiplex_active(was_multiplex)


def _backend():
    return OnePasswordLoginBackend({"binary_path": sys.executable, "account": "example"})


@pytest.mark.parametrize("mode", ["desktop", "manual", "service"])
def test_onepassword_authorization_uses_the_existing_profile_lease(fake_op, monkeypatch, mode):
    (fake_op / "state.json").write_text(json.dumps({"mode": mode}))
    secrets = {"OP_SERVICE_ACCOUNT_TOKEN": "synthetic-service-token"} if mode == "service" else {}
    with _profile(fake_op / "profile-a", secrets):
        backend = _backend()
        if mode == "service":
            assert backend.is_unlocked()
            assert backend.list_items()[0].id == "op:demo"
            assert backend.resolve_password("op:demo") == "synthetic-login-password"
            calls = [json.loads(line) for line in (fake_op / "calls.jsonl").read_text().splitlines()]
            assert all(call["command"] == "item" and call["account"] == "example" for call in calls)
            assert all(call["service"] == "synthetic-service-token" and call["session"] is None for call in calls)
            return
        assert not backend.is_unlocked()
        assert not (fake_op / "calls.jsonl").exists(), "status must not prompt in 1Password"
        unlock.set_current_session_id("owner-a")
        backend.unlock("" if mode == "desktop" else "synthetic master")
        assert backend.is_unlocked()
        # Backends are reconstructed for each RPC; the existing lease is the owner.
        assert _backend().list_items()[0].id == "op:demo"
        assert backend.resolve_password("op:demo") == "synthetic-login-password"
        with _profile(fake_op / "profile-b"):
            assert not _backend().is_unlocked()
            unlock.lock("onepassword")
        assert backend.is_unlocked()
        unlock.release_session("unrelated")
        assert backend.is_unlocked()
        unlock.release_session("owner-a")
        assert not backend.is_unlocked()
        with pytest.raises(UnlockRequired):
            backend.resolve_password("op:demo")
        backend.unlock("" if mode == "desktop" else "synthetic master")
        before = len((fake_op / "calls.jsonl").read_text().splitlines())
        unlock.lock("onepassword")
        assert backend.list_items() == []
        assert len((fake_op / "calls.jsonl").read_text().splitlines()) == before
        backend.unlock("" if mode == "desktop" else "synthetic master")
        now = unlock.time.monotonic()
        monkeypatch.setattr(unlock.time, "monotonic", lambda: now + unlock._IDLE_TTL_S + 1)
        assert not backend.is_unlocked()
    calls = [json.loads(line) for line in (fake_op / "calls.jsonl").read_text().splitlines()]
    assert all(call["account"] == "example" for call in calls)
    assert all("synthetic master" not in " ".join(call["args"]) for call in calls)
    assert all(call["stdin"] == "" for call in calls if call["command"] != "signin")
    if mode == "desktop":
        assert any(call["command"] == "vault" for call in calls)
        assert all(call["session"] is None and call["service"] is None for call in calls)
    else:
        assert all(call["session"] == "synthetic-session-token" for call in calls if call["command"] == "item")


@pytest.mark.parametrize("failure", ["signin_failure", "probe_failure", "invalid_json", "lock_race", "expired", "dismissed"])
def test_onepassword_unproven_or_revoked_app_authorization_stays_locked(fake_op, monkeypatch, failure):
    state = {"mode": "desktop", failure: failure not in {"lock_race", "expired", "dismissed"}}
    (fake_op / "state.json").write_text(json.dumps(state))
    with _profile(fake_op / "profile-a"):
        backend = _backend()
        if failure == "lock_race":
            from agent.vault_backends import onepassword
            run_cli = onepassword.run_cli

            def race(*args, **kwargs):
                proc = run_cli(*args, **kwargs)
                unlock.lock("onepassword")
                return proc

            monkeypatch.setattr(onepassword, "run_cli", race)
        if failure in {"expired", "dismissed"}:
            backend.unlock("")
            state[failure] = True
            (fake_op / "state.json").write_text(json.dumps(state))
            with pytest.raises(UnlockRequired):
                backend.list_items()
        else:
            with pytest.raises(RuntimeError):
                backend.unlock("")
        assert not backend.is_unlocked()
        with pytest.raises(UnlockRequired):
            backend.resolve_password("op:demo")
    calls = [json.loads(line) for line in (fake_op / "calls.jsonl").read_text().splitlines()]
    assert all(call["args"][:1] != ["get"] for call in calls), "no login secrets may be read to prove authorization"
