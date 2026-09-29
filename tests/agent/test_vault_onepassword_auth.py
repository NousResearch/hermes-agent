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
                            "service": os.environ.get("OP_SERVICE_ACCOUNT_TOKEN"),
                            "connect_host": os.environ.get("OP_CONNECT_HOST"),
                            "connect_token": os.environ.get("OP_CONNECT_TOKEN"),
                            "desktop_settings": os.environ.get("OP_LOAD_DESKTOP_APP_SETTINGS")}) + "\n")
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
if state.get("error"):
    sys.stderr.write("authorization rejected: " + os.environ.get("OP_CONNECT_TOKEN", ""))
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
            assert backend.auth_capabilities() == {
                "mode": "service_account", "methods": [],
                "native_app_eligible": False, "reason": "service_account_configured",
            }
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
        assert all(call["desktop_settings"] == "true" for call in calls if call["command"] in {"signin", "vault", "item"})
        assert all("--vault" not in call["args"] for call in calls if call["command"] == "item")
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


def test_complete_connect_pair_authenticates_without_manual_unlock(fake_op):
    (fake_op / "state.json").write_text(json.dumps({"mode": "connect"}))
    with _profile(fake_op / "profile-a", {
        "OP_CONNECT_HOST": "https://connect.invalid",
        "OP_CONNECT_TOKEN": "synthetic-connect-token",
    }):
        backend = _backend()
        assert backend.is_unlocked()
        assert backend.auth_capabilities() == {
            "mode": "connect", "methods": [], "native_app_eligible": False, "reason": None,
        }
        assert backend.list_items()[0].id == "op:demo"
        assert backend.resolve_password("op:demo") == "synthetic-login-password"
    calls = [json.loads(line) for line in (fake_op / "calls.jsonl").read_text().splitlines()]
    assert all(call["connect_host"] == "https://connect.invalid" for call in calls)
    assert all(call["connect_token"] == "synthetic-connect-token" for call in calls)
    assert all(call["service"] is None and call["session"] is None for call in calls)
    assert all(call["command"] != "signin" for call in calls)


def test_partial_connect_pair_fails_closed_ahead_of_service_account(fake_op):
    (fake_op / "state.json").write_text(json.dumps({"mode": "service"}))
    with _profile(fake_op / "profile-a", {
        "OP_CONNECT_HOST": "https://connect.invalid",
        "OP_SERVICE_ACCOUNT_TOKEN": "synthetic-service-token",
    }):
        backend = _backend()
        assert not backend.is_unlocked()
        assert backend.auth_capabilities() == {
            "mode": "unavailable", "methods": [],
            "native_app_eligible": False, "reason": "connect_incomplete",
        }
        with pytest.raises(UnlockRequired):
            backend.resolve_password("op:demo")
    assert not (fake_op / "calls.jsonl").exists(), "partial Connect must not fall through to another auth mode"


def test_connect_pair_is_resolved_for_each_active_profile(fake_op):
    (fake_op / "state.json").write_text(json.dumps({"mode": "connect"}))
    with _profile(fake_op / "profile-a", {
        "OP_CONNECT_HOST": "https://a.invalid", "OP_CONNECT_TOKEN": "token-a",
    }):
        backend = _backend()
        assert backend.is_unlocked()
        assert backend.resolve_password("op:demo") == "synthetic-login-password"
        with _profile(fake_op / "profile-b", {
            "OP_CONNECT_HOST": "https://b.invalid", "OP_CONNECT_TOKEN": "token-b",
        }):
            assert backend.is_unlocked()
            assert backend.resolve_password("op:demo") == "synthetic-login-password"
        assert backend.is_unlocked()
        assert backend.resolve_password("op:demo") == "synthetic-login-password"
    calls = [json.loads(line) for line in (fake_op / "calls.jsonl").read_text().splitlines()]
    assert [call["connect_host"] for call in calls] == [
        "https://a.invalid", "https://b.invalid", "https://a.invalid",
    ]
    assert [call["connect_token"] for call in calls] == ["token-a", "token-b", "token-a"]


def test_connect_auth_errors_redact_the_profile_token(fake_op):
    (fake_op / "state.json").write_text(json.dumps({"mode": "connect", "error": True}))
    with _profile(fake_op / "profile-a", {
        "OP_CONNECT_HOST": "https://connect.invalid",
        "OP_CONNECT_TOKEN": "synthetic-connect-token",
    }):
        backend = _backend()
        with pytest.raises(RuntimeError) as exc_info:
            backend.resolve_password("op:demo")
    assert "synthetic-connect-token" not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_vault_selector_is_used_for_list_password_and_otp(fake_op):
    (fake_op / "state.json").write_text(json.dumps({"mode": "service"}))
    with _profile(fake_op / "profile-a", {"OP_SERVICE_ACCOUNT_TOKEN": "synthetic-service-token"}):
        backend = OnePasswordLoginBackend({
            "binary_path": sys.executable, "account": "example", "vault": "Work Vault",
        })
        assert backend.list_items()[0].id == "op:demo"
        assert backend.resolve_password("op:demo") == "synthetic-login-password"
        assert backend.resolve_otp("op:demo") is None
    calls = [json.loads(line) for line in (fake_op / "calls.jsonl").read_text().splitlines()]
    assert [call["args"] for call in calls] == [
        ["list", "--categories", "Login", "--vault", "Work Vault", "--format", "json"],
        ["get", "demo", "--vault", "Work Vault", "--fields", "label=password", "--reveal"],
        ["get", "demo", "--vault", "Work Vault", "--otp"],
    ]


@pytest.mark.parametrize(
    ("host_os", "interactive", "cli_present", "expected"),
    [("win32", True, True, True), ("win32", False, True, False),
     ("darwin", True, True, True), ("linux", True, True, True),
     ("linux", False, True, False), ("win32", True, False, False)],
)
def test_native_app_eligibility_uses_backend_host_and_resolved_cli(
    fake_op, monkeypatch, host_os, interactive, cli_present, expected,
):
    from hermes_platform.host import facts

    monkeypatch.setattr(facts, "os_family", lambda: host_os)
    monkeypatch.setattr(facts, "interactive_session", lambda: interactive)
    binary = sys.executable if cli_present else str(fake_op / "missing-op")
    backend = OnePasswordLoginBackend({"binary_path": binary})
    with _profile(fake_op / "profile-a"):
        capability = backend.auth_capabilities()
    assert capability["native_app_eligible"] is expected
    assert backend.supports_app_unlock is expected
    assert capability["mode"] == ("interactive" if interactive else "unavailable")


def test_explicit_password_unlock_skips_desktop_app_probe(fake_op, monkeypatch):
    (fake_op / "state.json").write_text(json.dumps({"mode": "manual"}))
    from hermes_platform.host import facts

    monkeypatch.setattr(facts, "os_family", lambda: "linux")
    monkeypatch.setattr(facts, "interactive_session", lambda: True)
    with _profile(fake_op / "profile-a"):
        backend = _backend()
        backend.unlock("synthetic master", method="password")
        assert backend.resolve_password("op:demo") == "synthetic-login-password"
    calls = [json.loads(line) for line in (fake_op / "calls.jsonl").read_text().splitlines()]
    assert calls[0]["desktop_settings"] == "false"
    assert calls[0]["stdin"] == "synthetic master\n"


def test_explicit_password_with_empty_token_never_probes_native_app(fake_op, monkeypatch):
    (fake_op / "state.json").write_text(json.dumps({"mode": "desktop"}))
    from hermes_platform.host import facts

    monkeypatch.setattr(facts, "os_family", lambda: "linux")
    monkeypatch.setattr(facts, "interactive_session", lambda: True)
    with _profile(fake_op / "profile-a"):
        backend = _backend()
        with pytest.raises(RuntimeError, match="did not return a session token"):
            backend.unlock("synthetic master", method="password")
        assert not backend.is_unlocked()
    calls = [json.loads(line) for line in (fake_op / "calls.jsonl").read_text().splitlines()]
    assert [call["command"] for call in calls] == ["signin"]
    assert calls[0]["desktop_settings"] == "false"
    assert calls[0]["stdin"] == "synthetic master\n"


def test_explicit_app_unlock_does_not_fall_back_on_ineligible_host(fake_op, monkeypatch):
    (fake_op / "state.json").write_text(json.dumps({"mode": "desktop"}))
    from hermes_platform.host import facts

    monkeypatch.setattr(facts, "os_family", lambda: "linux")
    monkeypatch.setattr(facts, "interactive_session", lambda: False)
    with _profile(fake_op / "profile-a"):
        with pytest.raises(RuntimeError, match="unavailable on this backend host"):
            _backend().unlock(method="app")
    assert not (fake_op / "calls.jsonl").exists()
