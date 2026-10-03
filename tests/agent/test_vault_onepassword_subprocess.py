"""Exercise the login backend through run_cli and an offline executable, not mocks."""
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
import sys

import pytest

from agent import secret_scope
from agent.vault_backends.base import UnlockRequired
from agent.vault_backends.onepassword import OnePasswordLoginBackend
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def fake_op(tmp_path, monkeypatch, caplog):
    # Pin the executable and all home paths: never discover a developer's real op.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in list(os.environ):
        if key.startswith("OP_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "dummy-launch-token")
    monkeypatch.setenv("OP_CONNECT_TOKEN", "dummy-launch-connect-token")
    monkeypatch.setenv("OP_CONNECT_HOST", "https://invalid.example")
    monkeypatch.setenv("OP_SESSION_other", "dummy-launch-session")
    monkeypatch.setenv("UNRELATED_API_KEY", "dummy-unrelated-key")
    profiles = {}
    for name in ("a", "b", "empty"):
        home = tmp_path / f"profile-{name}"
        home.mkdir()
        home.joinpath(".env").write_text(
            "" if name == "empty" else f"OP_SERVICE_ACCOUNT_TOKEN=dummy-profile-{name}-token\n"
        )
        profiles[name] = home
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"mode": "ok", "vault": "vault-b"}))
    audit = tmp_path / "audit.jsonl"
    executable = tmp_path / "op"
    executable.write_text(f"#!{sys.executable}\n" + r'''
import json
import os
from pathlib import Path
import sys

root = Path(__file__).parent
state = json.loads((root / "state.json").read_text())
args = sys.argv[1:]
token = os.environ.get("OP_SERVICE_ACCOUNT_TOKEN")
identity = next((name for name in ("a", "b")
                 if token == "dummy-profile-" + name + "-token"), None)
record = {"argv": args, "identity": identity, "stdin_eof": sys.stdin.read() == "",
          "unexpected_env": sorted(k for k in os.environ
              if k.startswith("OP_CONNECT_") or k.startswith("OP_SESSION")
              or k == "UNRELATED_API_KEY")}
with (root / "audit.jsonl").open("a") as stream:
    stream.write(json.dumps(record) + "\n")
if identity is None or record["unexpected_env"]:
    print("authentication rejected", file=sys.stderr)
    sys.exit(1)
if args == ["item", "list", "--categories", "Login", "--format", "json"]:
    records = [{"id": "item-a", "title": "First", "vault": {"id": "vault-a"},
                "urls": [{"href": "https://first.example/login"}]},
               {"id": "item-b", "title": "Second", "vault": {"id": state["vault"]},
                "urls": [{"href": "https://second.example/login"}]}]
    if state["mode"] == "ambiguous":
        records.append(dict(records[1], vault={"id": "wrong-vault"}))
    print(json.dumps(records))
elif args[:3] == ["item", "get", "item-b"] and args[3:5] == ["--vault", state["vault"]]:
    if state["mode"] == "error":
        # Secret-bearing stdout from a failed command must not become diagnostics.
        print(token + " dummy-password")
        print("synthetic read failure", file=sys.stderr)
        sys.exit(1)
    if args[5:] == ["--fields", "label=password", "--reveal"]:
        print("dummy-password")
    elif args[5:] == ["--otp"]:
        print("123456")
    else:
        sys.exit(2)
else:
    print("unexpected selector", file=sys.stderr)
    sys.exit(2)
''')
    executable.chmod(0o700)
    caplog.set_level(logging.DEBUG)
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)

    @contextmanager
    def profile(name):
        home_token = set_hermes_home_override(profiles[name])
        scope_token = secret_scope.set_secret_scope(
            secret_scope.load_env_file(profiles[name] / ".env")
        )
        try:
            yield OnePasswordLoginBackend({"binary_path": str(executable)})
        finally:
            secret_scope.reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    def calls():
        return [json.loads(line) for line in audit.read_text().splitlines()] if audit.exists() else []

    try:
        yield profile, calls, state
    finally:
        secret_scope.set_multiplex_active(previous)


def assert_no_secret_diagnostics(capfd, caplog, *diagnostics):
    captured = capfd.readouterr()
    text = captured.out + captured.err + caplog.text + repr(diagnostics)
    for secret in ("dummy-profile-a-token", "dummy-profile-b-token", "dummy-launch-token",
                   "dummy-launch-connect-token", "dummy-launch-session", "dummy-password"):
        assert secret not in text


@pytest.mark.parametrize("fake_op", [
    pytest.param(None, marks=pytest.mark.linux_only, id="linux"),
    pytest.param(None, marks=pytest.mark.macos_only, id="macos"),
], indirect=True)
def test_real_subprocess_selects_fresh_vault_and_keeps_secrets_private(fake_op, capfd, caplog):
    profile, calls, state = fake_op
    with profile("a") as backend:
        metadata = backend.list_items()
        assert [item.id for item in metadata] == ["op:item-a", "op:item-b"]
        assert backend.get_meta("op:item-b").origin == "https://second.example"
        assert backend.resolve_password("op:item-b") == "dummy-password"
        # An already-issued handle must use newly listed metadata, not a cached vault.
        state.write_text(json.dumps({"mode": "ok", "vault": "vault-moved"}))
        assert backend.resolve_otp("op:item-b") == "123456"
        state.write_text(json.dumps({"mode": "error", "vault": "vault-moved"}))
        with pytest.raises(RuntimeError, match="synthetic read failure") as error:
            backend.resolve_password("op:item-b")
        assert backend.resolve_otp("op:item-b") is None
        before = len(calls())
        state.write_text(json.dumps({"mode": "ambiguous", "vault": "vault-moved"}))
        with pytest.raises(RuntimeError, match="missing or ambiguous") as ambiguous:
            backend.resolve_password("op:item-b")
        assert backend.resolve_otp("op:item-b") is None
        assert all(row["argv"][1] == "list" for row in calls()[before:])
    rows = calls()
    gets = [row["argv"] for row in rows if row["argv"][1] == "get"]
    assert gets == [
        ["item", "get", "item-b", "--vault", "vault-b", "--fields", "label=password", "--reveal"],
        ["item", "get", "item-b", "--vault", "vault-moved", "--otp"],
        ["item", "get", "item-b", "--vault", "vault-moved", "--fields", "label=password", "--reveal"],
        ["item", "get", "item-b", "--vault", "vault-moved", "--otp"],
    ]
    assert all(row["identity"] == "a" and row["stdin_eof"] and not row["unexpected_env"] for row in rows)
    assert_no_secret_diagnostics(capfd, caplog, metadata, str(error.value), str(ambiguous.value), rows)


@pytest.mark.parametrize("fake_op", [
    pytest.param(None, marks=pytest.mark.linux_only, id="linux"),
    pytest.param(None, marks=pytest.mark.macos_only, id="macos"),
], indirect=True)
def test_real_subprocess_auth_is_scoped_and_empty_profile_fails_closed(fake_op, capfd, caplog):
    profile, calls, _ = fake_op
    for name in ("a", "b", "a"):
        with profile(name) as backend:
            assert backend.resolve_password("op:item-b") == "dummy-password"
        assert [row["identity"] for row in calls()[-2:]] == [name, name]
    before = calls()
    with profile("empty") as backend:
        assert not backend.is_unlocked()
        assert backend.list_items() == []
        with pytest.raises(UnlockRequired) as error:
            backend.resolve_password("op:item-b")
        assert backend.resolve_otp("op:item-b") is None
    assert calls() == before  # No fallback to launch credentials or another profile.
    token = secret_scope.set_secret_scope(None)
    try:
        with pytest.raises(secret_scope.UnscopedSecretError) as unscoped:
            OnePasswordLoginBackend()
    finally:
        secret_scope.reset_secret_scope(token)
    assert calls() == before
    assert all(row["stdin_eof"] and not row["unexpected_env"] for row in before)
    assert_no_secret_diagnostics(capfd, caplog, str(error.value), str(unscoped.value), before)
