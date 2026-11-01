"""1Password backend: desktop-app integration unlock mode.

Regression contract: with "Connect with 1Password CLI" enabled, `op signin --raw`
delegates authentication to the unlocked desktop app and exits 0 with EMPTY stdout —
no classic session token exists. The unlock must recognize that mode (verify the app
answers, store the sentinel session, run calls without OP_SESSION) instead of failing
with "no session token", while the classic token flow and the unreachable-app error
stay byte-for-byte what they were.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.vault_backends import unlock as unlock_mod
from agent.vault_backends.onepassword import OnePasswordLoginBackend, _DESKTOP_SESSION


def _result(returncode: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture
def backend(monkeypatch):
    unlock_mod.lock("onepassword")
    backend = OnePasswordLoginBackend({"binary_path": "op"})
    # the fakes below replace both subprocess seams, so the locator never needs a real binary
    monkeypatch.setattr(backend, "_op", lambda: Path("op"))
    yield backend
    unlock_mod.lock("onepassword")


def _fake_op(monkeypatch, signin: SimpleNamespace, other: SimpleNamespace):
    monkeypatch.setattr("agent.vault_backends.onepassword.run_with_stdin_secret", lambda *a, **k: signin)
    monkeypatch.setattr("agent.vault_backends.onepassword.run_cli", lambda *a, **k: other)


def test_desktop_integration_unlock_stores_sentinel_and_runs_without_op_session(backend, monkeypatch):
    # signin exits 0 with empty stdout: the app integration answered instead of minting a token.
    calls: dict[str, dict] = {}

    def fake_run_cli(cmd, *, env, timeout, label, timeout_message, stdin):
        calls["probe"] = dict(timeout_message=timeout_message)
        return _result(0, "account list\n")

    monkeypatch.setattr("agent.vault_backends.onepassword.run_with_stdin_secret", lambda *a, **k: _result(0, "", ""))
    monkeypatch.setattr("agent.vault_backends.onepassword.run_cli", fake_run_cli)
    backend.unlock("never-seen")
    assert unlock_mod.is_unlocked("onepassword")
    token = unlock_mod.get_session_token("onepassword")
    assert token == _DESKTOP_SESSION
    # the sentinel must never reach op as OP_SESSION — the app holds the session, not a token
    assert "OP_SESSION" not in backend._env(token)
    # the liveness probe passes the required timeout_message (signature regression guard)
    assert calls["probe"]["timeout_message"] == "op timed out"


def test_desktop_integration_unreachable_still_reports_no_session_token(backend, monkeypatch):
    # signin exits 0 empty, but the app does not answer `account list`: nothing was unlocked.
    _fake_op(monkeypatch, signin=_result(0, "", ""), other=_result(1, "", "cannot reach the desktop app"))
    with pytest.raises(RuntimeError, match="no session token"):
        backend.unlock("never-seen")
    assert not unlock_mod.is_unlocked("onepassword")


def test_classic_signin_token_flow_unchanged(backend, monkeypatch):
    _fake_op(monkeypatch, signin=_result(0, "TOKEN-123\n", ""), other=_result(1))
    backend.unlock("never-seen")
    token = unlock_mod.get_session_token("onepassword")
    assert token == "TOKEN-123"
    assert backend._env(token)["OP_SESSION"] == "TOKEN-123"
