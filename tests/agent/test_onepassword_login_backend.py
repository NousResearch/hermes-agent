"""1Password login-backend unlock: tokenless desktop-app CLI integration vs fail-open.

``op signin --raw`` with desktop-app integration returns rc=0 and empty stdout
(no OP_SESSION token). That must unlock; a nonzero rc must not.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.vault_backends import unlock as unlock_mod
from agent.vault_backends.base import UnlockRequired
from agent.vault_backends.onepassword import OnePasswordLoginBackend


@pytest.fixture
def op_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    unlock_mod.lock()
    yield
    unlock_mod.lock()


def _backend() -> OnePasswordLoginBackend:
    return OnePasswordLoginBackend({"enabled": True, "binary_path": "/fake/op"})


def _proc(*, returncode: int, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _patch_op(signin_proc, run_cli_proc=None):
    """find_op + signin stdin helper + optional ``op`` run_cli (never a real binary)."""
    signin = patch(
        "agent.vault_backends.onepassword.run_with_stdin_secret",
        return_value=signin_proc,
    )
    find = patch(
        "agent.vault_backends.onepassword.find_op",
        return_value=Path("/fake/op"),
    )
    cli = patch(
        "agent.vault_backends.onepassword.run_cli",
        return_value=run_cli_proc or _proc(returncode=0, stdout="[]"),
    )
    return find, signin, cli


def test_unlock_tokenless_app_integration_succeeds(op_home):
    """Detection: rc=0 + empty stdout is an unlocked desktop-app session, not 'no session token'."""
    backend = _backend()
    find, signin, cli = _patch_op(_proc(returncode=0, stdout="", stderr=""))
    with find, signin as mock_signin, cli as mock_cli:
        backend.unlock("pw")
        assert mock_signin.call_args.kwargs["secret"] == "pw"
        argv = mock_signin.call_args.args[0]
        assert argv[1:] == ["signin", "--raw"]
        assert backend.is_unlocked()
        backend._run("item", "list", "--format", "json")
        child_env = mock_cli.call_args.kwargs["env"]
        assert "OP_SESSION" not in child_env
        assert not any(k.startswith("OP_SESSION_") for k in child_env)


def test_unlock_nonzero_returncode_still_fails(op_home):
    """CONTROL fail-open: rc=1 is still an unlock failure even with empty stdout."""
    backend = _backend()
    find, signin, cli = _patch_op(_proc(returncode=1, stdout="", stderr="invalid password"))
    with find, signin, cli:
        with pytest.raises(RuntimeError, match="invalid password"):
            backend.unlock("pw")
        assert not backend.is_unlocked()
        with pytest.raises(UnlockRequired):
            backend._run("item", "list")


def test_unlock_classic_token_sets_op_session(op_home):
    """CONTROL classic: rc=0 + token stdout stores that token and sets OP_SESSION."""
    backend = _backend()
    find, signin, cli = _patch_op(_proc(returncode=0, stdout="SESSION-TOKEN-XYZ\n"))
    with find, signin as mock_signin, cli as mock_cli:
        backend.unlock("pw")
        assert mock_signin.call_args.kwargs["secret"] == "pw"
        assert backend.is_unlocked()
        backend._run("item", "list", "--format", "json")
        child_env = mock_cli.call_args.kwargs["env"]
        assert child_env.get("OP_SESSION") == "SESSION-TOKEN-XYZ"
        assert backend._env("SESSION-TOKEN-XYZ").get("OP_SESSION") == "SESSION-TOKEN-XYZ"
        assert backend._env("").get("OP_SESSION") is None
