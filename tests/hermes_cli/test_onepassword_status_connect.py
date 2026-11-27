"""Regression tests for the 1Password status probe's Connect awareness (#95866).

``hermes secrets onepassword status`` treated ``OP_SERVICE_ACCOUNT_TOKEN`` or
a successful ``op whoami`` as the only valid authentication, but the backend
(``agent/secret_sources/onepassword.py``) also passes
``OP_CONNECT_HOST``/``OP_CONNECT_TOKEN`` through to the op child and resolves
references via ``op read`` — so on a Connect-authenticated install the status
command falsely warned "Hermes will warn and skip 1Password on next startup"
while startup was in fact succeeding through the same binary.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _run_status(monkeypatch, capsys, *, connect_env, whoami):
    monkeypatch.delenv("OP_CONNECT_HOST", raising=False)
    monkeypatch.delenv("OP_CONNECT_TOKEN", raising=False)
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    if connect_env:
        monkeypatch.setenv("OP_CONNECT_HOST", "https://connect.example")
        monkeypatch.setenv("OP_CONNECT_TOKEN", "connect-token-not-real")

    cfg = {
        "secrets": {
            "onepassword": {
                "enabled": True,
                "env": {"MY_SECRET": "op://Vault/Item/field"},
            }
        }
    }
    monkeypatch.setattr(
        "hermes_cli.onepassword_secrets_cli.load_config", lambda: cfg
    )
    monkeypatch.setattr(
        "hermes_cli.onepassword_secrets_cli.op_src.find_op",
        lambda binary_path: "/fake/op",
    )
    monkeypatch.setattr(
        "hermes_cli.onepassword_secrets_cli._op_version",
        lambda binary: "2.34.0",
    )
    monkeypatch.setattr(
        "hermes_cli.onepassword_secrets_cli._op_whoami",
        lambda binary, account, token_value=None: whoami,
    )

    from hermes_cli.onepassword_secrets_cli import cmd_status

    rc = cmd_status(SimpleNamespace())
    out = capsys.readouterr().out
    return rc, " ".join(out.split())


class TestStatusRecognizesConnect:
    def test_connect_env_suppresses_false_skip_warning(self, monkeypatch, capsys):
        """Connect credentials authenticate the op-read path; the skip
        warning must not fire even though whoami cannot verify Connect."""
        rc, out = _run_status(
            monkeypatch, capsys, connect_env=True, whoami=None
        )
        assert rc == 0
        assert "will warn and skip 1Password" not in out
        assert "1Password Connect credentials detected" in out

    def test_no_auth_still_warns(self, monkeypatch, capsys):
        """Without any credentials the original warning stands."""
        rc, out = _run_status(
            monkeypatch, capsys, connect_env=False, whoami=None
        )
        assert rc == 0
        assert "will warn and skip 1Password" in out

    def test_active_session_takes_precedence(self, monkeypatch, capsys):
        """A verified interactive session still reports the session line."""
        rc, out = _run_status(
            monkeypatch, capsys, connect_env=True, whoami="me@example"
        )
        assert rc == 0
        assert "Active op session" in out
        assert "1Password Connect credentials detected" not in out
