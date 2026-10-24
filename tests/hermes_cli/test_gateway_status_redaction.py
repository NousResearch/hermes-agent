"""Regression tests for `hermes gateway status` process-tree redaction (#143938).

`systemd_status()` used to stream `systemctl status` straight to the terminal
(``capture_output=False``). systemctl's own child-process tree prints each child's
complete command line, so an MCP server launched with a credential argument
(``--api-key <live-key>``) leaked into shared status output — bug reports,
screenshots, support logs. The output is now captured and passed through
``redact_debug_support_text()`` (the argv/header masking from #55576, which also
fails closed) before printing.
"""

from __future__ import annotations

import subprocess

from unittest.mock import patch

import pytest

import hermes_cli.gateway as gateway_module
from hermes_cli.gateway import systemd_status

_LIVE_KEY = "8f3a9c71d0e4b6a2f5c89017e4d2b3a6"
_TREE_OUTPUT = (
    "● hermes-gateway.service - Hermes Gateway\n"
    "     Loaded: loaded (/home/op/.config/systemd/user/hermes-gateway.service; enabled)\n"
    "     Active: active (running) since Fri 2026-10-23 09:00:00 UTC; 1 day ago\n"
    "   CGroup: /user.slice/user-1000.slice/user@1000.service/hermes-gateway.service\n"
    f"           └─4242 node /opt/mcp/server.js --email ops@example.com --api-key {_LIVE_KEY}\n"
)


def _fake_run_systemctl(responses):
    def _run(args, *, system=False, **kwargs):
        head = args[0] if args else ""
        if head == "status":
            return responses.get("status", subprocess.CompletedProcess(args, 0, stdout="", stderr=""))
        return responses.get(head, subprocess.CompletedProcess(args, 0, stdout="", stderr=""))

    return _run


@pytest.fixture()
def _status_env(monkeypatch, tmp_path):
    """Stub every systemd precondition systemd_status() checks before the status call."""
    unit_path = tmp_path / "hermes-gateway.service"
    unit_path.write_text("[Service]\n", encoding="utf-8")
    monkeypatch.setattr(gateway_module, "_select_systemd_scope", lambda system=False: False)
    monkeypatch.setattr(gateway_module, "get_systemd_unit_path", lambda system=False: unit_path)
    monkeypatch.setattr(gateway_module, "get_service_name", lambda: "hermes-gateway")
    monkeypatch.setattr(gateway_module, "has_conflicting_systemd_units", lambda: False)
    monkeypatch.setattr(gateway_module, "has_legacy_hermes_units", lambda: False)
    monkeypatch.setattr(gateway_module, "systemd_unit_is_current", lambda system=False: True)
    monkeypatch.setattr(gateway_module, "_systemd_cli_bits", lambda system=False: ("sudo ", " --system", " --user"))
    monkeypatch.setattr(gateway_module, "_read_systemd_user_from_unit", lambda path: None)


def test_status_tree_prints_redacted_credentials(capsys, _status_env):
    """The child-process tree from systemctl status is scrubbed before display (#143938)."""
    with patch.object(
        gateway_module,
        "_run_systemctl",
        _fake_run_systemctl({"status": subprocess.CompletedProcess([], 0, stdout=_TREE_OUTPUT, stderr=""),
                             "is-active": subprocess.CompletedProcess([], 0, stdout="active", stderr="")}),
    ):
        systemd_status()

    out = capsys.readouterr().out
    assert "--api-key [REDACTED]" in out, out
    assert _LIVE_KEY not in out, out
    assert "ops@example.com" not in out, out  # the email travels with the same tree line
    # The non-secret parts of the tree survive.
    assert "└─4242 node /opt/mcp/server.js" in out, out


def test_status_stderr_is_redacted_too(capsys, _status_env):
    stderr_tree = f"Warning: unit spawned └─99 node /srv/x.js --token {_LIVE_KEY}\n"
    with patch.object(
        gateway_module,
        "_run_systemctl",
        _fake_run_systemctl({"status": subprocess.CompletedProcess([], 0, stdout="", stderr=stderr_tree),
                             "is-active": subprocess.CompletedProcess([], 0, stdout="active", stderr="")}),
    ):
        systemd_status()

    err = capsys.readouterr().err
    assert _LIVE_KEY not in err, err
    assert "--token [REDACTED]" in err, err


def test_status_without_output_prints_nothing_extra(capsys, _status_env):
    """Empty stdout/stderr (e.g. unit missing) must not print blank redaction sentinels."""
    with patch.object(
        gateway_module,
        "_run_systemctl",
        _fake_run_systemctl({"is-active": subprocess.CompletedProcess([], 0, stdout="inactive", stderr="")}),
    ):
        systemd_status()

    out = capsys.readouterr().out
    assert "[REDACTED]" not in out, out
