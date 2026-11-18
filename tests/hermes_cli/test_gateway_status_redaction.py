"""`hermes gateway status` must not print child-process argv credentials (#143938).

systemctl's own CGroup tree lists every child's full command line, so the status path runs a
real stand-in ``systemctl`` process and asserts on what reaches the terminal.
"""
from __future__ import annotations

import secrets
import sys

import pytest

from hermes_cli import gateway

pytestmark = pytest.mark.platforms("linux")

_FAKE_SYSTEMCTL = """\
import os, sys
if "status" in sys.argv[1:]:
    print("● hermes-gateway.service - Hermes Agent Gateway")
    print("     CGroup: /user.slice/user-1000.slice/user@1000.service/app.slice/hermes-gateway.service")
    print(os.environ["FAKE_TREE_LINE"])
    print(os.environ["FAKE_TREE_LINE"], file=sys.stderr)
elif "is-active" in sys.argv[1:]:
    print("active")
"""


@pytest.fixture
def fake_systemctl(tmp_path, monkeypatch):
    script = tmp_path / "fake_systemctl.py"
    script.write_text(_FAKE_SYSTEMCTL, encoding="utf-8")
    unit = tmp_path / "hermes-gateway.service"
    unit.write_text("[Service]\n", encoding="utf-8")
    monkeypatch.setattr(gateway, "_systemctl_cmd", lambda system=False: [sys.executable, str(script)])
    monkeypatch.setattr(gateway, "_select_systemd_scope", lambda system=False: system)
    monkeypatch.setattr(gateway, "get_systemd_unit_path", lambda system=False: unit)
    monkeypatch.setattr(gateway, "has_conflicting_systemd_units", lambda: False)
    monkeypatch.setattr(gateway, "has_legacy_hermes_units", lambda: False)
    monkeypatch.setattr(gateway, "systemd_unit_is_current", lambda system=False: True)
    monkeypatch.setattr(gateway, "_print_runtime_health", lambda: None)
    monkeypatch.setattr(gateway, "_read_systemd_unit_properties", lambda system=False: {})
    monkeypatch.setattr(gateway, "get_systemd_linger_status", lambda username=None: (True, ""))
    return monkeypatch


def test_status_process_tree_masks_mcp_credential_argv(fake_systemctl, capfd):
    credential = secrets.token_hex(16)
    fake_systemctl.setenv("FAKE_TREE_LINE", f"  └─4242 node /opt/mcp/server.js --api-key {credential}")

    gateway.systemd_status()

    captured = capfd.readouterr()
    assert credential not in captured.out + captured.err
    assert "└─4242 node /opt/mcp/server.js --api-key" in captured.out
    assert "--api-key" in captured.err
    assert "gateway service is running" in captured.out
    assert "user@1000.service/app.slice" in captured.out  # a unit instance is not an email
