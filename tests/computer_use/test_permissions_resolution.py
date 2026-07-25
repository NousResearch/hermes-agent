"""Regression tests for Computer Use readiness under a thin GUI PATH."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX user-local path regression")
def test_status_finds_user_local_driver_when_path_omits_it(tmp_path, monkeypatch):
    """Desktop status must agree with the runtime resolver, not bare PATH."""
    from tools.computer_use import permissions

    driver = tmp_path / ".local" / "bin" / "cua-driver"
    driver.parent.mkdir(parents=True)
    driver.write_text("#!/bin/sh\nexit 0\n")
    driver.chmod(0o755)

    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")

    with patch("tools.computer_use.permissions.sys.platform", "darwin"), \
         patch("tools.computer_use.cua_backend.sys.platform", "darwin"), \
         patch.object(permissions, "_run", return_value=MagicMock(stdout="0.0.0")), \
         patch.object(permissions, "_doctor", return_value={"ok": True, "checks": []}), \
         patch.object(permissions, "_mac_permissions"):
        status = permissions.computer_use_status()

    assert status["installed"] is True
