"""Unreadable WSL interop dirs must not crash unit generation (#128255).

Split out of ``test_gateway_service.py``, which is at the 2000-line target.
"""

from pathlib import Path

import pytest

pwd = pytest.importorskip("pwd")
grp = pytest.importorskip("grp")

import hermes_cli.gateway as gateway_cli

class TestWslInteropUnreadablePaths:
    """Unreadable WSL interop dirs (e.g. /mnt/c) must not crash unit generation (#128255)."""

    def test_unreadable_fixed_candidate_treated_as_absent(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "is_wsl", lambda: True)
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        monkeypatch.setattr(gateway_cli.shutil, "which", lambda cmd: None)

        def _raise_permission(self):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(Path, "exists", _raise_permission)

        assert gateway_cli._build_wsl_interop_paths([]) == []
