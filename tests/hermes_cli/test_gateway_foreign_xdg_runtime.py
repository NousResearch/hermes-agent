"""Regression tests for a foreign/leaked ``XDG_RUNTIME_DIR`` in the user-systemd
preflight (#86558).

``runuser``/``su``/``sudo -u`` from a root shell leaks ``XDG_RUNTIME_DIR=/run/user/0``
into the child. The user-systemd preflight then stat-ed sockets under that
``0700 root:root`` directory with a bare ``Path.exists()``, which only suppresses
a subset of ``OSError`` (ENOENT/ENOTDIR/EBADF/ELOOP) — ``EACCES`` escaped as a raw
``PermissionError`` traceback instead of the documented
``UserSystemdUnavailableError`` remediation path.
"""

import os
from pathlib import Path

import pytest

import hermes_cli.gateway as gateway_cli


def _eacces(self):
    raise PermissionError(13, "Permission denied", str(self))


class TestPathExistsSafe:
    """_path_exists_safe() swallows the EACCES that Path.exists() re-raises."""

    def test_returns_false_on_permission_error(self, monkeypatch):
        monkeypatch.setattr(Path, "exists", _eacces)
        # A foreign /run/user/0/bus is unreadable, not "reachable".
        assert gateway_cli._path_exists_safe(Path("/run/user/0/bus")) is False

    def test_returns_true_when_present(self, monkeypatch):
        monkeypatch.setattr(Path, "exists", lambda self: True)
        assert gateway_cli._path_exists_safe(Path("/run/user/1001/bus")) is True

    def test_returns_false_when_absent(self, monkeypatch):
        monkeypatch.setattr(Path, "exists", lambda self: False)
        assert gateway_cli._path_exists_safe(Path("/run/user/1001/bus")) is False


class TestRuntimeDirIsOurs:
    """_runtime_dir_is_ours() separates our runtime dir from a leaked foreign one."""

    def test_true_when_owned_by_current_uid(self, tmp_path, monkeypatch):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        monkeypatch.setattr(os, "getuid", lambda: runtime.stat().st_uid)
        assert gateway_cli._runtime_dir_is_ours(str(runtime)) is True

    def test_false_when_owned_by_other_uid(self, tmp_path, monkeypatch):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        monkeypatch.setattr(os, "getuid", lambda: runtime.stat().st_uid + 1)
        assert gateway_cli._runtime_dir_is_ours(str(runtime)) is False

    def test_false_on_permission_error(self, monkeypatch):
        monkeypatch.setattr(Path, "stat", _eacces)
        assert gateway_cli._runtime_dir_is_ours("/run/user/0") is False

    def test_false_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(os, "getuid", lambda: 1001)
        assert gateway_cli._runtime_dir_is_ours(str(tmp_path / "nope")) is False


class TestUserSystemdSocketReadyForeignRuntime:
    """The readiness probe must not crash on an unreadable foreign XDG_RUNTIME_DIR."""

    def test_returns_false_on_eacces_instead_of_raising(self, monkeypatch):
        # su/sudo -u from root leaves XDG_RUNTIME_DIR=/run/user/0 (0700 root:root);
        # stat-ing a socket underneath it raises PermissionError.
        monkeypatch.setattr(Path, "exists", _eacces)
        # Previously raised PermissionError; must now report not-ready.
        assert gateway_cli._user_systemd_socket_ready() is False


class TestEnsureUserSystemdEnvForeignRuntime:
    """_ensure_user_systemd_env() drops a leaked foreign XDG_RUNTIME_DIR."""

    def test_replaces_foreign_leaked_xdg_runtime_dir(self, monkeypatch):
        # Fall back to our own dir so systemctl --user targets the right instance
        # instead of /run/user/0.
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/0")
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        monkeypatch.setattr(os, "getuid", lambda: 1001)
        monkeypatch.setattr(
            gateway_cli, "_runtime_dir_is_ours", lambda d: d == "/run/user/1001",
        )
        monkeypatch.setattr(
            gateway_cli, "_path_exists_safe", lambda p: str(p) == "/run/user/1001/bus",
        )

        gateway_cli._ensure_user_systemd_env()

        assert os.environ["XDG_RUNTIME_DIR"] == "/run/user/1001"
        assert os.environ["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/run/user/1001/bus"

    def test_keeps_own_xdg_runtime_dir(self, tmp_path, monkeypatch):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        monkeypatch.setattr(os, "getuid", lambda: runtime.stat().st_uid)

        gateway_cli._ensure_user_systemd_env()

        # A runtime dir that is genuinely ours must not be clobbered.
        assert os.environ["XDG_RUNTIME_DIR"] == str(runtime)

    def test_does_not_crash_when_foreign_bus_is_unreadable(self, monkeypatch):
        # Foreign XDG and no usable /run/user/{uid}: env stays as-is, no traceback.
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/0")
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        monkeypatch.setattr(os, "getuid", lambda: 1001)
        monkeypatch.setattr(gateway_cli, "_runtime_dir_is_ours", lambda d: False)
        monkeypatch.setattr(Path, "exists", _eacces)

        gateway_cli._ensure_user_systemd_env()  # must not raise

        assert "DBUS_SESSION_BUS_ADDRESS" not in os.environ


class TestPreflightForeignRuntimeNoLeak:
    """#86558: preflight surfaces a remediable error, not a raw PermissionError."""

    def test_foreign_xdg_runtime_dir_raises_unavailable_not_permission_error(self, monkeypatch):
        # runuser -u user -- hermes gateway restart, from a root shell.
        monkeypatch.setattr(gateway_cli, "_ensure_user_systemd_env", lambda: None)
        # Both socket paths resolve under the leaked /run/user/0 and are 0700 root.
        monkeypatch.setattr(Path, "exists", _eacces)
        monkeypatch.setattr(gateway_cli, "get_systemd_linger_status", lambda: (False, ""))
        monkeypatch.setattr(gateway_cli.shutil, "which", lambda _: "/usr/bin/loginctl")

        class _Denied:
            returncode = 1
            stdout = ""
            stderr = "Interactive authentication required."

        monkeypatch.setattr(gateway_cli.subprocess, "run", lambda *a, **kw: _Denied())

        with pytest.raises(gateway_cli.UserSystemdUnavailableError):
            gateway_cli._preflight_user_systemd()
