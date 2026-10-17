"""WSL: a Windows cua-driver reached through interop needs a Windows socket, not the host's Unix one.

The private daemon's ``--socket`` is consumed only by driver processes (``serve``, the ``mcp`` proxy,
``status``/``stop`` probes). When those processes are Windows — the normal WSL setup, where
``HERMES_CUA_DRIVER_CMD`` points at ``/mnt/c/.../cua-driver.exe`` and the driver's own manifest reports the
``C:\\...`` it was installed at — a Unix ``.sock`` path is rejected with ``os error 123``
(NousResearch/hermes-agent#80227). ``sys.platform`` describes the host, so it cannot answer this.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tests.waiting import wait_until
from tools.computer_use import cua_backend, cua_backend_daemon, cua_backend_driver

_ELF_MAGIC = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 56
_PE_MAGIC = b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 56


def _elf(path: Path) -> str:
    """A readable non-PE executable: a Linux driver, or a Linux driver misleadingly named ``*.exe``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_ELF_MAGIC)
    path.chmod(0o755)
    return str(path)


def _pe(path: Path) -> str:
    """A Windows executable (the DOS/PE ``MZ`` header the driver's own binary carries)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PE_MAGIC)
    path.chmod(0o755)
    return str(path)


def _pipe(value: str) -> bool:
    return value.startswith(cua_backend_daemon._WINDOWS_PIPE_PREFIX)


# ── 1. WSL calling a Windows .exe ───────────────────────────────────────────────────────────────────────
@pytest.mark.platforms("not windows")
def test_wsl_interop_path_to_windows_exe_is_windows(tmp_path):
    """The production shape: PM/env resolve a Windows driver to its ``/mnt/<drive>/...`` interop path."""
    command = _pe(tmp_path / "mnt" / "c" / "Users" / "alice" / "Programs" / "Cua" / "bin" / "cua-driver.exe")
    assert cua_backend_driver.cua_driver_command_is_windows(command)
    assert _pipe(cua_backend_daemon._EmbeddedCuaDaemon(command, "unrestricted").socket_path)


@pytest.mark.platforms("not windows")
def test_windows_manifest_notation_is_windows_without_touching_the_filesystem(tmp_path):
    """The ``C:\\...`` a Windows driver reports in its manifest, and the ``/mnt/c`` form it is translated to."""
    windows = r"C:\Users\alice\.cua-driver\packages\releases\0.23.2-x86_64-pc-windows-msvc\cua-driver.exe"
    assert cua_backend_driver.cua_driver_command_is_windows(windows)
    with patch("hermes_constants.is_wsl", return_value=True):
        translated = cua_backend_driver._wsl_windows_path_to_posix(windows)
    assert translated == "/mnt/c/Users/alice/.cua-driver/packages/releases/0.23.2-x86_64-pc-windows-msvc/cua-driver.exe"
    assert _pipe(cua_backend_daemon._EmbeddedCuaDaemon(translated, "unrestricted").socket_path)


@pytest.mark.platforms("not windows")
def test_live_interop_driver_uses_a_named_pipe():
    """The driver actually installed on this host, when it is a Windows build reached through interop."""
    import os

    from tools.computer_use.cua_backend_driver import _CUA_DRIVER_CMD_ENV

    configured = (os.environ.get(_CUA_DRIVER_CMD_ENV) or "").strip()
    if not configured.startswith("/mnt/") or not Path(configured).is_file():
        pytest.skip("no Windows cua-driver configured for WSL interop on this host")
    resolved = cua_backend_driver.resolve_cua_driver_cmd()
    assert resolved == configured and resolved is not None
    command, _args = cua_backend_driver._resolve_mcp_invocation(resolved, timeout=30.0)
    assert cua_backend_driver.cua_driver_command_is_windows(command)
    assert _pipe(cua_backend_daemon._EmbeddedCuaDaemon(resolved, "unrestricted").socket_path)


# ── 2. Native Linux drivers are unaffected ──────────────────────────────────────────────────────────────
@pytest.mark.platforms("not windows")
def test_posix_driver_keeps_a_unix_socket(tmp_path):
    command = _elf(tmp_path / "tools" / "cua-driver-0.21.0-linux-x64" / "cua-driver")
    assert not cua_backend_driver.cua_driver_command_is_windows(command)
    daemon = cua_backend_daemon._EmbeddedCuaDaemon(command, "unrestricted")
    assert daemon.socket_path.endswith(".sock") and daemon.socket_path.startswith(tempfile.gettempdir())
    assert not _pipe(daemon.socket_path)


@pytest.mark.platforms("not windows")
def test_linux_binary_named_exe_is_still_a_linux_driver(tmp_path):
    """The header decides, not the name: a Linux build dropped in as ``*.exe`` must not be sent a pipe."""
    command = _elf(tmp_path / "downloads" / "cua-driver.exe")
    assert not cua_backend_driver.cua_driver_command_is_windows(command)
    assert not _pipe(cua_backend_daemon._EmbeddedCuaDaemon(command, "unrestricted").socket_path)


@pytest.mark.platforms("not windows")
def test_bare_posix_name_and_unresolved_command_keep_a_unix_socket():
    for command in ("cua-driver", "", "/opt/nope/cua-driver"):
        assert not cua_backend_driver.cua_driver_command_is_windows(command)
    assert not _pipe(cua_backend_daemon._EmbeddedCuaDaemon("cua-driver", "unrestricted").socket_path)
    # An empty override defers to PM; until start() resolves it, the host's transport is the safe assumption.
    assert not _pipe(cua_backend_daemon._EmbeddedCuaDaemon("", "unrestricted").socket_path)


# ── 3. Native Windows path selection is unaffected ──────────────────────────────────────────────────────
def test_native_windows_always_gets_a_named_pipe(tmp_path, monkeypatch):
    """On Windows the socket is a pipe whatever the command looks like — the previous behaviour, kept."""
    command = _elf(tmp_path / "cua-driver")  # even a non-PE command cannot be a Unix socket there
    monkeypatch.setattr(cua_backend_daemon.sys, "platform", "win32")
    daemon = cua_backend_daemon._EmbeddedCuaDaemon(command, "unrestricted")
    assert _pipe(daemon.socket_path)
    assert daemon.socket_path == cua_backend_daemon._windows_pipe_path(daemon._socket_token)


@pytest.mark.platforms("not windows")
def test_windows_notation_and_bare_exe_are_windows(tmp_path):
    for command in (
        r"C:\Program Files\Cua\cua-driver\bin\cua-driver.exe",
        "C:/Program Files/Cua/cua-driver/bin/cua-driver.exe",
        r"\\wsl.localhost\share\cua-driver.exe",
        "cua-driver.exe",
        "cua-driver.com",
    ):
        assert cua_backend_driver.cua_driver_command_is_windows(command), command
        assert _pipe(cua_backend_daemon._EmbeddedCuaDaemon(command, "unrestricted").socket_path), command


# ── 4. Symlinks and paths containing spaces ─────────────────────────────────────────────────────────────
@pytest.mark.platforms("not windows")
def test_symlink_to_windows_exe_is_windows(tmp_path):
    target = _pe(tmp_path / "My Cua Driver" / "cua-driver.exe")
    link = tmp_path / "bin" / "cua-driver"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(target)
    except OSError:  # pragma: no cover - filesystems without symlink support
        pytest.skip("symlinks unavailable")
    assert cua_backend_driver.cua_driver_command_is_windows(str(link))
    assert _pipe(cua_backend_daemon._EmbeddedCuaDaemon(str(link), "unrestricted").socket_path)


@pytest.mark.platforms("not windows")
def test_symlink_to_exe_without_extension_is_windows(tmp_path):
    target = _pe(tmp_path / "vendor" / "driver-0.23.2")
    link = tmp_path / "vendor" / "current"
    try:
        link.symlink_to(target)
    except OSError:  # pragma: no cover - filesystems without symlink support
        pytest.skip("symlinks unavailable")
    assert cua_backend_driver.cua_driver_command_is_windows(str(link))


@pytest.mark.platforms("not windows")
def test_symlink_to_linux_exe_named_link_is_not_windows(tmp_path):
    target = _elf(tmp_path / "vendor" / "cua-driver")
    link = tmp_path / "vendor" / "cua-driver.exe"
    try:
        link.symlink_to(target)
    except OSError:  # pragma: no cover - filesystems without symlink support
        pytest.skip("symlinks unavailable")
    assert not cua_backend_driver.cua_driver_command_is_windows(str(link))


@pytest.mark.platforms("not windows")
def test_windows_exe_with_spaces_in_path_is_windows(tmp_path):
    command = _pe(tmp_path / "Program Files" / "Cua Driver 0.23.2" / "cua-driver.exe")
    assert " " in command
    assert cua_backend_driver.cua_driver_command_is_windows(command)
    assert _pipe(cua_backend_daemon._EmbeddedCuaDaemon(command, "unrestricted").socket_path)


# ── The socket follows the command start() resolves ─────────────────────────────────────────────────────
def _stub_start(monkeypatch, resolved_command: str) -> Mock:
    """Run ``start()`` for real (socket derivation included); stub only the spawn and the ready probe.

    ``_cua_no_overlay`` is pinned off for the same reason the 0.10 permissions test pins it: its
    environment-dependent auto-detect can add a ``--help`` capability probe the fixed run budget below
    does not account for (that probe is ``subprocess.run``'s third call)."""
    process = SimpleNamespace(stderr=(), poll=lambda: None, wait=lambda **kwargs: 0)
    probes = [SimpleNamespace(returncode=0, stdout="running", stderr=""), SimpleNamespace(returncode=0, stdout="", stderr="")]
    popen = Mock(return_value=process)
    monkeypatch.setattr(cua_backend.subprocess, "Popen", popen)
    monkeypatch.setattr(cua_backend.subprocess, "run", Mock(side_effect=probes))
    monkeypatch.setattr(cua_backend, "_cua_no_overlay", lambda: False)
    monkeypatch.setattr(cua_backend_driver, "_resolve_mcp_invocation", lambda cmd, **kwargs: (resolved_command, ["mcp"]))
    return popen


@pytest.mark.platforms("not windows")
def test_start_selects_the_pipe_when_pm_or_the_manifest_resolves_a_windows_driver(tmp_path, monkeypatch):
    """The failure being fixed: the constructor saw an empty/bare command, start() resolved a Windows one."""
    command = _pe(tmp_path / "mnt" / "c" / "Cua" / "cua-driver.exe")
    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", lambda *a, **k: command)
    daemon = cua_backend_daemon._EmbeddedCuaDaemon("", "unrestricted")
    assert not _pipe(daemon.socket_path)  # unresolved: the host's transport, for now
    popen = _stub_start(monkeypatch, command)

    daemon.start()
    try:
        serve = popen.call_args.args[0]
        assert serve[serve.index("--socket") + 1] == daemon.socket_path
        assert _pipe(daemon.socket_path)
        assert daemon.proxy_invocation()[1][-1] == daemon.socket_path
    finally:
        daemon.stop()


@pytest.mark.platforms("not windows")
def test_start_downgrades_to_a_unix_socket_when_resolution_lands_on_a_linux_driver(tmp_path, monkeypatch):
    """Re-derivation is not one-way: a Windows-looking override that resolves to a Linux binary stays native."""
    command = _elf(tmp_path / "opt" / "cua-driver")
    daemon = cua_backend_daemon._EmbeddedCuaDaemon(r"C:\stale\cua-driver.exe", "unrestricted")
    assert _pipe(daemon.socket_path)
    popen = _stub_start(monkeypatch, command)

    daemon.start()
    try:
        serve = popen.call_args.args[0]
        assert serve[serve.index("--socket") + 1] == daemon.socket_path
        assert not _pipe(daemon.socket_path) and daemon.socket_path.endswith(".sock")
    finally:
        daemon.stop()


@pytest.mark.platforms("not windows")
def test_stop_never_unlinks_a_windows_pipe(tmp_path, monkeypatch):
    """A pipe is the driver's to release; the host must not try to delete the name."""
    daemon = cua_backend_daemon._EmbeddedCuaDaemon(_pe(tmp_path / "cua-driver.exe"), "unrestricted")
    assert _pipe(daemon.socket_path)
    remove = Mock()
    monkeypatch.setattr(cua_backend_daemon.os.path, "exists", lambda path: True)  # even claiming it exists
    monkeypatch.setattr(cua_backend_daemon.os, "remove", remove)

    daemon.stop()

    assert not remove.called


# ── Startup/handshake budgets follow the same fact ─────────────────────────────────────────────────────
@pytest.mark.platforms("not windows")
def test_interop_driver_gets_a_cross_os_budget(tmp_path):
    """Each driver call to a Windows driver from a POSIX host is a fresh process launch (measured 1.9-2.8s
    for ``status --socket``), so the native 2.0s probe would report a daemon that is up as unreachable."""
    daemon = cua_backend_daemon._EmbeddedCuaDaemon(_pe(tmp_path / "cua-driver.exe"), "unrestricted")
    assert daemon._driver_via_interop()
    assert daemon._driver_call_timeout(2.0) == daemon._CROSS_OS_CALL_TIMEOUT_SECONDS > 2.8
    assert daemon._driver_call_timeout(3.0) == daemon._CROSS_OS_CALL_TIMEOUT_SECONDS
    assert daemon._start_timeout() == daemon._CROSS_OS_START_TIMEOUT_SECONDS


@pytest.mark.platforms("not windows")
def test_local_driver_keeps_the_native_budgets(tmp_path):
    daemon = cua_backend_daemon._EmbeddedCuaDaemon(_elf(tmp_path / "cua-driver"), "unrestricted")
    assert not daemon._driver_via_interop()
    assert daemon._driver_call_timeout(2.0) == 2.0
    assert daemon._driver_call_timeout(3.0) == 3.0
    assert daemon._start_timeout() == daemon._START_TIMEOUT_SECONDS == 15.0


def test_native_windows_host_keeps_the_native_budgets(tmp_path, monkeypatch):
    """On Windows a .exe driver is local: no interop, so no enlarged budget."""
    daemon = cua_backend_daemon._EmbeddedCuaDaemon(_pe(tmp_path / "cua-driver.exe"), "unrestricted")
    monkeypatch.setattr(cua_backend_daemon.sys, "platform", "win32")
    assert not daemon._driver_via_interop()
    assert daemon._driver_call_timeout(2.0) == 2.0
    assert daemon._start_timeout() == daemon._START_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    "platform,command_kind,expected",
    [("linux", "pe", "ready"), ("linux", "elf", "timed out"), ("win32", "pe", "timed out")],
)
def test_start_readiness_window_follows_the_driver(tmp_path, monkeypatch, platform, command_kind, expected):
    """A driver that takes 25s to answer is ready on the interop budget (60s) and out of time on the native
    one (15s). ``win32`` is the native control: the same ``.exe`` needs no interop launch there, so its
    window is left exactly as it was."""
    from tools.computer_use import cua_backend_daemon as daemon_module

    command = _pe(tmp_path / "cua-driver.exe") if command_kind == "pe" else _elf(tmp_path / "cua-driver")
    monkeypatch.setattr(daemon_module.sys, "platform", platform)
    _stub_start(monkeypatch, command)
    clock = {"now": 0.0}
    monkeypatch.setattr(daemon_module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(daemon_module.time, "sleep", lambda seconds: None)
    daemon = daemon_module._EmbeddedCuaDaemon(command, "unrestricted")

    def socket_ready(env):
        clock["now"] += 1.0
        return clock["now"] >= 25.0

    monkeypatch.setattr(daemon, "_socket_ready", socket_ready)
    if expected == "ready":
        daemon.start()
        try:
            assert daemon._running and clock["now"] >= 25.0
        finally:
            daemon.stop()
    else:
        with pytest.raises(RuntimeError, match="startup timed out"):
            daemon.start()


# ── 5. A bare name is resolved through PATH before the name decides ─────────────────────────────────────
@pytest.mark.platforms("not windows")
def test_bare_name_symlinked_on_path_to_a_windows_pe_is_windows(tmp_path, monkeypatch):
    """A PATH entry with no extension that points at a Windows PE — a WSL interop shim — is a Windows driver."""
    target = _pe(tmp_path / "vendor" / "cua-driver-0.23.2-x86_64-pc-windows-msvc" / "cua-driver")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    link = bin_dir / "cua-driver"
    try:
        link.symlink_to(target)
    except OSError:  # pragma: no cover - filesystems without symlink support
        pytest.skip("symlinks unavailable")
    monkeypatch.setenv("PATH", str(bin_dir))

    assert cua_backend_driver.cua_driver_command_is_windows("cua-driver")
    assert _pipe(cua_backend_daemon._EmbeddedCuaDaemon("cua-driver", "unrestricted").socket_path)


@pytest.mark.platforms("not windows")
def test_bare_name_symlinked_on_path_to_a_linux_elf_is_not_windows(tmp_path, monkeypatch):
    """Resolving a bare name through PATH must not turn a Linux ELF into a Windows driver."""
    target = _elf(tmp_path / "vendor" / "cua-driver-0.21.0-linux-x64" / "cua-driver")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    link = bin_dir / "cua-driver"
    try:
        link.symlink_to(target)
    except OSError:  # pragma: no cover - filesystems without symlink support
        pytest.skip("symlinks unavailable")
    monkeypatch.setenv("PATH", str(bin_dir))

    assert not cua_backend_driver.cua_driver_command_is_windows("cua-driver")
    assert not _pipe(cua_backend_daemon._EmbeddedCuaDaemon("cua-driver", "unrestricted").socket_path)


# ── 6. A failed start() leaves no daemon behind ─────────────────────────────────────────────────────────
_STAND_IN_SOURCE = """\
import os, sys, time

role = sys.argv[1] if len(sys.argv) > 1 else ""
if role == "serve":
    with open(os.environ["CUA_STANDIN_PID_FILE"], "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
if role == "status":
    open(os.environ["CUA_STANDIN_STATUS_MARKER"], "w", encoding="utf-8").close()
"""


def _stand_in_driver(path: Path, body: str) -> str:
    """An executable stand-in for cua-driver: a real process a test can spawn and then look for afterwards.

    The interpreter is the running one (a shebang on ``sys.executable``), so the spawn does not depend on the
    host's PATH. ``body`` runs for every verb, so a ``serve`` that never becomes ready is a ``time.sleep``. The
    ``serve`` daemon announces its pid and a ``status`` probe drops a marker, so a probe timeout is provable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n" + _STAND_IN_SOURCE + body + "\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def _spawnable_daemon(monkeypatch, command: str, *, start_timeout: float = 0.3, call_timeout: float = 0.15):
    """A daemon wired to a stand-in driver: it really spawns, but with no PM resolution, no config lookup and
    no real driver verbs, and with budgets small enough to keep a unit test fast."""
    monkeypatch.setattr(cua_backend, "_cua_no_overlay", lambda: False)
    monkeypatch.setattr(cua_backend_driver, "_resolve_mcp_invocation", lambda cmd, **kwargs: (cmd, []))
    daemon = cua_backend_daemon._EmbeddedCuaDaemon(command, "unrestricted")
    daemon._sanitized_env = lambda: dict(os.environ)
    daemon._start_timeout = lambda: start_timeout
    daemon._driver_call_timeout = lambda native: call_timeout
    return daemon


def _announced_pid(path: Path, timeout: float = 5.0) -> int:
    """The pid the ``serve`` stand-in wrote, once it is visible (the process is reaped by start()'s cleanup)."""
    announced = wait_until(lambda: path.read_text(encoding="utf-8-sig").strip() if path.exists() else "",
                           timeout=timeout, what=f"the stand-in daemon's pid file at {path}")
    return int(announced)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # windows-footgun: ok — only called by the platforms("not windows") tests below
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - another user's process
        return True
    return True


def _assert_reaped(daemon, pid_file: Path) -> None:
    """The failed launch released the handle and the runtime ownership, and its process is gone."""
    pid = _announced_pid(pid_file)
    assert daemon._process is None
    assert daemon._running is False and daemon._owns_runtime is False
    assert not _alive(pid), f"embedded cua-driver (pid {pid}) survived a failed start()"


@pytest.mark.platforms("not windows")
def test_start_cleans_up_when_the_driver_exits_before_becoming_ready(tmp_path, monkeypatch):
    """A driver that dies during startup must not leave its handle or ownership behind: this raise previously
    escaped the readiness loop with ``_process`` still set and ``_owns_runtime`` still true."""
    pid_file = tmp_path / "serve.pid"
    monkeypatch.setenv("CUA_STANDIN_PID_FILE", str(pid_file))
    monkeypatch.setenv("CUA_STANDIN_STATUS_MARKER", str(tmp_path / "status.called"))
    daemon = _spawnable_daemon(monkeypatch, _stand_in_driver(tmp_path / "cua-driver-dies", "sys.exit(3)"),
                               start_timeout=3.0)

    with pytest.raises(RuntimeError, match="exited during startup"):
        daemon.start()

    _assert_reaped(daemon, pid_file)


@pytest.mark.platforms("not windows")
def test_start_reaps_the_driver_that_never_becomes_ready(tmp_path, monkeypatch):
    """Persistently not ready: the readiness loop must stop() — and reap — the daemon it launched."""
    pid_file = tmp_path / "serve.pid"
    monkeypatch.setenv("CUA_STANDIN_PID_FILE", str(pid_file))
    monkeypatch.setenv("CUA_STANDIN_STATUS_MARKER", str(tmp_path / "status.called"))
    daemon = _spawnable_daemon(monkeypatch, _stand_in_driver(tmp_path / "cua-driver-stuck", "time.sleep(1.5)"))
    monkeypatch.setattr(daemon, "_socket_ready", lambda env: False)  # alive, but never ready

    with pytest.raises(RuntimeError, match="startup timed out"):
        daemon.start()

    _assert_reaped(daemon, pid_file)


@pytest.mark.platforms("not windows")
def test_start_reaps_the_driver_when_the_status_probe_times_out(tmp_path, monkeypatch):
    """The readiness probe itself hanging (``status`` never answers) is still a startup failure that must leave
    nothing behind: the real ``_socket_ready`` runs and times out against the stand-in."""
    pid_file = tmp_path / "serve.pid"
    status_marker = tmp_path / "status.called"
    monkeypatch.setenv("CUA_STANDIN_PID_FILE", str(pid_file))
    monkeypatch.setenv("CUA_STANDIN_STATUS_MARKER", str(status_marker))
    daemon = _spawnable_daemon(monkeypatch, _stand_in_driver(tmp_path / "cua-driver-silent", "time.sleep(1.5)"),
                               start_timeout=0.5)

    with pytest.raises(RuntimeError, match="startup timed out"):
        daemon.start()

    assert status_marker.exists(), "the readiness probe never ran, so this is not a probe-timeout case"
    _assert_reaped(daemon, pid_file)
