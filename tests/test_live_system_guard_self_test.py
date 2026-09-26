"""Self-test for the live-system guard fixture in tests/conftest.py.

This file is the canary. If anyone removes a guard or weakens it, these
tests fail. If anyone adds a NEW kill primitive to the codebase without
adding it to the guard, the corresponding test added here will fail too.

The guard exists to protect the developer's live ``hermes-gateway`` process
from being SIGTERMed by tests. See PR #23397 for the original incident
(5+ live gateway kills in 3 days). Per Teknium 2026-05-10:

  > "You better do such a deep scan and scrub of the tests that this
  >  never is possible ever again for all eternity."

Every primitive that can deliver a signal to a foreign process or mutate
the live systemd unit MUST be exercised below. Adding a new primitive to
the guard? Add a test here too.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import types

import pytest

# A guaranteed-foreign PID: PID 1 (init).  Owned by root, not us, and
# always exists. A sane guard refuses to signal it.
FOREIGN_PID = 1


# ──────────────────── fail-closed self-protection ──────────────
#
# This file executes REAL kill primitives — os.kill(-1, SIGTERM), os.killpg,
# pkill -f python — and depends entirely on the autouse ``_live_system_guard``
# fixture in tests/conftest.py to intercept them. That makes the canary
# fail-OPEN: in any collection context where this file is present but its home
# conftest is not, the primitives fire for real and ``os.kill(-1, SIGTERM)``
# SIGTERMs every process the invoking user owns (a full desktop-session kill was
# reported in the field — see issue #68311). Such contexts are not exotic:
# published sdists that ship ``tests/`` but not ``tests/conftest.py``, trees
# assembled by copying ``test*.py`` files (that glob does NOT match
# ``conftest.py``), ``pytest --noconftest``, or running from a foreign rootdir.
#
# The fixture below makes the canary fail-CLOSED instead: it refuses to run any
# test in this file unless the guard is provably active, so no collection
# context can ever detonate the primitives. The one thing the canary can detect
# about its own safety is that the guard monkeypatches ``os.kill`` with a plain
# Python function, whereas the unguarded primitive is a C builtin.


def _live_system_guard_is_active() -> bool:
    """True iff tests/conftest.py's ``_live_system_guard`` has patched os.kill.

    The guard replaces ``os.kill`` with a plain Python function; the raw,
    unguarded primitive is a C builtin (``types.BuiltinFunctionType``). If
    ``os.kill`` is still the builtin, the guard never loaded and every kill
    primitive in this file would fire for real.
    """
    return not isinstance(os.kill, types.BuiltinFunctionType)


@pytest.fixture(autouse=True)
def _refuse_to_fire_live_weapons(request):
    """Fail closed: refuse to run a canary test unless the guard is active.

    Tests genuinely marked ``@pytest.mark.live_system_guard_bypass`` opt out
    (they run the raw primitive deliberately and harmlessly, e.g. a signal-0
    liveness probe of our own PID), matching the guard's own bypass contract.
    """
    if request.node.get_closest_marker("live_system_guard_bypass"):
        yield
        return
    if not _live_system_guard_is_active():
        pytest.fail(
            "REFUSING TO RUN: the live-system guard from tests/conftest.py is "
            "not active in this interpreter (os.kill is still the raw C "
            "builtin). This canary file executes real kill primitives — "
            "os.kill(-1, SIGTERM), os.killpg, pkill -f python — and relies on "
            "the guard to intercept them; unguarded, they SIGTERM every process "
            "the current user owns. This usually means the file was collected "
            "without its home tests/conftest.py (note: a test*.py copy glob "
            "does NOT match conftest.py). See issue #68311.",
            pytrace=False,
        )
    yield


def test_fail_closed_probe_reports_guard_active():
    """In the real suite the guard is loaded, so the probe reports active and
    ``_refuse_to_fire_live_weapons`` stays out of the way (no false positives
    that would wedge CI)."""
    assert _live_system_guard_is_active() is True




# ──────────────────── kill primitives ─────────────────────────


@pytest.mark.platforms("linux")
def test_os_kill_blocks_foreign_pid():
    with pytest.raises(RuntimeError, match="live-system guard"):
        os.kill(FOREIGN_PID, signal.SIGTERM)


def test_os_kill_blocks_negative_one():
    """``os.kill(-1, sig)`` signals every process we can reach. Must be blocked."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        os.kill(-1, signal.SIGTERM)


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="killpg POSIX-only")
def test_os_killpg_blocks_foreign_pgid():
    with pytest.raises(RuntimeError, match="live-system guard"):
        os.killpg(FOREIGN_PID, signal.SIGTERM)


# ──────────────────── subprocess regex bypasses ────────────────


def test_subprocess_run_systemctl_restart_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["systemctl", "--user", "restart", "hermes-gateway"])


def test_subprocess_run_full_path_systemctl_blocked():
    """``/usr/bin/systemctl`` (full path) must be blocked too."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["/usr/bin/systemctl", "--user", "stop", "hermes-gateway"])


def test_subprocess_run_sudo_systemctl_blocked():
    """``sudo systemctl ...`` defeated the old head==systemctl check."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["sudo", "systemctl", "restart", "hermes-gateway"])


def test_subprocess_run_env_systemctl_blocked():
    """``env systemctl ...`` similarly defeated the old head check."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["env", "systemctl", "--user", "restart", "hermes-gateway"])


def test_subprocess_run_bash_c_systemctl_blocked():
    """``bash -c "systemctl ..."`` must also be caught."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["bash", "-c", "systemctl --user restart hermes-gateway"])




def test_subprocess_run_setsid_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["setsid", "systemctl", "kill", "hermes-gateway"])


def test_subprocess_run_string_shell_true_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(
            "systemctl --user restart hermes-gateway",
            shell=True,
        )


def test_subprocess_popen_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.Popen(["systemctl", "--user", "stop", "hermes-gateway"])


def test_subprocess_call_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.call(["systemctl", "--user", "restart", "hermes-gateway"])


def test_subprocess_check_call_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.check_call(["systemctl", "--user", "restart", "hermes-gateway"])


def test_subprocess_check_output_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.check_output(["systemctl", "--user", "restart", "hermes-gateway"])


def test_subprocess_getoutput_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.getoutput("systemctl --user restart hermes-gateway")


def test_subprocess_getstatusoutput_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.getstatusoutput("systemctl --user restart hermes-gateway")


# ──────────────────── os.system / os.popen ────────────────────


def test_os_system_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        os.system("systemctl --user restart hermes-gateway")


def test_os_popen_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        os.popen("systemctl --user restart hermes-gateway")


# ──────────────────── pty.spawn ────────────────────────────────


@pytest.mark.platforms("linux")
def test_pty_spawn_systemctl_blocked():
    import pty
    with pytest.raises(RuntimeError, match="live-system guard"):
        pty.spawn(["systemctl", "--user", "restart", "hermes-gateway"])


# ──────────────────── asyncio.create_subprocess_* ──────────────


def test_asyncio_create_subprocess_exec_systemctl_blocked():
    import asyncio

    async def _attempt():
        await asyncio.create_subprocess_exec(
            "systemctl", "--user", "restart", "hermes-gateway"
        )

    with pytest.raises(RuntimeError, match="live-system guard"):
        asyncio.run(_attempt())


def test_asyncio_create_subprocess_shell_systemctl_blocked():
    import asyncio

    async def _attempt():
        await asyncio.create_subprocess_shell(
            "systemctl --user restart hermes-gateway"
        )

    with pytest.raises(RuntimeError, match="live-system guard"):
        asyncio.run(_attempt())


# ──────────────────── pkill / killall / taskkill ───────────────


def test_subprocess_pkill_hermes_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["pkill", "-f", "hermes"])




def test_subprocess_pkill_python_dash_f_blocked():
    """``pkill -f python`` matches the gateway's "python -m hermes_cli.main"."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["pkill", "-f", "python"])


def test_subprocess_killall_hermes_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["killall", "hermes"])


# ──────────────────── pass-through cases (must NOT raise) ──────
















# ──────────────────── real gateway runtime spawn ─────────────────


def test_subprocess_popen_real_gateway_restart_blocked():
    """``python -m hermes_cli.main gateway restart`` is a detached child that
    inherits the pytest-tmp HERMES_HOME, resolves the developer's real
    ``hermes-gateway`` unit, and outlives the test (39 six-day orphans squatted
    the webhook port, 2026-09-03). Blocked at the spawn primitive."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.Popen(
            [sys.executable, "-m", "hermes_cli.main", "gateway", "restart"],
            start_new_session=True,
        )


def test_subprocess_popen_inline_source_restart_watcher_blocked():
    """``gateway._spawn_gateway_restart_watcher`` hides the real gateway argv behind
    ``python -c <src> <old_pid> …``. The identity matcher must ignore that trailing argv (#107002),
    but the guard reads it as SPAWN INTENT — otherwise the watcher sails through, waits out its
    120s deadline and leaves a real detached gateway squatting the webhook port."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(1)", "4242",
             sys.executable, "-m", "hermes_cli.main", "gateway", "run"],
            start_new_session=True,
        )


def test_subprocess_run_gateway_status_passes_through():
    """Only lifecycle verbs are blocked: ``gateway status`` (and every other
    read-only subcommand) must still spawn — via the canonical matcher, not an
    argv substring."""
    result = subprocess.run(
        [sys.executable, "-c", "import sys; print(sys.argv[1:])", "-m", "hermes_cli.main", "gateway", "status"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0


# ──────────────────── bypass marker ─────────────────────────────


@pytest.mark.live_system_guard_bypass
def test_bypass_marker_disables_guard():
    """The bypass marker exists for tests that genuinely need real signal delivery
    (e.g. PTY tests SIGINTing their own child). Verify it works.

    We use it harmlessly here by signaling our own PID 0 (own group) so we
    don't actually kill anything — but the call goes through real os.kill.
    """
    # With bypass, the guard yields without installing the monkeypatch,
    # so we get the real os.kill. Calling os.kill(os.getpid(), 0) just
    # checks that the PID exists — harmless.
    os.kill(os.getpid(), 0)  # No exception — guard is OFF.



# Native launchctl canaries use only `help`: a missing guard must never stop a service.
# This probe runs at collection time, before any autouse fixture can install protection.
_collection_launchctl_denied = False
if sys.platform == "darwin":
    try:
        subprocess.run(["/bin/launchctl", "help"], capture_output=True)
    except PermissionError:
        _collection_launchctl_denied = True


@pytest.mark.platforms("macos")
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS execution boundary")
@pytest.mark.live_system_guard_bypass
class TestLaunchctlExecutionBoundary:
    def test_guard_is_active_before_test_collection(self):
        assert _collection_launchctl_denied

    def test_system_ps_keeps_process_diagnostics_working(self):
        result = subprocess.run(
            ["/bin/ps", "-p", str(os.getpid()), "-o", "pid=,pgid="],
            capture_output=True, text=True, check=True,
        )
        pid, pgid = result.stdout.split()
        assert int(pid) == os.getpid()
        assert int(pgid) == os.getpgid(0)
        with pytest.raises(PermissionError):
            subprocess.run(["/bin/launchctl", "help"], capture_output=True)

    def test_direct_host_and_symlink_are_denied(self, tmp_path):
        import errno
        alias = tmp_path / "controller-alias"
        alias.symlink_to("/bin/launchctl")
        for executable in ("/bin/launchctl", str(alias)):
            with pytest.raises(PermissionError) as caught:
                subprocess.run([executable, "help"], capture_output=True)
            assert caught.value.errno in (errno.EPERM, errno.EACCES)

    @pytest.mark.parametrize("command", [
        ["/bin/bash", "-c", 'cmd=/bin/launchctl; "$cmd" help'],
        ["/bin/bash", "-cO", "extglob", "/bin/launchctl help"],
        ["/bin/bash", "-c", "cd /bin; ./launchctl help"],
        ["/bin/bash", "-c", "if true; then /bin/launchctl help; fi"],
        ["/bin/bash", "-c", "f() { /bin/launchctl help; }; f"],
        ["/bin/bash", "-c", '${CMD:-/bin/launchctl} help'],
        ["/bin/bash", "-c", "PATH=/bin launchctl help"],
    ])
    def test_shell_forms_cannot_execute_host_controller(self, command):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 0
        assert "Operation not permitted" in result.stderr or "Permission denied" in result.stderr

    def test_opaque_script_and_native_descendant_are_denied(self, tmp_path):
        script = tmp_path / "nested.sh"
        script.write_text('cmd=/bin/launchctl; "$cmd" help\n', encoding="utf-8")
        for argv in (["/bin/bash", str(script)],
                     ["/usr/bin/find", str(script), "-exec", "/bin/launchctl", "help", ";"]):
            result = subprocess.run(argv, capture_output=True, text=True)
            # find may return zero even when its -exec child was denied.
            assert "Operation not permitted" in result.stderr or "Permission denied" in result.stderr

    @pytest.mark.parametrize("primitive", [
        "os.execv('/bin/launchctl', ['launchctl', 'help'])",
        "os.posix_spawn('/bin/launchctl', ['launchctl', 'help'], {})",
        "subprocess.run(['/bin/launchctl', 'help'], env={})",
    ])
    def test_fresh_interpreter_inherits_guard_with_empty_environment(self, primitive):
        source = (
            "import errno, os, subprocess\n"
            "try:\n    " + primitive + "\n"
            "except OSError as error:\n"
            "    assert error.errno in (errno.EPERM, errno.EACCES)\n"
            "    print('denied')\n"
            "else:\n    raise AssertionError('host controller executed')\n"
        )
        result = subprocess.run([sys.executable, "-c", source], env={}, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "denied"

    def test_host_binary_cannot_be_copied_outside_protected_path(self):
        import errno
        from pathlib import Path
        with pytest.raises(PermissionError) as caught:
            Path("/bin/launchctl").read_bytes()
        assert caught.value.errno in (errno.EPERM, errno.EACCES)

    def test_fake_and_complex_harmless_shell_remain_usable(self, tmp_path):
        fake = tmp_path / "launchctl"
        fake.write_text("#!/bin/sh\nprintf 'fake-ok\\n'\n", encoding="utf-8")
        fake.chmod(0o755)
        source = tmp_path / "fixture.sh"
        source.write_text('say() { printf "shell-ok\\n"; }\n', encoding="utf-8")
        commands = [
            ([str(fake), "help"], "fake-ok"),
            (["/bin/bash", "-c", 'launchctl help',], "fake-ok"),
            (["/bin/bash", "-c", 'source "$1"; X=1; if [ "$X" = 1 ]; then say; fi', "bash", str(source)], "shell-ok"),
            (["/bin/echo", "/bin/launchctl help"], "/bin/launchctl help"),
        ]
        for argv, expected in commands:
            result = subprocess.run(argv, env={"PATH": str(tmp_path), "LC_ALL": "C"}, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip() == expected

    def test_child_can_reinstall_guard_after_inheritance(self):
        source = "from tests.launchctl_safety import install_launchctl_guard; install_launchctl_guard(); install_launchctl_guard()"
        result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


@pytest.mark.platforms("not macos")
def test_launchctl_guard_is_inert_off_darwin(monkeypatch):
    from tests import launchctl_safety
    monkeypatch.setattr(launchctl_safety, "_INSTALLED", False, raising=False)
    def unexpected(*args, **kwargs):
        raise AssertionError("non-Darwin must not load libsandbox")
    monkeypatch.setattr("ctypes.CDLL", unexpected)
    launchctl_safety.install_launchctl_guard()


@pytest.mark.platforms("macos")
def test_launchctl_guard_fails_explicitly_when_native_library_is_unavailable(monkeypatch):
    from tests import launchctl_safety
    monkeypatch.setattr(launchctl_safety, "_INSTALLED", False, raising=False)
    def unavailable(*args, **kwargs):
        raise OSError("test library unavailable")
    monkeypatch.setattr("ctypes.CDLL", unavailable)
    with pytest.raises(RuntimeError, match="launchctl test guard"):
        launchctl_safety.install_launchctl_guard()
    assert not launchctl_safety._INSTALLED


@pytest.mark.platforms("macos")
def test_launchctl_guard_fails_closed_on_profile_rejection(monkeypatch):
    from unittest.mock import Mock
    from tests import launchctl_safety
    monkeypatch.setattr(launchctl_safety, "_INSTALLED", False, raising=False)
    library = Mock()
    library.sandbox_init.return_value = -1
    monkeypatch.setattr("ctypes.CDLL", Mock(return_value=library))
    with pytest.raises(RuntimeError, match="launchctl test guard"):
        launchctl_safety.install_launchctl_guard()
    assert not launchctl_safety._INSTALLED
