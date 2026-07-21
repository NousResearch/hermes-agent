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
import types

import pytest

# A guaranteed-foreign PID: PID 1 (init).  Owned by root, not us, and
# always exists. A sane guard refuses to signal it.
FOREIGN_PID = 1


def _record_guarded_run(monkeypatch):
    """Replace only the guarded run wrapper's native delegate with a recorder."""
    calls = []

    def _native_recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    assert hasattr(subprocess.run, "__wrapped__"), "run wrapper is not installed"
    monkeypatch.setattr(subprocess.run, "__wrapped__", _native_recorder)
    return calls


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


def test_fail_closed_probe_classifies_raw_builtin_as_unguarded():
    """The probe's discriminator, exercised against real objects: a raw C
    builtin the guard never touches (``os.getpid``) is exactly what an
    unguarded ``os.kill`` looks like and must read as 'guard not active', while
    the loaded guard's ``os.kill`` is a plain Python function."""
    assert isinstance(os.getpid, types.BuiltinFunctionType)
    assert not isinstance(os.kill, types.BuiltinFunctionType)


# ──────────────────── kill primitives ─────────────────────────


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


def test_subprocess_run_sh_c_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["sh", "-c", "systemctl --user stop hermes-gateway"])


def test_subprocess_shell_wrapper_variants_are_blocked_before_execution():
    for command in (
        ["bash", "-lc", "systemctl restart hermes-gateway"],
        ["cmd.exe", "/c", "systemctl stop hermes-gateway"],
        ["powershell", "-Command", "Stop-Service hermes-gateway"],
    ):
        with pytest.raises(RuntimeError, match="live-system guard"):
            subprocess.run(command)


def test_popen_sh_c_detached_gateway_spawn_is_blocked_before_execution():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.Popen(["sh", "-c", "hermes gateway run &"])


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


def test_patched_popen_rejects_detached_gateway_before_native_execution():
    """The installed Popen wrapper must reject the spawn before its superclass."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.Popen(["hermes", "gateway", "run"], start_new_session=True)


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


def test_subprocess_pkill_hermes_gateway_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["pkill", "-f", "hermes-gateway"])


def test_subprocess_pkill_python_dash_f_blocked():
    """``pkill -f python`` matches the gateway's "python -m hermes_cli.main"."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["pkill", "-f", "python"])


def test_subprocess_killall_hermes_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["killall", "hermes"])


def test_patched_run_rejects_launchctl_gateway_mutations_before_native_execution():
    """The patched run wrapper blocks both launchd mutations without a native call."""
    for command in (
        ["launchctl", "kickstart", "gui/501/ai.hermes.gateway"],
        ["launchctl", "bootout", "gui/501/ai.hermes.gateway"],
    ):
        with pytest.raises(RuntimeError, match="live-system guard"):
            subprocess.run(command)


def test_patched_run_rejects_taskkill_foreign_pid_before_native_execution():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["taskkill", "/PID", str(FOREIGN_PID), "/T", "/F"])


def test_patched_run_rejects_taskkill_hermes_image_before_native_execution():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["taskkill.exe", "/IM", "hermes-gateway.exe", "/F"])


def test_patched_run_rejects_negative_kill_targets_before_native_execution():
    for command in (
        ["kill", "-9", "-1"],
        ["kill", "-9", "-987654321"],
        ["killpg", str(FOREIGN_PID), "9"],
    ):
        with pytest.raises(RuntimeError, match="live-system guard"):
            subprocess.run(command)


def test_patched_run_rejects_non_read_only_systemctl_verbs_before_execution():
    for command in (
        ["systemctl", "reenable", "hermes-gateway.service"],
        ["systemctl", "set-property", "hermes-gateway.service", "CPUQuota=1%"],
    ):
        with pytest.raises(RuntimeError, match="live-system guard"):
            subprocess.run(command)


# ──────────────────── pass-through cases (must NOT raise) ──────


def test_systemctl_status_passes_through(monkeypatch):
    """Read-only systemctl probes (status/show/list-units) are fine."""
    calls = _record_guarded_run(monkeypatch)
    r = subprocess.run(
        ["systemctl", "--user", "status", "hermes-gateway", "--no-pager"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert len(calls) == 1


def test_systemctl_show_passes_through(monkeypatch):
    calls = _record_guarded_run(monkeypatch)
    r = subprocess.run(
        ["systemctl", "--user", "show", "hermes-gateway", "--no-pager"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert len(calls) == 1


def test_required_pass_through_commands_are_delegated_without_execution(monkeypatch):
    calls = _record_guarded_run(monkeypatch)
    for command in (
        ["bash", "-lc", "echo hello"],
        ["kill", "-TERM", str(os.getpid())],
        ["systemctl", "--user", "status", "hermes-gateway"],
        ["systemctl", "show", "--property=Restart", "hermes-gateway"],
        ["taskkill", "/IM", "notepad.exe", "/F"],
    ):
        assert subprocess.run(command).returncode == 0
    assert len(calls) == 5


def test_systemctl_list_units_passes_through(monkeypatch):
    calls = _record_guarded_run(monkeypatch)
    r = subprocess.run(
        ["systemctl", "--user", "list-units", "fake-not-real-unit*", "--no-pager"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert len(calls) == 1


def test_systemctl_unrelated_unit_passes_through(monkeypatch):
    """systemctl restart of a non-hermes unit is allowed (we only protect hermes)."""
    calls = _record_guarded_run(monkeypatch)
    r = subprocess.run(
        ["systemctl", "--user", "show", "fake-not-real-unit"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert len(calls) == 1


def test_external_kill_of_own_subtree_passes_through(monkeypatch):
    """A numeric external kill of this test process is delegated, never run natively."""
    calls = _record_guarded_run(monkeypatch)
    r = subprocess.run(["kill", "-0", str(os.getpid())])
    assert r.returncode == 0
    assert len(calls) == 1


def test_launchctl_print_passes_through(monkeypatch):
    calls = _record_guarded_run(monkeypatch)
    r = subprocess.run(["launchctl", "print", "gui/501/ai.hermes.gateway"])
    assert r.returncode == 0
    assert len(calls) == 1


def test_taskkill_benign_image_passes_through(monkeypatch):
    calls = _record_guarded_run(monkeypatch)
    r = subprocess.run(["taskkill.exe", "/IM", "notepad.exe"])
    assert r.returncode == 0
    assert len(calls) == 1


def test_subprocess_pkill_with_unrelated_pattern_passes_through(monkeypatch):
    """``pkill -f some-unrelated-pattern`` (no hermes/python) is fine."""
    calls = _record_guarded_run(monkeypatch)
    r = subprocess.run(["pkill", "-f", "some-unrelated-pattern"], capture_output=True)
    assert r.returncode == 0
    assert len(calls) == 1


def test_normal_subprocess_run_passes_through(monkeypatch):
    """Plain non-systemctl subprocess.run should work normally."""
    calls = _record_guarded_run(monkeypatch)
    r = subprocess.run(["echo", "hello"], capture_output=True, text=True)
    assert r.returncode == 0
    assert len(calls) == 1


# ──────────────────── bypass marker ─────────────────────────────


@pytest.mark.live_system_guard_bypass
def test_bypass_marker_disables_guard():
    """The bypass marker exists for tests that genuinely need real signal delivery
    (e.g. PTY tests SIGINTing their own child). Verify it works.

    No signal is sent: the raw builtin identity is enough to prove bypass.
    """
    # With bypass, the guard yields without installing the monkeypatch.
    assert isinstance(os.kill, types.BuiltinFunctionType)
