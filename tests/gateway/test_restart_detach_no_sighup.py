"""Regression coverage for #29603 — gateway restart kills new instance on
non-systemd POSIX environments (Termux/Android).

The historical Unix restart path was::

    setsid bash -lc "while kill -0 OLD; do sleep 0.2; done; hermes gateway restart"

That made ``bash`` the session leader; bash exiting after the restart
subcommand returned tore the session down and SIGHUP'd the freshly
spawned daemon on bare-POSIX environments. The fix replaces the shell
with a Python watcher (matching the Windows branch + the
``hermes_cli.gateway.launch_detached_profile_gateway_restart`` helper)
that re-detaches the new gateway via ``start_new_session=True`` so the
watcher exiting can't take the new gateway with it.

These tests pin the architectural contract end-to-end:

* No ``bash``/``sh`` anywhere in the restart argv (the bug surface).
* The watcher is the live Python interpreter executing a real script.
* The watcher script preserves the original wait-for-old-pid loop.
* The watcher re-spawns the gateway with ``start_new_session=True``.
* The actual watcher script — run in a real subprocess — really does
  put its spawned child in a fresh session.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
from tests.gateway.restart_test_helpers import make_restart_runner


pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="The fix targets the POSIX restart path; the Windows branch is "
    "covered separately.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_resolve_and_pid(monkeypatch, *, pid: int = 9999):
    monkeypatch.setattr(gateway_run, "_resolve_hermes_bin", lambda: ["/usr/bin/hermes"])
    monkeypatch.setattr(gateway_run.os, "getpid", lambda: pid)


def _capture_popen(monkeypatch) -> list[tuple[list[str], dict]]:
    calls: list[tuple[list[str], dict]] = []

    def fake_popen(cmd, **kwargs):
        calls.append((list(cmd), dict(kwargs)))
        return MagicMock()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return calls


def _extract_watcher_script(argv: list[str]) -> str:
    """Pull the Python source out of a ``[..., python, "-c", <script>, …]`` argv."""
    for idx in range(len(argv) - 1):
        if argv[idx] == "-c":
            return argv[idx + 1]
    raise AssertionError(f"watcher argv missing '-c <script>': {argv!r}")


# ---------------------------------------------------------------------------
# Argv-shape regression tests
# ---------------------------------------------------------------------------


class TestRestartArgvShape:
    """The restart command never goes through bash again (#29603)."""

    @pytest.mark.asyncio
    async def test_no_bash_anywhere_in_restart_argv(self, monkeypatch):
        """Direct guardrail: ``bash -lc`` was the bug. Anything that
        reintroduces a shell wrapper should fail loudly here."""
        runner, _ = make_restart_runner()
        _patch_resolve_and_pid(monkeypatch)
        # Force the setsid-available branch — the more-common path.
        monkeypatch.setattr(
            shutil, "which",
            lambda cmd: "/usr/bin/setsid" if cmd == "setsid" else None,
        )
        calls = _capture_popen(monkeypatch)

        await runner._launch_detached_restart_command()

        assert len(calls) == 1
        argv, _kwargs = calls[0]
        flat = " ".join(argv)
        for needle in (" bash ", "bash ", " sh ", "/bin/sh", "-lc", "-c '", "-c \""):
            # ``-c`` on its own is fine (it follows ``sys.executable``);
            # but ``bash`` / ``sh`` / shell-quoted ``-c`` strings are not.
            pass
        assert not any(part in {"bash", "sh", "/bin/bash", "/bin/sh"} for part in argv), (
            f"shell wrapper reappeared in restart argv: {argv!r}"
        )
        assert "-lc" not in argv, f"login-shell flag reintroduced: {argv!r}"

    @pytest.mark.asyncio
    async def test_setsid_branch_uses_python_watcher(self, monkeypatch):
        runner, _ = make_restart_runner()
        _patch_resolve_and_pid(monkeypatch, pid=4242)
        monkeypatch.setattr(
            shutil, "which",
            lambda cmd: "/usr/bin/setsid" if cmd == "setsid" else None,
        )
        calls = _capture_popen(monkeypatch)

        await runner._launch_detached_restart_command()

        argv, kwargs = calls[0]
        # ``setsid --`` then the live interpreter ``-c <script>``.
        assert argv[0] == "/usr/bin/setsid"
        assert argv[1] == "--"
        assert argv[2] == sys.executable
        assert argv[3] == "-c"
        # PID + restart subcommand passed as argv to the watcher.
        tail = argv[5:]
        assert "4242" in tail
        assert "/usr/bin/hermes" in tail
        assert tail[-2:] == ["gateway", "restart"]
        # Top-level Popen is itself in a new session — defence in depth.
        assert kwargs["start_new_session"] is True
        assert kwargs["stdout"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.DEVNULL

    @pytest.mark.asyncio
    async def test_no_setsid_branch_still_uses_python_watcher(self, monkeypatch):
        """Termux is exactly the environment where ``setsid`` is often
        absent. The watcher contract must hold without it."""
        runner, _ = make_restart_runner()
        _patch_resolve_and_pid(monkeypatch, pid=4242)
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        calls = _capture_popen(monkeypatch)

        await runner._launch_detached_restart_command()

        argv, kwargs = calls[0]
        assert argv[0] == sys.executable
        assert argv[1] == "-c"
        # No setsid prefix at all.
        assert "setsid" not in argv[0]
        # PID + restart subcommand still threaded through.
        tail = argv[3:]
        assert "4242" in tail
        assert tail[-2:] == ["gateway", "restart"]
        assert kwargs["start_new_session"] is True

    @pytest.mark.asyncio
    async def test_resolve_failure_short_circuits(self, monkeypatch, caplog):
        """If we can't find the hermes binary, refuse to spawn anything
        — we'd just spin a watcher waiting to run a nonexistent command."""
        runner, _ = make_restart_runner()
        monkeypatch.setattr(gateway_run, "_resolve_hermes_bin", lambda: None)
        calls = _capture_popen(monkeypatch)

        with caplog.at_level("ERROR", logger="gateway.run"):
            await runner._launch_detached_restart_command()

        assert calls == []
        assert any("hermes binary" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_hermes_cmd_with_multiple_parts_threaded_unquoted(self, monkeypatch):
        """When ``_resolve_hermes_bin`` returns a multi-part argv (e.g.
        ``[python, /path/to/hermes-cli]``), the watcher must receive
        each part as a separate argv entry — never re-joined into a
        shell-quoted string the way the old ``shlex.join`` path did.
        """
        runner, _ = make_restart_runner()
        monkeypatch.setattr(
            gateway_run, "_resolve_hermes_bin",
            lambda: ["/usr/bin/python3", "/opt/hermes/cli.py", "--profile=prod"],
        )
        monkeypatch.setattr(gateway_run.os, "getpid", lambda: 7777)
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        calls = _capture_popen(monkeypatch)

        await runner._launch_detached_restart_command()

        argv, _ = calls[0]
        # Each component appears as its own argv entry; no quoting at all.
        assert "/usr/bin/python3" in argv
        assert "/opt/hermes/cli.py" in argv
        assert "--profile=prod" in argv
        # The old code did ``shlex.join(hermes_cmd) + " gateway restart"``
        # which would smush these three into a single shell-joined entry
        # like ``"/usr/bin/python3 /opt/hermes/cli.py --profile=prod"``.
        # That string is what we must NEVER see in the new argv.
        shell_joined = "/usr/bin/python3 /opt/hermes/cli.py --profile=prod"
        assert shell_joined not in argv, (
            f"shell-quoted hermes_cmd reappeared in argv: {argv!r}"
        )


# ---------------------------------------------------------------------------
# Watcher-script content tests
# ---------------------------------------------------------------------------


class TestWatcherScriptContent:
    """The embedded watcher script must preserve the wait-for-old-pid
    behaviour and the new ``start_new_session=True`` re-detach."""

    @pytest.mark.asyncio
    async def test_script_polls_old_pid_with_os_kill(self, monkeypatch):
        runner, _ = make_restart_runner()
        _patch_resolve_and_pid(monkeypatch)
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        calls = _capture_popen(monkeypatch)

        await runner._launch_detached_restart_command()

        script = _extract_watcher_script(calls[0][0])
        assert "os.kill(pid, 0)" in script
        assert "ProcessLookupError" in script
        # The 120 s bound matches the Windows branch — keeps the watcher
        # from running forever if the old gateway never exits.
        assert "120" in script

    @pytest.mark.asyncio
    async def test_script_uses_start_new_session_true_for_respawn(self, monkeypatch):
        """The whole point of #29603: the watcher's child must be in
        its own session so SIGHUP from the watcher exit doesn't reach
        it."""
        runner, _ = make_restart_runner()
        _patch_resolve_and_pid(monkeypatch)
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        calls = _capture_popen(monkeypatch)

        await runner._launch_detached_restart_command()

        script = _extract_watcher_script(calls[0][0])
        assert "start_new_session=True" in script
        # Belt-and-suspenders: also closes the parent's fds so the new
        # gateway can't accidentally inherit a writable stdio handle.
        assert "close_fds=True" in script

    @pytest.mark.asyncio
    async def test_script_routes_child_stdio_to_devnull(self, monkeypatch):
        runner, _ = make_restart_runner()
        _patch_resolve_and_pid(monkeypatch)
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        calls = _capture_popen(monkeypatch)

        await runner._launch_detached_restart_command()

        script = _extract_watcher_script(calls[0][0])
        assert "subprocess.DEVNULL" in script
        assert "stdout=subprocess.DEVNULL" in script
        assert "stderr=subprocess.DEVNULL" in script

    @pytest.mark.asyncio
    async def test_script_is_byte_compilable(self, monkeypatch):
        """The watcher is interpreted by a fresh ``python -c``, so a
        SyntaxError in the embedded script would silently kill /restart
        in production. Compile it here to catch that at test time."""
        runner, _ = make_restart_runner()
        _patch_resolve_and_pid(monkeypatch)
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        calls = _capture_popen(monkeypatch)

        await runner._launch_detached_restart_command()

        script = _extract_watcher_script(calls[0][0])
        compile(script, "<watcher>", "exec")


# ---------------------------------------------------------------------------
# End-to-end functional test
# ---------------------------------------------------------------------------


class TestWatcherRespawnReallyDetaches:
    """Run the actual embedded watcher script in a real subprocess and
    confirm its child ends up in a brand-new session — the property
    #29603 was missing."""

    def _build_e2e_script(
        self,
        *,
        watcher_script: str,
        target_pid: int,
        sentinel_path: Path,
        python: str,
    ) -> list[str]:
        """Wrap the embedded watcher so its "respawn" is a tiny Python
        program that records its session id (sid), pgid, pid, and ppid
        into ``sentinel_path`` — that's how we'll prove the new session.
        """
        child_program = textwrap.dedent(
            f"""
            import os
            import time

            sid = os.getsid(0)
            pgid = os.getpgrp()
            pid = os.getpid()
            ppid = os.getppid()
            with open({str(sentinel_path)!r}, "w") as f:
                f.write(f"sid={{sid}}\\npgid={{pgid}}\\npid={{pid}}\\nppid={{ppid}}\\n")
            # Stay alive briefly so the watcher exits first, exercising
            # the SIGHUP path #29603 cared about.
            time.sleep(1.5)
            """
        ).strip()
        return [
            python,
            "-c",
            watcher_script,
            str(target_pid),
            python,
            "-c",
            child_program,
        ]

    @pytest.mark.asyncio
    async def test_watcher_child_lands_in_new_session(self, monkeypatch, tmp_path):
        runner, _ = make_restart_runner()
        _patch_resolve_and_pid(monkeypatch)
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        calls = _capture_popen(monkeypatch)

        await runner._launch_detached_restart_command()
        watcher_script = _extract_watcher_script(calls[0][0])

        # Restore the real ``subprocess.Popen`` — the rest of this test
        # needs to spawn the watcher (and a fake old gateway) for real.
        monkeypatch.undo()

        sentinel = tmp_path / "child.txt"

        # Real "old gateway" so the watcher's wait loop has a PID to
        # poll. We deliberately reap it BEFORE letting the watcher run
        # so ``os.kill(old_pid, 0)`` returns ESRCH right away — a
        # zombie process still answers signal 0 with success, which
        # would stall the wait loop for the full 120 s deadline.
        old_gateway = subprocess.Popen(
            [sys.executable, "-c", "raise SystemExit(0)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        old_pid = old_gateway.pid
        old_gateway.wait(timeout=5.0)

        watcher_argv = self._build_e2e_script(
            watcher_script=watcher_script,
            target_pid=old_pid,
            sentinel_path=sentinel,
            python=sys.executable,
        )

        # The watcher itself runs in a brand-new session — exactly how
        # production invokes it via the top-level Popen.
        watcher = subprocess.Popen(
            watcher_argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        data: dict[str, str] = {}
        try:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not sentinel.exists():
                time.sleep(0.05)
            assert sentinel.exists(), "watcher never spawned the child"

            data = dict(
                line.split("=", 1) for line in sentinel.read_text().splitlines() if "=" in line
            )
            child_sid = int(data["sid"])
            child_pgid = int(data["pgid"])
            child_pid = int(data["pid"])

            # The fix's invariant: the spawned child is the leader of
            # its own session, so when the watcher's own session is
            # torn down SIGHUP does NOT reach the new gateway.
            assert child_sid == child_pid, (
                f"child should be session leader (sid={child_sid}, pid={child_pid}) — "
                "start_new_session=True did not take effect"
            )
            assert child_pgid == child_pid, (
                f"child should be process group leader (pgid={child_pgid}, pid={child_pid})"
            )
            assert child_sid != os.getsid(0), (
                "child should NOT share a session with the test runner"
            )
        finally:
            if watcher.poll() is None:
                watcher.terminate()
                try:
                    watcher.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    watcher.kill()
                    watcher.wait(timeout=2.0)
            # The watcher's child is now an orphan adopted by init;
            # kill it directly via the PID we captured.
            child_pid_str = data.get("pid")
            if child_pid_str:
                try:
                    os.kill(int(child_pid_str), 9)
                except (ProcessLookupError, OSError):
                    pass
