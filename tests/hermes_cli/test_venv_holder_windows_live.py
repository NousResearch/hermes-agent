"""LIVE Windows E2E for the venv-holder preflight (fleet-update #91277).

Runs ONLY on a real Windows host (the on-demand ``windows-venv-e2e.yml``
lane). Spawns REAL processes with realistic Hermes argv shapes and drives
the actual detection / classification / exemption code against the live
process table — no mocked psutil, no faked cmdlines.

Each test documents which cluster issue it probes. Tests written BEFORE
the consolidation fix intentionally pin the CORRECT behavior, so on
unfixed main the buggy ones fail — that failure on the Windows runner is
the empirical premise-check for each issue:

  #90778 — holder message mislabels `hermes dashboard` as the Desktop
           backend, and matches subcommands by substring ("--preserve"
           contains "serve").
  #78089 — pausable-gateway exemption vs. long managed-runtime
           interpreter paths (claimed fixed on main; verified here).
  #87594 — ancestor-exclusion hides the gateway from the scan when the
           updater is spawned BY the gateway (/update path).
  #81774 — serve backends have no pause path (documented behavior probe).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.skipif(sys.platform != "win32", reason="live Windows venv-holder E2E"),
    # ``_spawn`` sleepers carry a "gateway run" argv tail as inert data (the guard's real-gateway
    # spawn check matches it); every child is ``_kill``ed by the test.
    pytest.mark.spawns_gateway_lookalike,
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _spawn(args: list[str], cwd: Path | None = None) -> subprocess.Popen:
    """Spawn a real sleeper process whose argv carries the given tail.

    ``python -c "sleep" <tail...>`` — the tail is inert data to the child
    but fully visible to psutil cmdline scans, which is what the detection
    code classifies on.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)", *args],
        cwd=str(cwd or PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.8)  # let the process table settle
    assert proc.poll() is None, "sleeper died at spawn"
    return proc


def _detect() -> list[tuple[int, str, str]]:
    from hermes_cli.update_cmd import _detect_venv_python_processes

    return _detect_venv_python_processes()


def _kill(*procs: subprocess.Popen) -> None:
    for proc in procs:
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:
            pass


class TestDetection:
    def test_detects_hermes_argv_process(self):
        """Baseline: a live process running `-m hermes_cli.main serve` with
        cwd under the install root is detected as a venv holder."""
        proc = _spawn(["-m", "hermes_cli.main", "serve"])
        try:
            matches = _detect()
            pids = [pid for pid, _, _ in matches]
            assert proc.pid in pids, f"holder scan missed live process: {matches}"
            cmdline = next(c for p, _, c in matches if p == proc.pid)
            # Full cmdline, not a 120-char prefix (#78089 regression guard).
            assert "hermes_cli.main" in cmdline
        finally:
            _kill(proc)

    def test_foreign_python_not_detected(self):
        """A python process with no Hermes argv and cwd OUTSIDE the install
        must not be reported as a holder."""
        import tempfile

        outside = Path(tempfile.mkdtemp())
        proc = _spawn(["totally", "unrelated"], cwd=outside)
        try:
            pids = [pid for pid, _, _ in _detect()]
            assert proc.pid not in pids
        finally:
            _kill(proc)

    def test_long_runtime_path_gateway_detected_with_full_argv(self):
        """#78089: a gateway launched via a long managed-runtime interpreter
        path must surface with its FULL argv so the pausable exemption can
        see `gateway run` past the 120-char mark."""
        # Pad the argv front so `gateway run` sits beyond 120 chars.
        padding = os.path.join("C:\\", "Users", "x" * 90, ".hermes-runtime")
        proc = _spawn([padding, "-m", "hermes_cli.main", "gateway", "run"])
        try:
            matches = _detect()
            cmdline = next((c for p, _, c in matches if p == proc.pid), None)
            assert cmdline is not None, "long-path gateway missed by scan"
            assert "gateway run" in cmdline.lower(), (
                f"argv truncated before `gateway run`: {cmdline!r}"
            )
        finally:
            _kill(proc)


class TestAncestorExclusion:
    """#87594 — when the updater is a CHILD of the gateway (/update path),
    ancestor-exclusion must not hide the gateway from the scan entirely:
    the gateway must still be visible to the pause machinery."""

    def test_gateway_parent_visible_to_child_scan(self, tmp_path):
        # Simulate the /update topology: parent (gateway-argv process) spawns
        # a child python that runs the REAL detection and reports whether it
        # can see its gateway parent. The child's code lives in a FILE so the
        # parent's cmdline stays realistic (a real gateway's argv is clean
        # `... -m hermes_cli.main gateway run`, not a multi-line -c blob).
        child_file = tmp_path / "child_scan.py"
        child_file.write_text(
            "import json, os, sys\n"
            f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
            "from hermes_cli.update_cmd import _detect_venv_python_processes\n"
            "import psutil\n"
            "from gateway.status import looks_like_gateway_command_line\n"
            "# The venv shim makes every spawn a launcher/worker CHAIN, so the\n"
            "# gateway is an ANCESTOR, not necessarily the direct parent —\n"
            "# find it the same way the pause machinery would: by argv.\n"
            "gw = [int(a.pid) for a in psutil.Process().parents()\n"
            "      if looks_like_gateway_command_line(' '.join(a.cmdline() or []))]\n"
            "matches = _detect_venv_python_processes()\n"
            "print(json.dumps({'gateway_ancestors': gw,"
            " 'pids': [p for p, _, _ in matches]}))\n",
            encoding="utf-8",
        )
        parent_oneliner = (
            "import subprocess, sys;"
            f" r = subprocess.run([sys.executable, {str(child_file)!r}],"
            f" capture_output=True, text=True, cwd={str(PROJECT_ROOT)!r});"
            " print(r.stdout.strip());"
            " sys.stderr.write(r.stderr[-500:])"
        )
        # The parent's argv carries `gateway run` so it IS a gateway to any
        # cmdline classifier; it runs the child synchronously.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                parent_oneliner,
                "-m",
                "hermes_cli.main",
                "gateway",
                "run",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=120,
        )
        import json

        line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "{}"
        payload = json.loads(line)
        assert payload, f"child scan produced no output: {result.stderr[-500:]}"
        assert payload["gateway_ancestors"], (
            f"harness broke: no gateway-argv ancestor found: {payload}"
        )
        # The gateway ancestor must be visible to the scan so the pause
        # machinery can stop it (#87594). Blanket ancestor-exclusion hid it.
        visible = set(payload["gateway_ancestors"]) & set(payload["pids"])
        assert visible, (
            "gateway ancestor invisible to venv scan — /update from the "
            f"gateway can never pause it (#87594): {payload}"
        )


class TestUpdaterOwnedBackendDeferral:
    """#98336 — ledger-verified serve/dashboard holders defer to the CLI
    updater's stop/relaunch rungs instead of dead-ending the Desktop
    hand-off (and leaving hermes.exe locked → quarantine os error 32)."""

    @staticmethod
    def _ledger_write(pid: int, purpose: str, spawner_pid: int | None,
                      spawner_create: float | None) -> None:
        import psutil

        from hermes_cli.process_identity import (
            LedgerEntry,
            _append_entry,
            install_id,
        )

        entry = LedgerEntry(
            pid=pid,
            create_time=float(psutil.Process(pid).create_time()),
            purpose=purpose,
            install=install_id(),
            spawner_pid=spawner_pid,
            spawner_create=spawner_create,
            registered_at=time.time(),
            argv="",
        )
        assert _append_entry(entry)

    def test_dead_spawner_serve_is_deferred_live(self, tmp_path, monkeypatch):
        """A REAL serve-argv process, ledger-registered with a provably dead
        spawner, must classify as updater-owned (deferred, not a blocker)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        import psutil

        # A real spawner that has already exited — its (pid, create_time)
        # pair is provably dead.
        spawner = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(0.1)"],
        )
        spawner_create = float(psutil.Process(spawner.pid).create_time())
        spawner.wait(timeout=30)

        backend = _spawn(["-m", "hermes_cli.main", "serve", "--host", "127.0.0.1"])
        try:
            self._ledger_write(backend.pid, "serve", spawner.pid, spawner_create)

            from hermes_cli._scan_venv_blockers import _is_updater_owned_backend

            cmdline = " ".join(psutil.Process(backend.pid).cmdline())
            assert _is_updater_owned_backend(backend.pid, cmdline) is True, (
                "dead-spawner ledger serve must be deferred to the updater "
                "rungs (#98336)"
            )
        finally:
            _kill(backend)

    def test_live_foreign_spawner_serve_still_blocks_live(self, tmp_path, monkeypatch):
        """Same real serve process, but its recorded spawner is a LIVE
        process outside this scan's ancestry — must keep blocking."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        import psutil

        supervisor = _spawn(["fake-supervisor"])
        backend = _spawn(["-m", "hermes_cli.main", "serve", "--host", "127.0.0.1"])
        try:
            self._ledger_write(
                backend.pid,
                "serve",
                supervisor.pid,
                float(psutil.Process(supervisor.pid).create_time()),
            )

            from hermes_cli._scan_venv_blockers import _is_updater_owned_backend

            cmdline = " ".join(psutil.Process(backend.pid).cmdline())
            assert _is_updater_owned_backend(backend.pid, cmdline) is False, (
                "live foreign supervisor would respawn the backend — the "
                "scan must keep refusing"
            )
        finally:
            _kill(supervisor, backend)

    def test_unregistered_serve_still_blocks_live(self, tmp_path, monkeypatch):
        """A serve-argv process with NO ledger entry keeps blocking —
        positive identity only, never argv-shape alone."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        import psutil

        backend = _spawn(["-m", "hermes_cli.main", "serve", "--host", "127.0.0.1"])
        try:
            from hermes_cli._scan_venv_blockers import _is_updater_owned_backend

            cmdline = " ".join(psutil.Process(backend.pid).cmdline())
            assert _is_updater_owned_backend(backend.pid, cmdline) is False
        finally:
            _kill(backend)

    def test_handoff_desktop_ancestor_spawner_is_deferred_live(self, tmp_path, monkeypatch):
        """The #98336 field topology: the recorded spawner is the scan's own
        ANCESTOR (the Desktop performing the hand-off). Run the check in a
        real child process whose parent chain contains the spawner."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        import json as _json

        import psutil

        backend = _spawn(["-m", "hermes_cli.main", "serve", "--host", "127.0.0.1"])

        check_file = tmp_path / "check_owned.py"
        check_file.write_text(
            "import json, sys\n"
            f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
            "import psutil\n"
            "from hermes_cli._scan_venv_blockers import _is_updater_owned_backend\n"
            "pid = int(sys.argv[1])\n"
            "cmdline = ' '.join(psutil.Process(pid).cmdline())\n"
            "print(json.dumps({'owned': _is_updater_owned_backend(pid, cmdline)}))\n",
            encoding="utf-8",
        )
        # Fake-desktop parent: registers the backend in the ledger with
        # ITSELF as spawner, then runs the check in a child — exactly the
        # Desktop → scan-subprocess topology of the update preflight.
        parent_file = tmp_path / "fake_desktop.py"
        parent_file.write_text(
            "import json, os, subprocess, sys, time\n"
            f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
            "import psutil\n"
            "from hermes_cli.process_identity import LedgerEntry, _append_entry, install_id\n"
            "backend_pid = int(sys.argv[1])\n"
            "entry = LedgerEntry(pid=backend_pid,\n"
            "    create_time=float(psutil.Process(backend_pid).create_time()),\n"
            "    purpose='serve', install=install_id(),\n"
            "    spawner_pid=os.getpid(),\n"
            "    spawner_create=float(psutil.Process().create_time()),\n"
            "    registered_at=time.time(), argv='')\n"
            "assert _append_entry(entry)\n"
            f"r = subprocess.run([sys.executable, {str(check_file)!r}, sys.argv[1]],\n"
            f"    capture_output=True, text=True, cwd={str(PROJECT_ROOT)!r})\n"
            "print(r.stdout.strip())\n"
            "sys.stderr.write(r.stderr[-500:])\n",
            encoding="utf-8",
        )
        try:
            result = subprocess.run(
                [sys.executable, str(parent_file), str(backend.pid)],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=120,
                env={**os.environ, "HERMES_HOME": str(tmp_path)},
            )
            line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "{}"
            payload = _json.loads(line)
            assert payload.get("owned") is True, (
                "hand-off Desktop ancestor spawner must defer the backend "
                f"(#98336): stdout={result.stdout!r} stderr={result.stderr[-500:]!r}"
            )
        finally:
            _kill(backend)
