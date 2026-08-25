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

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="live Windows venv-holder E2E"
)

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


class TestClassification:
    def test_pausable_exemption_sees_long_path_gateway(self):
        """#78089 follow-through: `_leftover_pausable_gateway_pids` must
        classify the long-path gateway as pausable (not None)."""
        from hermes_cli.update_cmd import _leftover_pausable_gateway_pids

        padding = os.path.join("C:\\", "Users", "y" * 90, ".hermes-runtime")
        proc = _spawn([padding, "-m", "hermes_cli.main", "gateway", "run"])
        try:
            matches = [m for m in _detect() if m[0] == proc.pid]
            assert matches, "gateway not detected"
            pids = _leftover_pausable_gateway_pids(matches)
            assert pids == [proc.pid], (
                f"pausable exemption failed for long-path gateway: {pids}"
            )
        finally:
            _kill(proc)

    def test_serve_backend_not_classified_pausable(self):
        """#81774 premise probe: a serve backend is NOT pausable today —
        pinning current behavior so the consolidation change is visible."""
        from hermes_cli.update_cmd import _leftover_pausable_gateway_pids

        proc = _spawn(["-m", "hermes_cli.main", "serve"])
        try:
            matches = [m for m in _detect() if m[0] == proc.pid]
            assert matches, "serve backend not detected"
            assert _leftover_pausable_gateway_pids(matches) is None
        finally:
            _kill(proc)


class TestHolderMessage:
    """#90778 — the refusal message must name holders accurately."""

    def test_dashboard_not_labeled_desktop_backend(self):
        from hermes_cli.update_cmd import _format_venv_python_holders_message

        proc = _spawn(["-m", "hermes_cli.main", "dashboard"])
        try:
            matches = [m for m in _detect() if m[0] == proc.pid]
            assert matches, "dashboard process not detected"
            message = _format_venv_python_holders_message(matches)
            assert "close the desktop app" not in message.lower(), (
                "standalone `hermes dashboard` mislabeled as the Desktop "
                f"backend (#90778):\n{message}"
            )
        finally:
            _kill(proc)

    def test_substring_subcommand_not_mislabeled(self):
        """`--preserve-cache` contains 'serve'; the classifier must not
        label an unrelated subcommand as the Desktop backend (#90778)."""
        from hermes_cli.update_cmd import _format_venv_python_holders_message

        proc = _spawn(["-m", "hermes_cli.main", "kanban", "--preserve-cache"])
        try:
            matches = [m for m in _detect() if m[0] == proc.pid]
            assert matches, "kanban process not detected"
            message = _format_venv_python_holders_message(matches)
            assert "close the desktop app" not in message.lower(), (
                f"substring match mislabeled `--preserve-cache` (#90778):\n{message}"
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
