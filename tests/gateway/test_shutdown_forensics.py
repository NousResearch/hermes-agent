"""Tests for gateway.shutdown_forensics — fast snapshot + async diag spawn."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gateway import shutdown_forensics as sf


# ---------------------------------------------------------------------------
# _signal_name
# ---------------------------------------------------------------------------

class TestSignalName:

    def test_unknown_int_returns_signal_num_token(self):
        # Pick an integer extremely unlikely to ever be a real signal alias
        assert sf._signal_name(9999) == "signal#9999"


# ---------------------------------------------------------------------------
# snapshot_shutdown_context
# ---------------------------------------------------------------------------

class TestSnapshotShutdownContext:

    def test_handles_none_signal(self):
        ctx = sf.snapshot_shutdown_context(None)
        assert ctx["signal"] == "UNKNOWN"
        assert ctx["signal_num"] is None

    def test_includes_timestamps(self):
        before = time.time()
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        after = time.time()
        assert before <= ctx["ts"] <= after
        assert isinstance(ctx["ts_monotonic"], float)


    def test_under_systemd_false_without_invocation_id_and_normal_ppid(
        self, monkeypatch
    ):
        monkeypatch.delenv("INVOCATION_ID", raising=False)
        # We can't actually change ppid; skip if we happen to be reaped
        # by init (e.g. running under tini).
        if os.getppid() == 1:
            pytest.skip("test process is reaped by init")
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        assert ctx["under_systemd"] is False


    def test_detects_takeover_marker_for_self(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        marker = tmp_path / ".gateway-takeover.json"
        marker.write_text(
            f'{{"target_pid": {os.getpid()}, "replacer_pid": 99999}}',
            encoding="utf-8",
        )
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        assert "takeover_marker" in ctx
        assert ctx["takeover_marker_for_self"] is True


# ---------------------------------------------------------------------------
# format_context_for_log / context_as_json
# ---------------------------------------------------------------------------

class TestFormatters:


    def test_context_as_json_handles_unserialisable_values(self):
        ctx = {"signal": "SIGTERM", "weird": object()}
        payload = sf.context_as_json(ctx)
        # default=str means objects get repr'd, JSON stays valid
        decoded = json.loads(payload)
        assert decoded["signal"] == "SIGTERM"
        assert "weird" in decoded


# ---------------------------------------------------------------------------
# spawn_async_diagnostic
# ---------------------------------------------------------------------------

class TestResolveAncestorChain:
    """The chain must be resolved in the LIVE process, not by the detached diagnostic.

    The gateway exits the moment the handler returns, so a subprocess that walks
    `ps -p <gateway pid>` finds a dead process and records nothing — the one field that
    answers "who killed the gateway" would be empty exactly when it matters.
    """

    def test_chain_starts_at_self_and_reaches_an_ancestor(self):
        chain = sf.resolve_ancestor_chain(os.getpid())
        assert chain, "empty ancestor chain"
        assert chain[0]["pid"] == os.getpid()
        assert len(chain) >= 2, f"chain did not include a parent: {chain}"
        assert chain[1]["pid"] == os.getppid()

    def test_chain_is_depth_bounded(self):
        assert len(sf.resolve_ancestor_chain(os.getpid(), max_depth=2)) <= 2

    def test_chain_survives_dead_target(self):
        """A PID that exits before the walk must not raise — it degrades to a stub entry."""
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        chain = sf.resolve_ancestor_chain(proc.pid)
        assert isinstance(chain, list)  # no exception; contents may be a bare {"pid": N}

    def test_chain_text_embedded_in_diagnostic_survives_process_exit(self, tmp_path):
        """E2E-shaped: the emitting process dies immediately, log must still name the chain."""
        log_path = tmp_path / "diag-chain.log"
        code = (
            "import os, sys, time\n"
            f"sys.path.insert(0, {str(Path(sf.__file__).parents[1])!r})\n"
            "from pathlib import Path\n"
            "from gateway.shutdown_forensics import spawn_async_diagnostic\n"
            f"spawn_async_diagnostic(Path({str(log_path)!r}), 'SIGTERM', timeout_seconds=8.0)\n"
            "sys.exit(0)\n"  # die instantly — the diagnostic child is now orphaned
        )
        emitter = subprocess.Popen([sys.executable, "-c", code])
        emitter.wait(timeout=30)
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if log_path.exists() and b"=== end ===" in log_path.read_bytes():
                break
            time.sleep(0.1)
        contents = log_path.read_text(encoding="utf-8", errors="replace")
        chain_section = contents.split("--- parent chain of self ---")[-1].split("--- loadavg")[0]
        rows = [line for line in chain_section.splitlines() if line.strip()]
        assert rows, f"parent chain empty after emitter exit: {contents!r}"
        # The dead emitter's own PID must still appear — proof the chain was captured pre-exit.
        assert f"pid={emitter.pid}" in chain_section, chain_section


    def test_chain_persists_names_never_argv(self):
        """The chain is written to a world-readable log; argv routinely carries secrets.

        pid + executable name answers "who killed us" without persisting a connection string
        or token that happened to sit on a parent's command line (cf. upstream #59929).
        """
        chain = [
            {"pid": 10, "ppid": 11, "name": "python",
             "cmdline": "python app.py --token=SUPERSECRET123 --db=postgres://u:pw@h/d"},
            {"pid": 11, "ppid": 1, "name": "bash", "cmdline": "bash deploy.sh --key=AKIAXXXX"},
        ]
        out = sf._format_ancestor_chain(chain)
        assert "SUPERSECRET123" not in out
        assert "postgres://" not in out
        assert "AKIAXXXX" not in out
        assert len(out.splitlines()) == 2
        assert "pid=10 ppid=11 python" in out

    def test_multiline_name_stays_one_line_per_ancestor(self):
        """Control chars must not split one ancestor across lines (unparseable section)."""
        chain = [
            {"pid": 10, "ppid": 11, "name": "weird\nname\there"},
            {"pid": 11, "ppid": 1, "name": "bash"},
        ]
        out = sf._format_ancestor_chain(chain)
        assert len(out.splitlines()) == 2, f"expected 1 line per ancestor, got:\n{out}"

    def test_falls_back_to_placeholder(self):
        assert "bash" in sf._format_ancestor_chain([{"pid": 5, "ppid": 1, "name": "bash"}])
        assert sf._format_ancestor_chain([]) == "(chain unavailable)"


class TestProcSummary:
    """`_proc_summary` must name the parent on every platform, not just Linux.

    A gateway killed by SIGTERM logs `parent_cmdline` as the primary attribution field; when
    it degrades to '(unknown)' the shutdown log cannot answer "who killed it".
    """

    def test_reports_cmdline_and_name_for_live_process(self):
        summary = sf._proc_summary(os.getpid())
        assert summary["pid"] == os.getpid()
        # The identifying fields must be present on Linux (/proc) AND macOS/BSD (psutil).
        assert summary.get("cmdline"), f"no cmdline captured: {summary}"
        assert "python" in summary["cmdline"].lower() or "pytest" in summary["cmdline"].lower()
        assert summary.get("ppid") == os.getppid()

    def test_missing_process_degrades_without_raising(self):
        # PID 0 is the guard path; a never-allocated high PID exercises the lookup failure.
        assert sf._proc_summary(0) == {"pid": 0}
        summary = sf._proc_summary(2**22 - 1)
        assert summary["pid"] == 2**22 - 1  # fields absent, no exception

    def test_psutil_fallback_used_when_proc_absent(self, monkeypatch):
        """Simulate a non-Linux host: /proc reads fail, psutil must still fill the fields."""
        monkeypatch.setattr(sf, "_read_proc_field", lambda *a, **k: None)
        monkeypatch.setattr(sf.Path, "read_bytes", lambda self: (_ for _ in ()).throw(OSError()))
        summary = sf._proc_summary(os.getpid())
        assert summary.get("cmdline"), "psutil fallback did not populate cmdline"
        assert summary.get("ppid") == os.getppid()

    def test_snapshot_context_names_the_parent(self):
        """End-to-end: the formatted log line carries a real parent cmdline, not '(unknown)'."""
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        line = sf.format_context_for_log(ctx)
        assert "parent_cmdline='(unknown)'" not in line, line


# ---------------------------------------------------------------------------
# _diagnostic_timeout_argv
# ---------------------------------------------------------------------------

class TestDiagnosticTimeoutArgv:
    """Stock macOS has no `timeout` binary; hardcoding it made Popen raise FileNotFoundError
    and the diagnostic silently produced a 0-byte log."""

    def test_uses_timeout_when_present(self, monkeypatch):
        monkeypatch.setattr(sf.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "timeout" else None)
        assert sf._diagnostic_timeout_argv(5.0) == ["timeout", "5"]

    def test_falls_back_to_gtimeout(self, monkeypatch):
        monkeypatch.setattr(sf.shutil, "which", lambda name: "/opt/homebrew/bin/gtimeout" if name == "gtimeout" else None)
        assert sf._diagnostic_timeout_argv(5.0) == ["gtimeout", "5"]

    def test_returns_empty_when_no_timeout_binary(self, monkeypatch):
        monkeypatch.setattr(sf.shutil, "which", lambda name: None)
        assert sf._diagnostic_timeout_argv(5.0) == []


# ---------------------------------------------------------------------------
# spawn_async_diagnostic
# ---------------------------------------------------------------------------

class TestSpawnAsyncDiagnostic:
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only diagnostic")
    def test_spawns_subprocess_and_writes_output(self, tmp_path):
        log_path = tmp_path / "diag.log"
        pid = sf.spawn_async_diagnostic(log_path, "SIGTERM", timeout_seconds=3.0)
        assert pid is not None and pid > 0

        # Wait briefly for the subprocess to write — bounded by its own timeout.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if log_path.exists() and log_path.stat().st_size > 0:
                # Wait a touch longer for the script to finish writing
                time.sleep(0.2)
                break
            time.sleep(0.1)

        # Reap the subprocess so it doesn't show up as a zombie.
        try:
            os.waitpid(pid, 0)
        except (ChildProcessError, OSError):
            pass

        assert log_path.exists()
        contents = log_path.read_text(encoding="utf-8", errors="replace")
        assert "shutdown diagnostic" in contents
        assert "SIGTERM" in contents

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only diagnostic")
    def test_spawns_without_timeout_binary(self, tmp_path, monkeypatch):
        """No `timeout`/`gtimeout` on PATH (stock macOS) must still produce a diagnostic."""
        monkeypatch.setattr(sf.shutil, "which", lambda name: None)
        log_path = tmp_path / "diag-no-timeout.log"
        pid = sf.spawn_async_diagnostic(log_path, "SIGTERM", timeout_seconds=3.0)
        assert pid is not None and pid > 0
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if log_path.exists() and b"=== end ===" in log_path.read_bytes():
                break
            time.sleep(0.1)
        with contextlib.suppress(ChildProcessError, OSError):
            os.waitpid(pid, 0)
        contents = log_path.read_text(encoding="utf-8", errors="replace")
        assert "=== end ===" in contents, f"diagnostic did not complete: {contents!r}"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only diagnostic")
    def test_captures_real_process_rows_and_parent_chain(self, tmp_path):
        """The probes must yield DATA, not just section headers.

        `ps auxf --sort=-pcpu` is a usage error on BSD and `pstree` does not exist there, so
        the pre-fix script emitted empty sections on macOS — a diagnostic that looks like it
        never ran.
        """
        log_path = tmp_path / "diag-content.log"
        pid = sf.spawn_async_diagnostic(log_path, "SIGTERM", timeout_seconds=8.0)
        assert pid is not None
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if log_path.exists() and b"=== end ===" in log_path.read_bytes():
                break
            time.sleep(0.1)
        with contextlib.suppress(ChildProcessError, OSError):
            os.waitpid(pid, 0)
        contents = log_path.read_text(encoding="utf-8", errors="replace")
        sections = {}
        current = None
        for line in contents.splitlines():
            if line.startswith("--- ") and line.endswith(" ---"):
                current = line.strip("- ")
                sections[current] = []
            elif current and line.strip() and not line.startswith("==="):
                sections[current].append(line)
        ps_rows = next((v for k, v in sections.items() if k.startswith("ps ")), [])
        assert len(ps_rows) > 1, f"ps section empty on {sys.platform}: {contents!r}"
        chain = sections.get("parent chain of self", [])
        assert chain, f"parent chain section empty on {sys.platform}: {contents!r}"
        assert str(os.getpid()) in "\n".join(chain)


# ---------------------------------------------------------------------------
# parse_systemd_duration_to_us
# ---------------------------------------------------------------------------

class TestParseSystemdDuration:
    def test_seconds(self):
        assert sf.parse_systemd_duration_to_us("90s") == 90 * 1_000_000

    def test_minutes(self):
        assert sf.parse_systemd_duration_to_us("3min") == 180 * 1_000_000


# ---------------------------------------------------------------------------
# check_systemd_timing_alignment
# ---------------------------------------------------------------------------

class TestCheckSystemdTimingAlignment:

    def test_returns_none_when_unit_undeterminable(self, monkeypatch):
        monkeypatch.setenv("INVOCATION_ID", "abc")
        # /proc/self/cgroup likely doesn't end in .service for the test runner
        result = sf.check_systemd_timing_alignment(180.0)
        # Either None (we couldn't find a unit) or a dict with mismatch info
        # for whatever unit pytest IS in.  Both are valid; we just ensure
        # the function doesn't raise.
        assert result is None or isinstance(result, dict)
