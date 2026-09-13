"""Tests for ``hermes_cli/nssm_truth.py``.

Focus areas, in the order the review comments on the original PR asked for them:

* the dotted-module Windows argv shape (``pythonw.exe -m hermes_cli.main gateway run``)
  must be recognised, not just the script-path shape;
* liveness must come from the canonical ``gateway.status`` probe, including its
  runtime-lock, PID-reuse and profile-matching guards -- not from hand-rolled lock JSON;
* the module must be importable where ``ctypes.WinDLL`` does not exist;
* home resolution must be profile-aware, with no contributor-specific fallback.
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest

from hermes_cli import nssm_truth


class _Result:
    """Stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _RecordingRunner:
    """Fake ``subprocess.run`` that records argv and replays a canned result or error."""

    def __init__(self, result=None, error=None):
        self.calls: list[tuple[list[str], dict]] = []
        self._result = result
        self._error = error

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        if self._error is not None:
            raise self._error
        return self._result


# ---------------------------------------------------------------------------
# argv recognition (the dotted-module regression)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        # The form hermes_cli/gateway_windows.py renders. The original helper rejected
        # it because it only looked for the "hermes_cli/main" path fragment.
        [
            r"C:\Users\me\AppData\Local\hermes\hermes-agent\venv\Scripts\pythonw.exe",
            "-m",
            "hermes_cli.main",
            "gateway",
            "run",
        ],
        # Script-path form, which is what a live gateway.pid on this host records.
        [r"C:\Users\me\AppData\Local\hermes\hermes-agent\hermes_cli\main.py", "gateway", "run"],
        # Bare `hermes gateway` defaults to `run`.
        ["hermes", "gateway"],
        [r"C:\Users\me\scoop\shims\hermes.exe", "gateway", "run"],
        # `restart` hosts the runtime in-process when no service manager is used.
        ["python", "-m", "hermes_cli.main", "gateway", "restart"],
        # --profile / -p are stripped anywhere in argv (including before the subcommand).
        ["python", "-m", "hermes_cli.main", "--profile", "work", "gateway", "run"],
        ["python", "-m", "hermes_cli.main", "-p", "work", "gateway", "run"],
        [r"C:\x\hermes.exe", "--profile=work", "gateway", "run"],
    ],
)
def test_recognises_real_gateway_launch_shapes(argv):
    assert nssm_truth.looks_like_hermes_gateway_argv(argv) is True


@pytest.mark.parametrize(
    "argv",
    [
        None,
        [],
        ["python", "-m", "tui_gateway"],
        ["python", "-m", "hermes_cli.main", "gateway", "status"],
        ["nssm", "status", "HermesGateway"],
        ["python", "-m", "hermes_cli.main", "--help"],
        ["python", "some_other_script.py"],
    ],
)
def test_rejects_non_gateway_argv(argv):
    assert nssm_truth.looks_like_hermes_gateway_argv(argv) is False


def test_argv_predicate_defers_to_the_canonical_gateway_helper(monkeypatch):
    """Guard against the fix drifting into a second, hand-rolled heuristic."""
    seen: list[str] = []

    def _spy(command):
        seen.append(command)
        return True

    monkeypatch.setattr(nssm_truth, "looks_like_gateway_runtime_command_line", _spy)

    assert nssm_truth.looks_like_hermes_gateway_argv(["python", "-m", "hermes_cli.main", "gateway", "run"])
    assert seen == ["python -m hermes_cli.main gateway run"]


# ---------------------------------------------------------------------------
# portability
# ---------------------------------------------------------------------------


def test_import_does_not_require_ctypes_windll(monkeypatch):
    """The original helper called ``ctypes.WinDLL`` at import-adjacent time, which raises
    AttributeError on Linux."""
    import ctypes

    monkeypatch.delattr(ctypes, "WinDLL", raising=False)
    assert not hasattr(ctypes, "WinDLL")
    # Must not raise.
    importlib.reload(nssm_truth)


def test_query_skips_entirely_off_windows_without_spawning():
    runner = _RecordingRunner(_Result(stdout="SERVICE_RUNNING"))

    state = nssm_truth.query_nssm_state("HermesGateway", executable="nssm", runner=runner, is_windows=False)

    assert state is None
    assert runner.calls == []


# ---------------------------------------------------------------------------
# nssm status parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stdout, expected",
    [
        ("SERVICE_RUNNING\n", nssm_truth.SERVICE_RUNNING),
        ("SERVICE_PAUSED\r\n", nssm_truth.SERVICE_PAUSED),
        ("SERVICE_STOPPED\n", nssm_truth.SERVICE_STOPPED),
        ("SERVICE_START_PENDING\n", "SERVICE_START_PENDING"),
    ],
)
def test_query_parses_state_tokens(stdout, expected):
    runner = _RecordingRunner(_Result(stdout=stdout))

    assert nssm_truth.query_nssm_state("S", executable="nssm", runner=runner, is_windows=True) == expected


def test_query_reports_missing_service_as_not_installed():
    """NSSM 2.24 exits 0 and writes this to stdout, so the exit code proves nothing."""
    runner = _RecordingRunner(
        _Result(
            stdout=(
                "Can't open service!\n"
                "OpenService(): The specified service does not exist as an installed service.\n"
            )
        )
    )

    assert (
        nssm_truth.query_nssm_state("S", executable="nssm", runner=runner, is_windows=True)
        == nssm_truth.NOT_INSTALLED
    )


def test_query_returns_none_for_unrecognised_output():
    runner = _RecordingRunner(_Result(stdout="something else entirely\n"))

    assert nssm_truth.query_nssm_state("S", executable="nssm", runner=runner, is_windows=True) is None


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("nssm.exe"), subprocess.TimeoutExpired(cmd="nssm", timeout=10), OSError("spawn failed")],
)
def test_query_returns_none_when_the_spawn_fails(error):
    runner = _RecordingRunner(error=error)

    assert nssm_truth.query_nssm_state("S", executable="nssm", runner=runner, is_windows=True) is None


def test_query_returns_none_without_an_executable(monkeypatch):
    monkeypatch.delenv(nssm_truth.NSSM_EXECUTABLE_ENV, raising=False)
    monkeypatch.setattr(nssm_truth.shutil, "which", lambda _name: None)
    monkeypatch.setattr(nssm_truth, "_NSSM_FALLBACK_BIN_DIRS", ())

    assert nssm_truth.query_nssm_state("S", runner=_RecordingRunner(_Result()), is_windows=True) is None


# ---------------------------------------------------------------------------
# nssm executable resolution
# ---------------------------------------------------------------------------


def test_executable_resolution_prefers_the_env_override(tmp_path, monkeypatch):
    override = tmp_path / "nssm.exe"
    override.write_bytes(b"")
    monkeypatch.setenv(nssm_truth.NSSM_EXECUTABLE_ENV, str(override))
    monkeypatch.setattr(nssm_truth.shutil, "which", lambda _name: "ignored-from-path")

    assert nssm_truth.resolve_nssm_executable() == str(override)


def test_executable_resolution_rejects_a_bogus_override_instead_of_falling_back(monkeypatch):
    monkeypatch.setenv(nssm_truth.NSSM_EXECUTABLE_ENV, r"C:\nope\nssm.exe")
    monkeypatch.setattr(nssm_truth.shutil, "which", lambda _name: "from-path")

    assert nssm_truth.resolve_nssm_executable() is None


def test_executable_resolution_falls_back_to_path(monkeypatch):
    monkeypatch.delenv(nssm_truth.NSSM_EXECUTABLE_ENV, raising=False)
    monkeypatch.setattr(nssm_truth.shutil, "which", lambda name: f"/usr/local/bin/{name}")

    assert nssm_truth.resolve_nssm_executable() == "/usr/local/bin/nssm"


def test_no_contributor_specific_home_fallback():
    """The original hard-coded ``C:\\Users\\<contributor>``; home must come from Hermes."""
    source = Path(nssm_truth.__file__).read_text(encoding="utf-8")

    assert "bbask" not in source


# ---------------------------------------------------------------------------
# is_hermes_service_actually_running
# ---------------------------------------------------------------------------


def _patch_probe(monkeypatch, pid):
    calls: list[tuple[tuple, dict]] = []

    def _probe(*args, **kwargs):
        calls.append((args, kwargs))
        return pid

    monkeypatch.setattr(nssm_truth, "get_running_pid", _probe)
    return calls


def test_live_gateway_wins_and_never_consults_nssm(monkeypatch):
    _patch_probe(monkeypatch, 4321)
    nssm_calls: list[str] = []
    monkeypatch.setattr(nssm_truth, "query_nssm_state", lambda name: nssm_calls.append(name))

    running, detail = nssm_truth.is_hermes_service_actually_running("HermesGateway")

    assert running is True
    assert "4321" in detail
    assert nssm_calls == []


def test_unscoped_probe_is_read_only_by_default(monkeypatch):
    calls = _patch_probe(monkeypatch, None)
    monkeypatch.setattr(nssm_truth, "query_nssm_state", lambda name: None)

    nssm_truth.is_hermes_service_actually_running("HermesGateway")

    assert calls == [((), {"cleanup_stale": False})]


def test_explicit_home_is_a_scoped_probe_of_that_home(monkeypatch, tmp_path):
    calls = _patch_probe(monkeypatch, None)
    monkeypatch.setattr(nssm_truth, "query_nssm_state", lambda name: None)
    home = tmp_path / "profiles" / "work"

    nssm_truth.is_hermes_service_actually_running("HermesGateway", home)

    assert calls == [((home / "gateway.pid",), {"cleanup_stale": False})]


def test_cleanup_stale_is_opt_in(monkeypatch):
    calls = _patch_probe(monkeypatch, None)
    monkeypatch.setattr(nssm_truth, "query_nssm_state", lambda name: None)

    nssm_truth.is_hermes_service_actually_running("HermesGateway", cleanup_stale=True)

    assert calls == [((), {"cleanup_stale": True})]


@pytest.mark.parametrize(
    "state, expected_running, expected_fragment",
    [
        (nssm_truth.SERVICE_RUNNING, True, "trusting the service manager"),
        (nssm_truth.SERVICE_PAUSED, False, "nssm reports SERVICE_PAUSED"),
        (nssm_truth.SERVICE_STOPPED, False, "nssm reports SERVICE_STOPPED"),
        (nssm_truth.NOT_INSTALLED, False, "has no service named 'HermesGateway' installed"),
        (None, False, "could not report a state"),
    ],
)
def test_nssm_fallback_verdicts(monkeypatch, state, expected_running, expected_fragment):
    _patch_probe(monkeypatch, None)
    monkeypatch.setattr(nssm_truth, "query_nssm_state", lambda name: state)

    running, detail = nssm_truth.is_hermes_service_actually_running("HermesGateway")

    assert running is expected_running
    assert expected_fragment in detail
    if state == nssm_truth.SERVICE_RUNNING:
        # A service claiming RUNNING with no live process must say so, not imply a
        # verified gateway.
        assert "no live gateway process" in detail


def test_nssm_reader_is_injectable(monkeypatch):
    _patch_probe(monkeypatch, None)

    running, _detail = nssm_truth.is_hermes_service_actually_running(
        "HermesGateway", nssm_reader=lambda name: nssm_truth.SERVICE_RUNNING
    )

    assert running is True


def test_nssm_fallback_can_be_disabled(monkeypatch):
    _patch_probe(monkeypatch, None)
    monkeypatch.setattr(nssm_truth, "query_nssm_state", lambda name: pytest.fail("must not consult nssm"))

    running, detail = nssm_truth.is_hermes_service_actually_running("HermesGateway", fallback_to_nssm=False)

    assert running is False
    assert "NSSM check disabled" in detail


def test_pid_probe_can_be_disabled(monkeypatch):
    monkeypatch.setattr(nssm_truth, "get_running_pid", lambda *a, **k: pytest.fail("must not probe"))
    monkeypatch.setattr(nssm_truth, "query_nssm_state", lambda name: nssm_truth.SERVICE_PAUSED)

    running, _detail = nssm_truth.is_hermes_service_actually_running("HermesGateway", check_pid_file=False)

    assert running is False


def test_convenience_boolean(monkeypatch):
    monkeypatch.setattr(nssm_truth, "get_running_pid", lambda *a, **k: 99)

    assert nssm_truth.is_hermes_gateway_running() is True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _emit_into(lines: list[str]):
    return lines.append


def test_main_exit_codes():
    assert nssm_truth.main([], probe=lambda name: (True, "up"), emit=_emit_into([])) == 0
    assert nssm_truth.main([], probe=lambda name: (False, "down"), emit=_emit_into([])) == 1


def test_main_renders_the_detail_line():
    lines: list[str] = []

    status = nssm_truth.main(["--service", "OtherGateway"], probe=lambda name: (False, f"no {name}"), emit=_emit_into(lines))

    assert status == 1
    assert lines == ["is_hermes_service_actually_running('OtherGateway'): False (no OtherGateway)"]


def test_main_usage_errors():
    lines: list[str] = []
    assert nssm_truth.main(["--service"], emit=_emit_into(lines)) == 2
    assert "needs a value" in lines[0]

    lines = []
    assert nssm_truth.main(["--bogus"], emit=_emit_into(lines)) == 2
    assert "unrecognised argument" in lines[0]


def test_main_help():
    lines: list[str] = []

    assert nssm_truth.main(["--help"], emit=_emit_into(lines)) == 0
    assert lines[0].startswith("usage: python -m hermes_cli.nssm_truth")
