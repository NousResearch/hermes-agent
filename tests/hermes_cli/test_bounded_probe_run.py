"""``bounded_probe_run`` — deadlock-safe capture for fail-open probes (#87134).

On Windows, ``subprocess.run(..., capture_output=True, timeout=N)`` can hang
FOREVER after its timeout fires: run()'s cleanup kills the direct child and
then joins the pipe reader threads with an unbounded ``communicate()``.  A
descendant (``conhost.exe`` under wmic/powershell, ``git.exe`` under a
launcher shim) holding duplicated pipe handles keeps the pipes from EOF and
the join never returns.  ``hermes update`` wedged exactly there inside
``_scan_gateway_pids`` on machines where the full ``Win32_Process`` scan
exceeds its budget.

``bounded_probe_run`` is the shared, generalized form of the fix that
``bounded_git_probe`` already proved for git probes (#68609 / #66037):
explicit ``communicate(timeout)``, tree-kill on failure, bounded 1s drain,
then abandon.

These tests use REAL subprocesses (no mocks) for the semantics: a mock cannot
reproduce pipe-handle inheritance or timeout behavior.  The POSIX-only
descendant-survival cases live in ``test_git_probe_tree_kill.py`` and now
exercise the same code path through the ``bounded_git_probe`` delegation.
"""

import subprocess
import sys
import time

import pytest

from hermes_cli._subprocess_compat import bounded_git_probe, bounded_probe_run

_PY = sys.executable


def test_success_returns_completed_process():
    result = bounded_probe_run([_PY, "-c", "print('ok'); import sys; sys.exit(0)"], timeout=30)
    assert result is not None
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_nonzero_exit_is_returned_not_swallowed():
    """Unlike bounded_git_probe, callers see the real returncode — the gateway
    scan branches on ``returncode != 0`` to trip the wmic→powershell fallback."""
    result = bounded_probe_run([_PY, "-c", "print('partial'); import sys; sys.exit(3)"], timeout=30)
    assert result is not None
    assert result.returncode == 3
    assert result.stdout.strip() == "partial"


def test_spawn_failure_returns_none():
    result = bounded_probe_run(["definitely-not-a-real-binary-87134"], timeout=5)
    assert result is None


def test_timeout_returns_none_within_bounded_time():
    """A child that sleeps past the timeout must produce ``None`` promptly —
    timeout + tree-kill + 1s bounded drain, not an unbounded join."""
    start = time.monotonic()
    result = bounded_probe_run(
        [_PY, "-c", "import time; time.sleep(300)"],
        timeout=1.0,
    )
    elapsed = time.monotonic() - start
    assert result is None
    # 1s timeout + tree-kill + 1s drain + slack.  The pre-fix failure mode is
    # an indefinite hang, so any bound proves the property; keep it loose for
    # slow CI runners.
    assert elapsed < 30


def test_decode_errors_configurable():
    """The process scans pass errors='ignore' (wmic emits system code page);
    undecodable bytes must not raise or None out stdout (#17049 class)."""
    result = bounded_probe_run(
        [_PY, "-c", "import sys; sys.stdout.buffer.write(b'ok\\xff\\xfe')"],
        timeout=30,
        errors="ignore",
    )
    assert result is not None
    assert result.returncode == 0
    assert "ok" in result.stdout


def test_stdin_is_devnull_not_inherited():
    """A probe must never block reading the caller's stdin."""
    result = bounded_probe_run(
        [_PY, "-c", "import sys; print(repr(sys.stdin.read()))"],
        timeout=30,
    )
    assert result is not None
    assert result.returncode == 0
    assert result.stdout.strip() == "''"


def test_bounded_git_probe_delegates_same_contract():
    """The historical git-probe wrapper keeps its exact contract on top of
    bounded_probe_run: stripped stdout on rc==0, '' on any failure."""
    assert bounded_git_probe([_PY, "-c", "print('  x  ')"], timeout=30) == "x"
    assert bounded_git_probe([_PY, "-c", "import sys; sys.exit(1)"], timeout=30) == ""
    assert bounded_git_probe(["definitely-not-a-real-binary-87134"], timeout=5) == ""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group check")
def test_posix_child_gets_own_process_group():
    """POSIX spawns use process_group=0 so timeout cleanup can killpg the
    whole tree (same contract bounded_git_probe had)."""
    result = bounded_probe_run(
        [_PY, "-c", "import os; print(os.getpgid(0) == os.getpid())"],
        timeout=30,
    )
    assert result is not None
    assert result.stdout.strip() == "True"


# --------------------------------------------------------------------------
# Decoding contract (#90017 follow-up): the Windows-native probes -- wmic,
# tasklist, Windows PowerShell -- emit the machine's code page, not UTF-8.
# ``bounded_probe_run`` used to hardcode ``encoding="utf-8"``, so on a CP932
# host the process scans decoded CJK path segments to garbage before matching
# them against a path read from the filesystem.
# --------------------------------------------------------------------------


class _FakeProc:
    def __init__(self):
        self.returncode = 0

    def communicate(self, timeout=None):
        return "", ""


def _capture_popen(monkeypatch, captured):
    import hermes_cli._subprocess_compat as sc

    def fake_popen(argv, **kwargs):
        captured.update(kwargs)
        captured["argv"] = list(argv)
        return _FakeProc()

    monkeypatch.setattr(sc.subprocess, "Popen", fake_popen)


def test_bounded_probe_run_defaults_to_utf8(monkeypatch):
    """Existing callers (git probes, POSIX probes) keep UTF-8 decoding."""
    captured: dict = {}
    _capture_popen(monkeypatch, captured)

    bounded_probe_run(["anything"], timeout=5)

    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"


def test_bounded_probe_run_forwards_an_explicit_encoding(monkeypatch):
    captured: dict = {}
    _capture_popen(monkeypatch, captured)

    bounded_probe_run(["anything"], timeout=5, encoding="cp932", errors="ignore")

    assert captured["encoding"] == "cp932"
    assert captured["errors"] == "ignore"


def test_windows_probe_encoding_is_utf8_off_windows(monkeypatch):
    import hermes_cli._subprocess_compat as sc

    monkeypatch.setattr(sc, "IS_WINDOWS", False)
    assert sc.windows_probe_encoding() == "utf-8"


def test_windows_probe_encoding_falls_back_to_the_locale_code_page(monkeypatch):
    """When the OEM code page can't be read, use the ANSI one -- never
    ``getpreferredencoding(False)``, which follows Python UTF-8 Mode and would
    answer ``utf-8`` on a CP932 host that Hermes started in UTF-8 Mode."""
    import hermes_cli._subprocess_compat as sc

    monkeypatch.setattr(sc, "IS_WINDOWS", True)
    monkeypatch.setattr(sc.locale, "getencoding", lambda: "cp932")
    monkeypatch.setattr(sc.locale, "getpreferredencoding", lambda *a, **k: "utf-8")

    # ``ctypes.windll`` does not exist off Windows, so the OEM probe raises and
    # the locale fallback is what this asserts.
    assert sc.windows_probe_encoding() == "cp932"


def test_utf8_ignore_keeps_dbcs_trail_bytes_as_ascii():
    """Why the wrong codec is worse than a crash here.

    Measured on a ja-JP host: a real ``wmic`` scan returned a path segment
    whose CP932 bytes are ``90 66 92 66 83 7E``. Under ``utf-8`` with
    ``errors="ignore"`` each lead byte is dropped and each trail byte in the
    0x40-0x7E range survives as a literal ASCII character, so five characters
    became ``ff~`` -- no U+FFFD, nothing to notice, and every later
    ``in``-match against the real path fails.
    """
    segment = "診断ミレル"
    raw = segment.encode("cp932")

    assert raw == b"\x90\x66\x92\x66\x83\x7e\x83\x8c\x83\x8b"
    assert raw.decode("cp932") == segment
    assert raw.decode("utf-8", errors="ignore") == "ff~"
