"""A failed --version probe names the NTSTATUS fatal-exit code it returned.

Windows loader and CRT deaths surface to Python as bare unsigned exit codes:
ffmpeg's pinned win32-x64 build exited 3221225785 (0xC0000139,
STATUS_ENTRYPOINT_NOT_FOUND) on Windows Server 2016 (#130064), and ripgrep's
ARM64 rg.exe exited 3221225781 (0xC0000135, STATUS_DLL_NOT_FOUND) on fresh
Windows installs. The probe reason must translate both instead of forcing the
user to decode the number themselves.
"""

from __future__ import annotations

import subprocess

from pm.package import _probe_reason


def _proc(returncode: int, out: bytes = b"") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["ffmpeg", "--version"], returncode, out, b"")


def test_entrypoint_not_found_names_the_windows_gap(tmp_path):
    reason = _probe_reason(tmp_path / "ffmpeg.exe", _proc(3221225785))
    assert "exited 3221225785" in reason
    assert "0xC0000139 STATUS_ENTRYPOINT_NOT_FOUND" in reason
    assert "newer Windows" in reason


def test_dll_not_found_names_the_missing_runtime(tmp_path):
    reason = _probe_reason(tmp_path / "rg.exe", _proc(3221225781))
    assert "0xC0000135 STATUS_DLL_NOT_FOUND" in reason


def test_ordinary_exit_codes_stay_untouched(tmp_path):
    reason = _probe_reason(tmp_path / "ffmpeg", _proc(3, b"usage error"))
    assert reason.endswith("exited 3: usage error")


def test_posix_signal_deaths_are_not_misread_as_ntstatus(tmp_path):
    assert "STATUS_" not in _probe_reason(tmp_path / "ffmpeg", _proc(-11))


def test_unknown_ntstatus_codes_get_no_speculative_hint(tmp_path):
    assert "STATUS_" not in _probe_reason(tmp_path / "ffmpeg", _proc(0xC0000110))


def test_output_tail_is_still_appended_after_the_hint(tmp_path):
    reason = _probe_reason(tmp_path / "ffmpeg.exe", _proc(3221225785, b"dying words"))
    assert reason.endswith("dying words")
