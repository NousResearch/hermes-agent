"""Liveness handshake between the Windows .vbs launcher and ``gateway run`` (#136390).

A console-less Scheduled-Task python can exit before running any code (rc=0, no log), which used
to leave the task reporting success while nothing started. These tests pin the three sides of the
handshake: the launcher polls the heartbeat file's mtime, ``run_gateway`` rewrites that heartbeat
at its very first line, and ``hermes gateway status`` surfaces the failure marker the launcher
appends when the heartbeat never moved.
"""

import json
import re
from pathlib import Path

import pytest

from hermes_cli import gateway_windows


_VBS_KEYWORDS = {
    "Option",
    "Explicit",
    "Dim",
    "Set",
    "If",
    "Then",
    "Else",
    "End",
    "Not",
    "For",
    "To",
    "Next",
    "Exit",
    "On",
    "Error",
    "Resume",
    "GoTo",
    "Sub",
    "Function",
    "True",
    "False",
    "And",
    "Or",
    "ByVal",
    "ByRef",
    "Call",
    "Const",
    "Do",
    "Loop",
    "While",
    "Wend",
    "Select",
    "Case",
    "Each",
    "In",
    "Step",
    "Is",
    "Mod",
    "New",
    "Nothing",
    "Empty",
    "Null",
}
_VBS_BUILTINS = {"WScript", "Err", "Now", "CStr", "Len", "CreateObject"}


def _build_vbs() -> str:
    return gateway_windows._build_gateway_vbs_script(
        r"C:\venv\Scripts\python.exe",
        r"C:\Hermes",
        r"C:\Hermes",
        "--profile work",
    )


def test_vbs_polls_heartbeat_and_fails_the_task_when_it_never_moves():
    """The launcher must not let the task report success when its python died before reaching
    ``gateway run``: poll the heartbeat mtime, append a failure marker, quit non-zero."""
    content = _build_vbs()
    # Launch stays detached/async (never binds wscript's lifetime to the gateway).
    assert "sh.Run " in content and ", 0, False" in content
    # Snapshot-before / compare-after: a stale heartbeat from a previous boot is not success.
    assert 'beforeHb = ""' in content
    assert re.search(r"DateLastModified\) <> beforeHb", content)
    # Bounded polling at 1s granularity, driven by the shared timeout constant.
    assert f"For i = 1 To {gateway_windows._TASK_LAUNCH_LIVENESS_TIMEOUT_S}" in content
    assert "WScript.Sleep 1000" in content
    # Success clears a stale failure marker and exits 0.
    assert "If fso.FileExists(failurePath) Then fso.DeleteFile failurePath" in content
    assert "WScript.Quit 0" in content
    # Failure appends the marker (best-effort, never raises into a wscript error dialog) and
    # quits non-zero so Last Run Result goes red and RestartOnFailure retries.
    assert "On Error Resume Next" in content
    assert "If Err.Number = 0 Then" in content
    assert "WScript.Quit 3" in content


def test_vbs_paths_are_derived_from_the_configured_home():
    """Heartbeat and failure marker live under the launcher's HERMES_HOME logs dir, using the shared
    filenames (the heartbeat filename is mirrored as a literal in gateway/status.py)."""
    content = _build_vbs()
    assert r'logDir = "C:\Hermes\logs"' in content
    assert f'logDir & "\\{gateway_windows._TASK_HEARTBEAT_FILENAME}"' in content
    assert f'logDir & "\\{gateway_windows._TASK_FAILURE_MARKER_FILENAME}"' in content


def test_vbs_declares_every_variable_it_uses():
    """``Option Explicit`` makes an undeclared variable a hard runtime error — and a script error
    under wscript pops a dialog at every login (the Startup chain runs without //B). Lex the script
    (strings and comments stripped, member access excluded) and require every bare identifier to be
    declared by a ``Dim`` line, a keyword, or a builtin."""
    content = _build_vbs()
    code = re.sub(r"'[^\r\n]*", "", content)  # comments run to end-of-line
    code = re.sub(r'"(?:[^"]|"")*"', '""', code)  # quoted strings
    declared = set()
    for line in code.splitlines():
        if line.strip().startswith("Dim "):
            declared.update(name.strip() for name in line.strip()[4:].split(","))
    assert declared, "no Dim lines found — the parser drifted"
    used = {
        m.group(1) for m in re.finditer(r"(?<![.\w])([A-Za-z_][A-Za-z0-9_]*)", code)
    }
    undeclared = used - declared - _VBS_KEYWORDS - _VBS_BUILTINS
    assert not undeclared, (
        f"undeclared identifiers under Option Explicit: {sorted(undeclared)}"
    )


def test_run_gateway_heartbeat_rewrites_the_shared_filename(monkeypatch, tmp_path):
    """``run_gateway`` writes the heartbeat before any guard (an already-running gateway exits in a
    guard but still proves the task leg works), into the HERMES_HOME logs dir under the same filename the
    VBS polls — the literal mirror of ``_TASK_HEARTBEAT_FILENAME``."""
    import gateway.status as gw_status

    monkeypatch.setattr(gw_status, "_get_process_hermes_home", lambda: tmp_path)
    before = tmp_path / "logs" / gateway_windows._TASK_HEARTBEAT_FILENAME
    before.parent.mkdir(parents=True)
    before.write_text('{"pid": 1, "ts": 100.0}\n', encoding="utf-8")

    gw_status.write_task_launch_heartbeat()

    record = json.loads(before.read_text(encoding="utf-8"))
    assert record["pid"] > 1
    assert record["ts"] > 100.0


def test_run_gateway_heartbeat_never_raises(monkeypatch, tmp_path):
    """The heartbeat is written at the top of gateway startup: an unwritable home must not take
    the gateway down with it."""
    import gateway.status as gw_status

    def boom():
        raise OSError("home unreadable")

    monkeypatch.setattr(gw_status, "_get_process_hermes_home", boom)
    gw_status.write_task_launch_heartbeat()  # must not raise


def test_status_surfaces_the_failure_marker(monkeypatch, tmp_path, capsys):
    """``hermes gateway status`` must make a silent auto-start failure visible: print the marker
    path and its last entry when present, nothing when absent."""
    monkeypatch.setattr(gateway_windows, "_hermes_home", lambda: tmp_path)
    marker = tmp_path / "logs" / gateway_windows._TASK_FAILURE_MARKER_FILENAME
    marker.parent.mkdir(parents=True)
    marker.write_text(
        "2026-10-11 09:42:04 task-launch: gateway run was not reached within 30s\n"
        "2026-10-11 09:43:05 task-launch: gateway run was not reached within 30s\n",
        encoding="utf-8",
    )

    gateway_windows.print_task_launch_failure_warning()
    out = capsys.readouterr().out
    assert "⚠ Scheduled-Task launch failure marker present" in out
    assert "Last entry: 2026-10-11 09:43:05" in out
    assert str(marker) in out

    marker.unlink()
    gateway_windows.print_task_launch_failure_warning()
    assert capsys.readouterr().out == ""
