"""Console-less desktop backend: console-helper spawns must carry the hidden-window flag.

The desktop backend runs under ``pythonw.exe`` — a console-less (GUI-subsystem) parent. On Windows,
when such a process spawns a console-subsystem child (``git.exe``, ``tasklist.exe``,
``powershell.exe``) *without* ``CREATE_NO_WINDOW``, the OS allocates a brand-new **visible** console
for it (hosted by Windows Terminal on Windows 11), which is the "~7–8 console windows flash at
startup" symptom (#117781).

``hermes_cli._subprocess_compat.windows_hide_flags()`` is the codebase-wide remedy and returns
``0`` off Windows, so the assertions below are *wiring* checks (does the site thread the helper's
value into ``creationflags``?) rather than platform checks. Stubbing the helper to a known constant
keeps the Linux lane meaningful, mirroring the ``_patch_hide_flags`` pattern in
``tests/test_windows_subprocess_no_window_flags.py``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Tuple

import pytest

_CREATE_NO_WINDOW = 0x08000000


class _Completed:
    """Minimal ``CompletedProcess`` stand-in for the faked ``subprocess.run``."""

    def __init__(self, stdout: str | bytes = "ok\n", returncode: int = 0):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


def _capture(monkeypatch, module, *, stdout="ok\n", returncode=0) -> List[Tuple[list, dict]]:
    """Record every ``subprocess.run`` call *module* makes; return the spawn list."""
    captured: List[Tuple[list, dict]] = []

    def fake_run(cmd, **kwargs):
        captured.append((list(cmd), kwargs))
        return _Completed(stdout=stdout, returncode=returncode)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return captured


def _pin_hide_flags(monkeypatch, module) -> None:
    """Pin ``windows_hide_flags()`` to a known constant.

    The sites under test pass the helper's result straight through to ``creationflags``, so what is
    asserted is the WIRING, not the platform. Only the helper is stubbed — no ``IS_WINDOWS`` fake.
    """
    import hermes_cli._subprocess_compat as compat

    monkeypatch.setattr(compat, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)
    # Sites that imported the name directly (``from ... import windows_hide_flags``) must see the
    # stub too, so rebind the module-level reference where it exists.
    if hasattr(module, "windows_hide_flags"):
        monkeypatch.setattr(module, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)


def _spawns(captured, *needles):
    """Captured calls whose argv contains every needle — scoping out cross-talk from
    import-time daemon spawns (the ``_spawns`` pattern from the sibling ratchet file).

    A needle matches a whole argv token, or (case-insensitively) a substring of argv[0] — the
    Windows helper path (``...\\WindowsPowerShell\\v1.0\\powershell.EXE``) is not the bare name.
    """
    out = []
    for cmd, kw in captured:
        if not cmd:
            continue
        tokens = [str(c) for c in cmd]
        lowered_head = tokens[0].lower()
        if all(any(n == t for t in tokens) or n.lower() in lowered_head for n in needles):
            out.append((cmd, kw))
    return out


# ── hermes_cli/gitlock.py ───────────────────────────────────────────────────
#
# ``_git_proc_running`` (tasklist), ``is_ancestor_of_head`` (git merge-base),
# ``_git_stdout_lines`` (git rev-parse/reflog/...), ``_batch_missing_parents`` (git cat-file
# --batch / --batch-check) and ``repair_broken_shallow_boundaries`` (git rev-list --count --all
# --reflog). The last one is the exact spawn the headless repro captured at startup, titled
# ``D:\Program Files\Git\cmd\git.exe`` in the visible-console watcher log.


def test_gitlock_git_proc_running_hides_tasklist_window(monkeypatch):
    from hermes_cli import gitlock

    captured = _capture(monkeypatch, gitlock, stdout="\"git.exe\",\"1234\"\r\n")
    _pin_hide_flags(monkeypatch, gitlock)

    assert gitlock._git_proc_running() is True
    spawns = _spawns(captured, "tasklist")
    assert len(spawns) == 1, captured
    assert spawns[0][1]["creationflags"] == _CREATE_NO_WINDOW


def test_gitlock_is_ancestor_of_head_hides_git_window(monkeypatch, tmp_path):
    from hermes_cli import gitlock

    captured = _capture(monkeypatch, gitlock, stdout="", returncode=0)
    _pin_hide_flags(monkeypatch, gitlock)

    assert gitlock.is_ancestor_of_head(tmp_path, "HEAD~1") is True
    spawns = _spawns(captured, "git", "merge-base")
    assert len(spawns) == 1, captured
    assert spawns[0][1]["creationflags"] == _CREATE_NO_WINDOW


def test_gitlock_stdout_lines_hides_git_window(monkeypatch, tmp_path):
    from hermes_cli import gitlock

    captured = _capture(monkeypatch, gitlock, stdout="refs/heads/main\n")
    _pin_hide_flags(monkeypatch, gitlock)

    assert gitlock._git_stdout_lines(tmp_path, ["rev-parse", "HEAD"]) == ["refs/heads/main"]
    spawns = _spawns(captured, "git", "rev-parse")
    assert len(spawns) == 1, captured
    assert spawns[0][1]["creationflags"] == _CREATE_NO_WINDOW


def test_gitlock_batch_missing_parents_creationflags(monkeypatch, tmp_path):
    """Both ``git cat-file --batch`` and ``--batch-check`` must hide their console."""
    from hermes_cli import gitlock

    captured: List[Tuple[list, dict]] = []
    # ``git cat-file --batch`` output shape: ``<oid> <type> <size>\\n`` + raw object + ``\\n``.
    commit = b"b" * 40
    body = b"commit 67\nparent " + b"a" * 40 + b"\n\nsubject\n"
    batch_out = commit + b" commit %d\n" % len(body) + body + b"\n"
    # ``--batch-check``: ``<oid> <type> <size>`` per line, or ``<oid> missing``. gitlock does
    # ``check.stdout.decode(errors="replace")``, so this call must hand back bytes as well
    # (a str stdout would raise AttributeError, be swallowed, and look like a clean no-match).
    check_out = b"a" * 40 + b" missing\n"
    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        captured.append((list(cmd), kwargs))
        calls["n"] += 1
        # --batch has no text=True: its stdout is bytes, so _Completed must hand back bytes.
        if calls["n"] == 1:
            return _Completed(stdout=batch_out, returncode=0)
        return _Completed(stdout=check_out, returncode=0)

    monkeypatch.setattr(gitlock.subprocess, "run", fake_run)
    _pin_hide_flags(monkeypatch, gitlock)

    result = gitlock._batch_missing_parents(tmp_path, [commit.decode()])

    assert result == {commit.decode()}, result
    cat_file = _spawns(captured, "git", "cat-file")
    assert len(cat_file) == 2, captured
    for cmd, kwargs in cat_file:
        assert kwargs["creationflags"] == _CREATE_NO_WINDOW, cmd


def test_gitlock_shallow_repair_probe_hides_git_window(monkeypatch, tmp_path):
    """The startup flash the headless repro captured: ``git rev-list --count --all --reflog``.

    ``_shallow_file_path`` resolves ``.git/shallow`` first (its own git spawn); a non-repo
    ``tmp_path`` yields no shallow file, so the function returns before the rev-list probe. To
    reach the probe we make the shallow file exist by pointing ``_shallow_file_path`` at a real
    path, then let the rev-list probe's non-zero exit short-circuit the repair — which is enough
    to assert the spawn contract.
    """
    from hermes_cli import gitlock

    captured: List[Tuple[list, dict]] = []
    shallow = tmp_path / "shallow"
    shallow.write_text("deadbeef\n", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        captured.append((list(cmd), kwargs))
        return _Completed(stdout="0\n", returncode=1)

    monkeypatch.setattr(gitlock.subprocess, "run", fake_run)
    monkeypatch.setattr(gitlock, "_shallow_file_path", lambda root: shallow)
    _pin_hide_flags(monkeypatch, gitlock)

    assert gitlock.repair_broken_shallow_boundaries(tmp_path) == 0
    spawns = _spawns(captured, "git", "rev-list")
    assert len(spawns) == 1, captured
    assert spawns[0][1]["creationflags"] == _CREATE_NO_WINDOW


# ── hermes_cli/gateway.py ───────────────────────────────────────────────────
#
# ``_windows_scheduled_task_state`` — the Task-Scheduler COM probe run by the backend's startup
# supervisor sweep. The sibling process-scan PowerShell a few hundred lines above
# (``gateway.py:732``) already hides its window; this site was simply missed.


def test_windows_scheduled_task_state_hides_powershell_window(monkeypatch):
    from hermes_cli import gateway

    captured = _capture(monkeypatch, gateway, stdout="Ready\n", returncode=0)
    _pin_hide_flags(monkeypatch, gateway)
    monkeypatch.setattr(gateway, "is_windows", lambda: True)
    monkeypatch.setattr(gateway.shutil, "which", lambda name: r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.EXE")

    assert gateway._windows_scheduled_task_state("Hermes_Gateway") == "Ready"
    spawns = _spawns(captured, "powershell")
    assert len(spawns) == 1, captured
    assert spawns[0][1]["creationflags"] == _CREATE_NO_WINDOW


def test_windows_scheduled_task_state_no_window_flag_on_non_windows(monkeypatch):
    """The helper must never be reached off Windows — the probe returns None first."""
    from hermes_cli import gateway

    monkeypatch.setattr(gateway, "is_windows", lambda: False)
    assert gateway._windows_scheduled_task_state("Hermes_Gateway") is None


# ── hermes_cli/update_cmd.py ────────────────────────────────────────────────
#
# ``_git_run`` — the central git runner behind every ``hermes update`` / update-check git
# invocation. Console-less when the caller is the desktop backend.


def test_update_cmd_git_run_hides_git_window(monkeypatch):
    from hermes_cli import update_cmd

    captured = _capture(monkeypatch, update_cmd, stdout="origin\n", returncode=0)
    _pin_hide_flags(monkeypatch, update_cmd)

    result = update_cmd._git_run(["git"], ["remote", "get-url", "origin"], ".")
    assert result.returncode == 0
    spawns = _spawns(captured, "git", "remote")
    assert len(spawns) == 1, captured
    assert spawns[0][1]["creationflags"] == _CREATE_NO_WINDOW


def test_update_cmd_git_run_hides_git_window_on_network_path(monkeypatch):
    """The ``network=True`` branch splats extra kwargs — the flag must survive the splat."""
    from hermes_cli import update_cmd

    captured = _capture(monkeypatch, update_cmd, stdout="", returncode=0)
    _pin_hide_flags(monkeypatch, update_cmd)

    update_cmd._git_run(["git"], ["fetch", "origin", "main"], ".", network=True)
    spawns = _spawns(captured, "git", "fetch")
    assert len(spawns) == 1, captured
    assert spawns[0][1]["creationflags"] == _CREATE_NO_WINDOW


# ── hermes_cli/update_cmd_git.py ────────────────────────────────────────────
#
# ``_GIT_TEXT_KW`` — the shared kwargs dict for the update git plumbing (rev-parse, diff, config…).


def test_update_cmd_git_text_kw_carries_hidden_window_flag():
    """The shared kwargs dict must hide the console for every site that splats it."""
    from hermes_cli.update_cmd_git import _GIT_TEXT_KW

    assert _GIT_TEXT_KW["creationflags"] == _CREATE_NO_WINDOW


def test_update_cmd_git_rev_parse_hides_git_window(monkeypatch, tmp_path):
    """``_branch_head_label`` splats ``_GIT_TEXT_KW`` into both of its ``rev-parse`` calls."""
    from hermes_cli import update_cmd_git

    captured = _capture(monkeypatch, update_cmd_git, stdout="main\n", returncode=0)
    _pin_hide_flags(monkeypatch, update_cmd_git)

    # Both rev-parses return the same stdout here; the label shape is not under test.
    update_cmd_git._branch_head_label(["git"], tmp_path)
    spawns = _spawns(captured, "git", "rev-parse")
    assert len(spawns) == 2, captured
    for _cmd, kwargs in spawns:
        assert kwargs["creationflags"] == _CREATE_NO_WINDOW


def test_update_cmd_git_normalize_managed_eol_probe_hides_git_window(monkeypatch, tmp_path):
    """``_normalize_managed_eol``'s local ``_probe_run`` splats the same dict on a ``git -c …`` argv."""
    from hermes_cli import update_cmd_git

    captured: List[Tuple[list, dict]] = []

    def fake_run(cmd, **kw):
        captured.append((list(cmd), kw))
        # ``config --get core.autocrlf`` must report "true" for the EOL probe path to run at all.
        if "autocrlf" in [str(c) for c in cmd]:
            return _Completed(stdout="true\n", returncode=0)
        return _Completed(stdout="", returncode=0)

    monkeypatch.setattr(update_cmd_git.subprocess, "run", fake_run)
    _pin_hide_flags(monkeypatch, update_cmd_git)

    update_cmd_git._normalize_managed_eol(["git"], tmp_path)
    probe = _spawns(captured, "git", "config")
    assert len(probe) == 1, captured
    assert probe[0][1]["creationflags"] == _CREATE_NO_WINDOW
