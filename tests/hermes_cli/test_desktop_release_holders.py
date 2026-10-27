"""Release-folder holders are named, and closed only with consent (#101789).

Real child processes on any host: the cwd detection and the consent rules are not
Windows-specific (only the rename refusal is; that half runs in
``test_desktop_release_holder_windows_live.py`` on a Windows runner).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import psutil
import pytest

from hermes_cli import desktop_release_holders, main_desktop
from hermes_cli.desktop_release_holders import CONSENT_FILE_NAME, release_after_refused_rename, take_consent

_HELPER = "import time\nprint('ready', flush=True)\ntime.sleep(300)\n"


@pytest.fixture
def holder(tmp_path):
    release = tmp_path / "apps" / "desktop" / "release"
    (release / "win-unpacked").mkdir(parents=True)
    script = tmp_path / "foreign_helper.py"
    script.write_text(_HELPER, encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script)], cwd=release / "win-unpacked",
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"
    try:
        yield release, proc
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(10)


def _desktop_consent(release, proc, run_id, **override):
    """What the Desktop's "Close <program> and retry" writes for the retry run it starts."""
    ps = psutil.Process(proc.pid)
    seen = {"pid": proc.pid, "start_time": ps.create_time(), "exe": ps.exe(), **override}
    (release / CONSENT_FILE_NAME).write_text(json.dumps({
        "written_at": time.time(), "run_id": run_id, "exes": [seen["exe"]], "holders": [seen],
    }), encoding="utf-8")


def _never_asked(_prompt):
    raise AssertionError("a non-interactive run must never prompt")


def test_a_retry_that_fails_before_the_swap_leaves_no_consent_for_a_later_update(holder, tmp_path, monkeypatch):
    """Review N1: the consent belongs to one retry. Once that retry's swap attempt has run, a
    later update from any source (gateway /update, cron) must not close the program unasked."""
    release, proc = holder
    _desktop_consent(release, proc, run_id="desktop-retry-1")

    # The consented retry runs, but its swap fails before the rename (no launchable staged app).
    monkeypatch.setenv("HERMES_UPDATE_CORRELATION_ID", "desktop-retry-1")
    staging = tmp_path / "apps" / "desktop" / ".staging-failed"
    staging.mkdir()
    assert main_desktop._swap_staged_desktop_app(tmp_path / "apps" / "desktop", staging) is None
    assert not (release / CONSENT_FILE_NAME).exists(), "the swap attempt consumes the consent, failed or not"

    # Later, a non-interactive gateway /update hits the refused rename.
    monkeypatch.delenv("HERMES_UPDATE_CORRELATION_ID")
    holders, closed = release_after_refused_rename(release, input_fn=_never_asked)

    assert proc.pid in [h.pid for h in holders]
    assert closed == []
    assert proc.poll() is None, "closed without asking, on a consent given for another run"


def test_a_consent_closes_only_what_the_user_saw_in_the_run_it_was_given_for(holder, capsys):
    release, proc = holder
    started = psutil.Process(proc.pid).create_time()
    refused = {
        "another run's swap": ("desktop-retry-1", "gateway-update-7", {}),
        "a swap with no run id": ("desktop-retry-1", None, {}),
        "a recycled pid": ("desktop-retry-1", "desktop-retry-1", {"start_time": started - 60}),
        "another program": ("desktop-retry-1", "desktop-retry-1", {"exe": "C:/elsewhere/other.exe"}),
    }
    for case, (consented_run, swap_run, override) in refused.items():
        _desktop_consent(release, proc, consented_run, **override)
        consent = take_consent(release, swap_run)
        assert not (release / CONSENT_FILE_NAME).exists(), f"{case}: the attempt consumes the consent"
        holders, closed = release_after_refused_rename(release, consent=consent, input_fn=_never_asked)
        assert proc.pid in [h.pid for h in holders], case
        assert closed == [] and proc.poll() is None, f"closed without consent ({case})"
    assert f"{psutil.Process(proc.pid).name()} (pid {proc.pid}, " in capsys.readouterr().out

    _desktop_consent(release, proc, "desktop-retry-1")
    _holders, closed = release_after_refused_rename(
        release, consent=take_consent(release, "desktop-retry-1"), input_fn=_never_asked)

    assert proc.pid in closed
    assert proc.wait(10) is not None


def test_a_holder_whose_program_path_is_unreadable_is_left_for_the_user(holder, monkeypatch, capsys):
    """Review N3: no close offer that can never be honoured, at the terminal or in the Desktop
    (windows.ps1 hands the Desktop an empty exe for it, which gets no Close button)."""
    release, proc = holder
    # What psutil answers for a protected process (another user's, elevated): no program path.
    monkeypatch.setattr(desktop_release_holders, "_exe_or_empty", lambda _proc: "")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    _desktop_consent(release, proc, "desktop-retry-1", exe="")

    holders, closed = release_after_refused_rename(
        release, consent=take_consent(release, "desktop-retry-1"), input_fn=_never_asked)

    assert proc.pid in [h.pid for h in holders] and closed == []
    assert proc.poll() is None
    assert f"(pid {proc.pid}) yourself" in capsys.readouterr().out
    line = next(ln for ln in desktop_release_holders.describe_block(holders, release).splitlines()
                if f"(pid {proc.pid}, started " in ln)
    assert line.endswith("path unreadable)")
