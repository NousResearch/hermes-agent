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

from hermes_cli.desktop_release_holders import CONSENT_FILE_NAME, release_after_refused_rename

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


def _consent(release, exes, written_at=None):
    (release / CONSENT_FILE_NAME).write_text(
        json.dumps({"written_at": time.time() if written_at is None else written_at, "exes": exes}),
        encoding="utf-8")


def _never_asked(_prompt):
    raise AssertionError("a non-interactive run must never prompt")


def test_holder_is_named_and_left_running_without_a_matching_fresh_consent(holder, capsys):
    release, proc = holder
    exe = psutil.Process(proc.pid).exe()

    for consent in (None, ["C:/elsewhere/other.exe"], "stale"):
        if consent == "stale":
            _consent(release, [exe], written_at=time.time() - 2 * 3600)
        elif consent is not None:
            _consent(release, consent)
        holders, closed = release_after_refused_rename(release, input_fn=_never_asked)

        assert proc.pid in [h.pid for h in holders], consent
        assert closed == [], consent
        assert proc.poll() is None, f"closed without consent ({consent})"
        assert not (release / CONSENT_FILE_NAME).exists(), "a consent is consumed by the attempt that read it"
    assert f"Desktop app folder held by: {psutil.Process(proc.pid).name()} (pid {proc.pid}, {exe})" in capsys.readouterr().out


def test_consented_holder_is_closed(holder):
    release, proc = holder
    _consent(release, [psutil.Process(proc.pid).exe()])

    _holders, closed = release_after_refused_rename(release, input_fn=_never_asked)

    assert proc.pid in closed
    assert proc.wait(10) is not None
