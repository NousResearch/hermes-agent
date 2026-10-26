"""A program outside Hermes whose working folder is inside release/ blocks the Desktop swap (#101789).

Real processes on a real Windows host: a helper started with its cwd inside
``release/win-unpacked`` (the shape of Sogou IME's ``SogouCloud.exe``, spawned from a Hermes
window and inheriting its cwd) makes Windows refuse the promotion rename. The failure must name
that program, must never close it without consent, and a consented retry must close exactly it
and finish the swap.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from hermes_cli import main_desktop

pytestmark = pytest.mark.platforms("windows")

_HELPER = "import sys, time\nprint('ready', flush=True)\ntime.sleep(300)\n"


@pytest.fixture
def swap_fixture(tmp_path, monkeypatch):
    desktop_dir = tmp_path / "apps" / "desktop"
    live_root = desktop_dir / "release" / "win-unpacked"
    live_root.mkdir(parents=True)
    (live_root / "Hermes.exe").write_text("old", encoding="utf-8")
    staging = main_desktop._desktop_staging_dir(desktop_dir)
    (staging / "win-unpacked").mkdir(parents=True)
    (staging / "win-unpacked" / "Hermes.exe").write_text("new", encoding="utf-8")
    monkeypatch.setattr(main_desktop, "_DESKTOP_SWAP_RENAME_RETRY_DELAYS_S", (0.2,))
    script = tmp_path / "foreign_helper.py"
    script.write_text(_HELPER, encoding="utf-8")
    helper = subprocess.Popen(
        [sys.executable, str(script)], cwd=live_root, stdout=subprocess.PIPE, text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    # The child opens its cwd handle during its own startup: wait for it before asserting the lock.
    assert helper.stdout.readline().strip() == "ready"
    try:
        yield desktop_dir, staging, live_root, helper
    finally:
        if helper.poll() is None:
            helper.kill()
            helper.wait(10)


def _promote(desktop_dir: Path, staging: Path) -> tuple[Path | None, str]:
    try:
        return main_desktop._promote_staged_desktop_app(desktop_dir, staging, integrity_check=lambda _p: None), ""
    except RuntimeError as exc:
        return None, str(exc)


def test_blocked_swap_names_the_foreign_holder_and_leaves_it_running(swap_fixture, capsys):
    desktop_dir, staging, live_root, helper = swap_fixture

    promoted, error = _promote(desktop_dir, staging)

    out = capsys.readouterr().out
    assert promoted is None
    assert f"pid {helper.pid}" in error, error  # the update failure names the holder
    assert f"Desktop app folder held by: {psutil.Process(helper.pid).name()} (pid {helper.pid}, " in out, out
    assert helper.poll() is None, "a program outside Hermes was closed without consent"
    assert (live_root / "Hermes.exe").read_text(encoding="utf-8") == "old"


def test_consented_retry_closes_exactly_the_holder_and_promotes(swap_fixture, capsys):
    desktop_dir, staging, live_root, helper = swap_fixture
    # The venv python.exe may be a launcher with the real interpreter as its child, both inside
    # the folder; the Desktop consents to every program the failure named, so this does too.
    tree = [psutil.Process(helper.pid), *psutil.Process(helper.pid).children(recursive=True)]
    consent = desktop_dir / "release" / ".close-holders-consent.json"
    consent.write_text(json.dumps({"written_at": time.time(), "exes": [p.exe() for p in tree]}), encoding="utf-8")

    promoted, error = _promote(desktop_dir, staging)

    assert error == ""
    assert promoted == live_root / "Hermes.exe"
    assert promoted.read_text(encoding="utf-8") == "new"
    assert helper.wait(10) is not None
    assert not [p.pid for p in tree if p.is_running()]
    assert not consent.exists(), "a consent covers one retry only"
    assert "Closed with your consent: " in capsys.readouterr().out
