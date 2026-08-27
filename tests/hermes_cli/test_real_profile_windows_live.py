"""LIVE Windows E2E: consented auto-close makes real-profile work with a running browser.

windows-latest only. Proves the end state: with browser.real_profile_autoclose
on, a REAL running Chrome that share-locks its cookie DB is terminated by
snapshot_real_profile, the lock releases, and a valid signed-in-shaped copy is
produced. Also proves the default (autoclose off) fails fast, not hangs.

PROOF branch evidence — reverted before merge; never lands on main.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only live E2E")

_CHROME = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)


def _find_chrome():
    for p in _CHROME:
        if os.path.isfile(p):
            return p
    return shutil.which("chrome") or shutil.which("chrome.exe")


def _launch_chrome_on(user_data: Path):
    proc = subprocess.Popen(
        [_find_chrome(), "--headless=new", "--disable-gpu", "--no-first-run",
         "--no-default-browser-check", f"--user-data-dir={user_data}",
         "--remote-debugging-port=0", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    ck = None
    deadline = time.time() + 60
    while time.time() < deadline and not ck:
        for rel in (r"Default\Network\Cookies", r"Default\Cookies"):
            c = user_data / rel
            if c.is_file() and c.stat().st_size > 0:
                ck = c
                break
        time.sleep(1)
    return proc, ck


def _raw_copy_fails(path: str) -> bool:
    try:
        shutil.copy2(path, path + ".rc"); os.unlink(path + ".rc"); return False
    except OSError:
        return True


def test_autoclose_off_fails_fast(tmp_path):
    """Default (autoclose off): a running Chrome → fail fast (<30s) with the
    quit/autoclose guidance, never a hang, never a silent copy."""
    if not _find_chrome():
        pytest.skip("no chrome")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import hermes_cli.browser_connect as bc

    ud = tmp_path / "ud"; ud.mkdir()
    proc, ck = _launch_chrome_on(ud)
    try:
        assert ck is not None, "no cookie db"
        if not _raw_copy_fails(str(ck)):
            pytest.skip("Chrome did not share-lock on this runner")
        orig = bc.real_profile_data_dir
        bc.real_profile_data_dir = lambda b, system=None: str(ud)
        bc._real_profile_autoclose = lambda: False
        try:
            t0 = time.time()
            dst, err = bc.snapshot_real_profile("chrome", src=str(ud))
            elapsed = time.time() - t0
        finally:
            bc.real_profile_data_dir = orig
        assert dst is None and err
        assert "quit" in err.lower() and "real_profile_autoclose" in err
        assert elapsed < 30, f"hung {elapsed:.0f}s — must fail fast"
    finally:
        proc.terminate()
        try: proc.wait(timeout=15)
        except subprocess.TimeoutExpired: proc.kill()


def test_autoclose_on_closes_chrome_and_snapshots(tmp_path):
    """Consented auto-close: a running Chrome is terminated, the lock releases,
    and a valid cookie DB copy is produced — real-profile works WITH the browser
    initially running."""
    if not _find_chrome():
        pytest.skip("no chrome")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import hermes_cli.browser_connect as bc

    ud = tmp_path / "ud"; ud.mkdir()
    proc, ck = _launch_chrome_on(ud)
    try:
        assert ck is not None, "no cookie db"
        if not _raw_copy_fails(str(ck)):
            pytest.skip("Chrome did not share-lock on this runner")

        orig = bc.real_profile_data_dir
        bc.real_profile_data_dir = lambda b, system=None: str(ud)
        bc._real_profile_autoclose = lambda: True
        # Isolate the snapshot store under tmp.
        orig_home = bc.get_hermes_home
        bc.get_hermes_home = lambda: tmp_path / "hh"
        try:
            dst, err = bc.snapshot_real_profile("chrome", src=str(ud))
        finally:
            bc.real_profile_data_dir = orig
            bc.get_hermes_home = orig_home

        assert err is None, f"auto-close snapshot failed: {err}"
        assert dst is not None
        # Cookie DB may be at the modern Network/ location or the legacy path,
        # depending on the Chrome build — accept either, but it MUST exist and
        # be a valid SQLite cookies DB.
        candidates = [
            Path(dst) / "Default" / "Network" / "Cookies",
            Path(dst) / "Default" / "Cookies",
        ]
        copy_ck = next((c for c in candidates if c.is_file()), None)
        assert copy_ck is not None, (
            "cookie DB not copied after auto-close; Default contents: "
            + repr(sorted(os.listdir(Path(dst) / "Default")) if (Path(dst) / "Default").is_dir() else "NO Default")
        )
        # Valid SQLite with the cookies table.
        con = sqlite3.connect(str(copy_ck))
        try:
            names = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            con.close()
        assert "cookies" in names, f"copied DB missing cookies table: {names}"
        # The original Chrome should now be gone (we terminated its tree).
        assert proc.poll() is not None, "auto-close did not terminate Chrome"
    finally:
        try:
            if proc.poll() is None:
                proc.terminate(); proc.wait(timeout=15)
        except Exception:
            try: proc.kill()
            except Exception: pass
