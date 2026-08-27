"""LIVE Windows E2E: locked-profile blocks; explicit approved close then works.

windows-latest only. Proves the three-part contract:
  1. A running Chrome that share-locks its profile makes snapshot_real_profile
     BLOCK (never kill, never hang) with the [profile-locked] signal.
  2. The explicit, user-approved close step (close_browser_holding_profile,
     what `hermes browser close-profile` runs) terminates the browser and the
     lock releases.
  3. After the close, snapshot_real_profile succeeds and copies a valid DB.

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


def test_locked_blocks_then_approved_close_then_snapshots(tmp_path):
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

        orig_dd = bc.real_profile_data_dir
        orig_home = bc.get_hermes_home
        bc.real_profile_data_dir = lambda b, system=None: str(ud)
        bc.get_hermes_home = lambda: tmp_path / "hh"
        # autoclose armed → message OFFERS the close, but snapshot must NOT kill.
        bc._real_profile_autoclose = lambda: True
        try:
            # 1. Locked → blocks fast with the [profile-locked] signal, no kill.
            t0 = time.time()
            dst, err = bc.snapshot_real_profile("chrome", src=str(ud))
            elapsed = time.time() - t0
            assert dst is None and err
            assert err.startswith(bc._PROFILE_LOCKED_PREFIX), f"not the locked signal: {err}"
            assert elapsed < 30, f"blocked slowly ({elapsed:.0f}s) — must be fast"
            assert proc.poll() is None, "snapshot must NOT have killed Chrome on its own"

            # 2. Explicit approved close (the engine `hermes browser
            #    close-profile` runs) terminates Chrome; lock releases.
            closed, msg = bc.close_browser_holding_profile(str(ud))
            assert closed, f"approved close failed: {msg}"
            assert proc.poll() is not None, "close did not terminate Chrome"

            # 3. Snapshot now succeeds with a valid cookie DB.
            dst2, err2 = bc.snapshot_real_profile("chrome", src=str(ud))
            assert err2 is None, f"post-close snapshot failed: {err2}"
            assert dst2 is not None
            cands = [Path(dst2) / "Default" / "Network" / "Cookies",
                     Path(dst2) / "Default" / "Cookies"]
            copy_ck = next((c for c in cands if c.is_file()), None)
            assert copy_ck is not None, (
                "cookie DB not copied after approved close; Default: "
                + repr(sorted(os.listdir(Path(dst2) / "Default"))
                       if (Path(dst2) / "Default").is_dir() else "NONE"))
            con = sqlite3.connect(str(copy_ck))
            try:
                names = {r[0] for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
            finally:
                con.close()
            assert "cookies" in names, f"missing cookies table: {names}"
        finally:
            bc.real_profile_data_dir = orig_dd
            bc.get_hermes_home = orig_home
    finally:
        try:
            if proc.poll() is None:
                proc.terminate(); proc.wait(timeout=15)
        except Exception:
            try: proc.kill()
            except Exception: pass


def test_autoclose_off_blocks_with_quit_guidance(tmp_path):
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
            dst, err = bc.snapshot_real_profile("chrome", src=str(ud))
            assert dst is None and err and err.startswith(bc._PROFILE_LOCKED_PREFIX)
            assert "quit" in err.lower()
            assert proc.poll() is None, "must not kill Chrome when autoclose off"
        finally:
            bc.real_profile_data_dir = orig
    finally:
        proc.terminate()
        try: proc.wait(timeout=15)
        except subprocess.TimeoutExpired: proc.kill()
