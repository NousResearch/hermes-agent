"""LIVE Windows E2E: real-profile auth-DB copy under a real Chrome share-lock.

Runs ONLY on a windows-latest GitHub runner (see .github/workflows/
windows-realprofile-e2e.yml). It proves the thing the Linux lanes cannot:
that copying the SQLite auth DBs via the online-backup API succeeds while a
REAL Chrome process holds the cookie DB with a Windows OS-level share lock —
the exact "file in use by another application" failure the copy approach had
to solve.

Why this can't be a normal unit test: on Windows, Chrome opens
Cookies/Login Data with a share mode that makes a plain file copy raise
WinError 32. A Linux "open a write transaction" analog reproduces SQLite's
internal lock, NOT the Windows filesystem share lock, so only a real Chrome on
a real Windows runner exercises the failure this fix targets.
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

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows-only live share-lock E2E"
)

_CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)


def _find_chrome() -> str | None:
    for p in _CHROME_CANDIDATES:
        if os.path.isfile(p):
            return p
    which = shutil.which("chrome") or shutil.which("chrome.exe")
    return which


def _raw_copy_raises_while_locked(path: str) -> bool:
    """True if a plain copy of ``path`` fails (the WinError 32 we must beat)."""
    try:
        shutil.copy2(path, path + ".rawcopy")
        os.unlink(path + ".rawcopy")
        return False
    except OSError:
        return True


def test_real_chrome_locked_cookie_db_copies_via_backup(tmp_path):
    chrome = _find_chrome()
    if not chrome:
        pytest.skip("Chrome not installed on this runner")

    # Repo root on sys.path so we import the real module under test.
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from hermes_cli import browser_connect as bc

    user_data = tmp_path / "chrome-user-data"
    user_data.mkdir()

    # Launch a REAL Chrome on this user-data-dir so it creates + holds the
    # Default profile's Cookies DB open with a Windows share lock. Headless new
    # still creates the on-disk profile and holds the handles.
    proc = subprocess.Popen(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={user_data}",
            "--remote-debugging-port=0",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # Wait for Chrome to create the cookie DB (Network/Cookies on modern
        # Chrome; fall back to Default/Cookies).
        cookies = None
        deadline = time.time() + 60
        while time.time() < deadline:
            for rel in (r"Default\Network\Cookies", r"Default\Cookies"):
                cand = user_data / rel
                if cand.is_file() and cand.stat().st_size > 0:
                    cookies = cand
                    break
            if cookies:
                break
            time.sleep(1)
        assert cookies is not None, "Chrome never created a Cookies DB"

        # Precondition: while Chrome runs, a RAW copy of the locked DB must
        # actually fail — otherwise this test isn't exercising the lock and
        # would pass vacuously. (If it doesn't fail, the runner's Chrome build
        # didn't share-lock; skip rather than assert a false green.)
        if not _raw_copy_raises_while_locked(str(cookies)):
            pytest.skip(
                "Chrome did not share-lock the cookie DB on this runner; "
                "the lock-copy path is not being exercised"
            )

        # THE ASSERTION: our lock-aware copy succeeds while Chrome holds the
        # lock, and the copy is a readable SQLite DB.
        dst = tmp_path / "copy" / "Cookies"
        ok = bc._copy_auth_file(str(cookies), str(dst))
        assert ok is True, "lock-aware copy failed while Chrome held the DB"
        assert dst.is_file() and dst.stat().st_size > 0

        # The copy must be a real, openable SQLite DB with the cookies table —
        # proving we captured the committed snapshot, not an empty/torn file.
        con = sqlite3.connect(str(dst))
        try:
            names = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        finally:
            con.close()
        assert "cookies" in names, f"copied DB missing cookies table: {names}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
