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


def test_locked_profile_fails_closed_not_silent(tmp_path):
    """Windows contract: a running Chrome holds the cookie DB deny-all (proven
    live — even CreateFile with all share flags fails), so copy-while-running
    is impossible. ``snapshot_real_profile`` must FAIL FAST with an actionable
    'fully quit the browser' message — never hang, never a silent signed-out
    copy. (Real-profile browsing on Windows therefore requires the browser
    fully closed incl. background/tray; the live-drive path is #95669.)
    """
    import time as _t
    chrome = _find_chrome()
    if not chrome:
        pytest.skip("Chrome not installed on this runner")

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from hermes_cli import browser_connect as bc

    user_data = tmp_path / "chrome-user-data"
    user_data.mkdir()

    proc = subprocess.Popen(
        [
            chrome, "--headless=new", "--disable-gpu", "--no-first-run",
            "--no-default-browser-check", f"--user-data-dir={user_data}",
            "--remote-debugging-port=0", "about:blank",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
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

        if not _raw_copy_raises_while_locked(str(cookies)):
            pytest.skip("Chrome did not share-lock the cookie DB on this runner")

        import hermes_cli.browser_connect as bc_mod
        orig = bc_mod.real_profile_data_dir
        bc_mod.real_profile_data_dir = lambda browser, system=None: str(user_data)
        try:
            # Must return FAST (fail-fast lock probe), not hang. Assert both the
            # contract and that it took well under the old 24-min hang.
            t0 = _t.time()
            dst, err = bc.snapshot_real_profile("chrome", src=str(user_data))
            elapsed = _t.time() - t0
        finally:
            bc_mod.real_profile_data_dir = orig

        assert dst is None, "must not return a (silently broken) copy while locked"
        assert err is not None
        low = err.lower()
        assert "locked" in low or "running" in low, f"unclear error: {err}"
        assert "quit" in low or "close" in low, f"error must tell the user to quit: {err}"
        assert elapsed < 30, f"snapshot hung on a locked profile ({elapsed:.0f}s) — must fail fast"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_copy_works_when_chrome_closed(tmp_path):
    """Sanity: with NO live Chrome holding the dir, the copy succeeds on Windows
    (the supported path). Creates a profile with a real Chrome, closes it, then
    snapshots — cookies DB must copy and be a valid SQLite file."""
    chrome = _find_chrome()
    if not chrome:
        pytest.skip("Chrome not installed on this runner")

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from hermes_cli import browser_connect as bc

    user_data = tmp_path / "ud"
    user_data.mkdir()
    # One-shot Chrome run to materialize a profile, then it exits.
    subprocess.run(
        [
            chrome, "--headless=new", "--disable-gpu", "--no-first-run",
            "--no-default-browser-check", f"--user-data-dir={user_data}",
            "--dump-dom", "about:blank",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
    )
    # Chrome has exited; the DB is now unlocked.
    cookies = None
    for rel in (r"Default\Network\Cookies", r"Default\Cookies"):
        cand = user_data / rel
        if cand.is_file():
            cookies = cand
            break
    if cookies is None:
        pytest.skip("Chrome did not create a Cookies DB in the one-shot run")

    dst = tmp_path / "copy" / "Cookies"
    assert bc._copy_auth_file(str(cookies), str(dst)) is True
    assert dst.is_file()
    con = sqlite3.connect(str(dst))
    try:
        con.execute("SELECT name FROM sqlite_master LIMIT 1")
    finally:
        con.close()

