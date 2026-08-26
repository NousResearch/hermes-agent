"""DIAGNOSTIC (Windows live): which read strategy can open Chrome's locked DB?

Not a pass/fail test — it prints, for a cookie DB held open by a running
Chrome, which of several open strategies SUCCEED. This tells us empirically
whether ANY in-process read path exists (immutable=1, nolock, raw win32 share
flags, shutil) before we reach for VSS/admin. Runs on windows-latest only.
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

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only diagnostic")

_CHROME = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)


def _chrome():
    for p in _CHROME:
        if os.path.isfile(p):
            return p
    return shutil.which("chrome") or shutil.which("chrome.exe")


def test_diagnose_locked_db_read_strategies(tmp_path, capsys):
    chrome = _chrome()
    if not chrome:
        pytest.skip("no chrome")
    ud = tmp_path / "ud"; ud.mkdir()
    proc = subprocess.Popen(
        [chrome, "--headless=new", "--disable-gpu", "--no-first-run",
         "--no-default-browser-check", f"--user-data-dir={ud}",
         "--remote-debugging-port=0", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    results = []
    try:
        ck = None
        deadline = time.time() + 60
        while time.time() < deadline and not ck:
            for rel in (r"Default\Network\Cookies", r"Default\Cookies"):
                c = ud / rel
                if c.is_file() and c.stat().st_size > 0:
                    ck = c; break
            time.sleep(1)
        if not ck:
            pytest.skip("no cookie db materialized")

        def rec(name, fn):
            try:
                fn(); results.append((name, "OK"))
            except Exception as e:
                results.append((name, f"{type(e).__name__}: {str(e)[:80]}"))

        # 1. plain shutil copy (the original failing path)
        rec("shutil.copy2", lambda: shutil.copy2(str(ck), str(tmp_path / "c1")))
        # 2. open() read binary
        rec("open-rb", lambda: open(str(ck), "rb").read(64))
        # 3. sqlite mode=ro
        rec("sqlite mode=ro backup", lambda: _bk(f"file:{ck}?mode=ro", tmp_path / "c3"))
        # 4. sqlite immutable=1 (tells sqlite the file won't change; skips locking)
        rec("sqlite immutable=1 backup", lambda: _bk(f"file:{ck}?immutable=1", tmp_path / "c4"))
        # 5. sqlite mode=ro&nolock=1
        rec("sqlite ro+nolock backup", lambda: _bk(f"file:{ck}?mode=ro&nolock=1", tmp_path / "c5"))
        # 6. raw win32 CreateFile with full share flags, then read bytes
        rec("win32 share-all read", lambda: _win32_read(str(ck)))

        print("\n=== LOCKED-DB READ STRATEGY RESULTS ===")
        for name, outcome in results:
            print(f"  {name:32} -> {outcome}")
        # Surface in the CI log regardless of capture.
        sys.stderr.write("\n".join(f"{n} -> {o}" for n, o in results) + "\n")
    finally:
        proc.terminate()
        try: proc.wait(timeout=15)
        except subprocess.TimeoutExpired: proc.kill()


def _bk(uri, dst):
    src = sqlite3.connect(uri, uri=True, timeout=3)
    try:
        out = sqlite3.connect(str(dst))
        try:
            with out:
                src.backup(out)
        finally:
            out.close()
    finally:
        src.close()


def _win32_read(path):
    import ctypes
    from ctypes import wintypes
    GENERIC_READ = 0x80000000
    FILE_SHARE_ALL = 0x1 | 0x2 | 0x4  # READ|WRITE|DELETE
    OPEN_EXISTING = 3
    CreateFileW = ctypes.windll.kernel32.CreateFileW
    CreateFileW.restype = wintypes.HANDLE
    h = CreateFileW(path, GENERIC_READ, FILE_SHARE_ALL, None, OPEN_EXISTING, 0, None)
    if h == wintypes.HANDLE(-1).value or h is None:
        raise OSError(f"CreateFile failed err={ctypes.get_last_error()}")
    ctypes.windll.kernel32.CloseHandle(h)
