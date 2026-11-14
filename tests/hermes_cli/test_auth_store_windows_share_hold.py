"""auth.json reads ride out a real Windows no-share hold (AV scanners, other processes' exclusive opens).

Uses the production retry delays, unlike test_auth_store_read_failure.py, which zeroes them to count attempts.

Contributed by @zachariahismyname (from #133151) and verified on Windows 10/11
against main + this PR before being folded in.
"""

import ctypes
import json
import threading
import time

import pytest

import hermes_cli.auth as auth

pytestmark = pytest.mark.platforms("windows")


def _hold_without_sharing(path, seconds, ready):
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateFileW.restype = ctypes.c_void_p
    handle = kernel32.CreateFileW(
        str(path), 0x80000000, 0, None, 3, 0x80, None
    )  # GENERIC_READ, share none
    if handle in (None, ctypes.c_void_p(-1).value):
        return  # ``ready`` stays unset; the test fails on its wait instead of hanging.
    try:
        ready.set()
        time.sleep(seconds)
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


@pytest.mark.parametrize("hold_s, readable", [(0.3, True), (3.0, False)])
def test_auth_store_load_under_a_real_no_share_hold(tmp_path, hold_s, readable):
    store = {
        "version": auth.AUTH_STORE_VERSION,
        "providers": {"nous": {"api_key": "placeholder"}},
    }
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps(store), encoding="utf-8")
    ready = threading.Event()
    holder = threading.Thread(
        target=_hold_without_sharing, args=(auth_file, hold_s, ready), daemon=True
    )
    holder.start()
    try:
        assert ready.wait(10), "could not open auth.json without sharing"
        if readable:
            assert (
                auth._load_auth_store(auth_file)["providers"]["nous"]["api_key"]
                == "placeholder"
            )
        else:
            with pytest.raises(PermissionError):
                auth._load_auth_store(auth_file)
    finally:
        holder.join(10)
    # Fail-loud, never degrade: the store on disk is left untouched either way.
    assert auth_file.read_text(encoding="utf-8") == json.dumps(store)
