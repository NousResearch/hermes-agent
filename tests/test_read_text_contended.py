"""Readers must ride out another process's atomic replace on Windows.

On Windows, ``open()`` of a file that another process is atomically replacing
fails with a bare ``PermissionError`` (EACCES, no winerror: the C runtime maps
the sharing violation). Writers already retry in ``utils.atomic_replace``;
``utils.read_text_contended`` is the reader-side twin, used for ``auth.json``.
"""
from __future__ import annotations

import errno
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import utils
from utils import read_text_contended


def _flaky_read_text(monkeypatch, target: Path, failures: int, exc_factory):
    """Fail the first *failures* reads of *target*, then read normally."""
    real = Path.read_text
    calls = {"n": 0}

    def read_text(self, *args, **kwargs):
        if self == target:
            calls["n"] += 1
            if calls["n"] <= failures:
                raise exc_factory()
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(utils.time, "sleep", lambda _s: None)
    return calls


def _sharing_violation():
    return PermissionError(errno.EACCES, "Permission denied")


def test_windows_sharing_violation_is_retried_until_the_replace_lands(tmp_path, monkeypatch):
    target = tmp_path / "auth.json"
    target.write_text('{"version": 1}', encoding="utf-8")
    monkeypatch.setattr(utils, "_IS_WINDOWS", True)
    calls = _flaky_read_text(monkeypatch, target, failures=3, exc_factory=_sharing_violation)

    assert read_text_contended(target) == '{"version": 1}'
    assert calls["n"] == 4


def test_windows_persistent_denial_still_raises_after_bounded_retries(tmp_path, monkeypatch):
    target = tmp_path / "auth.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(utils, "_IS_WINDOWS", True)
    calls = _flaky_read_text(monkeypatch, target, failures=10_000, exc_factory=_sharing_violation)

    with pytest.raises(PermissionError):
        read_text_contended(target)
    assert calls["n"] == utils._READ_RETRY_ATTEMPTS


def test_permission_error_off_windows_raises_immediately(tmp_path, monkeypatch):
    target = tmp_path / "auth.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(utils, "_IS_WINDOWS", False)
    calls = _flaky_read_text(monkeypatch, target, failures=10_000, exc_factory=_sharing_violation)

    with pytest.raises(PermissionError):
        read_text_contended(target)
    assert calls["n"] == 1


def test_non_contention_errors_raise_immediately(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "_IS_WINDOWS", True)
    monkeypatch.setattr(utils.time, "sleep", lambda _s: pytest.fail("must not retry a missing file"))

    with pytest.raises(FileNotFoundError):
        read_text_contended(tmp_path / "missing.json")


def test_utf8_bom_is_stripped_like_the_previous_reader(tmp_path):
    target = tmp_path / "auth.json"
    target.write_bytes(b"\xef\xbb\xbf" + b'{"version": 1}')

    assert json.loads(read_text_contended(target)) == {"version": 1}


def test_auth_store_load_survives_a_transient_sharing_violation(tmp_path, monkeypatch):
    """Before the fix a single collision made _load_auth_store treat auth.json as unreadable."""
    from hermes_cli import auth as auth_mod

    auth_file = tmp_path / "auth.json"
    store = {"version": 1, "providers": {"example": {"access_token": "placeholder"}}}
    auth_file.write_text(json.dumps(store), encoding="utf-8")
    monkeypatch.setattr(utils, "_IS_WINDOWS", True)
    _flaky_read_text(monkeypatch, auth_file, failures=2, exc_factory=_sharing_violation)

    loaded = auth_mod._load_auth_store(auth_file)

    assert loaded.get("providers", {}).get("example", {}).get("access_token") == "placeholder"


def _hold_without_sharing(path, seconds, ready):
    """Open *path* with no sharing (the OS state a reader sees mid-replace), then release it."""
    import ctypes
    import time

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateFileW.restype = ctypes.c_void_p
    handle = kernel32.CreateFileW(str(path), 0x80000000, 0, None, 3, 0x80, None)  # GENERIC_READ, share none
    if handle in (None, ctypes.c_void_p(-1).value):
        return  # ``ready`` stays unset; the test fails on its wait instead of hanging.
    try:
        ready.set()
        time.sleep(seconds)
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("hold_s, readable", [(0.3, True), (2.0, False)])
def test_real_windows_sharing_violation(tmp_path, hold_s, readable):
    """A short hold (as an atomic replace is) is ridden out; one outlasting the retry fails closed."""
    import threading

    target = tmp_path / "auth.json"
    target.write_text('{"version": 1}', encoding="utf-8")
    ready = threading.Event()
    holder = threading.Thread(target=_hold_without_sharing, args=(target, hold_s, ready), daemon=True)
    holder.start()
    try:
        assert ready.wait(10), "could not open the file without sharing"
        if readable:
            assert read_text_contended(target) == '{"version": 1}'
        else:
            with pytest.raises(PermissionError):
                read_text_contended(target)
    finally:
        holder.join(10)
