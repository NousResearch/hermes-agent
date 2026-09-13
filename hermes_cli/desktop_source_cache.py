"""Reuse the desktop content hash while all source metadata is unchanged.

This is a build-freshness cache, not an integrity/security check. Normal launches
must not read thousands of source/test files (and trigger on-access AV scans).
Changed metadata always falls back to the existing content hash, so checkout
mtime churn alone still does not force a rebuild.
"""

import ctypes
import contextlib
import hashlib
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Callable


@lru_cache(maxsize=1)
def _windows_file_api():
    from ctypes import wintypes

    class FileBasicInfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_longlong) for name in (
            "creation", "access", "write", "change"
        )] + [("attributes", wintypes.DWORD)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                   ctypes.c_void_p, wintypes.DWORD]
    kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel, FileBasicInfo


def _change_time(path: Path, st: os.stat_result) -> int:
    if os.name != "nt":
        return st.st_ctime_ns
    # Windows stat().st_ctime is CREATION time on supported Python versions.
    # FILE_BASIC_INFO.ChangeTime also catches same-size edits with restored mtime.
    # Request metadata only: never FILE_READ_DATA or a read of source contents.
    kernel, info_type = _windows_file_api()
    handle = kernel.CreateFileW(str(path), 0x80, 7, None, 3, 0x02000000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        info = info_type()
        if not kernel.GetFileInformationByHandleEx(handle, 0, ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        return info.change
    finally:
        kernel.CloseHandle(handle)


def _source_metadata(project_root: Path, tree_dir: Path) -> str:
    from hermes_cli.main_web_build import _source_tree_files

    digest = hashlib.sha256()
    for path in _source_tree_files(project_root, tree_dir):
        st = path.stat()
        record = (str(path.relative_to(project_root)), st.st_dev, st.st_ino,
                  st.st_size, st.st_mtime_ns, _change_time(path, st))
        digest.update(json.dumps(record, ensure_ascii=True).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def cached_desktop_source_hash(project_root: Path, compute: Callable[[], str]) -> str:
    from hermes_constants import get_hermes_home

    tree_dir = project_root / "apps" / "desktop"
    cache_file = get_hermes_home() / "desktop-source-cache.json"
    scope = str(project_root.resolve())
    try:
        before = _source_metadata(project_root, tree_dir)
    except OSError:
        return compute()  # Metadata unavailable: keep the full content check.
    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
        if (isinstance(cache, dict) and cache.get("version") == 1
                and cache.get("root") == scope and cache.get("metadata") == before
                and isinstance(cache.get("contentHash"), str) and len(cache["contentHash"]) == 64):
            return cache["contentHash"]
    except (OSError, ValueError):
        pass

    content_hash = compute()
    temporary = None
    try:
        # Do not cache a digest if an editor/update changed inputs while we read.
        if before != _source_metadata(project_root, tree_dir):
            return content_hash
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_file.parent,
                                         prefix=".desktop-source-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({"version": 1, "root": scope, "metadata": before,
                       "contentHash": content_hash}, stream)
        temporary.replace(cache_file)
    except OSError:
        pass  # Read-only cache locations must not prevent building/launching.
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)
    return content_hash
