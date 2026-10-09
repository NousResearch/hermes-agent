"""Live Windows obstacles to removing a replaced PM entry.

A running Hermes process keeps the replaced interpreter's DLLs mapped (#124807),
and archives such as PortableGit ship read-only files; both make deleting the
old entry fail with WinError 5. The OS produces each obstacle here, no model.
"""

from __future__ import annotations

import ctypes
import shutil
import sys
from pathlib import Path

import pytest

from pm.lock import Facts
from tests.pm.test_pm_core import pm_env as pm_env
from tests.pm._fixtures import served as served


@pytest.mark.platforms("windows")
def test_mapped_dll_in_replaced_entry_does_not_fail_reinstall(pm_env):
    from pm.cli import cmd_gc
    from pm.install import ensure

    _, runtime, *_ = pm_env
    ensure("faketool", base_env={})
    entry = runtime / Facts(runtime / "facts.json").get("faketool")["entry"]
    dlls = Path(sys.base_prefix) / "DLLs"
    source = next(iter(sorted(dlls.glob("libcrypto-3*.dll")) or sorted(dlls.glob("*.dll"))))
    held = entry / "DLLs" / "held.dll"
    held.parent.mkdir()
    shutil.copy2(source, held)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LoadLibraryW.restype = ctypes.c_void_p
    kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
    handle = kernel32.LoadLibraryW(str(held))
    assert handle, ctypes.get_last_error()
    try:
        (entry / "bin/faketool").unlink()
        runner = ensure("faketool", base_env={})
        assert entry.name in runner.env["PATH"]
        assert (entry / "bin/faketool").read_bytes() == b"#!x"
        assert not (runtime / f".previous-{entry.name}").exists()
        assert [p.name for p in runtime.glob(".reclaim-*/DLLs/held.dll")]
    finally:
        kernel32.FreeLibrary(handle)
    cmd_gc(None)
    assert not list(runtime.glob(".reclaim-*"))


@pytest.mark.platforms("windows")
def test_read_only_file_in_replaced_entry_is_removed_on_reinstall(pm_env):
    import os
    import stat

    from pm.install import ensure

    _, runtime, *_ = pm_env
    ensure("faketool", base_env={})
    entry = runtime / Facts(runtime / "facts.json").get("faketool")["entry"]
    hosts = entry / "etc" / "hosts"
    hosts.parent.mkdir()
    hosts.write_bytes(b"127.0.0.1 localhost")
    os.chmod(hosts, stat.S_IREAD)
    (entry / "bin/faketool").unlink()
    ensure("faketool", base_env={})
    assert not (runtime / f".previous-{entry.name}").exists()
    assert not list(runtime.glob(".reclaim-*")), "a read-only file must not strand the replaced tree"
