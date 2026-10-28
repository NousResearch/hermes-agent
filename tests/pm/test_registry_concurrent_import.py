"""The built-in catalogue is complete for a thread that reads it during another thread's import.

``import`` puts a module in ``sys.modules`` before its body runs, and the built-in package
modules register their packages from that body. A reader that skipped the import because the
name was already in ``sys.modules`` got the registry without those packages and failed with
``unknown package`` while the first import was still running.
"""
import sys
import threading
import types

from pm import registry

_SLOW = "hermes_test_slow_builtin_packages"
_SYNC = "hermes_test_slow_builtin_packages_sync"


def test_reader_waits_for_a_builtin_import_in_progress(tmp_path, monkeypatch):
    sync = types.ModuleType(_SYNC)
    sync.started, sync.release = threading.Event(), threading.Event()
    monkeypatch.setitem(sys.modules, _SYNC, sync)
    (tmp_path / f"{_SLOW}.py").write_text(
        f"import {_SYNC} as sync\n"
        "from pm import registry\n"
        "sync.started.set()\n"
        "sync.release.wait(10)\n"
        "registry._packages['slow-tool'] = object()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(registry, "_BUILTIN_MODULES", (_SLOW,))
    monkeypatch.setattr(registry, "_packages", {})
    seen = []
    first = threading.Thread(target=registry._builtins_loaded)
    second = threading.Thread(target=lambda: seen.append("slow-tool" in registry._builtins_loaded()))
    try:
        first.start()
        assert sync.started.wait(10)
        second.start()
        second.join(0.5)  # returns at once without the fix; with it, blocks on the import lock
        sync.release.set()
        first.join(10)
        second.join(10)
    finally:
        sync.release.set()
        sys.modules.pop(_SLOW, None)

    assert seen == [True]
