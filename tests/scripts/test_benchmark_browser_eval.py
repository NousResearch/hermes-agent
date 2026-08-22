"""Regression tests for scripts/benchmark_browser_eval.py teardown safety.

``main()`` spawns Chrome and then, inside a ``try``/``finally``, imports
``SUPERVISOR_REGISTRY`` and runs the eval loop. The ``finally`` tears down the
supervisor (``stop_all()``) and then the Chrome process + temp profile.

Two failure surfaces this covers (issue #36650):

1. The supervisor name was bound *inside* the ``try``. A failed import left it
   unbound, so the ``finally`` raised a secondary ``UnboundLocalError`` (a
   ``NameError`` subclass) that masked the real error and skipped the Chrome
   teardown. Fixed by hoisting the import
   above ``_start_chrome()``.
2. The ``finally`` ran multi-resource teardown in one block, so if
   ``stop_all()`` raised, the Chrome ``proc.terminate()`` / profile ``rmtree``
   were skipped — leaking the browser and an orphaned ``/tmp`` profile. Fixed
   by isolating ``stop_all()`` in its own ``try``/``except``.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _load_bench():
    spec = importlib.util.spec_from_file_location(
        "_benchmark_browser_eval_under_test",
        Path(__file__).resolve().parents[2] / "scripts" / "benchmark_browser_eval.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_surfaces_setup_import_error_without_masking(monkeypatch):
    """A failed supervisor import must surface as the original ImportError, not
    a secondary UnboundLocalError from the finally.

    On the old code the import lived inside the ``try`` (after Chrome was
    started), so the failed import triggered an ``UnboundLocalError`` (a
    ``NameError`` subclass) in the finally and ``main()`` raised *that* — the
    ``pytest.raises(ImportError)`` below is what catches the regression. The
    ``started == []`` assertion additionally pins
    the fix's fail-fast behaviour: the import is now resolved before Chrome is
    spawned at all.
    """
    bench = _load_bench()

    started: list[int] = []
    fake_proc = MagicMock()

    def fake_start(port):
        started.append(port)
        return fake_proc, "/tmp/hermes-bench-fake-profile", "ws://fake"

    monkeypatch.setattr(bench, "_start_chrome", fake_start)

    # A stand-in module that lacks SUPERVISOR_REGISTRY makes
    # `from tools.browser_supervisor import SUPERVISOR_REGISTRY` raise ImportError.
    broken = types.ModuleType("tools.browser_supervisor")
    monkeypatch.setitem(sys.modules, "tools.browser_supervisor", broken)
    monkeypatch.setattr(sys, "argv", ["benchmark_browser_eval"])

    with pytest.raises(ImportError):
        bench.main()

    assert started == []
    fake_proc.terminate.assert_not_called()


def test_teardown_survives_supervisor_stop_failure(monkeypatch):
    """If stop_all() raises during teardown, the Chrome proc + profile cleanup
    must still run, and the original error (not the teardown error) propagates.
    """
    bench = _load_bench()

    fake_proc = MagicMock()
    profile_dir = "/tmp/hermes-bench-fake-profile"
    monkeypatch.setattr(
        bench, "_start_chrome", lambda port: (fake_proc, profile_dir, "ws://fake")
    )

    registry = MagicMock()
    registry.get_or_start.side_effect = RuntimeError("setup boom")
    registry.stop_all.side_effect = RuntimeError("stop boom")
    fake_mod = types.ModuleType("tools.browser_supervisor")
    fake_mod.SUPERVISOR_REGISTRY = registry
    monkeypatch.setitem(sys.modules, "tools.browser_supervisor", fake_mod)

    rmtree_calls: list[str] = []
    monkeypatch.setattr(bench.shutil, "rmtree", lambda p, **kw: rmtree_calls.append(p))
    monkeypatch.setattr(sys, "argv", ["benchmark_browser_eval"])

    # The setup failure propagates; the swallowed stop_all() error does not.
    with pytest.raises(RuntimeError, match="setup boom"):
        bench.main()

    # stop_all() blew up, but Chrome teardown still ran — no leaked proc/profile.
    fake_proc.terminate.assert_called_once()
    assert rmtree_calls == [profile_dir]


def test_start_chrome_timeout_cleans_up_before_raising(monkeypatch):
    """If Chrome never exposes CDP, ``_start_chrome()`` must own its cleanup —
    terminate + wait/kill the proc and rmtree the temp profile — before raising.

    This path fails *before* ``main()`` binds ``proc``/``profile`` at the call
    site, so ``main()``'s finally can never reach them. On the old code the
    timeout path only called ``proc.terminate()``, leaking the process (never
    reaped) and the ``/tmp`` profile dir (never removed).
    """
    bench = _load_bench()

    fake_proc = MagicMock()
    profile_dir = "/tmp/hermes-bench-timeout-profile"
    monkeypatch.setattr(bench, "_find_chrome", lambda: "/usr/bin/fake-chrome")
    monkeypatch.setattr(bench.tempfile, "mkdtemp", lambda **kw: profile_dir)
    monkeypatch.setattr(bench.subprocess, "Popen", lambda *a, **kw: fake_proc)

    # CDP never comes up: every probe raises, so the poll loop runs to deadline.
    monkeypatch.setattr(
        bench.urllib.request, "urlopen", MagicMock(side_effect=OSError("refused"))
    )

    # Drive the poll loop past its deadline without real waiting: one probe
    # iteration, then monotonic jumps past the 15s deadline.
    ticks = iter([0.0, 1.0, 100.0])
    monkeypatch.setattr(bench.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(bench.time, "sleep", lambda *_: None)

    rmtree_calls: list[str] = []
    monkeypatch.setattr(bench.shutil, "rmtree", lambda p, **kw: rmtree_calls.append(p))

    with pytest.raises(RuntimeError, match="Chrome didn't expose CDP"):
        bench._start_chrome(9333)

    # The timeout path reaped the process and removed the profile itself.
    fake_proc.terminate.assert_called_once()
    fake_proc.wait.assert_called_once()
    assert rmtree_calls == [profile_dir]
