"""Contract tests for _pid_is_descendant (#94084).

A second consumer (the update-path manual-gateway sweep) depends on this
primitive, so the semantics are locked with a fake psutil chain: direct
match, wrapper-descendant match, unrelated tree, PID-1 stop, depth
exhaustion, and the psutil-unavailable degradation to equality-only.
"""
from types import SimpleNamespace

import pytest

from hermes_cli import gateway as gw


class _FakeProcess:
    def __init__(self, pid, parent=None):
        self.pid = pid
        self._parent = parent

    def parent(self):
        return self._parent


def _chain(*pids):
    """Build a fake psutil.Process chain child->...->root from PID list."""
    proc = None
    for pid in reversed(pids):
        proc = _FakeProcess(pid, parent=proc)
    return proc


def _patch_psutil(monkeypatch, *, available=True):
    """Install a fake psutil module capturing Process() lookups."""
    calls = {"pids": []}

    if available:
        import psutil as real_psutil

        class _FakePsutil:
            Error = real_psutil.Error
            NoSuchProcess = real_psutil.NoSuchProcess

            @staticmethod
            def Process(pid):
                calls["pids"].append(pid)
                if pid in calls.get("chain", {}):
                    return calls["chain"][pid]
                raise real_psutil.NoSuchProcess(pid)

        fake = _FakePsutil
        import sys

        monkeypatch.setitem(sys.modules, "psutil", fake)
    else:
        # Make `import psutil` genuinely fail: remove the cached module and
        # point the loader at a path where it does not exist.
        import builtins
        import sys

        monkeypatch.delitem(sys.modules, "psutil", raising=False)
        real_import = builtins.__import__

        def _blocked_import(name, *args, **kwargs):
            if name == "psutil" or name.startswith("psutil."):
                raise ImportError("psutil blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked_import)
    return calls


def test_direct_pid_match_is_true(monkeypatch):
    """Equality is inclusive: ancestor-or-self, not strict descendant."""
    assert gw._pid_is_descendant(100, 100) is True


def test_child_of_wrapper_matches(monkeypatch):
    """The #94050 shape: gateway is a descendant of the launchd wrapper."""
    calls = _patch_psutil(monkeypatch)
    # gateway 200 -> wrapper 100
    calls["chain"] = {200: _chain(200, 100)}
    assert gw._pid_is_descendant(200, 100) is True


def test_unrelated_tree_is_false(monkeypatch):
    calls = _patch_psutil(monkeypatch)
    # gateway 300 -> some other parent 999 -> init 1
    calls["chain"] = {300: _chain(300, 999, 1)}
    assert gw._pid_is_descendant(300, 100) is False


def test_walk_stops_at_pid_1(monkeypatch):
    """Reaching init without seeing the ancestor ends the walk (not related)."""
    calls = _patch_psutil(monkeypatch)
    calls["chain"] = {300: _chain(300, 200, 1)}
    assert gw._pid_is_descendant(300, 100) is False


def test_depth_exhaustion_is_false(monkeypatch):
    """A chain deeper than max_depth is treated as unrelated (bounded walk)."""
    calls = _patch_psutil(monkeypatch)
    deep_chain = list(range(1000, 1000 + 40))  # 40-deep, ancestor never hit
    calls["chain"] = {deep_chain[0]: _chain(*deep_chain)}
    assert gw._pid_is_descendant(deep_chain[0], 1, max_depth=4) is False


def test_missing_psutil_degrades_to_equality(monkeypatch):
    """Without psutil the walk is impossible: only a direct PID match counts."""
    _patch_psutil(monkeypatch, available=False)
    assert gw._pid_is_descendant(200, 100) is False
    assert gw._pid_is_descendant(100, 100) is True


def test_dead_process_race_is_false(monkeypatch):
    """Process vanishing between read and walk start answers False (race)."""
    calls = _patch_psutil(monkeypatch)
    calls["chain"] = {}  # Process() raises NoSuchProcess for any pid
    assert gw._pid_is_descendant(200, 100) is False


def test_none_inputs_are_false():
    assert gw._pid_is_descendant(None, 100) is False
    assert gw._pid_is_descendant(200, None) is False
