"""Regression test: mem0 provider must not report available when SDK missing.

Bug: Mem0MemoryProvider.is_available() returned True based on config alone
(api_key/host present) without verifying the `mem0` SDK is importable. This
made `hermes memory status` report "available OK" for a provider whose
dependency was not installed — a false positive. The lazy-install path only
runs at first real use, so the status lied until then (and permanently if
allow_lazy_installs=false or offline).

This test asserts is_available() is False when `mem0` cannot be imported,
without performing any network install. (The positive case where both config
and SDK are present is covered by tests/agent/test_memory_provider.py.)
"""

import builtins
import os
import sys

import pytest


def _block_import(name):
    """Install a meta_path finder that makes `import name` fail."""

    class _Blocked:
        _repro_block = True

        def find_spec(self, fullname, path, target=None):
            if fullname is None:
                return None
            if fullname == name or fullname.startswith(name + "."):
                raise ImportError(f"blocked {fullname}")
            return None

    sys.meta_path.insert(0, _Blocked())


def _restore_meta_path():
    saved = [p for p in sys.meta_path if not getattr(p, "_repro_block", False)]
    sys.meta_path[:] = saved


@pytest.fixture
def mem0_provider(monkeypatch):
    """Provider with config present (api_key set) but SDK import blocked.

    This is the exact false-positive condition: config says 'ready' while the
    mem0 SDK is not importable. Pre-fix is_available() returned True here
    (the bug); post-fix returns False.
    """
    from plugins.memory.mem0 import Mem0MemoryProvider

    monkeypatch.setenv("MEM0_API_KEY", "test-key-not-real")
    yield Mem0MemoryProvider()
    _restore_meta_path()


def test_is_available_false_when_sdk_missing(mem0_provider):
    """is_available() must be False when the mem0 SDK is not importable.

    Reproduces #70979: with MEM0_API_KEY set (config 'ready') but the SDK
    unimportable, the provider must NOT report available. Pre-fix this
    asserted True (false positive); the fix adds an offline find_spec check.
    """
    _block_import("mem0")
    assert mem0_provider.is_available() is False
