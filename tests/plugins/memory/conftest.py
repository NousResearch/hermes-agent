"""Pytest config for memory tests - inject fake mem0 module.

The real mem0 package is lazy-installed (see tools/lazy_deps.py). Tests
that exercise code paths which import mem0 must inject a fake module
via sys.modules so the import resolves to a controllable shim.
"""
import sys
import types


# Create a fake mem0 module that satisfies `from mem0 import Memory`
# Tests that need a real shim use monkeypatch.setattr on the shim's
# `Memory` attribute. The shim's `from_config` classmethod is what the
# production code calls.
_fake_mem0 = types.ModuleType("mem0")
_fake_mem0.Memory = None  # placeholder; tests override per-case
sys.modules.setdefault("mem0", _fake_mem0)
