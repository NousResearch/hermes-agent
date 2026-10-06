"""Hermetic loader for cognition acceptance tests.

Loads plugin modules by file path under a synthetic package (so relative
imports work) and stubs the ``agent`` package with only ``memory_provider``
(its only dependency, stdlib-only) — no repo-wide imports, no network, no
hermes home. Runs under any Python 3.9+ interpreter.
"""

import importlib.util
import os
import sys
import types

COG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(
    os.path.join(COG_DIR, os.pardir, os.pardir, os.pardir)
)
PKG = "cogtest_under_test"


def _ensure_agent_stub():
    if "agent" not in sys.modules:
        pkg = types.ModuleType("agent")
        pkg.__path__ = [os.path.join(ROOT, "agent")]
        sys.modules["agent"] = pkg
    if "agent.memory_provider" not in sys.modules:
        path = os.path.join(ROOT, "agent", "memory_provider.py")
        spec = importlib.util.spec_from_file_location(
            "agent.memory_provider", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["agent.memory_provider"] = module
        spec.loader.exec_module(module)


def _ensure_pkg():
    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [COG_DIR]
        sys.modules[PKG] = pkg


def load(name):
    """Load cognition module ``name`` (e.g. 'queue') by file path."""
    _ensure_agent_stub()
    _ensure_pkg()
    full = PKG + "." + name
    if full in sys.modules:
        return sys.modules[full]
    path = os.path.join(COG_DIR, name + ".py")
    spec = importlib.util.spec_from_file_location(full, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module
