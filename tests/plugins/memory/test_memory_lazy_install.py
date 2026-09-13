"""Regression tests: supermemory + mem0 memory providers must lazy-install
their SDKs like honcho/hindsight.

Both providers ship a third-party SDK (``supermemory`` / ``mem0ai``) that is
NOT a core dependency. Before this fix they imported the SDK directly with no
``pm.ensure_import()`` preflight and had no extras mapping, so on hosted
instances the SDK was never installed and the provider silently reported
itself unavailable.

These tests pin the contract:

1. Both features are pyproject extras with an anchor in ``pm.extras.ANCHORS``
   (without a mapping, ``ensure_import()`` can't resolve the anchor import
   that proves the extra is installed — the original silent-dark bug).
2. Each provider's SDK-import chokepoint actually calls
   ``ensure_import(<extra>)``.
3. supermemory's ``is_available()`` no longer gates on the SDK being
   importable (the chicken-and-egg trap that stopped the provider loading at
   all on a sealed venv, so ``initialize()``/``ensure_import()`` never ran).
"""

from __future__ import annotations

import pytest

import pm
from pm.extras import ANCHORS


MEMORY_EXTRAS = {"supermemory": "supermemory", "mem0": "mem0"}


# ---------------------------------------------------------------------------
# 1. Extras contract — the core regression.
# ---------------------------------------------------------------------------


class TestExtrasAnchors:
    @pytest.mark.parametrize("extra,anchor", sorted(MEMORY_EXTRAS.items()))
    def test_extra_has_anchor(self, extra, anchor):
        # Without an anchor mapping, pm can't tell whether the extra is
        # installed, and ensure_import() can't prove a successful install.
        assert ANCHORS.get(extra) == anchor, (
            f"{extra!r} missing/wrong in pm.extras.ANCHORS — its SDK can "
            f"never lazy-install on a hosted instance."
        )


# ---------------------------------------------------------------------------
# 2. Import sites call ensure_import().
# ---------------------------------------------------------------------------


class TestSupermemoryEnsureCalled:
    def test_client_construction_calls_ensure(self, monkeypatch):
        """_SupermemoryClient.__init__ must call ensure_import('supermemory')
        before importing the SDK."""
        from plugins.memory.supermemory import _SupermemoryClient

        calls = []
        monkeypatch.setattr(
            pm, "ensure_import",
            lambda extra: calls.append(extra),
        )

        # Stub the SDK so construction doesn't need the real package. The
        # client does ``from supermemory import Supermemory`` right after
        # ensure_import(); inject a fake module.
        import sys
        import types

        fake = types.ModuleType("supermemory")
        fake.Supermemory = lambda **kw: object()
        monkeypatch.setitem(sys.modules, "supermemory", fake)

        _SupermemoryClient(api_key="k", timeout=5.0, container_tag="hermes")

        assert "supermemory" in calls, (
            "supermemory client did not call ensure_import('supermemory'); "
            f"calls={calls}"
        )


class TestMem0EnsureCalled:
    def test_create_backend_calls_ensure(self, monkeypatch):
        """The mem0 provider must call ensure_import('mem0') in
        _create_backend before importing the SDK."""
        from plugins.memory.mem0 import Mem0MemoryProvider

        calls = []
        monkeypatch.setattr(
            pm, "ensure_import",
            lambda extra: calls.append(extra),
        )

        prov = Mem0MemoryProvider()
        # Platform mode is the default; force a known mode and stub the backend
        # import so we isolate the ensure_import() call.
        prov._mode = "platform"
        prov._api_key = "k"

        import sys
        import types

        fake = types.ModuleType("mem0")
        fake.MemoryClient = lambda **kw: object()
        fake.Memory = object
        monkeypatch.setitem(sys.modules, "mem0", fake)
        # _backend imports ``from mem0 import MemoryClient`` lazily inside
        # PlatformBackend.__init__, so the fake module satisfies it.

        prov._create_backend()

        assert "mem0" in calls, (
            f"mem0 _create_backend did not call ensure_import('mem0'); "
            f"calls={calls}"
        )


# ---------------------------------------------------------------------------
# 3. supermemory is_available() chicken-and-egg fix.
# ---------------------------------------------------------------------------


class TestSupermemoryIsAvailable:
    def test_available_with_key_even_when_sdk_absent(self, monkeypatch):
        """With the key set but the SDK not importable, is_available() must
        still return True — otherwise the provider never loads on a sealed
        venv and ensure_import() (which installs the SDK) never runs."""
        from plugins.memory.supermemory import SupermemoryMemoryProvider
        import builtins

        monkeypatch.setenv("SUPERMEMORY_API_KEY", "sk-test")

        # Make any attempt to import the SDK fail, simulating the
        # not-yet-installed sealed-venv state.
        real_import = builtins.__import__

        def _no_supermemory(name, *args, **kwargs):
            if name == "supermemory" or name.startswith("supermemory."):
                raise ImportError("No module named 'supermemory'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_supermemory)

        prov = SupermemoryMemoryProvider()
        assert prov.is_available() is True

    def test_unavailable_without_key(self, monkeypatch):
        from plugins.memory.supermemory import SupermemoryMemoryProvider

        monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)
        prov = SupermemoryMemoryProvider()
        assert prov.is_available() is False
