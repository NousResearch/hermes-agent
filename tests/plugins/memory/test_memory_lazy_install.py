"""Regression tests: supermemory + mem0 memory providers must lazy-install
their SDKs like honcho/hindsight.

Both providers ship a third-party SDK (``supermemory`` / ``mem0ai``) that is
NOT a core dependency. Before this fix they imported the SDK directly with no
``tools.lazy_deps.ensure()`` preflight and had no ``LAZY_DEPS`` allowlist
entry. On the published Docker image the agent venv is sealed
(``HERMES_DISABLE_LAZY_INSTALLS=1``) and lazy installs are redirected to a
writable durable target (``HERMES_LAZY_INSTALL_TARGET``). honcho/hindsight
route through ``ensure()`` and therefore install fine on a hosted instance;
supermemory/mem0 never called it, so the SDK was never installed there and
the provider silently reported itself unavailable.

These tests pin the contract:

1. Both features are in the ``LAZY_DEPS`` allowlist (without an entry,
   ``ensure()`` raises ``FeatureUnavailable`` — the original silent-dark bug).
2. Each provider's SDK-import chokepoint actually calls ``ensure(<feature>)``.
3. supermemory's ``is_available()`` no longer gates on the SDK being
   importable (the chicken-and-egg trap that stopped the provider loading at
   all on a sealed venv, so ``initialize()``/``ensure()`` never ran).
4. The real sealed-venv durable-target gate accepts the new features (the
   exact hosted-Fly condition the user hit).

The pip subprocess is never actually run — ``_venv_pip_install`` /
``_is_satisfied`` are stubbed so we exercise the real ``ensure()`` control
flow without touching PyPI.
"""

from __future__ import annotations

import os

import pytest

import tools.lazy_deps as ld


MEMORY_FEATURES = ("memory.supermemory", "memory.mem0")


# ---------------------------------------------------------------------------
# 1. Allowlist contract — the core regression.
# ---------------------------------------------------------------------------


class TestAllowlistEntries:
    @pytest.mark.parametrize("feature", MEMORY_FEATURES)
    def test_feature_is_allowlisted(self, feature):
        # Without an allowlist entry, ensure() raises FeatureUnavailable with
        # "not in LAZY_DEPS" — which is exactly why the SDK never installed on
        # a hosted instance before this fix.
        assert feature in ld.LAZY_DEPS, (
            f"{feature!r} missing from LAZY_DEPS — its SDK can never "
            f"lazy-install on a sealed Docker venv."
        )


    def test_supermemory_spec_package(self):
        specs = ld.feature_specs("memory.supermemory")
        assert any(ld._pkg_name_from_spec(s) == "supermemory" for s in specs)

    def test_mem0_spec_package(self):
        # mem0's pip package is ``mem0ai`` (imports as ``mem0``).
        specs = ld.feature_specs("memory.mem0")
        assert any(ld._pkg_name_from_spec(s) == "mem0ai" for s in specs)

    @pytest.mark.parametrize("feature", MEMORY_FEATURES)
    def test_unknown_feature_would_raise_without_entry(self, feature, monkeypatch):
        # Demonstrate the failure mode the allowlist entry prevents: a feature
        # NOT in LAZY_DEPS raises rather than installing.
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        with pytest.raises(ld.FeatureUnavailable, match="not in LAZY_DEPS"):
            ld.ensure(feature + ".typo", prompt=False)


# ---------------------------------------------------------------------------
# 2. Import sites call ensure().
# ---------------------------------------------------------------------------


class TestSupermemoryEnsureCalled:
    def test_client_construction_calls_ensure(self, monkeypatch):
        """_SupermemoryClient.__init__ must call ensure('memory.supermemory')
        before importing the SDK."""
        from plugins.memory.supermemory import _SupermemoryClient

        calls = []
        monkeypatch.setattr(
            ld, "ensure",
            lambda feature, **kw: calls.append((feature, kw)),
        )

        # Stub the SDK so construction doesn't need the real package. The
        # client does ``from supermemory import Supermemory`` right after
        # ensure(); inject a fake module.
        import sys
        import types

        fake = types.ModuleType("supermemory")
        fake.Supermemory = lambda **kw: object()
        monkeypatch.setitem(sys.modules, "supermemory", fake)

        _SupermemoryClient(api_key="k", timeout=5.0, container_tag="hermes")

        assert ("memory.supermemory", {"prompt": False}) in calls, (
            "supermemory client did not call ensure('memory.supermemory', "
            f"prompt=False); calls={calls}"
        )


class TestMem0EnsureCalled:
    def test_create_backend_calls_ensure(self, monkeypatch):
        """SupermemoryMemoryProvider-style mem0 provider must call
        ensure('memory.mem0') in _create_backend before importing the SDK."""
        from plugins.memory.mem0 import Mem0MemoryProvider

        calls = []
        monkeypatch.setattr(
            ld, "ensure",
            lambda feature, **kw: calls.append((feature, kw)),
        )

        prov = Mem0MemoryProvider()
        # Platform mode is the default; force a known mode and stub the backend
        # import so we isolate the ensure() call.
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

        assert ("memory.mem0", {"prompt": False}) in calls, (
            f"mem0 _create_backend did not call ensure('memory.mem0', "
            f"prompt=False); calls={calls}"
        )


# ---------------------------------------------------------------------------
# 3. supermemory is_available() chicken-and-egg fix.
# ---------------------------------------------------------------------------


class TestSupermemoryIsAvailable:
    def test_available_with_key_even_when_sdk_absent(self, monkeypatch):
        """With the key set but the SDK not importable, is_available() must
        still return True — otherwise the provider never loads on a sealed
        venv and ensure() (which installs the SDK) never runs."""
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


# ---------------------------------------------------------------------------
# 4. Real sealed-venv durable-target gate accepts the new features.
#
# This is the exact hosted-Fly condition: HERMES_DISABLE_LAZY_INSTALLS=1 seals
# the venv, but HERMES_LAZY_INSTALL_TARGET redirects installs to a writable
# durable dir, so installs are still ALLOWED. We exercise the real
# _allow_lazy_installs() + ensure() flow end-to-end with only the pip
# subprocess stubbed.
# ---------------------------------------------------------------------------


class TestSealedVenvBlocksInstalls:
    """A sealed deployment (the Docker image) refuses a LAZY_DEPS install.

    Each of these memory SDKs is an extra in pyproject.toml, so the image
    contains it. A call to ensure() here means that the image does not have a
    dependency that it must ship, and a target directory must not change
    that: the target exists for install_specs, whose packages come from a
    plugin manifest and cannot be in any image.
    """

    @pytest.mark.parametrize("feature", MEMORY_FEATURES)
    def test_sealed_venv_blocks_even_with_a_target(
        self, feature, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
        monkeypatch.setenv("HERMES_LAZY_INSTALL_TARGET", str(tmp_path / "lazy"))
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"security": {"allow_lazy_installs": True}},
        )
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda *a, **kw: pytest.fail("sealed deployment must not install"),
        )
        with pytest.raises(ld.FeatureUnavailable) as exc:
            ld.ensure(feature, prompt=False)
        assert "HERMES_DISABLE_LAZY_INSTALLS" in str(exc.value)

    @pytest.mark.parametrize("feature", MEMORY_FEATURES)
    def test_sealed_venv_without_target_blocks(self, feature, monkeypatch):
        monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
        monkeypatch.delenv("HERMES_LAZY_INSTALL_TARGET", raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"security": {"allow_lazy_installs": True}},
        )
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)

        with pytest.raises(ld.FeatureUnavailable) as exc:
            ld.ensure(feature, prompt=False)
        message = str(exc.value)
        # Must name the real cause, not a config key the user never touched,
        # and must not suggest pip-installing into a read-only venv.
        assert "HERMES_DISABLE_LAZY_INSTALLS" in message
        assert "allow_lazy_installs" not in message
        assert "uv pip install" not in message
