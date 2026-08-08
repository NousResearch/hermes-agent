"""Tests for pip entry-point provider discovery (hermes_agent.plugins group).

Verifies that ``providers/__init__.py`` imports provider plugins exposed via a
distribution's ``hermes_agent.plugins`` entry point, supporting both a
``module:func`` callable target and a bare self-registering ``module`` target.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import providers


REPO_ROOT = Path(__file__).resolve().parents[2]


def _clear_provider_caches():
    providers._REGISTRY.clear()
    providers._ALIASES.clear()
    providers._PROVIDER_LIST_CACHE = None
    providers._discovered = False
    for mod in list(sys.modules.keys()):
        if mod.startswith("plugins.model_providers") or mod.startswith(
            "_hermes_user_provider"
        ):
            del sys.modules[mod]


@pytest.fixture(autouse=True)
def _restore_real_discovery():
    """Snapshot registry state; on teardown re-run REAL discovery.

    These tests monkeypatch ``importlib.metadata.entry_points`` and evict the
    ``plugins.model_providers`` submodules to force re-discovery. Without an
    explicit restore, the emptied registry / ``sys.modules`` would leak into
    later tests (e.g. ``from plugins.model_providers.custom import ...``).

    This fixture is autouse and declared before ``monkeypatch`` is requested,
    so it tears down LAST — after ``entry_points`` is restored to the real
    implementation — letting the final ``_discover_providers()`` repopulate
    both the registry and ``sys.modules`` from the real filesystem plugins.
    """
    yield
    _clear_provider_caches()
    providers._discover_providers()



class _FakeEP:
    def __init__(self, name, loader):
        self.name = name
        self.group = "hermes_agent.plugins"
        self._loader = loader

    def load(self):
        return self._loader()


class _FakeEntryPoints:
    def __init__(self, eps):
        self._eps = eps

    def select(self, group):
        return [e for e in self._eps if e.group == group]


def _register_via_callable():
    from providers.base import ProviderProfile

    def register():
        providers.register_provider(
            ProviderProfile(name="ep-callable", aliases=("epc",), base_url="https://a.test/v1")
        )

    return register  # ep.load() returns the callable; discovery invokes it


def _register_via_module():
    # ep.load() returns a non-callable object; the import side effect already
    # registered the profile (mirrors a bare ``module`` target).
    from providers.base import ProviderProfile

    providers.register_provider(
        ProviderProfile(name="ep-module", base_url="https://b.test/v1")
    )
    return object()  # non-callable → discovery must NOT try to call it


def test_entry_point_callable_and_module_targets(monkeypatch):
    fake_eps = _FakeEntryPoints(
        [
            _FakeEP("ep-callable", _register_via_callable),
            _FakeEP("ep-module", _register_via_module),
        ]
    )
    import importlib.metadata as md

    monkeypatch.setattr(md, "entry_points", lambda: fake_eps)
    _clear_provider_caches()
    try:
        assert providers.get_provider_profile("ep-callable") is not None
        assert providers.get_provider_profile("epc") is not None  # alias
        assert providers.get_provider_profile("ep-module") is not None
    finally:
        _clear_provider_caches()


def test_entry_point_failure_is_isolated(monkeypatch):
    def _boom():
        raise RuntimeError("broken plugin")

    fake_eps = _FakeEntryPoints(
        [
            _FakeEP("broken", _boom),
            _FakeEP("ep-callable", _register_via_callable),
        ]
    )
    import importlib.metadata as md

    monkeypatch.setattr(md, "entry_points", lambda: fake_eps)
    _clear_provider_caches()
    try:
        # A broken entry point must not prevent the good one from registering.
        assert providers.get_provider_profile("ep-callable") is not None
    finally:
        _clear_provider_caches()


def test_filesystem_plugins_win_over_entry_points(monkeypatch):
    """Entry points scan last, so a bundled/user profile of the same name wins."""
    from providers.base import ProviderProfile

    def _register_ep_openrouter():
        def register():
            providers.register_provider(
                ProviderProfile(name="openrouter", base_url="https://impostor.test/v1")
            )

        return register

    fake_eps = _FakeEntryPoints([_FakeEP("openrouter", _register_ep_openrouter)])
    import importlib.metadata as md

    monkeypatch.setattr(md, "entry_points", lambda: fake_eps)
    _clear_provider_caches()
    try:
        p = providers.get_provider_profile("openrouter")
        assert p is not None
        # The bundled OpenRouter profile (real base_url) must win, not the impostor.
        assert "impostor.test" not in (p.base_url or "")
    finally:
        _clear_provider_caches()
