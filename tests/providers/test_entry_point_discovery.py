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
    def __init__(self, name, loader, value=""):
        self.name = name
        self.group = "hermes_agent.plugins"
        self.value = value
        self._loader = loader

    def load(self):
        return self._loader()


def _enable(monkeypatch, *names, disabled=()):
    """Gate helper: mark entry-point names enabled/disabled in config.

    ``_discover_entry_point_providers`` enforces the PluginManager's
    ``plugins.enabled`` opt-in allow-list, so tests must enable their fake
    entry points explicitly.
    """
    import hermes_cli.plugins as hp

    monkeypatch.setattr(hp, "_get_enabled_plugins", lambda: set(names))
    monkeypatch.setattr(hp, "_get_disabled_plugins", lambda: set(disabled))


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
    _enable(monkeypatch, "ep-callable", "ep-module")
    _clear_provider_caches()
    try:
        assert providers.get_provider_profile("ep-callable") is not None
        assert providers.get_provider_profile("epc") is not None  # alias
        assert providers.get_provider_profile("ep-module") is not None
    finally:
        _clear_provider_caches()


def test_entry_point_not_enabled_is_skipped(monkeypatch):
    """Entry points honor the plugins.enabled opt-in gate — installed ≠ loaded."""
    fake_eps = _FakeEntryPoints([_FakeEP("ep-callable", _register_via_callable)])
    import importlib.metadata as md

    monkeypatch.setattr(md, "entry_points", lambda: fake_eps)
    _enable(monkeypatch, "some-other-plugin")  # ep-callable NOT enabled
    _clear_provider_caches()
    try:
        assert providers.get_provider_profile("ep-callable") is None
    finally:
        _clear_provider_caches()


def test_entry_point_disabled_wins_over_enabled(monkeypatch):
    """plugins.disabled is a deny-list that beats plugins.enabled."""
    fake_eps = _FakeEntryPoints([_FakeEP("ep-callable", _register_via_callable)])
    import importlib.metadata as md

    monkeypatch.setattr(md, "entry_points", lambda: fake_eps)
    _enable(monkeypatch, "ep-callable", disabled=("ep-callable",))
    _clear_provider_caches()
    try:
        assert providers.get_provider_profile("ep-callable") is None
    finally:
        _clear_provider_caches()


def test_general_plugin_register_ctx_not_invoked(monkeypatch):
    """A register(ctx)-style general plugin sharing the group is never called."""
    calls = []

    def _general_plugin_target():
        def register(ctx):  # requires an argument — PluginManager contract
            calls.append(ctx)

        return register

    fake_eps = _FakeEntryPoints([_FakeEP("general-plugin", _general_plugin_target)])
    import importlib.metadata as md

    monkeypatch.setattr(md, "entry_points", lambda: fake_eps)
    _enable(monkeypatch, "general-plugin")
    _clear_provider_caches()
    try:
        providers._discover_providers()
        assert calls == []  # never invoked (would have been a TypeError anyway)
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
    _enable(monkeypatch, "broken", "ep-callable")
    _clear_provider_caches()
    try:
        # A broken entry point must not prevent the good one from registering.
        assert providers.get_provider_profile("ep-callable") is not None
    finally:
        _clear_provider_caches()


def test_filesystem_plugins_win_over_entry_points(monkeypatch):
    """Entry points are discovered FIRST (lowest precedence): last-writer-wins
    in register_provider() means a bundled/user profile of the same name
    overrides a pip impostor."""
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
    _enable(monkeypatch, "openrouter")  # enabled, so precedence is what's tested
    _clear_provider_caches()
    try:
        p = providers.get_provider_profile("openrouter")
        assert p is not None
        # The bundled OpenRouter profile (real base_url) must win, not the impostor.
        assert "impostor.test" not in (p.base_url or "")
    finally:
        _clear_provider_caches()


def test_platform_named_entry_point_not_loaded(monkeypatch):
    """A ``<name>-platform`` entry point must not be imported by this scan
    (#98438).

    Platform adapters are owned by the PluginManager, which loads them
    lazily; importing one here runs its ``from gateway.config import
    Platform`` while ``gateway.config`` may still be half-initialized (this
    scan can run inside the ``gateway.config -> hermes_cli.config`` import
    chain), killing the platform with a circular-import error.
    """
    loads = []

    def _platform_side_effect():
        loads.append("stub-platform")  # the import itself would already fail
        return object()

    fake_eps = _FakeEntryPoints(
        [_FakeEP("stub-platform", _platform_side_effect, value="stubplat")]
    )
    import importlib.metadata as md

    monkeypatch.setattr(md, "entry_points", lambda: fake_eps)
    _enable(monkeypatch, "stub-platform")
    _clear_provider_caches()
    try:
        providers._discover_providers()
        assert loads == []  # never imported
    finally:
        _clear_provider_caches()


def test_memory_provider_entry_point_not_loaded(monkeypatch, tmp_path):
    """An enabled memory-provider entry point (``MemoryProvider`` source
    signature) is not imported here — memory providers are owned by the
    ``plugins/memory`` discovery system."""
    loads = []
    # A resolvable top-level module whose source carries the memory-provider
    # markers the PluginManager's import-free classifier looks for.
    (tmp_path / "stub_mem_plugin.py").write_text(
        "from plugins.memory import register_memory_provider\n"
        "class StubMemoryProvider(MemoryProvider):\n"
        "    pass\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    def _mem_side_effect():
        loads.append("stub-mem")
        return object()

    fake_eps = _FakeEntryPoints(
        [_FakeEP("stub-mem", _mem_side_effect, value="stub_mem_plugin")]
    )
    import importlib.metadata as md

    monkeypatch.setattr(md, "entry_points", lambda: fake_eps)
    _enable(monkeypatch, "stub-mem")
    _clear_provider_caches()
    try:
        providers._discover_providers()
        assert loads == []  # never imported
    finally:
        sys.modules.pop("stub_mem_plugin", None)
        _clear_provider_caches()


def test_unresolvable_entry_point_still_loaded(monkeypatch):
    """Fail-open: an entry point whose source cannot be classified is still
    loaded, so a real provider is never dropped because classification
    failed (unresolvable module, thin re-export package, ...)."""
    from providers.base import ProviderProfile

    def _register_unresolvable():
        def register():
            providers.register_provider(
                ProviderProfile(name="ep-unresolvable", base_url="https://c.test/v1")
            )

        return register

    fake_eps = _FakeEntryPoints(
        [
            _FakeEP(
                "ep-unresolvable",
                _register_unresolvable,
                value="no_such_module_zq",
            )
        ]
    )
    import importlib.metadata as md

    monkeypatch.setattr(md, "entry_points", lambda: fake_eps)
    _enable(monkeypatch, "ep-unresolvable")
    _clear_provider_caches()
    try:
        assert providers.get_provider_profile("ep-unresolvable") is not None
    finally:
        _clear_provider_caches()


def test_platform_import_marker_entry_point_not_loaded(monkeypatch, tmp_path):
    """A hand-written ``plugins.enabled`` entry whose name lacks the
    ``-platform`` suffix is still skipped when its source carries the
    documented platform-adapter import (#98438).

    The name convention only covers PluginManager-generated ids; the
    ``from gateway.config import Platform`` base import is the remaining
    import-free platform signal (model-provider plugins never touch
    gateway.config).
    """
    loads = []
    (tmp_path / "stub_platform_plugin.py").write_text(
        "from gateway.config import Platform, PlatformConfig\n"
        "class StubAdapter:\n"
        "    pass\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    def _adapter_side_effect():
        loads.append("stub-adapter")  # the import itself would already fail
        return object()

    fake_eps = _FakeEntryPoints(
        [
            _FakeEP(
                "stub-adapter",
                _adapter_side_effect,
                value="stub_platform_plugin",
            )
        ]
    )
    import importlib.metadata as md

    monkeypatch.setattr(md, "entry_points", lambda: fake_eps)
    _enable(monkeypatch, "stub-adapter")
    _clear_provider_caches()
    try:
        providers._discover_providers()
        assert loads == []  # never imported
    finally:
        sys.modules.pop("stub_platform_plugin", None)
        _clear_provider_caches()


def test_plain_plugin_without_markers_still_loaded(monkeypatch, tmp_path):
    """A source-bearing entry point with NO platform/memory markers keeps
    the historical load path — the marker scan must not over-skip."""
    from providers.base import ProviderProfile

    (tmp_path / "stub_plain_plugin.py").write_text(
        "# a plain plugin: no gateway import, no memory markers\n"
        "def register():\n"
        "    pass\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    def _plain_side_effect():
        def register():
            providers.register_provider(
                ProviderProfile(name="ep-plain", base_url="https://d.test/v1")
            )

        return register

    fake_eps = _FakeEntryPoints(
        [_FakeEP("ep-plain", _plain_side_effect, value="stub_plain_plugin")]
    )
    import importlib.metadata as md

    monkeypatch.setattr(md, "entry_points", lambda: fake_eps)
    _enable(monkeypatch, "ep-plain")
    _clear_provider_caches()
    try:
        assert providers.get_provider_profile("ep-plain") is not None
    finally:
        sys.modules.pop("stub_plain_plugin", None)
        _clear_provider_caches()
