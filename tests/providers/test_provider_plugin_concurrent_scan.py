"""Threads scanning one profile home at the same time all end up with its provider plugin registered.

Regression: a multiplex gateway resolves several sessions of one profile on different threads. Each thread's
first lookup scanned the home; the first imported the plugin into its staging layer, and every other thread
found the module already in ``sys.modules`` (still executing), skipped the import and published its own EMPTY
staging layer. That removed the provider from the home and recorded the tree as live, so no later scan imported
it again: "Unknown provider '<name>'" for that profile, and no row in the model picker, until a restart.
"""

from __future__ import annotations

import sys
import textwrap
import threading

import pytest

# The import is slow enough that every other thread reaches the scan while the first is still inside it.
_INIT = textwrap.dedent("""
    import time
    from providers import register_provider
    from providers.base import ProviderProfile

    time.sleep(0.3)
    register_provider(ProviderProfile(name="demo-slow", aliases=("demo-slow-alias",),
                                      base_url="process://demo-slow", auth_type="external_process"))
""")


def _reset_providers():
    import providers

    providers._REGISTRY.clear()
    providers._ALIASES.clear()
    providers._HOME_LAYERS.clear()
    providers._PROVIDER_LIST_CACHE = None
    providers._discovered = False
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _reset_providers()
    plugin = tmp_path / "plugins" / "demo-slow"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: demo-slow\nkind: model-provider\nversion: 1.0\n")
    (plugin / "__init__.py").write_text(_INIT)
    yield tmp_path
    _reset_providers()


def test_concurrent_first_scans_of_one_home_keep_its_provider_plugin(home):
    import providers
    from providers import get_provider_profile

    # Process-wide discovery first, then forget the home's layer: the threads below race only on the home scan.
    providers.list_providers()
    providers._HOME_LAYERS.clear()
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]

    threads = 8
    barrier = threading.Barrier(threads)
    seen: list[object] = []

    def lookup():
        barrier.wait()
        seen.append(get_provider_profile("demo-slow"))

    workers = [threading.Thread(target=lookup) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    profile = get_provider_profile("demo-slow")
    assert profile is not None  # the plugin survived the race ...
    assert get_provider_profile("demo-slow-alias") is profile
    assert seen == [profile] * threads  # ... and no thread was told the provider does not exist
