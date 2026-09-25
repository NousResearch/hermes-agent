"""A user-installed memory provider parked in ``plugins.disabled`` must not load.

The Plugins hub / `hermes plugins disable` write the deny-list; ``plugins/memory`` never read it, so
the UI said "disabled" while the provider kept loading at every agent init.
"""

from __future__ import annotations

import contextlib

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.memory import find_provider_dir, load_memory_provider

_PROVIDER = """
from agent.memory_provider import MemoryProvider

class P(MemoryProvider):
    name = "fakemem"
    def is_available(self):
        return True
    def initialize(self, *a, **kw):
        pass
    def get_tool_schemas(self):
        return []

def register(ctx):
    ctx.register_memory_provider(P())
"""


@pytest.fixture
def homes(tmp_path, monkeypatch):
    enabled, disabled = tmp_path / "enabled", tmp_path / "disabled"
    for hermes_home in (enabled, disabled):
        provider_dir = hermes_home / "plugins" / "fakemem"
        provider_dir.mkdir(parents=True)
        (provider_dir / "plugin.yaml").write_text(
            "name: fakemem-manifest\nkind: exclusive\n", encoding="utf-8"
        )
        (provider_dir / "__init__.py").write_text(_PROVIDER, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(enabled))
    return enabled, disabled


@contextlib.contextmanager
def _scoped_home(home):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def test_disabled_user_provider_is_found_but_never_loaded(homes):
    enabled, disabled = homes
    (enabled / "config.yaml").write_text(
        "memory:\n  provider: fakemem\nplugins:\n  disabled: []\n", encoding="utf-8"
    )
    (disabled / "config.yaml").write_text(
        "memory:\n  provider: fakemem\nplugins:\n  disabled: [fakemem-manifest]\n", encoding="utf-8"
    )

    with _scoped_home(enabled):
        assert load_memory_provider("fakemem") is not None
    with _scoped_home(disabled):
        # Still discoverable (installed, so no catalog re-clone at startup) ...
        assert find_provider_dir("fakemem") == disabled / "plugins" / "fakemem"
        # ... but this profile's deny-list wins over its memory.provider setting.
        assert load_memory_provider("fakemem") is None
    with _scoped_home(enabled):
        assert load_memory_provider("fakemem") is not None
