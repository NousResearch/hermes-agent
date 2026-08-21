"""The Discord authz mixin keeps the adapter's logger identity under both import paths.

``DiscordAuthorizationMixin`` was lifted out of
``plugins/platforms/discord/adapter.py``, and the log records emitted from
those methods must keep the name they had while they lived on the adapter.

That name is not a constant. The plugin manager imports directory plugins as
``hermes_plugins.<slug>`` (``hermes_cli/plugins.py``, ``_NS_PARENT``), so the
adapter's own ``logging.getLogger(__name__)`` resolves to
``hermes_plugins.<slug>.adapter`` there and to
``plugins.platforms.discord.adapter`` under the canonical package path. A
hard-coded logger name in the mixin is therefore correct on exactly one of the
two paths and silently wrong on the other, which is why the mixin derives its
logger from ``__package__``.

The relative ``from .authz_mixin import ...`` in the adapter is what keeps the
mixin inside whichever namespace is live. An absolute
``from plugins.platforms.discord.authz_mixin import ...`` would import and
initialize a second copy of the package under the canonical name even when the
plugin manager had already loaded it as ``hermes_plugins.<slug>``.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


DISCORD_PLUGIN_DIR = (
    Path(__file__).resolve().parents[3] / "plugins" / "platforms" / "discord"
)
NS_PARENT = "hermes_plugins"


def _load_authz_mixin_under(package_name: str):
    """Import authz_mixin as a submodule of *package_name*, like the manager does."""
    created = []

    parent = package_name.rpartition(".")[0]
    if parent and parent not in sys.modules:
        ns_pkg = types.ModuleType(parent)
        ns_pkg.__path__ = []
        ns_pkg.__package__ = parent
        sys.modules[parent] = ns_pkg
        created.append(parent)

    pkg = types.ModuleType(package_name)
    pkg.__path__ = [str(DISCORD_PLUGIN_DIR)]
    pkg.__package__ = package_name
    sys.modules[package_name] = pkg
    created.append(package_name)

    mod_name = f"{package_name}.authz_mixin"
    spec = importlib.util.spec_from_file_location(
        mod_name, DISCORD_PLUGIN_DIR / "authz_mixin.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    created.append(mod_name)
    spec.loader.exec_module(module)
    return module, created


@pytest.fixture
def cleanup_modules():
    names = []
    yield names
    for name in reversed(names):
        sys.modules.pop(name, None)


def test_logger_tracks_the_namespaced_plugin_package(cleanup_modules):
    """Loaded as hermes_plugins.<slug>, the mixin logs as <slug>.adapter."""
    slug = f"{NS_PARENT}.discord_authz_nstest"
    module, created = _load_authz_mixin_under(slug)
    cleanup_modules.extend(created)

    assert module.logger.name == f"{slug}.adapter"


def test_logger_tracks_the_canonical_package(cleanup_modules):
    """Loaded under the canonical path, the name is unchanged from before the lift."""
    canonical = "plugins.platforms.discord"
    mod_name = f"{canonical}.authz_mixin"
    sys.modules.pop(mod_name, None)

    spec = importlib.util.spec_from_file_location(
        mod_name, DISCORD_PLUGIN_DIR / "authz_mixin.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    cleanup_modules.append(mod_name)
    spec.loader.exec_module(module)

    assert module.logger.name == "plugins.platforms.discord.adapter"


def test_two_namespaces_do_not_share_a_logger_name(cleanup_modules):
    """A hard-coded name would collapse both loads onto one logger."""
    first, created_first = _load_authz_mixin_under(
        f"{NS_PARENT}.discord_authz_nstest_a"
    )
    cleanup_modules.extend(created_first)
    second, created_second = _load_authz_mixin_under(
        f"{NS_PARENT}.discord_authz_nstest_b"
    )
    cleanup_modules.extend(created_second)

    assert first.logger.name != second.logger.name


def test_namespaced_load_does_not_pull_in_the_canonical_package(cleanup_modules):
    """The mixin must not drag a second canonical Discord package into sys.modules.

    The canonical entries are evicted so the assertion can tell a fresh import
    from one that was already cached, then restored: other tests in the run
    hold references to the adapter module and patch attributes on it, so
    leaving it evicted would hand them a second, unpatched copy.
    """
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "plugins.platforms.discord"
        or name.startswith("plugins.platforms.discord.")
    }
    for name in saved:
        del sys.modules[name]

    try:
        slug = f"{NS_PARENT}.discord_authz_nstest_isolated"
        _module, created = _load_authz_mixin_under(slug)
        cleanup_modules.extend(created)

        leaked = [
            name
            for name in sys.modules
            if name == "plugins.platforms.discord"
            or name.startswith("plugins.platforms.discord.")
        ]
        assert leaked == []
    finally:
        sys.modules.update(saved)
