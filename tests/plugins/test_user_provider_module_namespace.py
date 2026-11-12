"""One process serves many profiles, so two profiles may each install a user provider with the
same directory name but different code. Each profile must get its own implementation and its
own import-time state, never the module another profile loaded first. Bundled providers keep
their ``plugins.<kind>.<name>`` module names."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_USER_NAMESPACES = ("_hermes_user_cron", "_hermes_user_context_engine", "_hermes_user_computer_use")

_CRON_SRC = dedent('''
    from cron.scheduler_provider import CronScheduler

    TAG = {tag!r}
    LOADS = []
    LOADS.append(TAG)

    class Demo(CronScheduler):
        @property
        def name(self):
            return TAG
        def start(self, stop_event, *, adapters=None, loop=None, interval=60):
            return None

    def register(ctx):
        ctx.register_cron_scheduler(Demo())
''')

_ENGINE_SRC = dedent('''
    from agent.context_engine import ContextEngine

    TAG = {tag!r}
    LOADS = []
    LOADS.append(TAG)

    class Demo(ContextEngine):
        @property
        def name(self):
            return TAG
        def update_from_response(self, usage):
            pass
        def should_compress(self, prompt_tokens=None):
            return False
        def compress(self, messages, current_tokens=None):
            return messages

    def register(ctx):
        ctx.register_context_engine(Demo())
''')

_COMPUTER_USE_SRC = dedent('''
    from tools.computer_use.backend import ComputerUseProvider

    TAG = {tag!r}
    LOADS = []
    LOADS.append(TAG)

    class Demo(ComputerUseProvider):
        name = TAG
        def create_backend(self, *, permission_mode):
            raise NotImplementedError

    def register(ctx):
        ctx.register_computer_use_provider(Demo())
''')


def _load_cron(name):
    from plugins.cron_providers import load_cron_scheduler
    return load_cron_scheduler(name)


def _load_engine(name):
    from plugins.context_engine import load_context_engine
    return load_context_engine(name)


def _load_computer_use(name):
    from plugins.computer_use import load_computer_use_provider
    return load_computer_use_provider(name)


_KINDS = {
    "cron": (_CRON_SRC, _load_cron),
    "context_engine": (_ENGINE_SRC, _load_engine),
    "computer_use": (_COMPUTER_USE_SRC, _load_computer_use),
}


@pytest.fixture(autouse=True)
def _clean_user_modules():
    yield
    for name in [m for m in sys.modules if m.split(".", 1)[0] in _USER_NAMESPACES]:
        del sys.modules[name]


def _profile(root: Path, label: str, plugin: str, src: str) -> Path:
    home = root / label
    plugin_dir = home / "plugins" / plugin
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(src.format(tag=f"impl-{label}"))
    return home


def _load_in(home: Path, load, name):
    token = set_hermes_home_override(str(home))
    try:
        return load(name)
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_same_named_user_provider_in_two_profiles_loads_each_profiles_own_code(tmp_path, kind):
    src, load = _KINDS[kind]
    home_a = _profile(tmp_path, "a", "shared_name", src)
    home_b = _profile(tmp_path, "b", "shared_name", src)

    first_a = _load_in(home_a, load, "shared_name")
    first_b = _load_in(home_b, load, "shared_name")
    again_a = _load_in(home_a, load, "shared_name")

    assert first_a is not None and first_b is not None
    assert first_a.name == "impl-a"
    assert first_b.name == "impl-b"  # not profile A's implementation
    assert again_a.name == "impl-a"

    mod_a = sys.modules[type(first_a).__module__]
    mod_b = sys.modules[type(first_b).__module__]
    assert mod_a is not mod_b
    assert Path(mod_a.__file__).resolve() == (home_a / "plugins" / "shared_name" / "__init__.py").resolve()
    assert Path(mod_b.__file__).resolve() == (home_b / "plugins" / "shared_name" / "__init__.py").resolve()
    # Import-time state is per profile, and a profile's module is reused rather than re-executed.
    assert mod_a.LOADS == ["impl-a"]
    assert mod_b.LOADS == ["impl-b"]


@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_an_alias_of_the_same_plugin_dir_reuses_one_module(tmp_path, kind):
    src, load = _KINDS[kind]
    home = _profile(tmp_path, "a", "aliased", src)
    alias = tmp_path / "alias-home"
    alias.symlink_to(home, target_is_directory=True)

    direct = _load_in(home, load, "aliased")
    via_alias = _load_in(alias, load, "aliased")

    assert type(direct).__module__ == type(via_alias).__module__
    assert sys.modules[type(direct).__module__].LOADS == ["impl-a"]


def test_user_module_name_is_namespaced_by_canonical_dir(tmp_path):
    from plugins.plugin_loader import user_module_name
    one = tmp_path / "one" / "plugins" / "dup"
    two = tmp_path / "two" / "plugins" / "dup"
    one.mkdir(parents=True)
    two.mkdir(parents=True)
    a, b = user_module_name("_hermes_user_cron", one), user_module_name("_hermes_user_cron", two)
    assert a != b
    assert a.startswith("_hermes_user_cron.") and a.endswith(".dup")
    assert user_module_name("_hermes_user_cron", one / ".." / "dup") == a


def test_bundled_cron_provider_keeps_its_module_name():
    from plugins.cron_providers import _load_provider_from_dir, find_provider_dir
    provider_dir = find_provider_dir("chronos")
    assert provider_dir is not None and provider_dir.parent.name == "cron_providers"
    from plugins.cron_providers import chronos
    instance = _load_provider_from_dir(provider_dir)
    assert instance is not None
    assert sys.modules["plugins.cron_providers.chronos"] is chronos  # same module, not reloaded
    assert type(instance).__module__ == "plugins.cron_providers.chronos"


def test_bundled_computer_use_provider_keeps_its_module_name():
    from plugins.computer_use import load_computer_use_provider
    instance = load_computer_use_provider("cua")
    assert instance is not None
    assert type(instance).__module__ == "plugins.computer_use.cua"


def test_shared_cache_never_hands_back_a_module_loaded_from_another_dir(tmp_path):
    """The loader's module cache is keyed by name; a cached module of that name loaded from a
    different file must not be returned as this plugin's module."""
    import logging

    from plugins.plugin_loader import load_plugin_module
    dirs = []
    for label in ("one", "two"):
        d = tmp_path / label / "dup"
        d.mkdir(parents=True)
        (d / "__init__.py").write_text(f"TAG = {label!r}\n")
        dirs.append(d)
    name = "_hermes_user_cron.cachecheck.dup"
    log = logging.getLogger("test")
    try:
        first = load_plugin_module(name, dirs[0], parents=(), logger=log, synthetic_namespace="_hermes_user_cron.cachecheck")
        assert first is not None and first.TAG == "one"
        assert load_plugin_module(name, dirs[0], parents=(), logger=log,
                                  synthetic_namespace="_hermes_user_cron.cachecheck") is first
        second = load_plugin_module(name, dirs[1], parents=(), logger=log, synthetic_namespace="_hermes_user_cron.cachecheck")
        assert second is None or second.TAG == "two"
    finally:
        for m in [m for m in sys.modules if m.startswith("_hermes_user_cron.cachecheck")]:
            del sys.modules[m]
