"""``$HERMES_HOME`` model-provider plugins resolve for the profile home bound at lookup time (#88143).

One process serves several profiles (multiplex gateway, Desktop ``serve``); discovery used to read the
plugins of whichever home was bound first and never look again, so a plugin installed in a secondary
profile was ``Unknown provider`` from Desktop while ``hermes -p <profile>`` in a terminal worked.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_PLUGIN = textwrap.dedent(
    """
    from providers import register_provider
    from providers.base import ProviderProfile

    register_provider(ProviderProfile(name="{name}", aliases=("{name}-alias",), auth_type="external_process",
                                      base_url="process://{name}", api_mode="chat_completions"))
    """
)


def _install(home: Path, name: str) -> None:
    plugin = home / "plugins" / name
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(f"name: {name}\nkind: model-provider\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(_PLUGIN.format(name=name), encoding="utf-8")


@pytest.fixture
def homes(tmp_path, monkeypatch):
    import providers

    launch = tmp_path / "launch"
    secondary = tmp_path / "profiles" / "scaleup"
    launch.mkdir()
    secondary.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_HOME_LAYERS", {}, raising=False)
    yield launch, secondary
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]


def _bound(home: Path, fn):
    token = set_hermes_home_override(home)
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


def test_secondary_profile_plugin_resolves_for_its_home_only(homes):
    import providers
    from hermes_cli.auth import resolve_provider

    launch, secondary = homes
    _install(secondary, "scaleup-only")

    assert providers.get_provider_profile("scaleup-only") is None  # launch home discovers first

    assert _bound(secondary, lambda: providers.get_provider_profile("scaleup-only")) is not None
    assert _bound(secondary, lambda: providers.get_provider_profile("scaleup-only-alias")) is not None
    assert _bound(secondary, lambda: providers.provider_source("scaleup-only")) == "user"
    assert "scaleup-only" in _bound(secondary, lambda: {p.name for p in providers.list_providers()})
    # The agent-build gate Desktop hits (``Unknown provider`` came from here).
    assert _bound(secondary, lambda: resolve_provider("scaleup-only")) == "scaleup-only"

    # Profiles are islands: the launch home still does not see the secondary's install.
    assert providers.get_provider_profile("scaleup-only") is None
    assert "scaleup-only" not in {p.name for p in providers.list_providers()}


def test_plugin_installed_after_discovery_is_found_without_a_restart(homes):
    import providers

    launch, _ = homes
    assert providers.get_provider_profile("late-install") is None

    _install(launch, "late-install")

    assert providers.get_provider_profile("late-install") is not None
    assert "late-install" in {p.name for p in providers.list_providers()}


def _rewrite(home: Path, name: str, url: str, *, alias: str | None = None) -> None:
    plugin = home / 'plugins' / name
    (plugin / '__init__.py').write_text(
        'from providers import register_provider\nfrom providers.base import ProviderProfile\n'
        f'register_provider(ProviderProfile(name={name!r}, aliases={((alias,) if alias else ())!r}, '
        f'auth_type="external_process", base_url={url!r}))\n', encoding='utf-8')


def test_explicit_reload_reimports_changed_body_and_replaces_removed_alias(homes):
    import providers
    home, _ = homes
    _install(home, 'warm-body')
    old = providers.get_provider_profile('warm-body')
    _rewrite(home, 'warm-body', 'process://changed', alias='new-alias')
    result = providers.reload_home_providers()
    assert result['reloaded'] is True
    assert providers.get_provider_profile('warm-body').base_url == 'process://changed'
    assert providers.get_provider_profile('new-alias') is providers.get_provider_profile('warm-body')
    assert providers.get_provider_profile('warm-body-alias') is None
    assert old.base_url == 'process://warm-body'  # existing clients retain their original profile


def test_explicit_reload_leaves_other_home_and_bundled_provider_objects_intact(homes):
    import providers
    a, b = homes
    _install(a, 'same-name'); _install(b, 'same-name')
    before_a = providers.get_provider_profile('same-name')
    before_b = _bound(b, lambda: providers.get_provider_profile('same-name'))
    bundled = providers.get_provider_profile('copilot-acp')
    module_b = providers._user_module_name(b / 'plugins/same-name', str(b.resolve()))
    # Use the actual canonical scope-key function, not a second key derivation.
    from hermes_constants import hermes_home_key
    module_b = providers._user_module_name(b / 'plugins/same-name', hermes_home_key(b))
    cached_b = sys.modules[module_b]
    _rewrite(a, 'same-name', 'process://updated-a')
    assert providers.reload_home_providers()['reloaded'] is True
    assert providers.get_provider_profile('same-name').base_url == 'process://updated-a'
    assert _bound(b, lambda: providers.get_provider_profile('same-name')) is before_b
    assert sys.modules[module_b] is cached_b
    assert providers.get_provider_profile('copilot-acp') is bundled
    assert before_a.base_url == 'process://same-name'


def test_explicit_reload_import_failure_preserves_known_previous_generation(homes):
    import providers
    home, _ = homes
    _install(home, 'failure-safe')
    old = providers.get_provider_profile('failure-safe')
    from hermes_constants import hermes_home_key
    name = providers._user_module_name(home / 'plugins/failure-safe', hermes_home_key(home))
    old_module = sys.modules[name]
    (home / 'plugins/failure-safe/__init__.py').write_text('raise RuntimeError("offline fixture")\n')
    result = providers.reload_home_providers()
    assert result['reloaded'] is False and result['errors']
    assert providers.get_provider_profile('failure-safe') is old
    assert sys.modules[name] is old_module


def test_explicit_reload_refuses_foreign_module_ownership_without_eviction(homes):
    import providers
    import types
    home, _ = homes
    _install(home, 'foreign-safe')
    old = providers.get_provider_profile('foreign-safe')
    from hermes_constants import hermes_home_key
    name = providers._user_module_name(home / 'plugins/foreign-safe', hermes_home_key(home))
    foreign = types.ModuleType(name); foreign.__file__ = str(home.parent / 'foreign.py')
    sys.modules[name] = foreign
    result = providers.reload_home_providers()
    assert result['reloaded'] is False
    assert sys.modules[name] is foreign
    assert providers.get_provider_profile('foreign-safe') is old


def test_explicit_reload_removes_deleted_user_plugin_without_losing_bundled(homes):
    import providers
    home, _ = homes
    _install(home, 'removed-user')
    assert providers.get_provider_profile('removed-user') is not None
    p = home / 'plugins/removed-user'
    (p / '__init__.py').unlink(); (p / 'plugin.yaml').unlink(); p.rmdir()
    assert providers.reload_home_providers()['reloaded'] is True
    assert providers.get_provider_profile('removed-user') is None
    assert providers.get_provider_profile('copilot-acp') is not None


def test_nested_provider_reload_reruns_existing_owned_dependency_invalidator(homes):
    import providers
    home, _ = homes
    p = home / 'plugins/model-providers/nested-cache'; p.mkdir(parents=True)
    (p / 'plugin.yaml').write_text('name: nested-cache\nkind: model-provider\n')
    (p / 'dependency.py').write_text('VALUE="old"\n')
    init = 'from .dependency import VALUE\nfrom providers import register_provider\nfrom providers.base import ProviderProfile\nregister_provider(ProviderProfile(name="nested-cache", base_url="process://"+VALUE))\n'
    (p / '__init__.py').write_text(init)
    assert providers.get_provider_profile('nested-cache').base_url == 'process://old'
    (p / 'dependency.py').write_text('VALUE="new-longer"\n')
    assert providers.reload_home_providers()['reloaded'] is True
    assert providers.get_provider_profile('nested-cache').base_url == 'process://new-longer'


def test_explicit_reload_refuses_busy_home_without_waiting_or_mutation(homes, monkeypatch):
    import providers
    from hermes_constants import hermes_home_key
    home, _ = homes
    _install(home, 'busy-safe'); old = providers.get_provider_profile('busy-safe')
    monkeypatch.setattr(providers, '_HOME_SCANS', {hermes_home_key(home)})
    assert providers.reload_home_providers()['reloaded'] is False
    assert providers.get_provider_profile('busy-safe') is old


def test_explicit_reload_refuses_symlink_outside_home_before_import(homes):
    import providers
    home, secondary = homes
    _install(home, 'symlink-safe'); old = providers.get_provider_profile('symlink-safe')
    external = secondary / 'external'; external.mkdir()
    (external / 'plugin.yaml').write_text('kind: model-provider\n')
    (external / '__init__.py').write_text('raise AssertionError("must never import")\n')
    (home / 'plugins/escape').symlink_to(external, target_is_directory=True)
    assert providers.reload_home_providers()['reloaded'] is False
    assert providers.get_provider_profile('symlink-safe') is old


def test_explicit_reload_scan_exception_restores_modules_and_previous_layer(homes, monkeypatch):
    import providers
    home, _ = homes
    _install(home, 'scan-safe'); old = providers.get_provider_profile('scan-safe')
    def fail(*args):
        raise OSError('offline fixture')
    monkeypatch.setattr(providers, '_scan_home_layer', fail)
    assert providers.reload_home_providers()['reloaded'] is False
    assert providers.get_provider_profile('scan-safe') is old


def test_explicit_reload_preserves_known_generation_if_mirror_raises(homes, monkeypatch):
    import providers
    home, _ = homes
    _install(home, 'mirror-safe'); old = providers.get_provider_profile('mirror-safe')
    _rewrite(home, 'mirror-safe', 'process://new-mirror')
    def fail():
        raise RuntimeError('offline fixture')
    monkeypatch.setattr(providers, '_sync_auth_registry', fail)
    assert providers.reload_home_providers()['reloaded'] is False
    assert providers.get_provider_profile('mirror-safe') is old


def test_explicit_reload_reruns_plugin_owned_script_invalidation(homes):
    import providers
    import types
    home, secondary = homes
    _install(home, 'owned-script')
    scripts = home / 'scripts'; scripts.mkdir()
    dep = scripts / 'owned_script.py'; dep.write_text('VALUE="old"\n')
    own = types.ModuleType('owned_script'); own.__file__ = str(dep)
    foreign = types.ModuleType('foreign_script'); foreign.__file__ = str(secondary / 'scripts/foreign_script.py')
    sys.modules['owned_script'] = own; sys.modules['foreign_script'] = foreign
    try:
        old = providers.get_provider_profile('owned-script')
        p = home / 'plugins/owned-script/__init__.py'
        p.write_text('import sys\nfrom pathlib import Path\n'
                     'home=Path(__file__).resolve().parents[2]\n'
                     'for name in ("owned_script", "foreign_script"):\n'
                     '    cached=sys.modules.get(name)\n'
                     '    if cached and Path(cached.__file__).resolve()==home/"scripts"/(name+".py"):\n'
                     '        del sys.modules[name]\n' + p.read_text())
        assert providers.reload_home_providers()['reloaded'] is True
        assert 'owned_script' not in sys.modules
        assert sys.modules['foreign_script'] is foreign
        assert old is not providers.get_provider_profile('owned-script')
    finally:
        sys.modules.pop('owned_script', None);sys.modules.pop('foreign_script', None)


def test_readers_keep_known_profiles_while_reload_imports_without_a_global_lock(homes):
    import providers
    import threading
    import types
    a, b = homes
    _install(a, 'concurrent-home'); _install(b, 'concurrent-home')
    old_a = providers.get_provider_profile('concurrent-home')
    old_b = _bound(b, lambda: providers.get_provider_profile('concurrent-home'))
    barrier = types.ModuleType('provider_reload_test_barrier')
    barrier.started = threading.Event(); barrier.release = threading.Event()
    sys.modules[barrier.__name__] = barrier
    p = a / 'plugins/concurrent-home/__init__.py'
    _rewrite(a, 'concurrent-home', 'process://concurrent-new')
    p.write_text('from provider_reload_test_barrier import started, release\nstarted.set()\nassert release.wait(5)\n' + p.read_text())
    results = []
    thread = threading.Thread(target=lambda: results.append(_bound(a, providers.reload_home_providers)))
    thread.start()
    try:
        assert barrier.started.wait(5)
        assert providers.get_provider_profile('concurrent-home') is old_a
        assert _bound(b, lambda: providers.get_provider_profile('concurrent-home')) is old_b
        assert providers.reload_home_providers()['reloaded'] is False
    finally:
        barrier.release.set(); thread.join(timeout=5)
        sys.modules.pop(barrier.__name__, None)
    assert not thread.is_alive() and results[0]['reloaded'] is True
    assert providers.get_provider_profile('concurrent-home').base_url == 'process://concurrent-new'
    assert _bound(b, lambda: providers.get_provider_profile('concurrent-home')) is old_b


@pytest.mark.parametrize('relative_dependency', [False, True])
def test_explicit_reload_ignores_timestamp_valid_owned_bytecode(homes, relative_dependency):
    import os
    import py_compile
    import providers
    home, _ = homes
    _install(home, 'bytecode-safe')
    plugin = home / 'plugins/bytecode-safe'
    if relative_dependency:
        source = plugin / 'dependency.py'
        source.write_text('VALUE="old"\n')
        (plugin / '__init__.py').write_text(
            'from .dependency import VALUE\nfrom providers import register_provider\n'
            'from providers.base import ProviderProfile\n'
            'register_provider(ProviderProfile(name="bytecode-safe", base_url="process://"+VALUE))\n')
    else:
        _rewrite(home, 'bytecode-safe', 'process://old')
        source = plugin / '__init__.py'
    old = providers.get_provider_profile('bytecode-safe')
    before = source.stat()
    py_compile.compile(str(source), doraise=True,
                       invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    source.write_text(source.read_text().replace('old', 'new'))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert source.stat().st_size == before.st_size
    assert providers.reload_home_providers()['reloaded'] is True
    assert providers.get_provider_profile('bytecode-safe').base_url == 'process://new'
    assert old.base_url == 'process://old'


def test_explicit_reload_restores_package_before_propagating_system_exit(homes):
    import providers
    from hermes_constants import hermes_home_key
    home, _ = homes
    _install(home, 'exit-safe')
    old = providers.get_provider_profile('exit-safe')
    name = providers._user_module_name(home / 'plugins/exit-safe', hermes_home_key(home))
    old_module = sys.modules[name]
    (home / 'plugins/exit-safe/__init__.py').write_text('raise SystemExit("offline fixture")\n')
    before_finders = tuple(sys.meta_path)
    with pytest.raises(SystemExit):
        providers.reload_home_providers()
    assert providers.get_provider_profile('exit-safe') is old
    assert sys.modules[name] is old_module
    assert tuple(sys.meta_path) == before_finders
    assert hermes_home_key(home) not in providers._HOME_SCANS


def test_reload_snapshots_modules_after_admission_including_completed_discovery(homes, monkeypatch):
    import providers
    import threading
    home, _ = homes
    _install(home, 'admission-safe')
    assert providers.get_provider_profile('admission-safe') is not None
    paused = threading.Event()
    scanned = threading.Event()
    real_lock = providers._HOME_LAYERS_LOCK
    class AdmissionBarrier:
        entries = 0
        def __enter__(self):
            if threading.current_thread().name == 'controlled-reloader':
                self.entries += 1
                if self.entries == 2:
                    paused.set()
                    assert scanned.wait(5)
            real_lock.acquire()
            return self
        def __exit__(self, *args):
            real_lock.release()
    monkeypatch.setattr(providers, '_HOME_LAYERS_LOCK', AdmissionBarrier())
    results = []
    failures = []
    def reload():
        try:
            results.append(_bound(home, providers.reload_home_providers))
        except BaseException as exc:
            failures.append(exc)
    thread = threading.Thread(target=reload, name='controlled-reloader')
    thread.start()
    try:
        assert paused.wait(5)
        _install(home, 'late-admission')
        assert providers.get_provider_profile('late-admission') is not None
    finally:
        scanned.set()
        thread.join(timeout=5)
    assert not thread.is_alive() and not failures
    assert results[0]['reloaded'] is True
    assert providers.get_provider_profile('late-admission') is not None
