"""Companions follow runtime source identity without activating a provider."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import sys
from threading import Barrier
import traceback

import pytest

import plugins.memory as memory
from agent import secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@contextmanager
def _scope(home):
    home_token = set_hermes_home_override(home)
    secret_token = secret_scope.set_secret_scope({})
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def _package(root, name, companion, value):
    package = root / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "# MemoryProvider\nraise AssertionError('provider activation is forbidden')\n",
        encoding="utf-8",
    )
    (package / "unrelated.py").write_text(
        "raise AssertionError('unrelated siblings must stay lazy')\n", encoding="utf-8"
    )
    (package / "support.py").write_text(f"VALUE = {value!r}\n", encoding="utf-8")
    (package / f"{companion}.py").write_text(
        "import importlib\n"
        "PACKAGE = importlib.import_module(__package__)\n"
        "from . import support\n"
        "from .support import VALUE\n"
        "from hermes_constants import get_hermes_home\n"
        "HOME_AT_IMPORT = get_hermes_home()\n",
        encoding="utf-8",
    )
    return package


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    monkeypatch.chdir(tmp_path)
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setattr(memory, "_MEMORY_PLUGINS_DIR", bundled)
    was_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    before = set(sys.modules)
    try:
        yield bundled
    finally:
        secret_scope.set_multiplex_active(was_multiplex)
        for name in set(sys.modules) - before:
            if name.startswith((memory._USER_NAMESPACE, "_hermes_memory_companions", "plugins.memory.surface_")):
                module = sys.modules.pop(name, None)
                parent, _, child = name.rpartition(".")
                parent_module = sys.modules.get(parent)
                if parent_module is not None and getattr(parent_module, child, None) is module:
                    delattr(parent_module, child)


@pytest.mark.parametrize("companion", ["clone", "settings", "oauth_flow"])
@pytest.mark.parametrize("name", ["surface_external", "surface_hyphen-provider"])
def test_companions_keep_runtime_precedence_and_source_identity(
    tmp_path, monkeypatch, isolated, companion, name
):
    from plugins.memory.surfaces import load_provider_companion

    homes = [tmp_path / "A", tmp_path / "B"]
    packages = [_package(home / "plugins", name, companion, home.name) for home in homes]
    project = tmp_path / ".hermes" / "plugins"
    _package(project, name, companion, "project duplicate")
    project_only = _package(project, "surface_project", companion, "project")
    # Real distribution metadata, including an alias different from the package
    # directory: runtime identity uses that directory's name, not the entry-point alias.
    site = tmp_path / "site"
    pip_package = _package(site, "surface_pip_package", companion, "pip")
    dist = site / "surface_fixture-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: surface-fixture\nVersion: 1.0\n")
    (dist / "entry_points.txt").write_text(
        "[hermes_agent.memory_providers]\n"
        f"{name} = surface_pip_package:register\n"
        "surface_pip_alias = surface_pip_package:register\n"
        "surface_project = surface_pip_package:register\n"
    )
    monkeypatch.syspath_prepend(str(site))
    bundled = _package(isolated, "surface_bundled", companion, "bundled")
    for home in homes:
        _package(home / "plugins", "surface_bundled", companion, "user duplicate")
    _package(project, "surface_bundled", companion, "project duplicate")

    loaded = []
    for home, package in zip([*homes, homes[0]], [*packages, packages[0]]):
        with _scope(home):
            module = load_provider_companion(name, companion)
            assert module.VALUE == home.name
            assert module.HOME_AT_IMPORT == home
            assert Path(module.__file__).parent == memory.find_provider_dir(name) == package
            assert module.PACKAGE.support is sys.modules[module.__package__ + ".support"]
            assert getattr(module.PACKAGE, companion) is module
            loaded.append(module)
            assert load_provider_companion("surface_bundled", companion).VALUE == "bundled"
    assert loaded[0] is loaded[2]
    assert loaded[0] is not loaded[1]
    assert f"plugins.memory.{name}" not in sys.modules
    assert memory._module_name(bundled, bundled.name) not in sys.modules

    with _scope(homes[0]):
        # Project is ignored until opted in, so the pip duplicate wins first.
        assert load_provider_companion("surface_project", companion).VALUE == "pip"
        monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "1")
        assert load_provider_companion("surface_project", companion).VALUE == "project"
        assert memory.find_provider_dir("surface_project") == project_only
        pip_module = load_provider_companion("surface_pip_alias", companion)
        assert Path(pip_module.__file__).parent == pip_package
        assert "surface_pip_package" not in sys.modules

    # Cold parallel imports under independent context-local homes must not share
    # relative-import modules, even though both providers have the same name.
    for home in homes:
        _package(home / "plugins", "surface_concurrent", companion, home.name)
    barrier = Barrier(2)

    def load(home):
        with _scope(home):
            barrier.wait(timeout=10)
            return load_provider_companion("surface_concurrent", companion)

    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(load, homes))
    assert [module.VALUE for module in parallel] == [home.name for home in homes]
    assert [module.HOME_AT_IMPORT for module in parallel] == homes
    assert parallel[0] is not parallel[1]

    # Companion lookup must not poison ordinary package imports or activation.
    monkeypatch.setattr(memory, "__path__", [str(isolated), *memory.__path__])
    for root in (isolated, homes[0] / "plugins"):
        for first in ("companion", "runtime"):
            runtime_name = f"{name}_lifecycle_{root.name}_{first}"
            runtime_dir = _package(root, runtime_name, companion, first)
            (runtime_dir / "__init__.py").write_text(
                "import importlib\n"
                "from agent.memory_provider import MemoryProvider\n"
                f"from . import {companion}\n"
                "PACKAGE = importlib.import_module(__name__)\n"
                f"assert PACKAGE.support is {companion}.support\n"
                "ACTIVATIONS = globals().get('ACTIVATIONS', 0) + 1\n"
                "class Provider(MemoryProvider):\n"
                "    name = 'fixture'\n"
                "    def is_available(self): return True\n"
                "    def initialize(self, *args, **kwargs): pass\n"
                "    def get_tool_schemas(self): return []\n"
                "def register(ctx):\n"
                f"    assert PACKAGE.{companion} is {companion}\n"
                f"    assert {companion}.PACKAGE is PACKAGE\n"
                "    provider = Provider()\n"
                f"    provider.value = {companion}.VALUE\n"
                "    ctx.register_memory_provider(provider)\n"
            )
            module_name = memory._module_name(runtime_dir, runtime_name)
            with _scope(homes[0]):
                if first == "companion":
                    lazy = load_provider_companion(runtime_name, companion)
                    assert module_name not in sys.modules
                if root == isolated:
                    runtime = importlib.import_module(module_name)
                    assert runtime.ACTIVATIONS == 1
                provider = memory.load_memory_provider(runtime_name, register_skills=False)
                assert provider is not None
                assert provider.value == first
                runtime = importlib.import_module(module_name)
                assert runtime.PACKAGE is runtime
                assert runtime.ACTIVATIONS == 1
                parent, _, child = module_name.rpartition(".")
                assert getattr(sys.modules[parent], child) is runtime
                if first == "runtime":
                    lazy = load_provider_companion(runtime_name, companion)
                assert load_provider_companion(runtime_name, companion) is lazy
                assert lazy.PACKAGE.support.VALUE == first
                assert getattr(runtime, companion).PACKAGE is runtime
                assert runtime.support is getattr(runtime, companion).support

    # Host validation must accept the very same names as companion discovery.
    # Restrict its installation census to this separate clone fixture.
    monkeypatch.setattr(memory, "_MEMORY_PLUGINS_DIR", tmp_path / "no-bundled")
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    (dist / "entry_points.txt").unlink()
    from hermes_cli.profile_clone import prepare_memory_clone
    source = tmp_path / "clone-source"
    clone_dir = _package(source / "plugins", name, "clone", "clone")
    (clone_dir / "clone.py").write_text(
        "def prepare_clone(*, staging_home, **kwargs):\n"
        "    (staging_home / 'prepared').write_text('yes')\n"
        "    return {'needs_auth': True}\n"
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "config.yaml").write_text(f"memory:\n  provider: {name}\n")
    env_before = dict(os.environ)
    report = prepare_memory_clone(source_home=source, source_name="source",
        staging_home=staging, destination_home=tmp_path / "destination",
        destination_name="destination", clone_all=True)
    assert name in report["needs_auth"]
    assert (staging / "prepared").read_text() == "yes"
    assert dict(os.environ) == env_before


@pytest.mark.parametrize("companion", ["clone", "settings", "oauth_flow"])
@pytest.mark.parametrize("import_style", ["relative", "importlib"])
def test_sibling_warmed_companion_uses_exact_file(tmp_path, isolated, companion, import_style):
    from plugins.memory.surfaces import load_provider_companion

    home = tmp_path / "A"
    first = "settings" if companion == "clone" else "clone"
    package = _package(home / "plugins", "surface_warm_collision", first, "file")
    (package / f"{companion}.py").write_text(
        "def prepare_clone(): return 'file hook'\n"
    )
    (package / companion).mkdir()
    (package / companion / "__init__.py").write_text("VALUE = 'wrong package'\n")
    # Ordinary helpers, including nested names matching the allowlist, retain
    # Python's package-before-module precedence.
    (package / "support").mkdir()
    (package / "support" / "__init__.py").write_text("from . import clone\n")
    (package / "support" / "clone.py").write_text("VALUE = 'wrong nested file'\n")
    (package / "support" / "clone").mkdir()
    (package / "support" / "clone" / "__init__.py").write_text("VALUE = 'nested package'\n")
    import_source = (
        f"from . import {companion} as sibling\n" if import_style == "relative" else
        f"import importlib\nsibling = importlib.import_module('.{companion}', __package__)\n"
    )
    (package / f"{first}.py").write_text(import_source + "from . import support\n")

    with _scope(home):
        warmed = load_provider_companion(package.name, first)
        exact = load_provider_companion(package.name, companion)
        assert warmed is not None and exact is not None
        assert exact.__file__ == str(package / f"{companion}.py")
        assert exact.__spec__ is not None
        assert exact.__spec__.origin == str(package / f"{companion}.py")
        assert warmed.sibling is exact
        assert exact.prepare_clone() == "file hook"
        assert load_provider_companion(package.name, companion) is exact
        assert warmed.support.clone.VALUE == "nested package"
        assert memory._module_name(package, package.name) not in sys.modules


def test_companion_rejects_wrong_cached_origin(tmp_path, monkeypatch, isolated):
    from plugins.memory.surfaces import ProviderCompanionLoadError, load_provider_companion

    home = tmp_path / "A"
    package = _package(home / "plugins", "surface_cache_origin", "clone", "file")
    with _scope(home):
        exact = load_provider_companion(package.name, "clone")
        assert exact is not None
        wrong = exact.support
        with monkeypatch.context() as cache:
            cache.setitem(sys.modules, exact.__name__, wrong)
            with pytest.raises(ProviderCompanionLoadError):
                load_provider_companion(package.name, "clone")
            # Refuse inconsistent identity rather than replacing existing handles.
            assert sys.modules[exact.__name__] is wrong
            assert exact.PACKAGE.clone is exact


@pytest.mark.parametrize("changed", ["clone", "support", "nested", "namespace"])
def test_companion_generation_tracks_content_not_timestamps(tmp_path, isolated, changed):
    import py_compile
    from plugins.memory.surfaces import load_provider_companion

    home = tmp_path / "A"
    package = _package(home / "plugins", "surface_generation", "clone", "old")
    (package / "nested").mkdir()
    (package / "nested" / "__init__.py").write_text("from .value import VALUE\n")
    if changed == "namespace":
        (package / "nested" / "__init__.py").unlink()
    (package / "nested" / "value.py").write_text("VALUE = 'old'\n")
    (package / "clone.py").write_text(
        "from .support import VALUE\n"
        "from .nested.value import VALUE as NESTED\n"
        "LABEL = 'old'\n"
        "def late():\n"
        "    from .late import VALUE\n"
        "    return VALUE\n"
        "def dynamic_late():\n"
        "    import importlib\n"
        "    return importlib.import_module('.dynamic', __package__).VALUE\n"
    )
    (package / "late.py").write_text("VALUE = 'old'\n")
    (package / "dynamic.py").write_text("VALUE = 'old'\n")
    changed_file = package / ("nested/value.py" if changed in {"nested", "namespace"} else changed + ".py")
    info = changed_file.stat()
    # A valid timestamp pyc for the old content must not win after a same-size edit.
    py_compile.compile(str(changed_file), doraise=True)
    with _scope(home):
        old = load_provider_companion(package.name, "clone")
        assert old is not None
        assert load_provider_companion(package.name, "clone") is old
        changed_file.write_text(changed_file.read_text().replace("old", "new"))
        os.utime(changed_file, ns=(info.st_atime_ns, info.st_mtime_ns))
        new = load_provider_companion(package.name, "clone")
        assert new is not None
        assert new is not old
        assert (new.LABEL, new.VALUE, new.NESTED) == {
            "clone": ("new", "old", "old"),
            "support": ("old", "new", "old"),
            "nested": ("old", "old", "new"),
            "namespace": ("old", "old", "new"),
        }[changed]
        assert load_provider_companion(package.name, "clone") is new
        (package / "late.py").write_text("VALUE = 'new'\n")
        (package / "dynamic.py").write_text("VALUE = 'new'\n")
        assert old.late() == new.late() == "old"
        assert old.dynamic_late() == new.dynamic_late() == "old"
        latest = load_provider_companion(package.name, "clone")
        assert latest is not None and latest.late() == "new"
        assert latest.dynamic_late() == "new"
        assert memory._module_name(package, package.name) not in sys.modules


@pytest.mark.parametrize("failure", ["dependency", "syntax", "runtime", "directory"])
def test_absence_is_not_a_broken_companion_and_failures_can_retry(
    tmp_path, monkeypatch, isolated, failure
):
    from plugins.memory.surfaces import ProviderCompanionLoadError, load_provider_companion

    home = tmp_path / "A"
    package = _package(home / "plugins", "surface_broken", "clone", "fixed")
    companion_file = package / "clone.py"
    companion_file.unlink()

    with _scope(home):
        assert load_provider_companion("surface_missing", "clone") is None
        assert load_provider_companion(package.name, "clone") is None
        assert memory._module_name(package, package.name) not in sys.modules
        from hermes_cli.profile_clone import _provider_name
        for invalid in ("../surface_broken", "surface_broken.clone", "", "a/b", "a\\b", "a\n", None):
            assert not _provider_name(invalid)
            with pytest.raises(ValueError):
                load_provider_companion(invalid, "clone")
        for invalid in ("__init__", "cli", "../clone", "clone.py", "", None):
            with pytest.raises(ValueError):
                load_provider_companion(package.name, invalid)
        if failure == "directory":
            companion_file.mkdir()
        else:
            companion_file.write_text({
                "dependency": "from .not_installed import VALUE\n",
                "syntax": "def broken(: # PRIVATE_SENTINEL\n",
                "runtime": "raise RuntimeError('PRIVATE_SENTINEL')\n",
            }[failure], encoding="utf-8")
        with pytest.raises(ProviderCompanionLoadError) as caught:
            load_provider_companion(package.name, "clone")
        diagnostic = "".join(traceback.format_exception(caught.value))
        assert "PRIVATE_SENTINEL" not in diagnostic
        assert package.name in str(caught.value)
        assert "clone" in str(caught.value)
        assert "install" in str(caught.value).lower()
        assert not any(n.endswith(".clone") and getattr(m, "__file__", None) == str(companion_file)
                       for n, m in sys.modules.copy().items())
        if failure == "directory":
            companion_file.rmdir()
        companion_file.write_text("from .support import VALUE\n", encoding="utf-8")
        importlib.invalidate_caches()
        assert load_provider_companion(package.name, "clone").VALUE == "fixed"

        # Exact-file resolution wins even when normal imports prefer a package.
        collision = _package(home / "plugins", "surface_collision", "settings", "file")
        (collision / "settings").mkdir()
        (collision / "settings" / "__init__.py").write_text(
            "raise AssertionError('wrong settings package')\n"
        )
        barrier = Barrier(2)
        def load_exact(_):
            with _scope(home):
                barrier.wait(timeout=10)
                return load_provider_companion(collision.name, "settings")
        with ThreadPoolExecutor(max_workers=2) as pool:
            modules = list(pool.map(load_exact, range(2)))
        assert modules[0] is modules[1]
        assert modules[0].VALUE == "file"
        assert Path(modules[0].__file__) == collision / "settings.py"

        assert not _provider_name("caf\u00e9")
        with pytest.raises(ValueError):
            load_provider_companion("caf\u00e9", "clone")

        site = tmp_path / "site"
        site.mkdir()
        (site / "surface_single.py").write_text("raise AssertionError('must stay lazy')\n")
        dist = site / "single_fixture-1.0.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: single-fixture\nVersion: 1.0\n")
        (dist / "entry_points.txt").write_text(
            "[hermes_agent.memory_providers]\n"
            "surface_single = surface_single:register\n"
            "surface_unresolved = surface_not_installed:register\n"
        )
        monkeypatch.syspath_prepend(str(site))
        assert load_provider_companion("surface_single", "clone") is None
        assert "surface_single" not in sys.modules
        with pytest.raises(ProviderCompanionLoadError):
            load_provider_companion("surface_unresolved", "clone")
