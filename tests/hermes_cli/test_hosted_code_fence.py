"""The hosted code fence: one process-wide audit-hook enforcement point for code loading.

Every mechanism that can put executable code into the process — a source import, a
``.pyc``-only import, a zip member, an extension module, ``runpy``, ``exec(compile(...))``,
a dotted entry point's parent package, discovery and warm-up paths that call private loaders —
is refused when the file it would execute resolves outside the deployment image, and the home
code never runs. The positive controls: the bundled tree, an installed package's entry point,
the stdlib and the core tools load exactly as before. Locally the fence is never installed by
the boot and stays dormant in this file's process once the verdict flips back.

Which audit event carries a code file's path was established empirically on this interpreter
(CPython 3.14) and is asserted by the tests: ``compile`` for every source load, ``exec`` on the
code object (the only signal for ``.pyc``-only imports), and ``import`` for dynamically loaded
extension modules (whose ``.so`` path it carries). See the fence module's docstring.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import logging
import runpy
import shutil
import sys
import sysconfig
import zipfile
from pathlib import Path

import pytest

from hermes_cli import hosted_mode
from hermes_cli.hosted_code_fence import (
    HostedCodeRefused,
    hosted_code_refusal,
    hosted_fence_installed,
    install_hosted_code_fence,
    prune_sys_path_outside_image,
)


@pytest.fixture
def hosted(monkeypatch):
    """A frozen hosted verdict, with the process-wide code fence installed (idempotent: the
    first hosted test in this file installs it for the process, and the hook consults the
    live verdict, so it goes dormant again whenever a test flips the verdict away)."""
    monkeypatch.setitem(hosted_mode._VERDICT, "hosted", True)
    monkeypatch.setattr("hermes_cli.plugin_isolation.isolation_mode", lambda *a, **k: "in_process")
    install_hosted_code_fence()
    assert hosted_fence_installed()


def _home_module(home: Path, name: str, marker: Path, text: str = "executed") -> Path:
    """A module under *home* whose module-level code writes *marker*; loading it is the
    behaviour every refusal below must prevent."""
    home.mkdir(parents=True, exist_ok=True)
    module = home / f"{name}.py"
    module.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text({text!r})\n",
        encoding="utf-8")
    return module


# --------------------------------------------------------------------------------------------------
# Every code-loading mechanism is refused when the file resolves outside the image


def test_a_home_source_import_is_refused(tmp_path, monkeypatch, hosted):
    """A plain ``import`` found through ``sys.path``: importlib compiles the file before
    executing it, and the fence refuses at that compile — the module never runs and never
    lands in ``sys.modules``."""
    home = tmp_path / "home"
    marker = home / "executed"
    _home_module(home, "fence_home_mod", marker)
    monkeypatch.syspath_prepend(str(home))
    try:
        with pytest.raises(HostedCodeRefused, match="deployment image"):
            importlib.import_module("fence_home_mod")
        assert not marker.exists()
        assert "fence_home_mod" not in sys.modules
    finally:
        sys.modules.pop("fence_home_mod", None)


def test_a_home_pyc_only_import_is_refused(tmp_path, monkeypatch, hosted):
    """A ``.pyc`` with no source beside it loads no source, so the ``compile`` signal never
    fires; the fence's ``exec`` check on the code object's ``co_filename`` refuses it before
    the module body runs. The ``.pyc`` is built while the process is not hosted (a tenant
    drops a prebuilt bytecode file), then imported once it is."""
    home = tmp_path / "home"
    home.mkdir()
    marker = home / "executed"
    src = home / "fence_pyc_mod.py"
    src.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n", encoding="utf-8")
    monkeypatch.setitem(hosted_mode._VERDICT, "hosted", False)  # local behaviour: the pyc builds
    import py_compile

    py_compile.compile(src, cfile=home / "fence_pyc_mod.pyc")
    src.unlink()
    monkeypatch.setitem(hosted_mode._VERDICT, "hosted", True)  # back to hosted
    monkeypatch.syspath_prepend(str(home))
    try:
        with pytest.raises(ImportError):
            importlib.import_module("fence_pyc_mod")
        assert not marker.exists()  # the module body never ran
        assert "fence_pyc_mod" not in sys.modules
    finally:
        sys.modules.pop("fence_pyc_mod", None)


def test_a_home_zipimport_is_refused(tmp_path, monkeypatch, hosted):
    """A module inside a zip archive under the home: the compile signal's filename is the
    ``<archive>/<member>`` pseudo-path, whose realpath resolves to the archive — outside the
    image, so the member is refused before it executes."""
    home = tmp_path / "home"
    home.mkdir()
    marker = home / "executed"
    archive = home / "fence_home.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("fence_zip_mod.py",
                    f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n")
    monkeypatch.syspath_prepend(str(archive))
    try:
        with pytest.raises(ImportError, match="deployment image"):
            importlib.import_module("fence_zip_mod")
        assert not marker.exists()
        assert "fence_zip_mod" not in sys.modules
    finally:
        sys.modules.pop("fence_zip_mod", None)


def _a_compiled_module() -> Path:
    """A real extension module (``.so``) this interpreter's own environment ships, for the
    extension-load pair below."""
    purelib = Path(sysconfig.get_paths()["purelib"])
    for pattern in ("charset_normalizer/md*.so", "frozenlist/_frozenlist*.so",
                    "markupsafe/_speedups*.so"):
        found = sorted(purelib.glob(pattern))
        if found:
            return found[0]
    pytest.fail("no compiled module in this environment's site-packages")


def test_a_home_extension_module_is_refused(tmp_path, monkeypatch, hosted):
    """Extension loads carry their ``.so`` path on the ``import`` audit event (the load goes
    through ``_imp.create_dynamic``): a module outside the image is refused before the dynamic
    library is even opened."""
    so = _a_compiled_module()
    home = tmp_path / "home"
    home.mkdir()
    copy = home / so.name
    shutil.copy2(so, copy)
    spec = importlib.util.spec_from_file_location("fence_ext_probe", copy)
    assert spec is not None and spec.loader is not None
    with pytest.raises(HostedCodeRefused, match="deployment image"):
        importlib.util.module_from_spec(spec)  # loads through _imp.create_dynamic
    assert "fence_ext_probe" not in sys.modules


def test_an_image_extension_module_still_loads(hosted):
    """The same path, image side: an extension module inside this interpreter's own
    site-packages (where the image's packages live) loads exactly as before."""
    so = _a_compiled_module()
    # The spec's last name component must match the library's init function, as with any
    # extension import; the directory it loads from is what the fence judges.
    spec = importlib.util.spec_from_file_location(f"_fence_probe.{so.name.split('.')[0]}", so)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    assert module is not None
    assert getattr(module.__spec__, "origin", None) == str(so)  # the loader ran, nothing was refused


def test_spec_from_file_location_exec_module_is_refused(tmp_path, hosted):
    """The explicit loader every directory-plugin seam uses: refused at the compile of the
    file it was pointed at, whatever code called it."""
    home = tmp_path / "home"
    marker = home / "executed"
    module = _home_module(home, "fence_spec_mod", marker)
    spec = importlib.util.spec_from_file_location("fence_spec_mod", module)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    with pytest.raises(HostedCodeRefused, match="deployment image"):
        spec.loader.exec_module(mod)
    assert not marker.exists()


def test_runpy_run_path_is_refused(tmp_path, hosted):
    """``runpy.run_path`` compiles the file it was pointed at with the real path: refused
    before execution."""
    home = tmp_path / "home"
    marker = home / "executed"
    module = _home_module(home, "fence_runpy_mod", marker)
    with pytest.raises(HostedCodeRefused, match="deployment image"):
        runpy.run_path(str(module))
    assert not marker.exists()


def test_exec_compile_of_home_source_is_refused(tmp_path, hosted):
    """``exec(compile(open(p).read(), p, 'exec'))`` — the file is read as data, but the
    compile names the real path, and the fence refuses it there."""
    home = tmp_path / "home"
    marker = home / "executed"
    module = _home_module(home, "fence_exec_mod", marker)
    with pytest.raises(HostedCodeRefused, match="deployment image"):
        exec(compile(module.read_text(encoding="utf-8"), str(module), "exec"))
    assert not marker.exists()


# --------------------------------------------------------------------------------------------------
# The review's demonstrated bypasses are closed by the same fence


def test_a_dotted_entry_points_parent_package_never_runs(tmp_path, monkeypatch, hosted):
    """``find_spec('parent.child')`` imports the parent: with the fence installed first, the
    parent's compile is refused, so its module-level code never runs — and the boot's
    manifest pre-check reports the entry point as refused."""
    from hermes_cli.plugins_manifest import PluginManifest

    home = tmp_path / "home"
    parent = home / "fence_parent"
    parent.mkdir(parents=True)
    marker = home / "executed"
    (parent / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('parent ran')\n", encoding="utf-8")
    (parent / "child.py").write_text("def register(ctx): pass\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(home))
    manifest = PluginManifest(name="fence-parent", source="entrypoint",
                              path="fence_parent.child:register")
    monkeypatch.setattr(hosted_mode, "_plugins_that_would_load_inprocess", lambda: [manifest])
    # The gate would park it as not-enabled; the pre-check must judge provenance anyway.
    monkeypatch.setattr("hermes_cli.plugins_discovery.gate_manifest",
                        lambda *a, **k: type("Gate", (), {"action": "load", "error": None})())
    try:
        with pytest.raises(hosted_mode.HostedBootRefused, match="fence-parent"):
            hosted_mode._refuse_uncurated_inprocess_code({"plugins": {}})
        assert not marker.exists()  # the parent's module-level code never ran
        assert "fence_parent" not in sys.modules
    finally:
        sys.modules.pop("fence_parent", None)
        sys.modules.pop("fence_parent.child", None)


def test_a_symlinked_init_in_an_allowed_directory_is_refused(tmp_path, monkeypatch, hosted):
    """A directory check alone never sufficed: a genuine allowed directory whose
    ``__init__.py`` is a symlink to a home file is refused at the compile of the file that
    actually executes, through the real PluginManager loader."""
    import hermes_cli.plugins as plugins_mod
    from hermes_cli.plugins_manifest import PluginManifest
    from hermes_cli.plugins import PluginManager

    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    marker = tmp_path / "executed"
    _home_module(tmp_path, "home", marker)
    plugin_dir = bundled / "fence-init-symlink"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").symlink_to(tmp_path / "home.py")
    with pytest.raises(HostedCodeRefused, match="deployment image"):
        PluginManager()._load_directory_module(
            PluginManifest(name="fence-init-symlink", source="bundled", path=str(plugin_dir)))
    assert not marker.exists()  # the home file behind the symlink never ran


def test_an_eagerly_executed_symlinked_sibling_is_refused(tmp_path, monkeypatch, hosted):
    """The shared directory loader eagerly executes every sibling ``*.py``: a sibling that is
    a symlink to a home file is refused at its own compile, before either the sibling or the
    plugin's ``__init__`` runs."""
    import hermes_cli.plugins as plugins_mod
    from plugins.plugin_loader import load_plugin_module

    bundled = tmp_path / "bundled"
    plugin_dir = bundled / "fence-eager"
    plugin_dir.mkdir(parents=True)
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    marker = tmp_path / "executed"
    _home_module(tmp_path, "sibling_home.py", marker)
    (plugin_dir / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (plugin_dir / "sibling.py").symlink_to(tmp_path / "sibling_home.py")
    module_name = "_fence_eager.plugin"
    try:
        load_plugin_module(module_name, plugin_dir, parents=(),
                           logger=logging.getLogger(__name__))
        assert not marker.exists()  # the home file behind the symlink never ran
        assert f"{module_name}.sibling" not in sys.modules  # the sibling never loaded
    finally:
        sys.modules.pop(module_name, None)
        sys.modules.pop(module_name + ".sibling", None)


def test_an_allowed_package_importing_a_symlinked_child_is_refused(tmp_path, monkeypatch, hosted):
    """The transitive boundary: an allowed package whose ``__init__`` imports a child that is
    a symlink to a home file — the parent loads, the child's compile is refused, and the home
    code never runs."""
    import sysconfig as _sysconfig

    image = tmp_path / "image-site-packages"
    (image / "fence_pkg").mkdir(parents=True)
    marker = tmp_path / "executed"
    _home_module(tmp_path, "child_home.py", marker)
    (image / "fence_pkg" / "child.py").symlink_to(tmp_path / "child_home.py")
    (image / "fence_pkg" / "__init__.py").write_text("from . import child\n", encoding="utf-8")
    real_get_paths = _sysconfig.get_paths
    monkeypatch.setattr(_sysconfig, "get_paths", lambda *a, **k: {
        **real_get_paths(), "purelib": str(image), "platlib": str(image)})
    monkeypatch.syspath_prepend(str(image))
    try:
        with pytest.raises(ImportError):
            importlib.import_module("fence_pkg")  # the child's compile was refused
        assert not marker.exists()  # the home file behind the child symlink never ran
    finally:
        sys.modules.pop("fence_pkg", None)
        sys.modules.pop("fence_pkg.child", None)


@pytest.mark.parametrize("kind", ["memory", "context_engine"])
def test_availability_discovery_refuses_a_home_provider(tmp_path, monkeypatch, hosted, kind):
    """Discovery's availability probe used to call the private loaders directly, past every
    per-loader check: the fence refuses at the compile, the provider reports unavailable, and
    the home code never runs."""
    package = importlib.import_module("plugins." + kind)
    provider = tmp_path / "home" / ("fence_" + kind)
    provider.mkdir(parents=True)
    marker = provider.parent / "executed"
    (provider / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('discovery ran')\n",
        encoding="utf-8")
    if kind == "memory":
        monkeypatch.setattr(package, "_iter_provider_dirs", lambda: [(provider.name, provider)])
        monkeypatch.setattr(package, "_iter_entry_points", list)
        results = package.discover_memory_providers()
    else:
        monkeypatch.setattr(package, "_iter_engine_dirs", lambda: [(provider.name, provider)])
        results = package.discover_context_engines()
    assert [available for _, _, available in results] == [False]
    assert not marker.exists()


def test_memory_warmup_and_import_helper_refuse_a_home_provider(tmp_path, monkeypatch, hosted):
    """The warm-up path and the public ``import_provider_module`` helper reach the same
    shared loader; the fence refuses both, and the home code never runs."""
    from plugins import memory

    provider = tmp_path / "home" / "fence_memory_helpers"
    provider.mkdir(parents=True)
    marker = provider.parent / "executed"
    (provider / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('helpers ran')\n",
        encoding="utf-8")
    monkeypatch.setattr(memory, "find_provider_dir", lambda name: provider)
    monkeypatch.setattr(memory, "_NATIVE_WARM_IMPORTS", ())
    name = memory._module_name(provider, provider.name)
    try:
        assert memory.import_memory_provider_module(provider.name) is False
        assert not marker.exists()
        with pytest.raises(ImportError):
            memory.import_provider_module(provider.name)
        assert not marker.exists()
    finally:
        sys.modules.pop(name, None)


def test_the_model_provider_entry_point_scan_refuses_a_home_entry_point(tmp_path, monkeypatch, hosted):
    """The provider entry-point scanner called ``ep.load()`` behind only the isolation-mode
    check: the fence refuses the import, the scan logs the failure, and the home code never
    runs."""
    import providers

    src = tmp_path / "home"
    marker = src / "executed"
    _home_module(src, "fence_model", marker)
    monkeypatch.syspath_prepend(str(src))
    entry_point = importlib.metadata.EntryPoint(
        name="fence-model", value="fence_model", group="hermes_agent.plugins")
    monkeypatch.setattr(importlib.metadata, "entry_points",
                        lambda: importlib.metadata.EntryPoints([entry_point]))
    monkeypatch.setattr("hermes_cli.plugins._get_enabled_plugins", lambda: {"fence-model"})
    monkeypatch.setattr("hermes_cli.plugins._get_disabled_plugins", lambda: set())
    try:
        assert hosted_code_refusal(None, module="fence_model")
        providers._discover_entry_point_providers()
        assert not marker.exists()  # the home module never ran
        assert "fence_model" not in sys.modules
    finally:
        sys.modules.pop("fence_model", None)


def test_a_memory_schema_symlinked_from_the_bundled_tree_is_refused(tmp_path, monkeypatch, hosted):
    """The schema loader judged a directory lexically bundled and executed its
    ``config_schema.py`` whatever the file really was: the fence refuses the symlinked file at
    its compile, the panel has no schema, and the home code never runs."""
    from plugins import memory
    from plugins.memory import config_schema

    root = tmp_path / "image" / "memory"
    root.mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    marker = home / "executed"
    (home / "config_schema.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('schema ran')\n",
        encoding="utf-8")
    link = root / "fence_schema"
    link.symlink_to(home, target_is_directory=True)
    monkeypatch.setattr(memory, "_MEMORY_PLUGINS_DIR", root)
    monkeypatch.setattr(memory, "find_provider_dir", lambda name: link)
    monkeypatch.setattr(config_schema, "_SCHEMA_CACHE", {})
    try:
        assert config_schema.get_provider_config_schema("fence_schema") is None
        assert not marker.exists()
    finally:
        sys.modules.pop("_hermes_memory_config_schema.fence_schema", None)


def test_a_platform_declaration_symlinked_into_the_bundled_tree_is_refused(tmp_path, monkeypatch, hosted):
    """The lightweight bundled-declaration loader executed a ``gateway_env.py`` link without
    any provenance check: the fence refuses the symlinked file at its compile, and the home
    code never runs."""
    from gateway import platform_env
    from gateway.config import Platform

    root = Path(platform_env.__file__).resolve().parent.parent / "plugins" / "platforms"
    probe = root / "_fence_env"
    probe.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    marker = home / "executed"
    (home / "env.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('platform env ran')\n",
        encoding="utf-8")
    (probe / "gateway_env.py").symlink_to(home / "env.py")
    monkeypatch.setattr(Platform, "_scan_bundled_plugin_platforms", lambda: ({"_fence_env"}, set()))
    try:
        with pytest.raises(HostedCodeRefused, match="deployment image"):
            platform_env.bundled_sibling("_fence_env")
        assert not marker.exists()  # the home file behind the symlink never ran
    finally:
        platform_env.bundled_sibling.cache_clear()
        (probe / "gateway_env.py").unlink(missing_ok=True)
        probe.rmdir()


def test_a_real_preflight_refuses_a_dotted_home_entrypoint_without_importing_the_parent(
        tmp_path, monkeypatch):
    """End to end against the real preflight: a home ``.dist-info`` with an enabled dotted
    entry point reaches ``hosted_boot_preflight``, which refuses the boot naming the plugin —
    and the parent package's module-level code never runs on the way there."""
    from hermes_constants import get_hermes_home
    from hermes_cli.plugins_bundled import get_bundled_plugins_dir as real_root

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    src = home / "src"
    parent = src / "fence_real_parent"
    parent.mkdir(parents=True)
    marker = home / "executed"
    (parent / "__init__.py").write_text(
        "from pathlib import Path\n"
        "from hermes_cli.hosted_mode import hosted_verdict_frozen\n"
        f"Path({str(marker)!r}).write_text(str(hosted_verdict_frozen()))\n",
        encoding="utf-8")
    (parent / "child.py").write_text("def register(ctx): pass\n", encoding="utf-8")
    dist = src / "fence-real-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: fence-real\nVersion: 1.0\n", encoding="utf-8")
    (dist / "entry_points.txt").write_text(
        "[hermes_agent.plugins]\nfence-real = fence_real_parent.child:register\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(src))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(home / "fake-bundled"))
    (home / "config.yaml").write_text(
        "hosted:\n  enabled: true\nplugins:\n  enabled: [fence-real]\n", encoding="utf-8")
    monkeypatch.setattr(hosted_mode, "_VERDICT",
                        {"hosted": None, "config": None, "plugins_permitted": False})
    try:
        with pytest.raises(hosted_mode.HostedBootRefused, match="fence-real"):
            hosted_mode.hosted_boot_preflight(env={"PATH": "/usr/bin"})
        assert not marker.exists()  # the parent package never ran, so no marker
        assert hosted_mode.hosted_verdict_frozen() is True
        assert real_root() != home / "fake-bundled"  # the bundled root ignores the environment
        assert hosted_mode._VERDICT["plugins_permitted"] is False
    finally:
        sys.modules.pop("fence_real_parent", None)
        sys.modules.pop("fence_real_parent.child", None)


# --------------------------------------------------------------------------------------------------
# sys.path pruning: nothing outside the image stays searchable


def test_preflight_prunes_out_of_image_sys_path_and_records_it(tmp_path, monkeypatch, caplog, hosted):
    """A ``.pth`` file inside an allowed site root can still add the home to ``sys.path`` at
    interpreter startup; the preflight removes such entries, records what it removed in the
    boot log, and afterwards the home's modules cannot even be found."""
    import site

    from hermes_constants import get_hermes_home

    home = tmp_path / "home"
    home.mkdir()
    _home_module(home, "fence_pth_mod", home / "executed", text="raise AssertionError('must never import')")
    image_site = tmp_path / "image-site"
    image_site.mkdir()
    (image_site / "fence.pth").write_text(f"{home}\n", encoding="utf-8")
    real_get_paths = sysconfig.get_paths
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {
        **real_get_paths(), "purelib": str(image_site), "platlib": str(image_site)})
    before = sys.path[:]
    (get_hermes_home() / "config.yaml").write_text("hosted:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setattr(hosted_mode, "_VERDICT",
                        {"hosted": None, "config": None, "plugins_permitted": False})
    try:
        site.addsitedir(str(image_site))
        importlib.invalidate_caches()
        assert str(home) in sys.path  # the .pth added it
        with caplog.at_level(logging.INFO):
            hosted_mode.hosted_boot_preflight(env={"PATH": "/usr/bin"})
        assert str(home) not in sys.path  # the boot pruned it
        assert "fence.pth" in " ".join(caplog.messages) or str(home) in " ".join(caplog.messages)
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("fence_pth_mod")
        assert not (home / "executed").exists()
    finally:
        sys.path[:] = before
        importlib.invalidate_caches()
        sys.modules.pop("fence_pth_mod", None)


def test_pruning_keeps_the_images_own_paths(hosted):
    """The interpreter's own paths survive the prune: the venv's site-packages, the stdlib
    and the work tree are all inside the image, so a hosted boot loses nothing it ships."""
    import sysconfig

    paths = sysconfig.get_paths()
    pruned = prune_sys_path_outside_image()
    assert not [entry for entry in pruned
                if Path(entry or Path.cwd()).resolve() in (Path(paths[key]).resolve()
                                                           for key in ("stdlib", "purelib"))]
    assert any(Path(entry or Path.cwd()).resolve() == Path(paths["purelib"]).resolve()
               for entry in sys.path)


# --------------------------------------------------------------------------------------------------
# The shared predicate's entry-point semantics


def test_a_namespace_module_is_refused(tmp_path, monkeypatch, hosted):
    """A namespace package has no origin file to judge, so no trust: refused."""
    (tmp_path / "fence_namespace").mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    assert importlib.util.find_spec("fence_namespace").origin is None
    assert hosted_code_refusal(None, module="fence_namespace")


def test_a_zip_origin_outside_the_roots_is_refused(tmp_path, monkeypatch, hosted):
    """An entry point whose module resolves inside a zip archive outside the image is refused
    by the same predicate the boot consults."""
    archive = tmp_path / "tenant.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("fence_zip_entry.py", "VALUE = 1\n")
    monkeypatch.syspath_prepend(str(archive))
    assert hosted_code_refusal(None, module="fence_zip_entry")


def test_an_allowed_package_submodule_symlink_is_refused(tmp_path, monkeypatch, hosted):
    """A dotted entry point's parent may be image code while the child it targets is a
    symlink to a home file: the predicate refuses the child by its resolved file, not by the
    parent's origin."""
    import hermes_cli.plugins as plugins_mod

    bundled = tmp_path / "bundled"
    pkg = bundled / "fence_pkg_probe"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    evil = tmp_path / "evil.py"
    evil.write_text("VALUE = 1\n", encoding="utf-8")
    (pkg / "child.py").symlink_to(evil)
    monkeypatch.syspath_prepend(str(bundled))
    try:
        assert hosted_code_refusal(None, module="fence_pkg_probe") == ""
        assert hosted_code_refusal(None, module="fence_pkg_probe.child")
    finally:
        sys.modules.pop("fence_pkg_probe", None)
        shutil.rmtree(pkg, ignore_errors=True)  # imports cached bytecode beside the package


def test_a_custom_meta_path_loader_claiming_an_allowed_origin_still_executes(tmp_path, monkeypatch, hosted):
    """The documented boundary, unchanged: a custom meta-path finder hands importlib an
    invented origin inside the image and executes — installing the finder already requires
    trusted in-process code, which the fence does not pretend to sandbox."""
    import importlib.abc
    import importlib.machinery

    marker = tmp_path / "executed"

    class Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            marker.write_text("custom loader ran")

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "fence_custom":
                return importlib.machinery.ModuleSpec(
                    fullname, Loader(), origin=str(Path(__file__).parent / "invented.py"))

    finder = Finder()
    sys.meta_path.insert(0, finder)
    try:
        assert hosted_code_refusal(None, module="fence_custom") == ""
        importlib.import_module("fence_custom")
        assert marker.exists()
    finally:
        sys.meta_path.remove(finder)
        sys.modules.pop("fence_custom", None)


# --------------------------------------------------------------------------------------------------
# Positive controls: the image's own code loads exactly as before


def test_bundled_plugin_code_loads_under_the_fence(tmp_path, monkeypatch, hosted):
    """A plugin inside the tree the image bundles with Hermes imports through the real
    directory loader and runs."""
    import hermes_cli.plugins as plugins_mod
    from hermes_cli.plugins_manifest import PluginManifest
    from hermes_cli.plugins import PluginManager

    bundled = tmp_path / "bundled"
    plugin_dir = bundled / "fence-bundled"
    plugin_dir.mkdir(parents=True)
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    marker = tmp_path / "executed"
    (plugin_dir / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bundled ran')\n"
        "def register(ctx):\n    pass\n", encoding="utf-8")
    module = PluginManager()._load_directory_module(
        PluginManifest(name="fence-bundled", source="bundled", path=str(plugin_dir)))
    assert module is not None
    assert marker.read_text(encoding="utf-8") == "bundled ran"


def test_an_installed_package_entry_point_loads_under_the_fence(tmp_path, monkeypatch, hosted):
    """An installed distribution's entry point — its module resolves inside the image's
    site-packages — loads exactly as before."""
    import sysconfig as _sysconfig

    site_dir = tmp_path / "image-site-packages"
    site_dir.mkdir()
    marker = tmp_path / "executed"
    (site_dir / "fence_image_entry.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('image code ran')\n"
        "def register(ctx):\n    pass\n", encoding="utf-8")
    real_get_paths = _sysconfig.get_paths
    monkeypatch.setattr(_sysconfig, "get_paths", lambda *a, **k: {
        **real_get_paths(), "purelib": str(site_dir), "platlib": str(site_dir)})
    monkeypatch.syspath_prepend(str(site_dir))
    try:
        module = importlib.import_module("fence_image_entry")
        assert module is not None
        assert marker.read_text(encoding="utf-8") == "image code ran"
    finally:
        sys.modules.pop("fence_image_entry", None)


def test_the_stdlib_and_core_tools_import_under_the_fence(hosted):
    """The stdlib and this checkout's own tool modules are image code: fresh imports under
    the fence behave exactly as before."""
    importlib.import_module("dataclasses")  # stdlib
    importlib.import_module("tools.file_tools")  # the Hermes tree's core tools
    importlib.import_module("plugins.memory")  # the bundled plugins tree


# --------------------------------------------------------------------------------------------------
# Local behaviour: no fence is installed by the boot, and a dormant fence changes nothing


def test_install_refuses_to_act_without_a_frozen_hosted_verdict(monkeypatch):
    """The install is gated on the frozen hosted verdict: a local install (verdict unset or
    false) installs nothing, even when called."""
    monkeypatch.setattr(hosted_mode, "_VERDICT", {"hosted": None})
    assert install_hosted_code_fence() is False


def test_the_fence_is_dormant_when_the_verdict_flips_back(tmp_path, monkeypatch):
    """A process that was hosted and installed the fence keeps local behaviour once the
    verdict is not hosted: a home plugin imports exactly as before, byte for byte."""
    from plugins.plugin_loader import load_plugin_module

    monkeypatch.setattr(hosted_mode, "_VERDICT", {"hosted": False})
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    marker = tmp_path / "executed"
    (plugin_dir / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('local ran')\n",
        encoding="utf-8")
    module_name = "_fence_local.plugin"
    try:
        module = load_plugin_module(module_name, plugin_dir, parents=(),
                                    logger=logging.getLogger(__name__))
        assert module is not None
        assert marker.read_text(encoding="utf-8") == "local ran"
    finally:
        sys.modules.pop(module_name, None)
