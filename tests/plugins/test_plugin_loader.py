"""Regression tests for directory-plugin module loading."""

from __future__ import annotations

import importlib
import json
import logging
import site
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from plugins.plugin_loader import load_plugin_module


def test_failed_sibling_is_removed_before_init_handles_missing_import(tmp_path):
    """A failed eager sibling import must remain catchable as ModuleNotFoundError."""
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "broken.py").write_text(
        "from .missing_dependency import value\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "try:\n"
        "    from .broken import value\n"
        "except ModuleNotFoundError:\n"
        "    fallback_used = True\n",
        encoding="utf-8",
    )
    module_name = "test_plugin_loader_package.failed_sibling"

    try:
        module = load_plugin_module(
            module_name,
            plugin_dir,
            parents=(),
            logger=logging.getLogger(__name__),
        )

        assert module is not None
        assert module.fallback_used is True
        assert f"{module_name}.broken" not in sys.modules
    finally:
        for name in tuple(sys.modules):
            if name == module_name or name.startswith(f"{module_name}."):
                sys.modules.pop(name, None)


def test_successful_sibling_remains_available_on_loaded_module(tmp_path):
    """Cleaning failed siblings must not alter the eager success path."""
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "helper.py").write_text("value = 42\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "from .helper import value\n",
        encoding="utf-8",
    )
    module_name = "test_plugin_loader_package.successful_sibling"

    try:
        module = load_plugin_module(
            module_name,
            plugin_dir,
            parents=(),
            logger=logging.getLogger(__name__),
        )

        assert module is not None
        assert module.value == 42
        assert module.helper.value == 42
    finally:
        for name in tuple(sys.modules):
            if name == module_name or name.startswith(f"{module_name}."):
                sys.modules.pop(name, None)


def test_concurrent_load_waits_for_the_first_load(tmp_path, monkeypatch):
    """A second caller never receives the half-built shell published mid-load. An agent build waits
    out a slow first import; a per-turn caller in bounded_load_wait() refuses after the bound instead."""
    import plugins.plugin_loader as loader

    monkeypatch.setattr(loader, "_CONCURRENT_LOAD_WAIT_SECS", 0.3)
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    gate_name = "_test_plugin_loader_race_gate"
    gate = SimpleNamespace(entered=threading.Event(), release=threading.Event())
    sys.modules[gate_name] = gate
    (plugin_dir / "slow.py").write_text(
        f"import sys\ngate = sys.modules[{gate_name!r}]\n"
        "gate.entered.set()\ngate.release.wait(10)\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    module_name = "test_plugin_loader_package.concurrent_load"
    results = {}

    def load(key):
        results[key] = load_plugin_module(module_name, plugin_dir, parents=(),
                                          logger=logging.getLogger(__name__))

    def bounded(key):
        with loader.bounded_load_wait():
            load(key)

    first = threading.Thread(target=load, args=("first",))
    build = threading.Thread(target=load, args=("build",))
    turn = threading.Thread(target=bounded, args=("turn",))
    try:
        first.start()
        assert gate.entered.wait(5)
        build.start()
        build.join(1)  # well past the bound
        assert build.is_alive(), f"agent build gave up on a slow import: {results.get('build')!r}"
        turn.start()
        turn.join(5)
        assert not turn.is_alive(), "per-turn caller blocked on the hung first load"
        assert results["turn"] is None
        bounded("stalled")  # later bounded callers refuse at once while the import is still hung
        assert results["stalled"] is None
        gate.release.set()
        first.join(10)
        build.join(10)
        assert hasattr(results["first"], "register")
        assert results["build"] is results["first"]
        bounded("after")
        assert results["after"] is results["first"]

        # A cross-thread cycle through importlib's own module lock (thread A imports a helper that
        # loads the plugin; thread B loads the plugin, which imports the helper) must not deadlock.
        monkeypatch.setattr(loader, "_CONCURRENT_LOAD_WAIT_SECS", 60)
        cyc_dir, helper = tmp_path / "cyc", "_test_plugin_loader_cycle_helper"
        cyc_dir.mkdir()
        cyc_name = "test_plugin_loader_package.cycle"
        gate.helper, gate.loader = threading.Event(), threading.Event()
        monkeypatch.syspath_prepend(str(tmp_path))
        (tmp_path / f"{helper}.py").write_text(
            f"import sys, logging, pathlib\nfrom plugins.plugin_loader import load_plugin_module\n"
            f"gate = sys.modules[{gate_name!r}]\ngate.helper.set()\ngate.loader.wait(10)\n"
            f"mod = load_plugin_module({cyc_name!r}, pathlib.Path({str(cyc_dir)!r}), parents=(),"
            " logger=logging.getLogger('x'))\n", encoding="utf-8")
        (cyc_dir / "__init__.py").write_text(
            f"import sys\nsys.modules[{gate_name!r}].loader.set()\nimport {helper}\n"
            "def register(ctx):\n    pass\n", encoding="utf-8")
        a = threading.Thread(target=importlib.import_module, args=(helper,), daemon=True)
        b = threading.Thread(target=load_plugin_module, args=(cyc_name, cyc_dir),
                             kwargs={"parents": (), "logger": logging.getLogger(__name__)}, daemon=True)
        a.start()
        assert gate.helper.wait(5)
        b.start()
        a.join(5)
        b.join(5)
        assert not a.is_alive() and not b.is_alive(), "loader lock and import lock deadlocked"
        assert hasattr(sys.modules[cyc_name], "register")
    finally:
        gate.release.set()
        getattr(gate, "loader", threading.Event()).set()
        sys.modules.pop(gate_name, None)
        sys.modules.pop("_test_plugin_loader_cycle_helper", None)
        for name in tuple(sys.modules):
            if name.startswith("test_plugin_loader_package."):
                sys.modules.pop(name, None)


def test_load_named_refuses_a_home_dir_in_a_hosted_process(tmp_path, monkeypatch):
    """The shared named-provider seam: in a hosted process a provider directory outside the
    tree bundled with Hermes is refused BEFORE its loader (which would import it) runs — the
    one rule is that only what the deployment image ships loads in-process."""
    from hermes_cli import hosted_mode
    from plugins.plugin_loader import NamedProviderRefused, load_named

    monkeypatch.setattr(hosted_mode, "is_hosted", lambda: True)

    def _must_not_load(provider_dir):
        raise AssertionError("a refused provider's loader ran")

    with pytest.raises(NamedProviderRefused, match="deployment image"):
        load_named("probe", tmp_path / "user-dir", _must_not_load, kind="Probe provider",
                   noun="provider", logger=logging.getLogger(__name__))


def test_load_named_loads_a_bundled_dir_in_a_hosted_process(tmp_path, monkeypatch):
    """The same seam, bundled side: a provider inside the tree the image bundles with Hermes
    loads exactly as before when the process is hosted."""
    import hermes_cli.plugins as plugins_mod
    from hermes_cli import hosted_mode
    from plugins.plugin_loader import load_named

    monkeypatch.setattr(hosted_mode, "is_hosted", lambda: True)
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)

    def _load(provider_dir):
        return object()

    assert load_named("probe", bundled, _load, kind="Probe provider", noun="provider",
                      logger=logging.getLogger(__name__)) is not None


def test_a_symlink_wearing_the_bundled_label_is_refused(tmp_path, monkeypatch):
    """The bundled tree is resolved from Hermes's own installed code, and a directory that is
    a symlink — or whose realpath leaves that tree — does not get the bundled trust, whatever
    its label says."""
    import hermes_cli.plugins as plugins_mod
    from hermes_cli import hosted_mode
    from plugins.plugin_loader import NamedProviderRefused, hosted_inprocess_refusal, load_named

    monkeypatch.setattr(hosted_mode, "is_hosted", lambda: True)
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (bundled / "impostor").symlink_to(elsewhere)

    assert "deployment image" in hosted_inprocess_refusal(bundled / "impostor")
    assert hosted_inprocess_refusal(bundled) == ""  # the genuine tree itself is eligible

    def _must_not_load(provider_dir):
        raise AssertionError("a refused provider's loader ran")

    with pytest.raises(NamedProviderRefused, match="deployment image"):
        load_named("impostor", bundled / "impostor", _must_not_load, kind="Probe provider",
                   noun="provider", logger=logging.getLogger(__name__))


def test_the_plugin_managers_directory_loader_refuses_a_home_dir_in_a_hosted_process(tmp_path, monkeypatch):
    """PluginManager's in-process import seam — a directory plugin's ``__init__.py`` — answers
    the same one hosted rule: a directory outside the tree the image bundles with Hermes never
    imports in a hosted process, whatever its install record says."""
    from hermes_cli import hosted_mode
    from hermes_cli.plugins_manifest import PluginManifest
    from hermes_cli.plugins import PluginManager

    monkeypatch.setattr(hosted_mode, "is_hosted", lambda: True)
    plugin_dir = tmp_path / "home-plugin"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(tmp_path / 'imported')!r}).write_text('x')\n",
        encoding="utf-8",
    )
    manifest = PluginManifest(name="home-plugin", source="user", path=str(plugin_dir))

    with pytest.raises(RuntimeError, match="deployment image"):
        PluginManager()._load_directory_module(manifest)
    assert not (tmp_path / "imported").exists()  # the plugin module never ran


# --------------------------------------------------------------------------------------------------
# Installed entry points: the same one rule judges the file the entry point's module resolves to


def _planted_home_distribution(tmp_path, kind):
    """A distribution whose entry point targets code in the deployment home, discovered the way
    a planted distribution is: ``kind='pth'`` puts the distribution metadata in the home itself
    with a ``.pth`` file adding the home's source dir to the path; ``kind='editable'`` puts an
    editable record in the image's site-packages pointing into the home. Returns the marker the
    home code would write and the site dir to remove from ``sys.path`` afterwards."""
    home = tmp_path / "home"
    src = home / "src"
    src.mkdir(parents=True)
    marker = home / "executed"
    (src / "rv3_home_entry.py").write_text(
        f'from pathlib import Path\nPath({str(marker)!r}).write_text("home code ran")\n\n\ndef register(ctx):\n    pass\n',
        encoding="utf-8")
    site_dir = tmp_path / "image-site-packages"
    site_dir.mkdir()
    dist = (src if kind == "pth" else site_dir) / "rv3_home_entry-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: rv3-home-entry\nVersion: 1.0\n", encoding="utf-8")
    (dist / "entry_points.txt").write_text(
        "[hermes_agent.plugins]\nrv3-home-entry = rv3_home_entry:register\n", encoding="utf-8")
    if kind == "editable":
        (dist / "direct_url.json").write_text(
            json.dumps({"url": src.as_uri(), "dir_info": {"editable": True}}), encoding="utf-8")
    (site_dir / "rv3.pth").write_text(f"{src}\n", encoding="utf-8")
    return marker, site_dir


@pytest.mark.parametrize("kind", ["pth", "editable"])
def test_an_entry_point_whose_module_resolves_into_the_home_is_refused(tmp_path, monkeypatch, kind):
    """Distribution metadata is not provenance. A hosted process judges an installed
    distribution's entry point by where its module's file really is — resolved WITHOUT
    importing it — so a ``.pth``-discovered home distribution and an image-side editable
    record pointing into the home are both refused before ``ep.load()`` can import anything:
    the home's code never runs."""
    from hermes_cli import hosted_mode
    from hermes_cli.plugins import PluginManager
    from hermes_cli.plugins_manifest import PluginManifest
    from plugins.plugin_loader import hosted_inprocess_refusal

    monkeypatch.setattr(hosted_mode, "is_hosted", lambda: True)
    marker, site_dir = _planted_home_distribution(tmp_path, kind)
    before_path = sys.path[:]
    try:
        site.addsitedir(str(site_dir))
        importlib.invalidate_caches()
        manifest = PluginManifest(name="rv3-home-entry", source="entrypoint",
                                  path="rv3_home_entry:register")
        assert hosted_inprocess_refusal(None, module="rv3_home_entry")
        with pytest.raises(RuntimeError, match="deployment image"):
            PluginManager()._load_entrypoint_module(manifest)
        assert not marker.exists()  # the home's code never ran
    finally:
        sys.path[:] = before_path
        sys.modules.pop("rv3_home_entry", None)
        importlib.invalidate_caches()


def test_an_entry_point_of_a_package_in_the_images_environment_still_loads(tmp_path, monkeypatch):
    """The same rule, image side: an entry point whose module resolves inside this
    interpreter's own site-packages — where the image's packages are installed — is eligible
    and loads exactly as before in a hosted process."""
    import sysconfig

    from hermes_cli import hosted_mode
    from hermes_cli.plugins import PluginManager
    from hermes_cli.plugins_manifest import PluginManifest
    from plugins.plugin_loader import hosted_inprocess_refusal

    monkeypatch.setattr(hosted_mode, "is_hosted", lambda: True)
    site_dir = tmp_path / "image-site-packages"
    site_dir.mkdir()
    marker = tmp_path / "executed"
    (site_dir / "rv3_image_entry.py").write_text(
        f'from pathlib import Path\nPath({str(marker)!r}).write_text("image code ran")\n\n\ndef register(ctx):\n    pass\n',
        encoding="utf-8")
    dist = site_dir / "rv3_image_entry-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: rv3-image-entry\nVersion: 1.0\n", encoding="utf-8")
    (dist / "entry_points.txt").write_text(
        "[hermes_agent.plugins]\nrv3-image-entry = rv3_image_entry:register\n", encoding="utf-8")
    real_get_paths = sysconfig.get_paths
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {
        **real_get_paths(), "purelib": str(site_dir), "platlib": str(site_dir)})
    before_path = sys.path[:]
    try:
        site.addsitedir(str(site_dir))
        importlib.invalidate_caches()
        manifest = PluginManifest(name="rv3-image-entry", source="entrypoint",
                                  path="rv3_image_entry:register")
        assert hosted_inprocess_refusal(None, module="rv3_image_entry") == ""
        assert PluginManager()._load_entrypoint_module(manifest) is not None
        assert marker.read_text(encoding="utf-8") == "image code ran"
    finally:
        sys.path[:] = before_path
        sys.modules.pop("rv3_image_entry", None)
        importlib.invalidate_caches()


def test_the_bundled_root_cannot_be_moved_by_the_environment_in_a_hosted_process(tmp_path, monkeypatch):
    """``HERMES_BUNDLED_PLUGINS`` keeps its local meaning (a packaged install's read-only store
    path), but a hosted process ignores it: the tree in-process trust resolves from is always
    the one inside the installed Hermes package, so the environment cannot move the root the
    boot judges provenance from."""
    from hermes_cli import hosted_mode
    from hermes_cli.plugins import get_bundled_plugins_dir
    from plugins.plugin_loader import bundled_dir_is_genuine, hosted_inprocess_refusal

    fake_root = tmp_path / "operator-named-root"
    fake_root.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(fake_root))
    monkeypatch.setitem(hosted_mode._VERDICT, "hosted", False)
    assert get_bundled_plugins_dir() == fake_root  # a local install honours the override

    monkeypatch.setitem(hosted_mode._VERDICT, "hosted", True)  # the boot froze: hosted
    assert get_bundled_plugins_dir() != fake_root  # a hosted process ignores the environment
    assert not bundled_dir_is_genuine(fake_root)
    assert hosted_inprocess_refusal(fake_root)  # and its directories are refused
