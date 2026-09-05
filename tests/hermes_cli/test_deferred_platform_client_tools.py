"""Deferred platform plugins must still register their *client* tools.

Issue #78050: a bundled ``kind: platform`` plugin is registered as a deferred
loader so ``hermes chat`` doesn't import ~20 gateway SDKs. The a2a plugin ships
two independent things behind that one deferral — an inbound adapter (heavy)
and outbound client tools (``a2a_call``, ``a2a_get_task``, ``a2a_discover``,
``a2a_list``, ``a2a_history``, ``a2a_orchestrate``). Deferring the plugin deferred both, so
in a CLI/TUI process the client tools never registered at all:
``resolve_toolset("a2a")`` returned ``[]`` and the toolset was absent from the
``hermes tools`` checklist. The same tools worked in gateway/web processes only
because those materialize every platform at startup.

Client tools that live in a dedicated ``tools`` submodule are now registered at
discovery time; the adapter stays deferred.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import yaml


A2A_CLIENT_TOOLS = {
    "a2a_call",
    "a2a_discover",
    "a2a_get_task",
    "a2a_history",
    "a2a_list",
    "a2a_orchestrate",
}


# ── synthetic platform plugin helpers ──────────────────────────────────────


def _write_platform_plugin(
    root: Path,
    platform: str,
    *,
    with_tools_module: bool,
    declares_provides_tools: "bool | None" = None,
) -> "object":
    """Create a bundled-style platform plugin and return its manifest.

    The adapter import is the expensive thing we must NOT trigger: it is
    modelled as ``adapter.py`` setting a module-level sentinel, imported from
    inside ``register()`` exactly as the real a2a plugin does.

    ``declares_provides_tools`` controls the manifest opt-in independently of
    whether a ``tools.py`` exists on disk, so a test can pin what actually
    triggers pre-registration. Defaults to following ``with_tools_module``,
    which is the shape a real plugin ships.
    """
    from hermes_cli.plugins import PluginManifest

    if declares_provides_tools is None:
        declares_provides_tools = with_tools_module
    provides_tools = [f"{platform}_call"] if declares_provides_tools else []

    plugin_dir = root / platform
    plugin_dir.mkdir(parents=True, exist_ok=True)
    manifest_data = {
        "name": f"{platform}-platform",
        "kind": "platform",
        "version": "1.0.0",
    }
    if provides_tools:
        manifest_data["provides_tools"] = provides_tools
    (plugin_dir / "plugin.yaml").write_text(
        yaml.dump(manifest_data),
        encoding="utf-8",
    )

    # Sentinels let the tests prove what was and wasn't imported.
    (plugin_dir / "adapter.py").write_text(
        "import _deferred_probe\n"
        "_deferred_probe.adapter_imports += 1\n",
        encoding="utf-8",
    )
    init_body = [
        "import _deferred_probe",
        "_deferred_probe.package_execs += 1",
        "",
        "def register(ctx):",
        "    from . import adapter  # noqa: F401  (heavy import, deferred)",
    ]
    if with_tools_module:
        (plugin_dir / "tools.py").write_text(
            "import _deferred_probe\n"
            "_deferred_probe.tools_execs += 1\n"
            "\n"
            "\n"
            "def _handler(**kwargs):\n"
            "    return 'ok'\n"
            "\n"
            "\n"
            "def register_tools(ctx):\n"
            f"    ctx.register_tool(\n"
            f"        name='{platform}_call',\n"
            f"        toolset='{platform}',\n"
            "        schema={'type': 'function', 'function': {'name': "
            f"'{platform}_call', 'description': 'call a peer', 'parameters': "
            "{'type': 'object', 'properties': {}}}},\n"
            "        handler=_handler,\n"
            "        description='call a peer',\n"
            "    )\n",
            encoding="utf-8",
        )
        init_body.append("    from .tools import register_tools")
        init_body.append("    register_tools(ctx)")
    (plugin_dir / "__init__.py").write_text("\n".join(init_body) + "\n", encoding="utf-8")

    return PluginManifest(
        name=f"{platform}-platform",
        kind="platform",
        source="bundled",
        path=str(plugin_dir),
        key=f"{platform}-platform",
        provides_tools=provides_tools,
    )


@pytest.fixture
def probe(monkeypatch):
    """A module the synthetic plugin can count imports into."""
    import types

    mod = types.ModuleType("_deferred_probe")
    mod.package_execs = 0
    mod.adapter_imports = 0
    mod.tools_execs = 0
    monkeypatch.setitem(sys.modules, "_deferred_probe", mod)
    return mod


@pytest.fixture
def clean_registry():
    """Undo everything a synthetic plugin leaves behind.

    Each test writes a fresh plugin to its own tmp_path but reuses the
    ``probeplat`` name, so the imported ``hermes_plugins.*`` modules have to go
    too — otherwise the next test's ``import_module`` returns the previous
    test's cached submodule instead of reading the new file.
    """
    from gateway.platform_registry import platform_registry
    from tools.registry import registry

    before_tools = set(registry._tools)
    before_modules = set(sys.modules)
    yield
    for name in set(registry._tools) - before_tools:
        registry._tools.pop(name, None)
    for platform in ("probeplat", "barefoot", "quietplat", "promiseplat"):
        platform_registry.unregister(platform)
    for name in set(sys.modules) - before_modules:
        if name.startswith("hermes_plugins."):
            sys.modules.pop(name, None)


# ── the reported symptom, against the real a2a plugin ──────────────────────


class TestA2AClientToolsInCliProcess:
    """The issue's exact repro: a CLI/TUI process, no gateway startup."""

    def test_manifest_declares_the_client_tools(self):
        """The opt-in lives in the manifest, so it is pinned like any contract.

        Dropping ``provides_tools`` from plugin.yaml silently reverts a2a to
        the deferred-and-invisible behaviour of #78050, with every other test
        here still passing on the synthetic plugins — so assert it directly.
        """
        manifest_path = (
            Path(__file__).resolve().parents[2]
            / "plugins" / "platforms" / "a2a" / "plugin.yaml"
        )
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

        assert set(manifest.get("provides_tools") or []) == A2A_CLIENT_TOOLS

    def test_a2a_toolset_resolves_without_materializing_the_platform(self):
        from hermes_cli.plugins import PluginManager
        from toolsets import resolve_toolset

        mgr = PluginManager()
        mgr.discover_and_load()

        a2a = mgr._plugins.get("a2a-platform")
        assert a2a is not None, "bundled a2a platform plugin was not discovered"

        # The whole point of the deferral is preserved: the inbound adapter is
        # still not imported in a CLI process.
        assert a2a.deferred is True

        # ...but the outbound client tools are now reachable. Before the fix
        # this was [] until a gateway/web process called all_entries().
        assert set(resolve_toolset("a2a")) == A2A_CLIENT_TOOLS

    def test_a2a_appears_in_the_hermes_tools_checklist(self):
        """`a2a` is in _DEFAULT_OFF_TOOLSETS, so it must be tickable.

        Every other member of that set (homeassistant, spotify, video_gen,
        x_search, ...) renders a checkbox; a2a rendered nothing, so the
        documented opt-in path had nothing to tick.
        """
        from hermes_cli.plugins import discover_plugins, get_plugin_toolsets

        # get_plugin_toolsets() reads the process-wide manager, which is what
        # the `hermes tools` checklist does.
        discover_plugins()

        assert "a2a" in {key for key, _, _ in get_plugin_toolsets()}

    def test_platform_bundle_includes_the_client_tools(self):
        """``hermes-a2a`` sessions get the client tools too.

        The bundle path read the tool registry behind a deliberately cheap
        ``is_registered()`` check, so a deferred platform's own tools were
        dropped from its bundle as well.
        """
        from hermes_cli.plugins import PluginManager
        from toolsets import resolve_toolset

        mgr = PluginManager()
        mgr.discover_and_load()

        assert A2A_CLIENT_TOOLS.issubset(set(resolve_toolset("hermes-a2a")))


# ── the general mechanism ──────────────────────────────────────────────────


class TestDeferredPlatformToolPreregistration:
    def test_tools_module_registers_without_importing_the_adapter(
        self, tmp_path, probe, clean_registry
    ):
        from hermes_cli.plugins import PluginManager
        from toolsets import resolve_toolset

        manifest = _write_platform_plugin(tmp_path, "probeplat", with_tools_module=True)

        mgr = PluginManager()
        mgr._register_deferred_platform(manifest)

        assert resolve_toolset("probeplat") == ["probeplat_call"]
        # The expensive half stayed deferred — that's what makes this safe.
        assert probe.adapter_imports == 0
        assert probe.tools_execs == 1
        assert mgr._plugins["probeplat-platform"].deferred is True

    def test_plugin_without_tools_module_stays_fully_deferred(
        self, tmp_path, probe, clean_registry
    ):
        """No ``tools.py`` means no behaviour change at all — nothing imported."""
        from hermes_cli.plugins import PluginManager

        manifest = _write_platform_plugin(tmp_path, "barefoot", with_tools_module=False)

        mgr = PluginManager()
        mgr._register_deferred_platform(manifest)

        assert probe.package_execs == 0
        assert probe.adapter_imports == 0
        assert mgr._plugins["barefoot-platform"].tools_registered == []

    def test_tools_module_alone_does_not_opt_a_platform_in(
        self, tmp_path, probe, clean_registry
    ):
        """``provides_tools`` is the trigger, not the presence of a file.

        A platform is free to keep internal helpers in ``tools.py``; without
        the manifest declaring what it publishes, discovery must not import
        the package at all. Otherwise a plugin opts into an eager import by
        naming a file, and the contract is invisible to anyone reading the
        manifest.
        """
        from hermes_cli.plugins import PluginManager

        manifest = _write_platform_plugin(
            tmp_path,
            "quietplat",
            with_tools_module=True,
            declares_provides_tools=False,
        )

        mgr = PluginManager()
        mgr._register_deferred_platform(manifest)

        assert probe.package_execs == 0
        assert probe.tools_execs == 0
        assert probe.adapter_imports == 0
        assert mgr._plugins["quietplat-platform"].deferred is True
        assert mgr._plugins["quietplat-platform"].tools_registered == []

    def test_package_body_runs_once_across_discovery_and_materialization(
        self, tmp_path, probe, clean_registry
    ):
        """Pre-importing the package must not double-execute it later.

        Discovery imports ``<plugin>/__init__.py`` to reach ``tools.py``; when
        the gateway later materializes the adapter, ``_load_plugin`` reuses
        that module instead of re-running its body.
        """
        from gateway.platform_registry import platform_registry
        from hermes_cli.plugins import PluginManager

        manifest = _write_platform_plugin(tmp_path, "probeplat", with_tools_module=True)

        mgr = PluginManager()
        mgr._register_deferred_platform(manifest)
        assert probe.package_execs == 1

        # What gateway/web startup does.
        platform_registry.get("probeplat")

        assert probe.package_execs == 1
        assert probe.adapter_imports == 1

    def test_tools_stay_attributed_after_materialization(
        self, tmp_path, probe, clean_registry
    ):
        """`hermes plugins list` must still credit the pre-registered tools.

        ``_load_plugin`` attributes tools by diffing the registry around
        ``register()``. Tools registered at discovery are already in the
        "before" snapshot, so the diff alone would report zero.
        """
        from gateway.platform_registry import platform_registry
        from hermes_cli.plugins import PluginManager

        manifest = _write_platform_plugin(tmp_path, "probeplat", with_tools_module=True)

        mgr = PluginManager()
        mgr._register_deferred_platform(manifest)
        assert mgr._plugins["probeplat-platform"].tools_registered == ["probeplat_call"]

        platform_registry.get("probeplat")

        loaded = mgr._plugins["probeplat-platform"]
        assert loaded.tools_registered == ["probeplat_call"]
        assert loaded.enabled is True

    def test_broken_tools_module_does_not_break_discovery(
        self, tmp_path, probe, clean_registry, caplog
    ):
        """A plugin whose ``tools.py`` raises degrades to the old behaviour.

        Degrading quietly is not enough: the degraded state IS the #78050
        symptom (declared tools absent from the session), so it has to be
        visible without enabling debug logging to find it.
        """
        from hermes_cli.plugins import PluginManager

        manifest = _write_platform_plugin(tmp_path, "probeplat", with_tools_module=True)
        (Path(manifest.path) / "tools.py").write_text(
            "raise RuntimeError('boom')\n", encoding="utf-8"
        )

        mgr = PluginManager()
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            mgr._register_deferred_platform(manifest)  # must not raise

        assert mgr._plugins["probeplat-platform"].deferred is True
        assert mgr._plugins["probeplat-platform"].tools_registered == []
        assert any(
            "probeplat-platform" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        ), caplog.text

    def test_partially_registered_tools_are_still_attributed(
        self, tmp_path, probe, clean_registry, caplog
    ):
        """Tools registered before a mid-way failure are live — credit them.

        `register_tools` is not transactional: whatever it registered before
        raising stays in the registry. Leaving those unattributed makes
        `hermes plugins list` under-report what the process is carrying, and
        `_load_plugin`'s own diff cannot recover them later because they are
        already inside its "before" snapshot.
        """
        from hermes_cli.plugins import PluginManager

        manifest = _write_platform_plugin(tmp_path, "probeplat", with_tools_module=True)
        tools_py = (Path(manifest.path) / "tools.py").read_text(encoding="utf-8")
        (Path(manifest.path) / "tools.py").write_text(
            tools_py + "    raise RuntimeError('boom after the first tool')\n",
            encoding="utf-8",
        )

        mgr = PluginManager()
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            mgr._register_deferred_platform(manifest)  # must not raise

        # Attribution without a live tool would be a lie, so check the registry
        # itself rather than only the bookkeeping maps.
        from toolsets import resolve_toolset

        assert resolve_toolset("probeplat") == ["probeplat_call"]
        assert mgr._plugins["probeplat-platform"].tools_registered == ["probeplat_call"]
        assert mgr._predeclared_tools["probeplat-platform"] == ["probeplat_call"]
        assert mgr._plugins["probeplat-platform"].deferred is True
        assert any(r.levelno == logging.WARNING for r in caplog.records), caplog.text

    def test_failed_materialization_tears_down_pre_registered_tools(
        self, tmp_path, probe, clean_registry, caplog
    ):
        """A failed materialize takes the pre-registered tools down with it.

        The synthetic plugin's ``register()`` calls the same broken
        ``register_tools`` without catching, so materializing raises.

        ``_load_plugin_scoped``'s failure path sweeps the *whole* ownership
        ledger for this plugin key — not the ``registration_start:`` slice —
        and disposes it, so the discovery-time client tools go with the failed
        adapter. Attribution and the registry therefore agree at zero: `hermes
        plugins list` reports no tools because the process really is serving
        none.

        ``enabled`` stays False on purpose: the adapter genuinely did not load.
        """
        from gateway.platform_registry import platform_registry
        from hermes_cli.plugins import PluginManager
        from toolsets import resolve_toolset

        manifest = _write_platform_plugin(tmp_path, "probeplat", with_tools_module=True)
        tools_py = (Path(manifest.path) / "tools.py").read_text(encoding="utf-8")
        (Path(manifest.path) / "tools.py").write_text(
            tools_py + "    raise RuntimeError('boom after the first tool')\n",
            encoding="utf-8",
        )

        mgr = PluginManager()
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            mgr._register_deferred_platform(manifest)
            platform_registry.get("probeplat")  # gateway startup; register() raises

        loaded = mgr._plugins["probeplat-platform"]
        assert resolve_toolset("probeplat") == []
        assert loaded.tools_registered == []
        assert loaded.enabled is False
        assert loaded.error
        # The bookkeeping entry must not outlive the failed load attempt.
        assert "probeplat-platform" not in mgr._predeclared_tools

    def test_declared_tools_with_no_tools_module_warns(
        self, tmp_path, probe, clean_registry, caplog
    ):
        """A manifest promising tools it cannot deliver must say so.

        Returning silently here leaves the operator with exactly the bug this
        path fixes and no thread to pull on.
        """
        from hermes_cli.plugins import PluginManager

        manifest = _write_platform_plugin(
            tmp_path,
            "promiseplat",
            with_tools_module=False,
            declares_provides_tools=True,
        )

        mgr = PluginManager()
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            mgr._register_deferred_platform(manifest)

        assert probe.package_execs == 0
        assert mgr._plugins["promiseplat-platform"].tools_registered == []
        assert any(
            "promiseplat-platform" in r.message and "provides_tools" in r.message
            for r in caplog.records
        ), caplog.text
