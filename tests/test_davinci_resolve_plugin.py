from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path

import yaml


PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugins" / "video" / "davinci_resolve"


def _load_plugin():
    sys.modules.pop("davinci_resolve_plugin", None)
    spec = importlib.util.spec_from_file_location(
        "davinci_resolve_plugin",
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_tools():
    plugin = _load_plugin()
    return sys.modules[f"{plugin.__name__}.tools"]


class _Ctx:
    def __init__(self) -> None:
        self.tools = []

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)


def test_davinci_resolve_plugin_registers_expected_tools():
    plugin = _load_plugin()
    ctx = _Ctx()

    plugin.register(ctx)

    names = [tool["name"] for tool in ctx.tools]
    assert names == [
        "resolve_capabilities",
        "resolve_launch",
        "resolve_probe",
        "resolve_project_summary",
        "resolve_import_media",
        "resolve_create_timeline",
        "resolve_append_to_current_timeline",
        "resolve_add_timeline_marker",
        "resolve_scan_media_folder",
        "resolve_create_scripted_timeline",
        "resolve_render_timeline",
        "resolve_render_status",
        "resolve_generate_fcpxml_timeline",
        "resolve_generate_marker_csv",
    ]
    assert {tool["toolset"] for tool in ctx.tools} == {"davinciresolve"}
    assert all(tool["check_fn"]() is (platform.system() == "Darwin") for tool in ctx.tools)


def test_davinci_resolve_dry_run_handlers_return_json(tmp_path):
    tools = _load_tools()
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"not real media, but enough for path validation")

    timeline = json.loads(
        tools.handle_create_scripted_timeline(
            {
                "name": "Hermes Dry Run",
                "clips": [{"path": str(media), "start_frame": 0, "end_frame": 24}],
                "dry_run": True,
            }
        )
    )
    assert timeline["ok"] is True
    assert timeline["dry_run"] is True
    assert timeline["plan"]["existing_paths"] == [str(media.resolve())]

    render = json.loads(
        tools.handle_render_timeline(
            {
                "target_dir": str(tmp_path),
                "custom_name": "hermes-render",
                "dry_run": True,
            }
        )
    )
    assert render["ok"] is True
    assert render["plan"]["render_format"] == "mov"
    assert render["plan"]["render_codec"] == "H264"


def test_davinci_resolve_interchange_generators_default_to_dry_run(tmp_path):
    tools = _load_tools()
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"not real media, but enough for path validation")
    fcpxml_output = tmp_path / "timeline.fcpxml"
    marker_output = tmp_path / "markers.csv"

    fcpxml = json.loads(
        tools.handle_generate_fcpxml_timeline(
            {
                "name": "Hermes FCPXML",
                "media_paths": [str(media)],
                "output_path": str(fcpxml_output),
            }
        )
    )
    markers = json.loads(
        tools.handle_generate_marker_csv(
            {
                "markers": [{"frame": 0, "name": "Start"}],
                "output_path": str(marker_output),
            }
        )
    )

    assert fcpxml["ok"] is True
    assert fcpxml["dry_run"] is True
    assert markers["ok"] is True
    assert markers["dry_run"] is True
    assert not fcpxml_output.exists()
    assert not marker_output.exists()


def test_davinci_resolve_scan_media_folder(tmp_path):
    tools = _load_tools()
    (tmp_path / "a.mov").write_bytes(b"x")
    (tmp_path / "ignore.txt").write_text("ignore", encoding="utf-8")

    result = json.loads(
        tools.handle_scan_media_folder(
            {
                "folder_path": str(tmp_path),
                "recursive": False,
            }
        )
    )

    assert result["ok"] is True
    assert result["returned_count"] == 1
    assert result["counts"]["video"] == 1
    assert result["files"][0]["name"] == "a.mov"


def test_davinci_resolve_plugin_manager_loads_with_registry_gating(tmp_path, monkeypatch):
    from hermes_cli.plugins import PluginManager
    from tools.registry import registry

    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["video/davinci_resolve"]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    manager = PluginManager()
    manager.discover_and_load()

    loaded = manager._plugins["video/davinci_resolve"]
    assert loaded.enabled is True
    assert loaded.error is None
    assert "resolve_probe" in loaded.tools_registered

    entry = registry.get_entry("resolve_probe")
    assert entry is not None
    assert entry.toolset == "davinciresolve"
    assert entry.check_fn is not None
    assert entry.check_fn() is (platform.system() == "Darwin")


def test_davinci_resolve_loads_through_plugin_manager():
    """Load the plugin through the real PluginManager directory loader.

    Covers the loader-level contract from review: the plugin imports as a
    package under the ``hermes_plugins`` namespace (package-relative imports
    resolve without any ``sys.path`` mutation), registration succeeds through
    the real ``PluginContext``, and the plugin's ``tools.py`` submodule does
    not shadow Hermes' top-level ``tools`` package.
    """
    import tools as hermes_tools
    from hermes_cli.plugins import PluginManager, PluginManifest

    for name in [
        m for m in list(sys.modules) if m.startswith("hermes_plugins.video__davinci_resolve")
    ]:
        sys.modules.pop(name, None)

    manager = PluginManager()
    manifest = PluginManifest(
        name="davinci-resolve",
        version="1.0.0",
        description="DaVinci Resolve integration",
        author="",
        requires_env=[],
        provides_tools=[],
        provides_hooks=[],
        source="project",
        path=str(PLUGIN_DIR),
        kind="standalone",
        key="video/davinci_resolve",
    )
    manager._load_plugin(manifest)

    loaded = manager._plugins["video/davinci_resolve"]
    assert loaded.error is None, f"plugin failed to load: {loaded.error}"
    assert loaded.enabled is True
    assert "resolve_capabilities" in loaded.tools_registered
    assert "resolve_import_media" in loaded.tools_registered

    # Imported as a namespaced package, not a bare top-level module.
    assert loaded.module.__name__ == "hermes_plugins.video__davinci_resolve"
    assert "hermes_plugins.video__davinci_resolve.tools" in sys.modules

    # Hermes' top-level ``tools`` package is untouched by the plugin's tools.py.
    import tools as tools_after

    assert tools_after is hermes_tools
    assert Path(tools_after.__file__).resolve() != (PLUGIN_DIR / "tools.py").resolve()
