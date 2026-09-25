"""Configured-feature warnings must describe the selected update child (#10651).

PM's atomic dependency union cannot validate a configured but unselected SDK.
These tests use real PM publication and a fresh interpreter; only expensive
frontend compilation is replaced. SDK anchors exist only in the target tree.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import sysconfig
import subprocess

import pytest

import pm
from pm.environments import activation_environment, runtime_facts_path, selected_venv, site_packages
from pm.environments import venv_python
from tests.pm._fixtures import isolated_python  # noqa: F401
from tests.pm.test_source_update_launch import source_launch  # noqa: F401


@pytest.fixture
def configured_update(source_launch, tmp_path, monkeypatch):
    root, _, _ = source_launch
    repository = Path(__file__).resolve().parents[2]
    pm.sync_venv(["launch-extra"], explicit=True, project_root=root)
    selected = selected_venv(root)
    site = site_packages(selected)
    # Reuse installed core libraries without exposing the updater's site tree:
    # neither SDK, executable .pth hooks, nor editable package finders cross over.
    for path in Path(sysconfig.get_path("purelib")).iterdir():
        if (path.name in {"lark_oapi", "mcp", "__pycache__"}
                or path.name.startswith(("lark_oapi-", "mcp-", "__editable__"))
                or path.suffix == ".pth" or (site / path.name).exists()):
            continue
        (site / path.name).symlink_to(path, target_is_directory=path.is_dir())
    # Application source is unchanged; the toy manifest keeps PM resolution
    # offline while exercising the real selected-environment launch boundary.
    for path in repository.iterdir():
        if path.name in {"hermes_cli", "gateway", "agent", "tools", "plugins", "pm"} or path.suffix == ".py":
            if (root / path.name).exists():
                # source_launch already stubs hermes_cli/source_completion.py; link the rest of that
                # package's modules beside it.
                for module in path.iterdir():
                    if not (root / path.name / module.name).exists():
                        (root / path.name / module.name).symlink_to(module, target_is_directory=module.is_dir())
                continue
            (root / path.name).symlink_to(path, target_is_directory=path.is_dir())
    def build_in_child():
        script = (
            "import json, sys; from pathlib import Path\n"
            "from hermes_cli import source_build\n"
            "source_build.source_build_env = lambda **kwargs: {}\n"
            "source_build.prepare_source_dependencies = lambda *args, **kwargs: None\n"
            "source_build.build_source_tui = lambda *args, **kwargs: None\n"
            "source_build.build_source_web = lambda *args, **kwargs: None\n"
            f"source_build.build_update_products(Path({str(root)!r}), desktop=False)\n"
            "print('TARGET=' + json.dumps({'prefix': sys.prefix, 'python': sys.executable}))\n"
        )
        subprocess.run([str(venv_python(selected)), "-c", script],
                       cwd=root, env=activation_environment(root), check=True)
    home = tmp_path / "home"
    config = home / "config.yaml"
    config.write_text(
        "platforms:\n  feishu:\n    enabled: true\n    extra:\n      app_id: cli_x\n      app_secret: y\n"
        "mcp_servers:\n  fixture:\n    command: must-not-run\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    # A stale positive verdict in the updater must not conceal missing child SDKs.
    from types import ModuleType
    for name in ("lark_oapi", "mcp"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    return root, selected, site, config, build_in_child


@pytest.mark.platforms("posix")
def test_update_names_missing_configured_features_from_selected_child(configured_update, capfd):
    root, selected, site, _, build = configured_update
    facts = runtime_facts_path(root).read_bytes()
    build()
    out = capfd.readouterr().out
    assert "fail to load them on restart" in out
    assert "Feishu / Lark" in out and "MCP servers" in out
    assert "hermes setup" in out and "hermes pm" in out
    target = json.loads(next(line.removeprefix("TARGET=") for line in out.splitlines() if line.startswith("TARGET=")))
    assert Path(target["prefix"]) == selected
    assert Path(target["python"]).parent.parent == selected
    assert runtime_facts_path(root).read_bytes() == facts
    assert not (root / ".update-incomplete").exists()

    # Only the target gains the anchors; no updater module/cache is repaired.
    for name in ("lark_oapi", "mcp"):
        (site / f"{name}.py").write_text("# Passive dependency-probe fixture.\n", encoding="utf-8")
    build()
    assert "fail to load them on restart" not in capfd.readouterr().out
    assert runtime_facts_path(root).read_bytes() == facts


@pytest.mark.platforms("posix")
def test_unconfigured_or_disabled_features_are_quiet(configured_update, capfd):
    root, _, _, config, build = configured_update
    for body in (
        "platforms:\n  feishu:\n    enabled: false\n    extra:\n      app_id: cli_x\n      app_secret: y\n",
        "platforms:\n  feishu:\n    enabled: true\n",
    ):
        config.write_text(body, encoding="utf-8")
        build()
        out = capfd.readouterr().out
        assert "TARGET=" in out
        assert "fail to load them on restart" not in out
        assert "MCP servers" not in out
        assert "Feishu / Lark:" not in out
        assert not (root / ".update-incomplete").exists()
