"""Source orchestration uses real node-deps/npm in an isolated checkout.

Only PM's tool acquisition is substituted with the host's node/npm. Small
workspace scripts stand in for the expensive UI compilers; subprocess failures,
locked dependency selection, environment propagation and publication are real.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import pm
from pm.package import Runner


@pytest.fixture
def source_checkout(tmp_path, monkeypatch):
    node, npm = shutil.which("node"), shutil.which("npm")
    assert node and npm, "source-build integration requires node and npm"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.setenv("npm_config_cache", str(tmp_path / "npm-cache"))
    monkeypatch.setenv("ESBUILD_BINARY_PATH", "/wrong/esbuild")
    monkeypatch.setenv("HERMES_PYTHON", "/wrong/python")
    (home / "npmrc").write_text("fund=false\n", encoding="utf-8")
    monkeypatch.delenv("NPM_CONFIG_USERCONFIG", raising=False)
    acquired = []

    def acquire(name, *, base_env=None, explicit=False):
        acquired.append(name)
        assert name == "npm"
        assert explicit
        return Runner(name, {**(base_env or os.environ), "PATH": os.pathsep.join(
            [str(Path(node).parent), str(Path(npm).parent), os.environ["PATH"]])})

    monkeypatch.setattr(pm, "ensure", acquire)
    root = tmp_path / "source with spaces"
    root.mkdir()
    workspaces = ["ui-tui", "web", "apps/desktop", "unrelated"]
    manifest = {"name": "build-fixture", "private": True, "version": "1.0.0",
                "workspaces": workspaces, "scripts": {"postinstall": "node log.mjs deps"}}
    (root / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
    for workspace in workspaces:
        directory = root / workspace
        directory.mkdir(parents=True)
        (directory / "package.json").write_text(json.dumps({
            "name": workspace.replace("/", "-"), "version": "1.0.0",
            "scripts": {"pack": "node ../../scripts/build/package-desktop.mjs"}
            if workspace == "apps/desktop" else {},
        }), encoding="utf-8")
    (root / "log.mjs").write_text(
        "import { appendFileSync } from 'node:fs';\n"
        "appendFileSync('events.jsonl', JSON.stringify({step: process.argv[2], "
        "python: process.env.HERMES_PYTHON, ci: process.env.CI, "
        "esbuild: process.env.ESBUILD_BINARY_PATH, "
        "npmrc: process.env.NPM_CONFIG_USERCONFIG}) + '\\n');\n",
        encoding="utf-8",
    )
    subprocess.run([npm, "install", "--package-lock-only", "--ignore-scripts", "--offline",
                    "--no-audit", "--no-fund"], cwd=root, check=True)
    scripts = root / "scripts" / "build"
    scripts.mkdir(parents=True)
    repository = Path(__file__).resolve().parents[2]
    shutil.copy2(repository / "scripts/build/node-deps.mjs", scripts / "node-deps.mjs")
    (root / ".gitignore").write_text("node_modules/\n**/dist/\n", encoding="utf-8")
    return root, acquired


@pytest.fixture
def source_products(source_checkout):
    root, acquired = source_checkout
    (root / "product.mjs").write_text(
        "import { appendFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';\n"
        "import { dirname, join } from 'node:path';\n"
        "import { fileURLToPath } from 'node:url';\n"
        "const root = dirname(fileURLToPath(import.meta.url));\n"
        "export function build(step, output) {\n"
        "  appendFileSync(join(root, 'events.jsonl'), JSON.stringify({step}) + '\\n');\n"
        "  if (existsSync(join(root, 'fail-' + step))) throw new Error('fixture ' + step + ' failure');\n"
        "  if (step === 'web' && !existsSync(join(root, 'web/public/favicon.ico'))) throw new Error('icons missing');\n"
        "  const path = join(root, output); mkdirSync(dirname(path), { recursive: true });\n"
        "  writeFileSync(path, step);\n"
        "}\n",
        encoding="utf-8",
    )
    for script, step, output in [
        ("generate-icons.mjs", "icons", "web/public/favicon.ico"),
        ("build/tui.mjs", "tui", "ui-tui/dist/entry.js"),
        ("build/web.mjs", "web", "hermes_cli/web_dist/index.html"),
    ]:
        relative = "../../" if script.startswith("build/") else "../"
        (root / "scripts" / script).write_text(
            f"import {{ build }} from '{relative}product.mjs'; build({step!r}, {output!r});\n",
            encoding="utf-8",
        )
    (root / "scripts/build/package-desktop.mjs").write_text(
        "import { relative } from 'node:path';\n"
        "import { build } from '../../product.mjs';\n"
        "const flag = '-c.directories.output=';\n"
        "const staging = process.argv.find(arg => arg.startsWith(flag)).slice(flag.length);\n"
        "build('desktop', relative('../..', staging) + '/linux-unpacked/hermes');\n",
        encoding="utf-8",
    )
    return root, acquired


def _events(root):
    path = root / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.platforms("posix")
def test_preparation_reuses_only_the_exact_completed_workspace_union(source_checkout):
    from hermes_cli.source_build import prepare_source_dependencies, source_build_env

    root, acquired = source_checkout
    before = (root / "package-lock.json").read_bytes()
    env = source_build_env()
    prepare_source_dependencies(root, ("ui-tui", "web"), env=env)
    first = _events(root)
    assert [event["step"] for event in first] == ["deps"]
    assert first[0]["python"] == sys.executable
    assert first[0]["ci"] == "1"
    assert "esbuild" not in first[0]
    assert first[0]["npmrc"] == str(Path(os.environ["HERMES_HOME"]) / "npmrc")
    assert (root / "node_modules/ui-tui").exists()
    assert (root / "node_modules/web").exists()
    assert not (root / "node_modules/apps-desktop").exists()
    assert not (root / "node_modules/unrelated").exists()

    prepare_source_dependencies(root, ("ui-tui", "web"), env=env)
    assert _events(root) == first
    prepare_source_dependencies(root, ("ui-tui", "web", "apps/desktop"), env=env)
    assert len(_events(root)) == 2
    assert (root / "node_modules/apps-desktop").exists()
    assert not (root / "node_modules/unrelated").exists()
    assert (root / "package-lock.json").read_bytes() == before
    assert acquired == ["npm"]

    # A lock failure must not fall back to npm install and rewrite the lock.
    (root / "package-lock.json").write_text("not json", encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError):
        prepare_source_dependencies(root, ("ui-tui", "web"), env=env)
    assert (root / "package-lock.json").read_text() == "not json"
    assert len(_events(root)) == 2
    assert acquired == ["npm"]

    # Exercise PM rather than hiding a provisioning error behind system npm.
    from unittest.mock import patch
    with patch.object(pm, "ensure", side_effect=pm.InstallError("npm", "unavailable")):
        with pytest.raises(pm.InstallError, match="unavailable"):
            source_build_env()
    assert len(_events(root)) == 2


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("desktop", [False, True])
def test_update_builds_selected_products_after_one_union_preparation(source_products, desktop):
    from hermes_cli.source_build import build_update_products
    from hermes_cli.main_web_build import _web_ui_build_needed

    root, acquired = source_products
    app = root / "apps/desktop/release/linux-unpacked/hermes"
    app.parent.mkdir(parents=True)
    app.write_text("previous app")
    build_update_products(root, desktop=desktop)
    steps = [event["step"] for event in _events(root)]
    assert steps == ["deps", "tui", "icons", "web"] + (["desktop"] if desktop else [])
    assert acquired == ["npm"]
    assert (root / "ui-tui/dist/entry.js").read_text() == "tui"
    assert (root / "hermes_cli/web_dist/index.html").read_text() == "web"
    assert not _web_ui_build_needed(root / "web")
    assert (root / "node_modules/apps-desktop").exists() == desktop
    assert not (root / "node_modules/unrelated").exists()
    assert app.read_text() == ("desktop" if desktop else "previous app")
    assert not list((root / "apps/desktop").glob(".staging-*"))


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("step", ["tui", "icons", "web", "desktop"])
def test_update_failure_raises_without_retries_or_replacing_live_app(source_products, step):
    from hermes_cli.source_build import build_update_products

    root, acquired = source_products
    app = root / "apps/desktop/release/linux-unpacked/hermes"
    app.parent.mkdir(parents=True)
    app.write_text("previous app")
    (root / f"fail-{step}").touch()
    with pytest.raises(subprocess.CalledProcessError):
        build_update_products(root, desktop=True)
    assert app.read_text() == "previous app"
    assert not list((root / "apps/desktop").glob(".staging-*"))
    order = ["deps", "tui", "icons", "web", "desktop"]
    assert [event["step"] for event in _events(root)] == order[:order.index(step) + 1]
    assert acquired == ["npm"]
    assert not (Path(os.environ["HERMES_HOME"]) / "desktop-build-stamp.json").exists()


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("desktop", [False, True])
def test_module_cli_builds_the_requested_products(source_products, desktop, monkeypatch):
    import runpy

    root, acquired = source_products
    monkeypatch.setattr(sys, "argv", ["source_build", "--source", str(root)] + (["--desktop"] if desktop else []))
    # run_module exercises __main__ while substituting only tool acquisition.
    monkeypatch.delitem(sys.modules, "hermes_cli.source_build", raising=False)
    runpy.run_module("hermes_cli.source_build", run_name="__main__")
    assert acquired == ["npm"]
    assert (root / "hermes_cli/web_dist/index.html").is_file()
    assert (root / "apps/desktop/release/linux-unpacked/hermes").exists() == desktop
