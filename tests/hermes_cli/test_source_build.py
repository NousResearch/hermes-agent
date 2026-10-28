"""Source orchestration uses real node-deps/npm in an isolated checkout.

Only PM's tool acquisition (and its installed-node record) is substituted with the host's node/npm. Small
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
from types import SimpleNamespace

import pytest

import pm
from pm.package import Runner


def use_host_node_as_pm_node(monkeypatch):
    """Freshness reads run only under PM's recorded Node; stand the host's node in for it."""
    node, real = Path(shutil.which("node")), pm.installed_package
    monkeypatch.setattr(pm, "installed_package", lambda name, **kwargs: (
        SimpleNamespace(binary=node) if name == "node" else real(name, **kwargs)))


def copy_freshness_scripts(root):
    repository = Path(__file__).resolve().parents[2]
    scripts = root / "scripts/build"
    scripts.mkdir(parents=True, exist_ok=True)
    for name in ("freshness.mjs", "frontend-common.mjs"):
        shutil.copy2(repository / "scripts/build" / name, scripts / name)


def stamp_product(root, product, out):
    script = (root / "scripts/build/freshness.mjs").as_uri()
    subprocess.run([shutil.which("node"), "--input-type=module", "-e",
                    f"import {{recordProduct, buildInputs}} from {json.dumps(script)};"
                    "const [source, product, out] = process.argv.slice(1);"
                    "recordProduct({source, product, out, inputs: buildInputs(source, product)});",
                    str(root), product, str(out)], check=True)


def test_source_build_uses_selected_python_for_isolated_icon_child(tmp_path, monkeypatch):
    from hermes_cli.source_build import source_build_env
    from pm import paths
    from pm.environments import site_packages, venv_python

    root = tmp_path / "source"
    root.mkdir()
    venv = root / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    selected = site_packages(venv)
    (selected / "icon_dependency.py").write_text("ready = True\n", encoding="utf-8")
    monkeypatch.setattr(paths, "repo_root", lambda: root)
    monkeypatch.setattr(pm, "ensure", lambda name, **kwargs: Runner(name, kwargs["base_env"]))
    monkeypatch.syspath_prepend(str(selected))

    env = source_build_env()
    assert env["HERMES_PYTHON"] == str(venv_python(venv))
    subprocess.run([env["HERMES_PYTHON"], "-I", "-c",
                    "import icon_dependency; assert icon_dependency.ready"], check=True)


def test_automatic_build_preserves_pm_admission_intent(monkeypatch):
    from hermes_cli.source_build import source_build_env

    intent = []
    def acquire(name, *, base_env, explicit, verify=True):
        intent.append(explicit)
        return Runner(name, base_env)
    monkeypatch.setattr(pm, "ensure", acquire)
    source_build_env()
    assert intent == [False]


def test_installed_npm_does_not_authorize_missing_workspace_dependencies(source_checkout, monkeypatch):
    from hermes_cli.source_build import prepare_source_dependencies, source_build_env

    root, _ = source_checkout
    env = source_build_env()
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    with pytest.raises(subprocess.CalledProcessError):
        prepare_source_dependencies(root, ("ui-tui", "web"), env=env)
    assert _events(root) == []
    prepare_source_dependencies(root, ("ui-tui", "web"), env=env, explicit=True)
    events = _events(root)
    prepare_source_dependencies(root, ("ui-tui", "web"), env=env)
    assert _events(root) == events


@pytest.fixture
def source_checkout(tmp_path, monkeypatch):
    node, npm = shutil.which("node"), shutil.which("npm")
    assert node and npm, "source-build integration requires node and npm"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "0")
    monkeypatch.setenv("npm_config_cache", str(tmp_path / "npm-cache"))
    monkeypatch.setenv("ESBUILD_BINARY_PATH", "/wrong/esbuild")
    monkeypatch.setenv("HERMES_PYTHON", "/wrong/python")
    (home / "npmrc").write_text("fund=false\n", encoding="utf-8")
    monkeypatch.delenv("NPM_CONFIG_USERCONFIG", raising=False)
    acquired = []

    def acquire(name, *, base_env=None, explicit=False, verify=True):
        acquired.append(name)
        assert name == "npm"

        return Runner(name, {**(base_env or os.environ), "PATH": os.pathsep.join(
            [str(Path(npm).parent), str(Path(node).parent), os.environ["PATH"]])})

    monkeypatch.setattr(pm, "ensure", acquire)
    use_host_node_as_pm_node(monkeypatch)
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
            "scripts": {"build": "node ../../scripts/build/build-desktop.mjs",
                        "builder": "node ../../scripts/build/package-desktop.mjs"}
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
    for name in ("node-deps.mjs", "electron-runtime.mjs", "freshness.mjs", "frontend-common.mjs"):
        shutil.copy2(repository / "scripts/build" / name, scripts / name)
    (root / ".gitignore").write_text("node_modules/\n**/dist/\n", encoding="utf-8")
    return root, acquired


@pytest.fixture
def source_products(source_checkout):
    root, acquired = source_checkout
    (root / "product.mjs").write_text(
        "import { appendFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';\n"
        "import { dirname, join } from 'node:path';\n"
        "import { fileURLToPath } from 'node:url';\n"
        "import { recordProduct, buildInputs } from './scripts/build/freshness.mjs';\n"
        "const root = dirname(fileURLToPath(import.meta.url));\n"
        "export function build(step, output) {\n"
        "  appendFileSync(join(root, 'events.jsonl'), JSON.stringify({step}) + '\\n');\n"
        "  if (existsSync(join(root, 'fail-' + step))) throw new Error('fixture ' + step + ' failure');\n"
        "  if (step === 'web' && !existsSync(join(root, 'web/public/favicon.ico'))) throw new Error('icons missing');\n"
        "  const path = join(root, output); mkdirSync(dirname(path), { recursive: true });\n"
        "  writeFileSync(path, step);\n"
        "  if (step !== 'icons') {\n"
        "    let out = dirname(path);\n"
        "    if (step === 'desktop') { out = join(out, 'resources/app.asar.unpacked/dist');\n"
        "      mkdirSync(out, {recursive: true}); writeFileSync(join(out, 'index.html'), 'renderer'); }\n"
        "    recordProduct({source: root, product: step, out, inputs: buildInputs(root, step)});\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    for script, step, output in [
        ("generate-icons.mjs", "icons", "never-rendered-at-install"),
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
    (root / "scripts/build/build-desktop.mjs").write_text(
        "if (!process.argv.includes('--icons')) await import('../generate-icons.mjs');\n",
        encoding="utf-8",
    )
    # Default-brand icons are committed; updates consume them without rendering.
    (root / "web/public").mkdir(parents=True, exist_ok=True)
    (root / "web/public/favicon.ico").write_bytes(b"committed icon")
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
    assert first[0]["python"] == env["HERMES_PYTHON"]
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
    assert steps == ["deps", "tui", "web"] + (["desktop"] if desktop else [])
    assert acquired == ["npm"]
    assert (root / "ui-tui/dist/entry.js").read_text() == "tui"
    assert (root / "hermes_cli/web_dist/index.html").read_text() == "web"
    assert not _web_ui_build_needed(root / "web")
    assert (root / "node_modules/apps-desktop").exists() == desktop
    assert not (root / "node_modules/unrelated").exists()
    assert app.read_text() == ("desktop" if desktop else "previous app")
    assert not list((root / "apps/desktop").glob(".staging-*"))


@pytest.mark.platforms("linux")
def test_update_recompiles_only_products_whose_inputs_changed(source_products):
    from hermes_cli.source_build import build_update_products

    root, _ = source_products

    def products():
        return [event["step"] for event in _events(root) if event["step"] != "deps"]

    build_update_products(root, desktop=True)
    (root / "events.jsonl").unlink()
    build_update_products(root, desktop=True)
    assert products() == ["desktop"]

    (root / "events.jsonl").unlink()
    (root / "web/src").mkdir(parents=True, exist_ok=True)
    (root / "web/src/changed.ts").write_text("export {}\n", encoding="utf-8")
    build_update_products(root, desktop=False)
    assert products() == ["web"]


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("step", ["tui", "web", "desktop"])
def test_update_failure_raises_without_retries_or_replacing_live_app(source_products, step):
    from hermes_cli.source_build import ProductBuildError, build_update_products

    root, acquired = source_products
    app = root / "apps/desktop/release/linux-unpacked/hermes"
    app.parent.mkdir(parents=True)
    app.write_text("previous app")
    (root / f"fail-{step}").touch()
    with pytest.raises(ProductBuildError) as failure:
        build_update_products(root, desktop=True)
    # The failure is still raised, naming the one product that failed (no retries) ...
    assert len(failure.value.failures) == 1
    assert isinstance(failure.value.failures[0][1], subprocess.CalledProcessError)
    # ... but it no longer skips the independent products after it (was order[:index + 1]).
    assert [event["step"] for event in _events(root)] == ["deps", "tui", "web", "desktop"]
    assert acquired == ["npm"]
    assert not list((root / "apps/desktop").glob(".staging-*"))
    if step == "desktop":
        # A failed desktop build never replaces the live app nor stamps it.
        assert app.read_text() == "previous app"
        assert not (Path(os.environ["HERMES_HOME"]) / "desktop-build-stamp.json").exists()


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("failure", ["npm-unavailable", "desktop-after-long-feature-failure"])
def test_unbuilt_desktop_is_named_on_one_whole_line(source_products, monkeypatch, capsys, failure):
    # The Desktop hand-off keys on this line: the receipt follow-up is truncated and leads with
    # whichever product failed first, and an npm acquisition failure used to escape un-named.
    import hermes_cli.main_install_repair as install_repair
    from hermes_cli.source_build import ProductBuildError, build_update_products

    root, _ = source_products

    def fail(error):
        def raiser(*_args, **_kwargs):
            raise error
        return raiser

    if failure == "npm-unavailable":
        monkeypatch.setattr(pm, "ensure", fail(pm.InstallError("npm", "unavailable")))
        expected = "Desktop app build owed: Node dependencies failed"
    else:
        monkeypatch.setattr(install_repair, "_install_configured_features_missing_deps",
                            fail(RuntimeError("pip install failed: " + "x" * 600)))
        (root / "fail-desktop").touch()
        expected = "Desktop app build owed: desktop app build failed"
    with pytest.raises(ProductBuildError):
        build_update_products(root, desktop=True)
    lines = [line.strip() for line in capsys.readouterr().out.splitlines()]
    assert [line for line in lines if line.startswith("Desktop app build owed:")] == [expected]


@pytest.mark.platforms("linux")
def test_desktop_dependency_failure_still_builds_the_tui_and_web_ui(source_products, capsys):
    # A host whose compiler rejects a Desktop-only native addon (node-pty on g++ 9) used to lose
    # every frontend with it: the union npm ci failed and nothing was built.
    from hermes_cli.source_build import ProductBuildError, build_update_products

    root, _ = source_products
    manifest = root / "apps/desktop/package.json"
    manifest.write_text(json.dumps({**json.loads(manifest.read_text()), "scripts": {"postinstall": "exit 1"}}))
    with pytest.raises(ProductBuildError) as failure:
        build_update_products(root, desktop=True)
    assert [name for name, _ in failure.value.failures] == ["desktop dependencies"]
    assert [event["step"] for event in _events(root)] == ["deps", "tui", "web"]
    assert not (root / "node_modules/apps-desktop").exists()
    out = capsys.readouterr().out
    assert "Desktop app build owed: desktop dependencies failed" in out
    assert "hermes uninstall --gui" in out


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


@pytest.mark.platforms("posix")
def test_packaged_desktop_is_reused_only_while_it_names_head(tmp_path, monkeypatch):
    """The update skips the desktop build only when the shipped app's baked commit is HEAD
    and its receipt is current; a moved HEAD or an unreadable stamp means build."""
    from hermes_cli import main_desktop

    root = tmp_path / "checkout"
    root.mkdir()
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git[:3], "init", "-q"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "one"], check=True)
    desktop_dir = root / "apps/desktop"
    exe = desktop_dir / "release/app/Hermes"
    monkeypatch.setattr(main_desktop, "_desktop_packaged_executable", lambda _d: exe)
    resources = main_desktop._packaged_resources_dir(desktop_dir)
    (resources / "app.asar.unpacked/dist").mkdir(parents=True)
    receipt_current = {"value": True}
    monkeypatch.setattr(main_desktop, "_desktop_build_needed",
                        lambda *_a, **_k: not receipt_current["value"])

    def head():
        return subprocess.run([*git[:3], "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()

    def reused():
        return main_desktop._packaged_desktop_current_for_head(desktop_dir, root)

    assert not reused()  # no baked stamp
    (resources / "install-stamp.json").write_text(json.dumps({"commit": head()}), encoding="utf-8")
    assert reused()
    receipt_current["value"] = False
    assert not reused()
    receipt_current["value"] = True
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "two"], check=True)
    assert not reused()


@pytest.mark.parametrize("enabled", [False, True])
def test_update_rebuilds_the_webapp_bundle_a_home_serves(tmp_path, monkeypatch, enabled):
    """``hermes update`` rebuilds ``apps/desktop/dist-webapp``, with the function ``hermes webapp``
    uses, when any home sharing this install has ``webapp.enabled``: that home's gateway serves the
    bundle as built, so an old one would talk to the updated backend. Otherwise it is left alone."""
    import hermes_cli.left_core_migration as left_core
    import hermes_cli.main_install_repair as install_repair
    import hermes_cli.memory_provider_migration as memory_migration
    from hermes_cli import update_stage, webapp
    from hermes_cli.source_build import build_update_products
    from pm import plugins_state

    root = tmp_path / "checkout"
    (root / "apps/desktop/scripts").mkdir(parents=True)
    (root / "apps/desktop/scripts/build-webapp.mjs").write_text("", encoding="utf-8")
    homes = [tmp_path / "default", tmp_path / "profiles" / "work"]
    for home in homes:
        home.mkdir(parents=True)
    (homes[1] / "config.yaml").write_text(f"webapp:\n  enabled: {str(enabled).lower()}\n", encoding="utf-8")
    builds = []
    monkeypatch.setattr(install_repair, "_install_configured_features_missing_deps", lambda _root: None)
    monkeypatch.setattr(memory_migration, "migrate_all_homes", list)
    monkeypatch.setattr(left_core, "migrate_all_homes", list)
    monkeypatch.setattr(update_stage, "publish_stage", lambda _stage: None)
    monkeypatch.setattr(plugins_state, "dependency_homes", lambda: homes)
    monkeypatch.setattr(webapp, "prepare_webapp_renderer",
                        lambda project_root, **options: builds.append((project_root, options)))

    build_update_products(root, desktop=False)

    assert builds == ([(root, {"explicit": True})] if enabled else [])


@pytest.mark.parametrize("built", [True, False])
def test_a_failed_webapp_rebuild_in_update_marks_the_served_bundle_out_of_date(tmp_path, monkeypatch, capsys, built):
    """When ``hermes update`` cannot rebuild the Webapp bundle a home serves, the previous bundle
    stays on disk but is marked out of date: its gateway reports the Webapp unavailable and says to
    run ``hermes webapp`` (which rebuilds it) instead of serving it against the updated backend. The
    update still reports the failure. With no bundle built there is nothing to mark."""
    import hermes_cli.left_core_migration as left_core
    import hermes_cli.main_install_repair as install_repair
    import hermes_cli.memory_provider_migration as memory_migration
    from gateway.webapp_listener import NOT_BUILT, bundle_missing_reason
    from hermes_cli import update_stage, webapp
    from hermes_cli.source_build import ProductBuildError, build_update_products
    from pm import plugins_state

    root = tmp_path / "checkout"
    (root / "apps/desktop/scripts").mkdir(parents=True)
    (root / "apps/desktop/scripts/build-webapp.mjs").write_text("", encoding="utf-8")
    dist = webapp.webapp_dist_dir(root)
    dist.mkdir()
    if built:
        (dist / "index.html").write_text("<!doctype html>", encoding="utf-8")
    home = tmp_path / "default"
    home.mkdir()
    (home / "config.yaml").write_text("webapp:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setattr(install_repair, "_install_configured_features_missing_deps", lambda _root: None)
    monkeypatch.setattr(memory_migration, "migrate_all_homes", list)
    monkeypatch.setattr(left_core, "migrate_all_homes", list)
    monkeypatch.setattr(update_stage, "publish_stage", lambda _stage: None)
    monkeypatch.setattr(plugins_state, "dependency_homes", lambda: [home])

    def fail(_project_root, **_options):
        raise webapp.WebappBuildError("Browser-hosted Desktop build failed: npm ci exited 1")

    monkeypatch.setattr(webapp, "prepare_webapp_renderer", fail)
    before = sorted(root.rglob("*"))

    with pytest.raises(ProductBuildError, match="Webapp build"):
        build_update_products(root, desktop=False)

    reason = bundle_missing_reason(dist)
    assert reason is not None and reason.startswith(NOT_BUILT) and "hermes webapp" in reason
    if built:
        assert (dist / "index.html").read_text(encoding="utf-8") == "<!doctype html>"
        assert "hermes webapp" in capsys.readouterr().out
    else:
        assert sorted(root.rglob("*")) == before
