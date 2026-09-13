"""TUI launch consumes the shared dependency preparation and compiler."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from hermes_cli import main_tui_launch
from tests.hermes_cli.test_source_build import source_checkout, source_products, _events  # noqa: F401


def _touch_tui_entry(root: Path) -> None:
    entry = root / "dist" / "entry.js"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("console.log('tui')")


def test_need_rebuild_when_tui_bundle_missing(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "entry.tsx").write_text("console.log('src')")

    assert main_tui_launch._tui_need_rebuild(tmp_path) is True


def test_unreceipted_bundle_is_stale_even_when_newer(tmp_path: Path) -> None:
    _touch_tui_entry(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    (src / "entry.tsx").write_text("console.log('src')")
    os.utime(src / "entry.tsx", (100, 100))
    os.utime(tmp_path / "dist" / "entry.js", (200, 200))

    assert main_tui_launch._tui_need_rebuild(tmp_path) is True



@pytest.fixture
def tui_source(source_products, monkeypatch):
    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.delenv("HERMES_TUI_FORCE_BUILD", raising=False)
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setenv("PREFIX", "/usr")
    monkeypatch.setenv("HERMES_NODE", shutil.which("node"))
    monkeypatch.setattr(main_tui_launch, "_find_bundled_tui", lambda: None)
    return source_products


@pytest.mark.platforms("posix")
def test_source_launch_prepares_base_union_but_compiles_only_tui(tui_source):
    root, acquired = tui_source
    argv, cwd = main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=False)
    assert cwd == root / "ui-tui"
    assert argv[-1] == str(cwd / "dist/entry.js")
    assert (cwd / "dist/entry.js").read_text() == "tui"
    assert [event["step"] for event in _events(root)] == ["deps", "tui"]
    assert acquired == ["npm"]
    assert (root / "node_modules/ui-tui").exists()
    assert (root / "node_modules/web").exists()
    assert not (root / "node_modules/apps-desktop").exists()


@pytest.mark.platforms("posix")
def test_source_compile_failure_stops_launch_without_reinstall(tui_source):
    root, acquired = tui_source
    (root / "fail-tui").touch()
    with pytest.raises(subprocess.CalledProcessError):
        main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=False)
    assert [event["step"] for event in _events(root)] == ["deps", "tui"]
    assert acquired == ["npm"]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("termux", [False, True])
def test_fresh_bundle_does_not_prepare_or_compile(tui_source, monkeypatch, termux):
    root, acquired = tui_source
    _touch_tui_entry(root / "ui-tui")
    from tests.hermes_cli.test_source_build import stamp_product
    stamp_product(root, "tui", root / "ui-tui/dist")
    if termux:
        monkeypatch.setenv("TERMUX_VERSION", "test")
    argv, cwd = main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=False)
    assert argv[-1] == str(cwd / "dist/entry.js")
    assert _events(root) == []
    assert acquired == []


@pytest.mark.platforms("posix")
def test_prebuilt_bundle_launch_does_not_touch_missing_source(tmp_path, monkeypatch):
    bundled = tmp_path / "bundle/entry.js"
    bundled.parent.mkdir()
    bundled.write_text("console.log('prebuilt')")
    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setenv("HERMES_NODE", shutil.which("node"))
    monkeypatch.setattr(main_tui_launch, "_find_bundled_tui", lambda: bundled)
    argv, cwd = main_tui_launch._make_tui_argv(tmp_path / "missing-source", tui_dev=False)
    result = subprocess.run(argv, cwd=cwd, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "prebuilt"
    assert not (tmp_path / "missing-source").exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("local_tsx", [False, True])
def test_dev_launch_builds_ink_before_running_source(tui_source, local_tsx):
    root, acquired = tui_source
    ink = root / "ui-tui/packages/hermes-ink"
    ink.mkdir(parents=True)
    (ink / "package.json").write_text(json.dumps({
        "name": "fixture-ink", "version": "1.0.0", "scripts": {"build": "node build.mjs"},
    }))
    (ink / "build.mjs").write_text("import { writeFileSync } from 'node:fs'; writeFileSync('built', 'ink');")
    manifest = json.loads((root / "package.json").read_text())
    manifest["workspaces"].append("ui-tui/packages/hermes-ink")
    (root / "package.json").write_text(json.dumps(manifest))
    subprocess.run([shutil.which("npm"), "install", "--package-lock-only", "--ignore-scripts", "--offline"], cwd=root, check=True)
    tsx = root / "ui-tui/node_modules/.bin/tsx"
    if local_tsx:
        hook = root / "log.mjs"
        hook.write_text(hook.read_text() + "\nimport { mkdirSync, writeFileSync } from 'node:fs'; "
                        "mkdirSync('ui-tui/node_modules/.bin', {recursive: true}); "
                        "writeFileSync('ui-tui/node_modules/.bin/tsx', 'fixture tsx');\n")
    argv, cwd = main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=True)
    assert (ink / "built").read_text() == "ink"
    assert argv == ([str(tsx), "src/entry.tsx"] if local_tsx else [shutil.which("npm"), "start"])
    assert cwd == root / "ui-tui"
    assert acquired == ["npm"]
    assert [event["step"] for event in _events(root)] == ["deps"]


@pytest.mark.platforms("posix")
def test_runtime_node_comes_from_pm_without_legacy_repair(tmp_path, monkeypatch):
    import pm
    from pm.package import Runner

    node = shutil.which("node")
    managed = tmp_path / "bin/node"
    managed.parent.mkdir()
    managed.symlink_to(node)
    monkeypatch.delenv("HERMES_NODE", raising=False)
    acquired = []

    def acquire(name):
        acquired.append(name)
        return Runner(name, {"PATH": str(managed.parent)})

    monkeypatch.setattr(pm, "ensure", acquire)
    assert main_tui_launch._tui_node_bin("node") == str(managed)
    assert acquired == ["node"]


@pytest.mark.platforms("linux")
def test_tui_rebuild_preserves_the_prepared_desktop_and_web_union(tui_source):
    from hermes_cli.source_build import build_update_products

    root, acquired = tui_source
    build_update_products(root, desktop=True)
    before = _events(root)
    main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=False)
    assert _events(root) == before, "fresh launch must not prepare or rebuild"
    src = root / "ui-tui/src/entry.tsx"
    src.parent.mkdir()
    src.write_text("changed tui source")
    os.utime(root / "ui-tui/dist/entry.js", (1, 1))
    main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=False)
    assert _events(root) == [*before, {"step": "tui"}]
    assert acquired == ["npm", "npm"]
    assert (root / "node_modules/web").exists()
    assert (root / "node_modules/apps-desktop").exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("changed", ["ui-tui/src/entry.tsx", "package.json", "package-lock.json", "apps/shared/shared.ts"])
def test_tui_rebuild_tracks_root_and_shared_inputs(tui_source, changed):
    root, _ = tui_source
    tui = root / "ui-tui"
    changed_file = root / changed
    changed_file.parent.mkdir(parents=True, exist_ok=True)
    if not changed_file.exists():
        changed_file.write_text("original input\n", encoding="utf-8")
    main_tui_launch._make_tui_argv(tui, tui_dev=False)
    assert not main_tui_launch._tui_need_rebuild(tui), "must have a current receipt before invalidation"
    before = changed_file.stat()
    changed_file.write_text(changed_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    # A content change must invalidate even when mtimes are restored.
    os.utime(changed_file, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert main_tui_launch._tui_need_rebuild(tui)
