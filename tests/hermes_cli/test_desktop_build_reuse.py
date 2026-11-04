"""``hermes update`` reuses the packaged desktop app across commits that changed no desktop input.

A real git checkout carries the real receipt reader (``scripts/build/freshness.mjs``), the real
stamp resolver (``apps/desktop/scripts/write-build-stamp.mjs``) and the real reuse probe; only the
expensive Electron build itself is replaced by a recorder (the live A/B in the PR runs it for real).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import pm

REPOSITORY = Path(__file__).resolve().parents[2]
COPIED = (
    "scripts/build/freshness.mjs", "scripts/build/frontend-common.mjs", "scripts/build/desktop-reuse.mjs",
    "apps/desktop/scripts/write-build-stamp.mjs", "apps/desktop/scripts/bundle-env.mjs",
    "apps/desktop/scripts/utils.mjs", "apps/desktop/product-identity.cjs",
    "scripts/msix-shared.mjs", "scripts/release-content-types.json",
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def _node(*args: str, cwd: Path) -> None:
    subprocess.run([shutil.which("node"), "--input-type=module", "-e", *args], cwd=cwd, check=True)


@pytest.fixture
def packaged_checkout(tmp_path, monkeypatch):
    """A checkout whose packaged app was built (stamp baked, receipt recorded) at its first commit."""
    node = shutil.which("node")
    assert node, "the desktop reuse probe runs under Node"
    real = pm.installed_package
    monkeypatch.setattr(pm, "installed_package", lambda name, **kwargs: (
        SimpleNamespace(binary=Path(node)) if name == "node" else real(name, **kwargs)))
    monkeypatch.setattr(pm, "env_for", lambda _name, **_kwargs: dict(os.environ))
    root = tmp_path / "checkout"
    for name in COPIED:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY / name, root / name)
    (root / "apps/desktop/src").mkdir(parents=True)
    (root / "apps/desktop/src/main.tsx").write_text("export {}\n", encoding="utf-8")
    (root / "hermes_cli").mkdir()
    (root / "hermes_cli/banner.py").write_text("BANNER = 1\n", encoding="utf-8")
    (root / ".gitignore").write_text("apps/desktop/build/\napps/desktop/release/\n", encoding="utf-8")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "one")

    desktop = root / "apps/desktop"
    stamp = desktop / "build/install-stamp.json"
    dist = desktop / "release/linux-unpacked/resources/app.asar.unpacked/dist"
    dist.mkdir(parents=True)
    (desktop / "release/linux-unpacked/hermes").write_text("", encoding="utf-8")
    (dist / "electron-main.mjs").write_text("// baked\n", encoding="utf-8")
    _node("import {resolveStamp, buildStampPayload, writeDesktopStamp} from './apps/desktop/scripts/write-build-stamp.mjs';"
          "writeDesktopStamp('apps/desktop/build', buildStampPayload(resolveStamp({repoRoot: process.cwd()}), process.env, process.platform, null))",
          cwd=root)
    shutil.copy2(stamp, dist.parents[1] / "install-stamp.json")
    _node("import {recordProduct, buildInputs} from './scripts/build/freshness.mjs';"
          f"recordProduct({{source: '.', product: 'desktop', out: {json.dumps(str(dist))},"
          f" inputs: buildInputs('.', 'desktop', {{stamp: {json.dumps(str(stamp))}}})}})", cwd=root)
    return root


def _update_builds_desktop(root: Path, monkeypatch) -> bool:
    from hermes_cli import desktop_build_lock, main_desktop, source_build

    builds = []
    monkeypatch.setattr(main_desktop, "build_prepared_desktop", lambda *_a, **_k: builds.append(1))
    monkeypatch.setattr(main_desktop, "_refresh_installed_desktop_apps", lambda _d: None)
    monkeypatch.setattr(desktop_build_lock.DesktopBuildLock, "acquire", lambda self, **_k: None)
    monkeypatch.setattr(desktop_build_lock.DesktopBuildLock, "release", lambda self: None)
    source_build._build_desktop_product(root, {"PATH": "/usr/bin:/bin"}, lambda _stage: None)
    return bool(builds)


@pytest.mark.platforms("linux")
def test_update_reuses_the_packaged_app_until_a_desktop_input_changes(packaged_checkout, monkeypatch):
    root = packaged_checkout
    built_at = _git(root, "rev-parse", "HEAD")
    (root / "hermes_cli/banner.py").write_text("BANNER = 2\n", encoding="utf-8")
    _git(root, "commit", "-q", "-am", "python only")

    assert not _update_builds_desktop(root, monkeypatch), "a Python-only commit rebuilt the desktop app"
    baked = root / "apps/desktop/release/linux-unpacked/resources/install-stamp.json"
    assert json.loads(baked.read_text(encoding="utf-8"))["commit"] == built_at, "the reused app must keep its real commit"

    (root / "apps/desktop/src/main.tsx").write_text("export const changed = 1\n", encoding="utf-8")
    _git(root, "commit", "-q", "-am", "desktop change")
    assert _update_builds_desktop(root, monkeypatch), "a desktop source change must rebuild"


@pytest.mark.platforms("linux")
def test_a_stamp_a_build_would_bake_differently_still_rebuilds(packaged_checkout, monkeypatch):
    """Same inputs, but a build at HEAD would stamp another branch: the shipped identity changed."""
    root = packaged_checkout
    _git(root, "checkout", "-q", "-b", "other")
    (root / "hermes_cli/banner.py").write_text("BANNER = 3\n", encoding="utf-8")
    _git(root, "commit", "-q", "-am", "python only, on another branch")

    assert _update_builds_desktop(root, monkeypatch)
