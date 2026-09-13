"""Browser builds must leave the native workspace installation untouched."""
from pathlib import Path
from types import SimpleNamespace
import json
import shutil

import pytest

from hermes_cli import webapp


def test_webapp_installs_and_builds_in_a_private_workspace(tmp_path, monkeypatch):
    project = tmp_path / "checkout"
    desktop = project / "apps" / "desktop"
    shared = project / "apps" / "shared"
    desktop.mkdir(parents=True)
    shared.mkdir()
    (project / "package.json").write_text(json.dumps({"workspaces": ["apps/*"]}), encoding="utf-8")
    (project / "package-lock.json").write_text("locked", encoding="utf-8")
    (project / ".npmrc").write_text("engine-strict=true\n", encoding="utf-8")
    (project / ".gitignore").write_text("node_modules/\ndist/\n", encoding="utf-8")
    (desktop / "package.json").write_text('{"name":"desktop"}', encoding="utf-8")
    (shared / "package.json").write_text('{"name":"shared"}', encoding="utf-8")
    (desktop / "source.ts").write_text("renderer", encoding="utf-8")
    (shared / "source.ts").write_text("shared", encoding="utf-8")
    binary = project / "node_modules" / "electron" / "dist" / "electron.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"existing native install")
    native = desktop / "dist" / "index.html"
    native.parent.mkdir()
    native.write_text("existing native renderer", encoding="utf-8")
    roots = []

    def install(_npm, cwd, **_kwargs):
        roots.append(cwd)
        # Model npm ci's destructive removal at the actual requested destination.
        shutil.rmtree(cwd / "node_modules", ignore_errors=True)
        assert (cwd / ".npmrc").read_text() == "engine-strict=true\n"
        assert (cwd / "package-lock.json").read_text() == "locked"
        assert (cwd / "apps/shared/source.ts").read_text() == "shared"
        return SimpleNamespace(returncode=0)

    def build(argv, *, cwd, env):
        assert cwd == roots[0]
        assert (cwd / "apps/desktop/source.ts").read_text() == "renderer"
        assert not (cwd / "apps/desktop/dist").exists()
        staging = Path(argv[-1])
        staging.mkdir()
        (staging / "index.html").write_text("browser renderer", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(webapp, "_resolve_node_runtime_npm", lambda: "npm")
    monkeypatch.setattr(webapp, "_run_npm_install_deterministic", install)
    monkeypatch.setattr(webapp, "_run_with_idle_timeout", build)
    monkeypatch.setattr(webapp, "_write_stamp", lambda _root: None)

    result = webapp.prepare_webapp_renderer(project, force=True)

    assert binary.read_bytes() == b"existing native install"
    assert native.read_text() == "existing native renderer"
    assert result == desktop / "dist-webapp"
    assert (result / "index.html").read_text() == "browser renderer"
    assert roots[0] != project
    assert not roots[0].exists()


def test_failed_webapp_install_does_not_change_native_dependencies(tmp_path, monkeypatch):
    project = tmp_path / "checkout"
    desktop = project / "apps/desktop"
    desktop.mkdir(parents=True)
    (project / "package.json").write_text('{"workspaces":["apps/*"]}', encoding="utf-8")
    (desktop / "package.json").write_text('{}', encoding="utf-8")
    binary = project / "node_modules/electron/dist/electron.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"native")
    roots = []

    def fail_install(_npm, cwd, **_kwargs):
        roots.append(cwd)
        shutil.rmtree(cwd / "node_modules", ignore_errors=True)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(webapp, "_resolve_node_runtime_npm", lambda: "npm")
    monkeypatch.setattr(webapp, "_run_npm_install_deterministic", fail_install)
    with pytest.raises(webapp.WebappBuildError, match="dependency install failed"):
        webapp.prepare_webapp_renderer(project, force=True)
    assert binary.read_bytes() == b"native"
    assert not roots[0].exists()
