"""pm.workspace: the generated uv-workspace root for plugin deps.

The workspace root is a pm-GENERATED project (never the committed
pyproject.toml — sealed installs are read-only and member lists are
machine-specific). Its pyproject = core's pyproject verbatim +
``[tool.uv.workspace] members`` pointing at each snapshotted plugin.
``uv lock`` unions core + plugin deps into ONE lock; conflict = loud refusal.
"""

from __future__ import annotations

import os
import subprocess
import shutil
import sys
from pathlib import Path

import pytest

import pm.workspace as ws
from pm.environment import managed_environment
from tests.pm.test_environment_build import locked_project  # noqa: F401




@pytest.fixture
def layout(locked_project, tmp_path, monkeypatch):
    core, uv, _ = locked_project
    manifest = core / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace('[tool.uv.workspace]\nmembers=["member"]\n', ""))
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (uv, Path(sys.executable)))
    monkeypatch.setattr(ws.paths, "repo_root", lambda: core)
    return tmp_path, core, core / "member", tmp_path / "store"




def test_preparation_refuses_existing_workspace_without_mutating_it(layout):
    from pm.package import InstallError

    tmp, core, plug_a, _ = layout
    root = tmp / "workspace"
    environment = managed_environment(tmp / "env")
    kwargs = dict(root=root, source=core, seed_lock=None,
                  environment=environment)
    ws.lock_and_sync([plug_a], [], **kwargs)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    (plug_a / "pyproject.toml").write_text('changed after publication')
    with pytest.raises(InstallError, match="fresh"):
        ws.lock_and_sync([], [], **kwargs)
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_missing_explicit_seed_cannot_silently_resolve_new_versions(layout):
    tmp, core, plug_a, _ = layout
    with pytest.raises(FileNotFoundError):
        ws.lock_and_sync([plug_a], [], root=tmp / "workspace", source=core,
                         seed_lock=tmp / "missing.lock", environment=managed_environment(tmp / "env"))
    assert not (tmp / "env").exists()


def test_core_quarantine_covers_core_packages_and_not_plugin_ones(layout, locked_project):
    """Regression for #120076: Hermes's 14-day cutoff must not filter a plugin's own deps.

    A global ``exclude-newer`` in the generated root made a catalog pin floored on a fresh
    release unresolvable. The cutoff now travels per package: every registry package in
    core's lock keeps it (or core's explicit exemption), plugin-only packages follow the
    plugin's policy, and the rewritten root still locks and syncs.
    """
    import tomllib

    tmp, core, plug_a, _ = layout
    _, uv, env = locked_project
    manifest = core / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace("[tool.uv]\n", '[tool.uv]\nexclude-newer="14 days"\n', 1)
                        + '[tool.uv.exclude-newer-package]\nBase_Dep = false\n')
    subprocess.run([str(uv), "lock", "--python", sys.executable], cwd=core, env=env, check=True,
                   capture_output=True)
    core_packages = {p["name"] for p in tomllib.loads((core / "uv.lock").read_text())["package"]
                     if "registry" in p.get("source", {})}
    assert {"base-dep", "chosen-dep"} <= core_packages and "member-dep" not in core_packages

    root = tmp / "workspace"
    ws.lock_and_sync([plug_a], [], root=root, source=core, seed_lock=core / "uv.lock",
                     environment=managed_environment(tmp / "env"))

    policy = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["uv"]
    assert "exclude-newer" not in policy
    per_package = policy["exclude-newer-package"]
    assert per_package == {name: False if name == "base-dep" else "14 days" for name in core_packages}
    locked = {p["name"] for p in tomllib.loads((root / "uv.lock").read_text())["package"]}
    assert "member-dep" in locked and "member-dep" not in per_package






# --- classified failures + staging surface (FINAL-RUNTIME-CONTRACT) ---


def test_classify_network_failure_stays_generic():
    from pm.workspace import ResolutionConflict, classify_uv_failure

    err = classify_uv_failure("lock", 1, "error: Failed to fetch https://pypi.org (timed out)")
    assert not isinstance(err, ResolutionConflict)


def test_sync_failure_is_never_a_conflict(layout, monkeypatch):
    from pm.package import InstallError

    tmp, core, _, _ = layout
    environment = managed_environment(tmp / "candidate")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs:
                        subprocess.CompletedProcess(cmd, 1, "", "Failed to download wheel"))
    with pytest.raises(InstallError) as excinfo:
        ws.lock_and_sync([], [], root=tmp / "workspace", source=core,
                         seed_lock=None, environment=environment, frozen=True)
    assert not isinstance(excinfo.value, ws.ResolutionConflict)


def test_staging_root_and_env_are_honored_without_live_mutation(layout, monkeypatch):
    tmp, core, _, _ = layout
    staging = tmp / "staging-ws"
    monkeypatch.setenv("PM_WORKSPACE_TEST_SENTINEL", "live")
    environment = managed_environment(tmp / "staging-venv", env={
        "PATH": "/staged/bin", "PM_WORKSPACE_TEST_SENTINEL": "staged",
    })
    # The prepared environment is authoritative; workspace never discovers tools.
    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: pytest.fail("PATH discovery"))
    seen = []
    def run(cmd, **kwargs):
        seen.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", run)
    ws.lock_and_sync([], [], root=staging, source=core, seed_lock=None, environment=environment)
    assert [cmd[1] for cmd, _ in seen] == ["lock", "sync"]
    for cmd, kwargs in seen:
        assert Path(cmd[0]) == environment.uv
        assert Path(kwargs["cwd"]) == staging
        assert kwargs["env"]["PM_WORKSPACE_TEST_SENTINEL"] == "staged"
        assert kwargs["env"]["UV_CACHE_DIR"] == str(environment.cache)
        assert kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(environment.destination)
        assert kwargs["env"]["UV_PYTHON"] == str(environment.python)
    assert os.environ["PM_WORKSPACE_TEST_SENTINEL"] == "live"


def test_plugin_generation_resolves_host_without_unrelated_foreign_platform_conflict(layout):
    """A runtime generation preserves selections even when a foreign fork cannot lock."""
    import tomllib

    tmp, core, plugin, _ = layout
    # The local index has no compatible release for this foreign-only extra.
    foreign_platform = "android" if sys.platform != "android" else "darwin"
    manifest = core / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace(
        'other=["other-dep==1.0"]',
        f'other=["unavailable-on-foreign-platform==1.0; sys_platform == {foreign_platform!r}"]'))
    environment = managed_environment(tmp / "env")
    from pm.workspace import ResolutionConflict
    with pytest.raises(ResolutionConflict, match="unavailable-on-foreign-platform"):
        environment.lock(core, timeout=30)

    root = tmp / "workspace"
    ws.lock_and_sync([plugin], ["chosen"], root=root, source=core,
                     seed_lock=core / "uv.lock", environment=environment)
    locked = {p["name"] for p in tomllib.loads((root / "uv.lock").read_text())["package"]}
    assert {"base-dep", "chosen-dep", "member-dep"} <= locked
    assert "other-dep" not in locked
    # The original cross-platform declaration remains available to other hosts.
    assert f"sys_platform == {foreign_platform!r}" in manifest.read_text()


@pytest.mark.parametrize("host", ["android", "linux", "darwin", "win32"])
def test_runtime_environments_intersect_core_without_excluding_supported_hosts(host):
    from packaging.markers import Marker
    markers = [Marker(marker) for marker in
               ws._runtime_environments(["python_version >= '3.14'"], host)]
    env = {"sys_platform": host, "platform_machine": "aarch64", "python_version": "3.14"}
    assert any(marker.evaluate(env) for marker in markers)
    assert not any(marker.evaluate({**env, "python_version": "3.13"}) for marker in markers)
    assert not any(marker.evaluate({**env, "sys_platform": "other-host"}) for marker in markers)
    assert Marker(ws._runtime_environments(None, host)[0]).evaluate(env)
