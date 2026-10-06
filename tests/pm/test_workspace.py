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


def _plugin_pyproject(tmp: Path, name: str, body: str) -> Path:
    directory = tmp / name
    directory.mkdir()
    (directory / "pyproject.toml").write_text(body, encoding="utf-8")
    (directory / "plugin.yaml").write_text(f"name: {name}\n", encoding="utf-8")
    return directory


def test_plugin_extras_and_dev_groups_never_constrain_hermes(layout):
    """Hermes installs neither a plugin's extras nor its dev group, so their pins (here an
    unsatisfiable one against core's ``base-dep==1.0``) must not make the plugin uninstallable."""
    import tomllib

    tmp, core, _, _ = layout
    plugin = _plugin_pyproject(tmp, "pinned-dev", (
        '[project]\nname = "pinned-dev"\nversion = "1"\nrequires-python = ">=3.11"\n'
        'dependencies = ["member-dep==1.0"]\n'
        '[project.optional-dependencies]\ndev = ["base-dep==9.9"]\n'
        '[dependency-groups]\ndev = ["other-dep==9.9"]\n[tool.uv]\npackage = false\n'))
    root = tmp / "workspace"
    ws.lock_and_sync([plugin], [], root=root, source=core, seed_lock=core / "uv.lock",
                     environment=managed_environment(tmp / "env"))
    locked = {p["name"]: p["version"] for p in tomllib.loads((root / "uv.lock").read_text())["package"]}
    assert locked["base-dep"] == "1.0" and locked["member-dep"] == "1.0"


def test_tool_config_pyproject_leaves_the_manifest_in_charge_of_dependencies(layout):
    """A pyproject holding only tool settings (ruff, pytest) is not a package definition:
    the plugin's manifest dependencies still install, instead of uv refusing a [project]
    table PM had to invent."""
    import tomllib

    tmp, core, _, _ = layout
    plugin = tmp / "lint-only"
    plugin.mkdir()
    (plugin / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n", encoding="utf-8")
    (plugin / "plugin.yaml").write_text("name: lint-only\npython_dependencies:\n  - member-dep==1.0\n",
                                        encoding="utf-8")
    root = tmp / "workspace"
    ws.lock_and_sync([plugin], [], root=root, source=core, seed_lock=core / "uv.lock",
                     environment=managed_environment(tmp / "env"))
    locked = {p["name"] for p in tomllib.loads((root / "uv.lock").read_text())["package"]}
    assert "member-dep" in locked


class _TimedIndex:
    """A PEP 691 JSON index with per-file ``upload-time``.

    uv applies ``exclude-newer`` only to files that carry an upload time, which
    find-links wheels never do, so the quarantine needs a real index to bite.
    """

    def __init__(self, wheels: Path, uploaded: dict[str, str]):
        import hashlib
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def _reply(self, send_body: bool):
                parts = self.path.strip("/").split("/")
                if parts[0] == "simple":
                    name = parts[1]
                    files = [{"filename": wheel.name, "url": f"/files/{wheel.name}",
                              "hashes": {"sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()},
                              "upload-time": uploaded[wheel.name]}
                             for wheel in sorted(wheels.glob(name.replace("-", "_") + "-*.whl"))]
                    body = json.dumps({"meta": {"api-version": "1.1"}, "name": name, "files": files,
                                       "versions": sorted({f["filename"].split("-")[1] for f in files})}).encode()
                    kind = "application/vnd.pypi.simple.v1+json"
                else:
                    body, kind = (wheels / parts[1]).read_bytes(), "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if send_body:
                    self.wfile.write(body)

            def do_GET(self):
                self._reply(True)

            def do_HEAD(self):
                self._reply(False)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/simple"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def quarantined(tmp_path, monkeypatch):
    """Core with a 14-day quarantine: ``core-dep`` 1.0 is old, 2.0 and ``fresh-dep`` are a day old."""
    import datetime as dt

    from tests.pm._fixtures import _wheel

    uv = shutil.which("uv")
    assert uv, "workspace tests require uv on PATH"
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    now = dt.datetime.now(dt.timezone.utc)
    stamp = lambda days: (now - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    uploaded = {}
    for name, version, age in (("core_dep", "1.0", 400), ("core_dep", "2.0", 1), ("fresh_dep", "1.0", 1)):
        _wheel(wheels, name, version)
        uploaded[f"{name}-{version}-py3-none-any.whl"] = stamp(age)
    index = _TimedIndex(wheels, uploaded)
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text(
        '[project]\nname="quarantine-core"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=["core-dep>=1.0"]\n[tool.uv]\npackage=false\nexclude-newer="14 days"\n'
        f'[[tool.uv.index]]\nurl="{index.url}"\ndefault=true\n', encoding="utf-8")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("UV_", "PYTHON")) and key != "VIRTUAL_ENV"}
    env.update(UV_CACHE_DIR=str(tmp_path / "cache"), XDG_CONFIG_HOME=str(tmp_path / "config"))
    subprocess.run([uv, "lock", "--python", sys.executable], cwd=core, env=env, check=True, capture_output=True)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable)))
    monkeypatch.setattr(ws.paths, "repo_root", lambda: core)

    def plugin(name: str, dependencies: list[str], exempt: tuple[str, ...] = ()) -> Path:
        import json

        directory = tmp_path / name
        directory.mkdir()
        exemptions = "".join(f'{package} = false\n' for package in exempt)
        (directory / "pyproject.toml").write_text(
            f'[project]\nname="{name}"\nversion="1"\nrequires-python=">=3.11"\n'
            f'dependencies={json.dumps(dependencies)}\n[tool.uv]\npackage=false\n'
            + (f"[tool.uv.exclude-newer-package]\n{exemptions}" if exempt else ""), encoding="utf-8")
        return directory

    def lock(plugin_dir: Path, workspace: str):
        root = tmp_path / f"{workspace}-workspace"
        ws.lock_and_sync([plugin_dir], [], root=root, source=core, seed_lock=core / "uv.lock",
                         environment=managed_environment(tmp_path / f"{workspace}-env"))
        import tomllib
        return {p["name"]: p["version"] for p in tomllib.loads((root / "uv.lock").read_text())["package"]}

    yield plugin, lock
    index.close()


def test_plugin_deps_wait_out_the_quarantine_unless_the_plugin_exempts_them(quarantined):
    """A plugin's fresh dependency is held by core's 14-day cutoff like any other, and the
    plugin's own ``[tool.uv] exclude-newer-package`` exemption (which uv ignores on a
    workspace member) is what lets it through."""
    plugin, lock = quarantined
    with pytest.raises(ws.ResolutionConflict, match="fresh-dep"):
        lock(plugin("held", ["fresh-dep==1.0"]), "held")
    assert lock(plugin("exempt", ["fresh-dep==1.0"], exempt=("fresh-dep",)), "exempt")["fresh-dep"] == "1.0"


def test_plugin_exemption_cannot_move_a_core_package_past_the_quarantine(quarantined):
    """A plugin may only exempt what it alone brings in: exempting a package core locks
    would drag core's dependency onto a release nobody has waited out."""
    plugin, lock = quarantined
    assert lock(plugin("loose", ["core-dep>=1.0"], exempt=("core-dep",)), "loose")["core-dep"] == "1.0"
    with pytest.raises(ws.ResolutionConflict, match="core-dep"):
        lock(plugin("grabby", ["core-dep>=2.0"], exempt=("core-dep",)), "grabby")






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
