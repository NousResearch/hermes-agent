"""Offline behavior contracts for PM's caller-owned build operations."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from pm.package import InstallError
from tests.pm.test_environment_build import _run, _wheel


@pytest.fixture
def build_tools(tmp_path, monkeypatch):
    uv = shutil.which("uv")
    assert uv, "PM build tests require real uv"
    # Worker transport is covered separately. These public operation tests use
    # real engine/tools without downloading a second toolchain.
    monkeypatch.setattr("pm.client.is_runtime", lambda: True)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable)))
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("UV_", "PYTHON")) and key != "VIRTUAL_ENV"}
    home = tmp_path / "home"
    home.mkdir()
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(home / ".hermes"))
    monkeypatch.setenv("HERMES_HOME", env["HERMES_HOME"])
    monkeypatch.setattr(Path, "home", lambda: home)
    return env


@pytest.fixture
def locked_source(tmp_path, build_tools):
    from pm.operations import lock_project

    wheels = tmp_path / "wheelhouse"
    wheels.mkdir()
    for name in ("base_dep", "chosen_dep", "dev_dep"):
        _wheel(wheels, name)
    source = tmp_path / "source with spaces"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname="export-root"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=["base-dep==1.0"]\n'
        '[project.optional-dependencies]\nchosen=["chosen-dep==1.0; sys_platform == \'win32\'"]\n'
        '[dependency-groups]\ndev=["dev-dep==1.0"]\n'
        '[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding="utf-8",
    )
    lock_project(source, python=Path(sys.executable), cache=tmp_path / "cache",
                 env=build_tools, offline=True)
    return source


@pytest.mark.parametrize("lock_state", ["current", "stale", "missing"])
def test_check_lock_never_changes_source(locked_source, tmp_path, build_tools, lock_state):
    from pm import check_project_lock

    source = locked_source
    if lock_state == "stale":
        manifest = source / "pyproject.toml"
        manifest.write_text(manifest.read_text().replace('version="1"', 'version="2"'), encoding="utf-8")
    elif lock_state == "missing":
        (source / "uv.lock").unlink()
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    if lock_state == "current":
        check_project_lock(source, python=Path(sys.executable), cache=tmp_path / "cache",
                           env=build_tools, offline=True)
    else:
        with pytest.raises(InstallError):
            check_project_lock(source, python=Path(sys.executable), cache=tmp_path / "cache",
                               env=build_tools, offline=True)
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
    assert not (source / ".venv").exists()


def test_frozen_export_keeps_markers_and_excludes_build_metadata(locked_source, tmp_path, build_tools):
    from packaging.requirements import Requirement
    from pm import export_requirements

    source = locked_source
    # Frozen export must use yesterday's resolution, not refresh today's edits.
    manifest = source / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace("base-dep==1.0", "base-dep==2.0"), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    out = tmp_path / "export directory" / "requirements.txt"
    export_requirements(source, out, extras=["chosen", "chosen"], python=Path(sys.executable),
                        cache=tmp_path / "cache", env=build_tools)
    text = out.read_text(encoding="utf-8")
    requirements = {r.name: r for r in map(Requirement, text.splitlines())}
    assert set(requirements) == {"base-dep", "chosen-dep"}
    assert str(requirements["base-dep"].specifier) == "==1.0"
    marker = requirements["chosen-dep"].marker
    assert marker is not None
    assert marker.evaluate({"sys_platform": "win32"})
    assert not marker.evaluate({"sys_platform": "linux"})
    assert "#" not in text and "--hash" not in text
    assert {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()} == before
    assert not (source / ".venv").exists()


def test_frozen_export_preserves_git_commit_pin(locked_source, tmp_path, build_tools):
    from packaging.requirements import Requirement
    from pm import export_requirements, lock_project

    dependency = tmp_path / "git-dependency"
    dependency.mkdir()
    (dependency / "pyproject.toml").write_text(
        '[project]\nname="git-dep"\nversion="1.0"\nrequires-python=">=3.11"\n'
        'dependencies=[]\n', encoding="utf-8",
    )
    _run(["git", "init", "--quiet"], cwd=dependency, env=build_tools)
    _run(["git", "add", "pyproject.toml"], cwd=dependency, env=build_tools)
    _run(["git", "-c", "user.name=PM test", "-c", "user.email=pm@example.invalid",
          "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={tmp_path / 'empty-hooks'}",
          "commit", "--quiet", "-m", "fixture"], cwd=dependency, env=build_tools)
    revision = _run(["git", "rev-parse", "HEAD"], cwd=dependency, env=build_tools)
    url = f"git+{dependency.as_uri()}@{revision}"
    manifest = locked_source / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace('"base-dep==1.0"', json.dumps(f"git-dep @ {url}")),
                        encoding="utf-8")
    lock_project(locked_source, python=Path(sys.executable), cache=tmp_path / "cache", env=build_tools)
    before = (locked_source / "uv.lock").read_bytes()
    out = tmp_path / "git-requirements.txt"
    export_requirements(locked_source, out, cache=tmp_path / "cache", env=build_tools)
    requirement = Requirement(out.read_text(encoding="utf-8").strip())
    assert requirement.name == "git-dep"
    assert requirement.url == url
    assert (locked_source / "uv.lock").read_bytes() == before


@pytest.mark.parametrize("sealed", [False, True])
def test_requirements_build_installs_offline_markers_and_seals_only_build_pth(tmp_path, build_tools, sealed):
    from pm import build_requirements_environment

    wheels = tmp_path / "wheels with spaces"
    wheels.mkdir()
    _wheel(wheels, "leaf_dep")
    _wheel(wheels, "app_dep", requirements=["leaf-dep==1.0"])
    out = tmp_path / "destination with spaces"
    # Ambient project and interpreter settings cannot escape the explicit build.
    (tmp_path / "uv.toml").write_text('required-version="<0.1"\n', encoding="utf-8")
    poison = tmp_path / "poison"
    env = dict(build_tools, UV_PROJECT_ENVIRONMENT=str(poison), VIRTUAL_ENV=str(poison),
               UV_PYTHON=str(poison), UV_CACHE_DIR=str(poison), UV_SYSTEM_PYTHON="true")
    before = dict(os.environ)
    executable = build_requirements_environment(
        ["app-dep==1.0; python_version >= '3'", "missing-dep==1.0; python_version < '2'"],
        out=out, python=Path(sys.executable), wheelhouse=wheels, cache=tmp_path / "cache",
        env=env, offline=True, sealed=sealed,
    )
    result = json.loads(_run(
        [str(executable), "-I", "-c", "import sys, json, app_dep, leaf_dep, importlib.util; "
         "print(json.dumps([app_dep.__version__, leaf_dep.__version__, sys.base_prefix, "
         "importlib.util.find_spec('missing_dep') is None]))"], cwd=tmp_path, env=build_tools,
    ))
    assert result == ["1.0", "1.0", sys.base_prefix, True]
    assert executable.parent.parent == out
    assert bool(list(out.rglob("_virtualenv.pth"))) is not sealed
    assert not poison.exists()
    assert not Path(build_tools["HERMES_HOME"]).exists()
    assert dict(os.environ) == before


@pytest.mark.parametrize("failure", ["create", "install", "check"])
def test_failed_requirement_build_removes_only_its_candidate(tmp_path, build_tools, failure):
    import zipfile
    from pm import build_requirements_environment

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel = _wheel(wheels, "app_dep")
    previous = tmp_path / "previous"
    executable = build_requirements_environment(["app-dep==1.0"], out=previous, wheelhouse=wheels,
                                               env=build_tools, cache=tmp_path / "cache", offline=True)
    previous_cfg = (previous / "pyvenv.cfg").read_bytes()
    python = Path(sys.executable)
    if failure == "create":
        python = tmp_path / "missing-python"
    elif failure == "install":
        wheel.unlink()
    else:
        # The archive carries an extra distribution invisible to resolution.
        # pip check must reject its unsatisfied dependency after installation.
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr("app_dep-1.0.data/purelib/ghost-1.0.dist-info/METADATA",
                             "Metadata-Version: 2.1\nName: ghost\nVersion: 1.0\n"
                             "Requires-Dist: missing-dep==1.0\n")
    out = tmp_path / "candidate"
    with pytest.raises(InstallError, match="dependency validation" if failure == "check" else None):
        build_requirements_environment(["app-dep==1.0"], out=out, python=python, wheelhouse=wheels,
                                       env=build_tools, cache=tmp_path / "cold-cache", offline=True)
    assert not out.exists()
    with pytest.raises(FileExistsError):
        build_requirements_environment(["app-dep==1.0"], out=previous, wheelhouse=wheels,
                                       env=build_tools, cache=tmp_path / "cache", offline=True)
    assert (previous / "pyvenv.cfg").read_bytes() == previous_cfg
    assert _run([str(executable), "-I", "-c", "import app_dep; print(app_dep.__version__)"],
                cwd=tmp_path, env=build_tools) == "1.0"


def test_wheelhouse_cannot_fall_back_to_index_or_build_source(tmp_path, build_tools):
    import base64
    import io
    import tarfile
    from pm import build_requirements_environment

    index = tmp_path / "index" / "leaf-dep"
    index.mkdir(parents=True)
    wheel = _wheel(index, "leaf_dep")
    (index / "index.html").write_text(f'<a href="{wheel.name}">{wheel.name}</a>', encoding="utf-8")
    wheels = tmp_path / "wheelhouse"
    wheels.mkdir()
    # A real, self-contained sdist: relaxing binary-only would make it install.
    backend = (
        "from pathlib import Path\nimport base64\n"
        "def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n"
        f"    Path(wheel_directory, {wheel.name!r}).write_bytes(base64.b64decode("
        f"{base64.b64encode(wheel.read_bytes())!r}))\n"
        f"    return {wheel.name!r}\n"
    )
    entries = {
        "pyproject.toml": '[project]\nname="leaf-dep"\nversion="1.0"\n'
        '[build-system]\nrequires=[]\nbuild-backend="backend"\nbackend-path=["."]\n',
        "backend.py": backend,
    }
    with tarfile.open(wheels / "leaf_dep-1.0.tar.gz", "w:gz") as archive:
        for name, text in entries.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(f"leaf_dep-1.0/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    env = dict(build_tools, UV_DEFAULT_INDEX=index.parent.as_uri())
    out = tmp_path / "candidate"
    with pytest.raises(InstallError):
        build_requirements_environment(["leaf-dep==1.0"], out=out, wheelhouse=wheels,
                                       env=env, cache=tmp_path / "cache", offline=True)
    assert not out.exists()
    # Both alternate sources really work when there is no wheelhouse restriction.
    for name, settings in (("index", env), ("source", dict(build_tools, UV_NO_INDEX="1", UV_FIND_LINKS=str(wheels)))):
        executable = build_requirements_environment(
            ["leaf-dep==1.0"], out=tmp_path / f"from-{name}", env=settings,
            cache=tmp_path / f"{name}-cache", offline=True,
        )
        assert _run([str(executable), "-I", "-c", "import leaf_dep; print(leaf_dep.__version__)"],
                    cwd=tmp_path, env=build_tools) == "1.0"


@pytest.mark.parametrize("ci", [False, True])
def test_prune_cache_preserves_downloaded_wheels_unless_ci(tmp_path, build_tools, ci):
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from pm import build_requirements_environment, prune_cache

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel = _wheel(wheels, "cached_dep")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(wheels)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cache = tmp_path / "cache"
    requirement = f"cached-dep @ http://127.0.0.1:{server.server_port}/{wheel.name}"
    try:
        executable = build_requirements_environment(
            [requirement], out=tmp_path / "first", cache=cache, env=build_tools,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert _run([str(executable), "-I", "-c", "import cached_dep; print(cached_dep.__version__)"],
                cwd=tmp_path, env=build_tools) == "1.0"
    prune_cache(cache, ci=ci)
    out = tmp_path / "second"
    if ci:
        with pytest.raises(InstallError):
            build_requirements_environment([requirement], out=out, cache=cache, env=build_tools, offline=True)
        assert not out.exists()
    else:
        second = build_requirements_environment([requirement], out=out, cache=cache, env=build_tools, offline=True)
        assert _run([str(second), "-I", "-c", "import cached_dep; print(cached_dep.__version__)"],
                    cwd=tmp_path, env=build_tools) == "1.0"
