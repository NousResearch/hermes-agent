"""Real, offline uv construction below PM selection and distribution packaging."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from tests.pm._fixtures import (
    _run,
    _wheel,
    build_worker as build_worker,
    client as client,
    isolated_python as isolated_python,
)


def test_prune_site_pth_keeps_only_load_bearing_pth(tmp_path):
    """The payload venv must not carry uv's venv-marker or editable-install
    .pth files: the launcher addsitedirs the venv, and those two would repoint
    sys.prefix / shadow the repo snapshot with build-machine paths. Everything
    else (pywin32.pth!) must survive — it is what makes `import pywintypes`
    resolve on Windows bundles."""
    from pm.environment import prune_site_pth

    # Windows layout (Scripts/ present).
    win_venv = tmp_path / "win-venv"
    (win_venv / "Scripts").mkdir(parents=True)
    win_site = win_venv / "Lib" / "site-packages"
    win_site.mkdir(parents=True)
    # POSIX layout (bin/ present, versioned site-packages).
    posix_venv = tmp_path / "posix-venv"
    (posix_venv / "bin").mkdir(parents=True)
    posix_site = posix_venv / "lib" / "python3.14" / "site-packages"
    posix_site.mkdir(parents=True)

    for site in (win_site, posix_site):
        (site / "pywin32.pth").write_text("win32\nwin32\\lib\nimport pywin32_bootstrap\n", encoding="utf-8")
        (site / "_virtualenv.pth").write_text("import _virtualenv\n", encoding="utf-8")
        (site / "__editable__.hermes_agent-0.21.1.pth").write_text(
            "import __editable___hermes_agent_0_21_1_finder\n", encoding="utf-8"
        )

    prune_site_pth(win_venv)
    prune_site_pth(posix_venv)

    assert sorted(p.name for p in win_site.glob("*.pth")) == ["pywin32.pth"]
    assert sorted(p.name for p in posix_site.glob("*.pth")) == ["pywin32.pth"]


@pytest.fixture
def locked_project(tmp_path):
    uv = shutil.which("uv")
    assert uv, "environment construction tests require uv on PATH"
    source = tmp_path / "source with spaces"
    source.mkdir()
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for name in ("base_dep", "member_dep", "chosen_dep", "other_dep"):
        _wheel(wheels, name)
    (source / "pyproject.toml").write_text(
        '[project]\nname="construction-root"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=["base-dep==1.0"]\n'
        '[project.optional-dependencies]\nchosen=["chosen-dep==1.0"]\nother=["other-dep==1.0"]\n'
        '[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n'
        '[tool.uv.workspace]\nmembers=["member"]\n', encoding="utf-8",
    )
    member = source / "member"
    member.mkdir()
    (member / "pyproject.toml").write_text(
        '[project]\nname="plugin-member"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=["member-dep==1.0"]\n[tool.uv]\npackage=false\n', encoding="utf-8",
    )
    home = tmp_path / "isolated-home"
    home.mkdir()
    config = tmp_path / "config"
    config.mkdir()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("UV_", "PYTHON")) and key != "VIRTUAL_ENV"}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(home / ".hermes"),
               XDG_CONFIG_HOME=str(config), XDG_CONFIG_DIRS=str(config),
               UV_CACHE_DIR=str(tmp_path / "cache"), UV_PYTHON=sys.executable, UV_OFFLINE="1")
    _run([uv, "lock", "--python", sys.executable], cwd=source, env=env)
    return source, Path(uv), env


@pytest.fixture
def installable_project(locked_project, build_worker):
    source, uv, env = locked_project
    metadata = source / "pyproject.toml"
    metadata.write_text(metadata.read_text().replace("package=false", "package=true") +
                        '\n[build-system]\nrequires=[]\nbuild-backend="local_backend"\nbackend-path=["."]\n')
    (source / "root_app.py").write_text("VALUE = 'installed from the explicit source'\n")
    # A local PEP 517/660 backend: no registry or build-tool downloads in this fixture.
    (source / "local_backend.py").write_text('''
from pathlib import Path
from zipfile import ZipFile

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    Path("root_app.py").read_text()  # Full builds require the application source layer.
    name = "construction_root-1-py3-none-any.whl"
    dist = "construction_root-1.dist-info"
    entries = {
        "construction_root.pth": str(Path.cwd()) + "\\n",
        dist + "/METADATA": "Metadata-Version: 2.1\\nName: construction-root\\nVersion: 1\\nRequires-Dist: base-dep==1.0\\n",
        dist + "/WHEEL": "Wheel-Version: 1.0\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n",
    }
    entries[dist + "/RECORD"] = "".join(path + ",,\\n" for path in entries)
    with ZipFile(Path(wheel_directory) / name, "w") as wheel:
        for path, body in entries.items():
            wheel.writestr(path, body)
    return name

build_editable = build_wheel
''')
    _run([str(uv), "lock", "--python", sys.executable], cwd=source, env=env)
    return source, uv, env


@pytest.mark.parametrize("sealed", [False, True])
def test_public_build_installs_all_extras_at_explicit_destination(installable_project, tmp_path, monkeypatch, sealed):
    from pm import build_environment
    import pm.paths
    import pm.workspace

    source, uv, env = installable_project
    monkeypatch.setattr(pm.paths, "repo_root", lambda: tmp_path / "unrelated-project")
    monkeypatch.setattr(pm.workspace, "enabled_member_dirs", lambda: pytest.fail("user plugins"))
    before = dict(os.environ)
    locked = (source / "uv.lock").read_bytes()
    executable = build_environment(explicit=True,
        source=source, python=Path(sys.executable), out=tmp_path / "native environment",
        cache=tmp_path / "cache", env=env, all_extras=True, offline=True,
        sealed=sealed,
    )
    assert _run([str(executable), "-I", "-c",
                 "import root_app, member_dep, chosen_dep, other_dep; print(root_app.VALUE)"],
                cwd=tmp_path, env=env) == "installed from the explicit source"
    assert executable.parent.parent == tmp_path / "native environment"
    from hermes_cli.runtime_paths import site_packages

    site = site_packages(executable.parent.parent)
    assert (site / "_virtualenv.pth").exists() is not sealed
    assert (site / "construction_root.pth").is_file(), "load-bearing .pth must survive sealing"
    assert (source / "uv.lock").read_bytes() == locked
    assert dict(os.environ) == before
    assert not Path(env["HERMES_HOME"]).exists()


@pytest.mark.parametrize("selection, expected", [
    ({"extras": ["chosen", "chosen"]}, [True, False]),
    ({"all_extras": True}, [True, True]),
    ({}, [False, False]),
])
def test_public_dependency_only_build_needs_no_application_source(installable_project, tmp_path, selection, expected):
    source, uv, env = installable_project
    (source / "root_app.py").unlink()
    (source / "local_backend.py").unlink()
    locked = (source / "uv.lock").read_bytes()
    from pm import build_environment

    executable = build_environment(explicit=True, source=source, python=Path(sys.executable),
                                   out=tmp_path / "docker env", cache=tmp_path / "cache", env=env,
                                   no_install_project=True, offline=True, **selection)
    assert executable.is_file()
    result = json.loads(_run(
        [str(executable), "-I", "-c", "import json, importlib.util, importlib.metadata, base_dep; "
         "print(json.dumps([importlib.util.find_spec('chosen_dep') is not None, "
         "importlib.util.find_spec('other_dep') is not None, "
         "'construction-root' in [d.metadata['Name'] for d in importlib.metadata.distributions()]]))"],
        cwd=tmp_path, env=env,
    ))
    assert result == [*expected, False]
    assert (source / "uv.lock").read_bytes() == locked
    assert not Path(env["HERMES_HOME"]).exists()


def test_group_only_build_excludes_application_dependencies(locked_project, tmp_path, build_worker):
    import pm

    source, _, env = locked_project
    metadata = source / "pyproject.toml"
    metadata.write_text(metadata.read_text() + '\n[dependency-groups]\nicons=["chosen-dep==1.0"]\n')
    pm.lock_project(source, python=Path(sys.executable), cache=tmp_path / "cache", env=env,
                    offline=True, explicit=True)
    python = pm.build_environment(source=source, out=tmp_path / "icons", groups=["icons"],
                                  only_groups=True, python=Path(sys.executable), cache=tmp_path / "cache",
                                  env=env, offline=True, explicit=True)
    assert _run([str(python), "-I", "-c", "import chosen_dep, importlib.util; "
                 "assert importlib.util.find_spec('base_dep') is None; print(chosen_dep.__version__)"],
                cwd=tmp_path, env=env) == "1.0"


def test_worker_sync_reuses_unions_and_reports_real_lock_drift(locked_project, build_worker, tmp_path, monkeypatch):
    import pm
    from hermes_cli.runtime_paths import selected_venv, runtime_facts_path
    from pm.lock import Facts, Lockfile
    from pm import paths

    source, _, env = locked_project
    manifest = source / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace('[tool.uv.workspace]\nmembers=["member"]\n', ""), encoding="utf-8")
    monkeypatch.setattr(paths, "repo_root", lambda: source)
    pm.lock_project(source, offline=True, explicit=True)
    assert pm.check() == []
    parent_path, parent_env = list(sys.path), dict(os.environ)

    pm.sync_venv(["chosen"], explicit=True, plugin_dirs=[])
    first = selected_venv(source)
    first_fact = Facts(runtime_facts_path(source)).get("venv")
    pm.sync_venv(["chosen"], explicit=True, plugin_dirs=[])
    assert selected_venv(source) == first
    assert Facts(runtime_facts_path(source)).get("venv") == first_fact
    assert _run([str(first / ("Scripts/python.exe" if os.name == "nt" else "bin/python")), "-I", "-c",
                 "import base_dep, chosen_dep, importlib.util; assert importlib.util.find_spec('other_dep') is None; print(chosen_dep.__version__)"],
                cwd=tmp_path, env=env) == "1.0"

    pm.sync_venv(["other"], explicit=True, plugin_dirs=[])
    second = selected_venv(source)
    assert second != first
    assert Facts(runtime_facts_path(source)).get("venv")["extras"] == ["chosen", "other"]
    pm.sync_venv(["chosen"], explicit=True, plugin_dirs=[])
    assert selected_venv(source) == second
    assert _run([str(second / ("Scripts/python.exe" if os.name == "nt" else "bin/python")), "-I", "-c",
                 "import chosen_dep, other_dep; print(chosen_dep.__version__, other_dep.__version__)"],
                cwd=tmp_path, env=env) == "1.0 1.0"
    assert pm.check() == []
    _wheel(tmp_path / "wheels", "base_dep", "1.1")
    manifest.write_text(manifest.read_text().replace("base-dep==1.0", "base-dep==1.1"), encoding="utf-8")
    pm.lock_project(source, offline=True, explicit=True)
    assert pm.check() == ["venv: out of sync with uv.lock"]
    lock = Lockfile(paths.lockfile_path())
    lock.set_pin("node", "fixture", {"any": {"url": "https://example.invalid/node", "sha256": "0" * 64}})
    lock.save()
    assert "node: not installed or outdated" in pm.check()
    assert selected_venv(source) == second
    assert list(sys.path) == parent_path and dict(os.environ) == parent_env
    assert not {"base_dep", "chosen_dep", "other_dep"} & sys.modules.keys()


def test_child_output_is_live_and_keeps_explicit_index_credentials(tmp_path):
    import io
    from pm.environment import PythonEnvironment

    released = tmp_path / "release-child"

    class AcknowledgingLog(io.StringIO):
        def write(self, text):
            if "child-started" in text:
                released.touch()
            return super().write(text)

    output = AcknowledgingLog()
    child_env = dict(os.environ, UV_INDEX_PRIVATE_USERNAME="fixture-user",
                     UV_INDEX_PRIVATE_PASSWORD="fixture-secret",
                     UV_DEFAULT_INDEX="https://fixture.invalid/simple",
                     UV_PROJECT="/poison/project", UV_VENV_SEED="true", UV_SYSTEM_PYTHON="true",
                     UV_PYTHON="/poison/python", UV_CACHE_DIR="/poison/cache")
    environment = PythonEnvironment(
        uv=Path(sys.executable), python=Path(sys.executable), destination=tmp_path / "venv",
        cache=tmp_path / "cache", env=child_env, output=output,
    )
    script = (
        "import os, time; from pathlib import Path; "
        "print('child-started', flush=True); "
        f"release = Path({str(released)!r}); deadline = time.monotonic() + 5\n"
        "while not release.exists() and time.monotonic() < deadline: time.sleep(.01)\n"
        "assert release.exists(), 'log was hidden until process exit'\n"
        "assert os.environ['UV_INDEX_PRIVATE_USERNAME'] == 'fixture-user'\n"
        "assert os.environ['UV_INDEX_PRIVATE_PASSWORD'] == 'fixture-secret'\n"
        "assert os.environ['UV_DEFAULT_INDEX'] == 'https://fixture.invalid/simple'\n"
        "assert not {'UV_PROJECT', 'UV_VENV_SEED', 'UV_SYSTEM_PYTHON'} & os.environ.keys()\n"
        f"assert os.environ['UV_PYTHON'] == {sys.executable!r}\n"
        f"assert os.environ['UV_CACHE_DIR'] == {str(tmp_path / 'cache')!r}\n"
        "print('child-complete', flush=True)\n"
    )
    result = environment._run(["-c", script], cwd=tmp_path, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "child-started" in output.getvalue()
    assert "child-complete" in output.getvalue()
    assert "child-complete" in result.stderr, "failure diagnostics must retain a bounded output tail"


@pytest.fixture(params=["environment", "cli"])
def streaming_runner(request, tmp_path):
    import contextlib
    import io
    from pm.cli import _run_live
    from pm.environment import PythonEnvironment

    output = io.StringIO()
    environment = PythonEnvironment(
        uv=Path(sys.executable), python=Path(sys.executable), destination=tmp_path / "venv",
        cache=tmp_path / "cache", env=dict(os.environ), output=output,
    )

    def run(script, timeout):
        if request.param == "cli":
            with contextlib.redirect_stdout(output):
                return _run_live([sys.executable, "-c", script], cwd=tmp_path,
                                 env=dict(os.environ), timeout=timeout)
        return environment._run(["-c", script], cwd=tmp_path, timeout=timeout)

    return run, output, RuntimeError if request.param == "cli" else subprocess.TimeoutExpired


@pytest.mark.parametrize("parent_exits", [True, False])
def test_streaming_deadline_includes_inherited_stdout(tmp_path, parent_exits, streaming_runner):
    import threading
    import time

    run, output, timeout_error = streaming_runner
    release = tmp_path / "release-descendant"
    finished = tmp_path / "descendant-finished"
    # The descendant inherits stdout even when the direct child has already exited.
    # Its finite lifetime also bounds this regression on the broken implementation.
    descendant = (
        "import time; from pathlib import Path; "
        "print('inherited-output', flush=True); "
        f"release = Path({str(release)!r}); deadline = time.monotonic() + 10\n"
        "while not release.exists() and time.monotonic() < deadline: time.sleep(.01)\n"
        f"Path({str(finished)!r}).touch()\n"
    )
    parent = (
        "import subprocess, sys; "
        f"child = subprocess.Popen([sys.executable, '-c', {descendant!r}]); "
        + ("sys.exit(0)" if parent_exits else "child.wait()")
    )
    threads_before = set(threading.enumerate())
    timeout = 2
    started = time.monotonic()
    try:
        with pytest.raises(timeout_error):
            run(parent, timeout)
        assert time.monotonic() - started < timeout + 2, "draining stdout restarted the timeout"
        assert "inherited-output" in output.getvalue()
        assert set(threading.enumerate()) <= threads_before, "timeout leaked an output reader"
    finally:
        # Cooperatively stop the orphan: its parent is no longer in the test subtree.
        release.touch()
        deadline = time.monotonic() + 5
        while not finished.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert finished.exists(), "descendant did not acknowledge cleanup"
        # Clean up even a broken implementation's reader after closing its writer.
        for thread in set(threading.enumerate()) - threads_before:
            thread.join(timeout=5)


def test_streaming_eof_does_not_restart_process_wait_timeout(tmp_path, streaming_runner):
    import time

    run, output, timeout_error = streaming_runner
    # Spend most of the budget before EOF, then leave the process alive without
    # any pipe writers. Waiting for exit must use only the remaining budget.
    script = (
        "import os, time; print('before-eof', flush=True); time.sleep(3); "
        "os.close(1); os.close(2); time.sleep(10)"
    )
    timeout = 4
    started = time.monotonic()
    with pytest.raises(timeout_error):
        run(script, timeout)
    assert time.monotonic() - started < timeout + 2, "EOF restarted the process wait budget"
    assert "before-eof" in output.getvalue()


@pytest.mark.parametrize("damage", ["source", "check", "lock"])
def test_failed_build_removes_only_its_candidate(installable_project, tmp_path, damage):
    from pm import build_environment
    from pm.package import InstallError

    source, uv, env = installable_project
    previous = tmp_path / "previous"
    executable = build_environment(explicit=True, source=source, python=Path(sys.executable),
                                          out=previous, env=env, cache=tmp_path / "cache", offline=True)
    cfg = (previous / "pyvenv.cfg").read_bytes()
    source_lock = (source / "uv.lock").read_bytes()
    # Check destination refusal with valid inputs. The contract does not specify
    # which error comes first when the source is also damaged.
    with pytest.raises(FileExistsError, match="already exists"):
        build_environment(explicit=True, source=source, python=Path(sys.executable),
                          out=previous, env=env, cache=tmp_path / "cache", offline=True)
    if damage == "source":
        (source / "root_app.py").unlink()
    elif damage == "check":
        backend = source / "local_backend.py"
        backend.write_text(backend.read_text().replace("base-dep==1.0", "base-dep==2.0"))
        # Same-size edits within one timestamp tick otherwise reuse the backend's .pyc.
        shutil.rmtree(source / "__pycache__", ignore_errors=True)
    else:
        (source / "uv.lock").unlink()
    candidate = tmp_path / "candidate"
    with pytest.raises(InstallError, match="dependency validation" if damage == "check" else "uv sync|frozen build requires a lock"):
        build_environment(explicit=True, source=source, python=Path(sys.executable),
                                 out=candidate, env=env, cache=tmp_path / "cold-cache", offline=True)
    assert not candidate.exists()
    assert (previous / "pyvenv.cfg").read_bytes() == cfg
    assert _run([str(executable), "-I", "-c", "import base_dep; print(base_dep.__version__)"],
                cwd=tmp_path, env=env) == "1.0"
    if damage != "lock":
        assert (source / "uv.lock").read_bytes() == source_lock
    else:
        assert not (source / "uv.lock").exists(), "frozen builds must not manufacture a lock"


def test_lock_upgrade_and_group_selection_use_the_same_environment(locked_project, tmp_path):
    from pm.environment import PythonEnvironment
    import tomllib

    source, uv, env = locked_project
    manifest = source / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace('base-dep==1.0', 'base-dep>=1,<2') +
                        '\n[dependency-groups]\nqa=["other-dep==1.0"]\n')
    environment = PythonEnvironment(uv=uv, python=Path(sys.executable), destination=tmp_path / "candidate",
                                    cache=tmp_path / "cache", env=env, offline=True)
    environment.lock(source, timeout=60)
    _wheel(tmp_path / "wheels", "base_dep", "1.1")
    environment.lock(source, timeout=60)
    packages = tomllib.loads((source / "uv.lock").read_text())["package"]
    assert next(p["version"] for p in packages if p["name"] == "base-dep") == "1.0"
    environment.lock(source, upgrade=True, timeout=60)
    locked = (source / "uv.lock").read_bytes()
    environment.create()
    environment.sync(source, groups=["qa", "qa"], timeout=60)
    assert _run([str(environment.executable), "-I", "-c",
                 "import base_dep, other_dep; print(base_dep.__version__, other_dep.__version__)"],
                cwd=tmp_path, env=env) == "1.1 1.0"
    assert (source / "uv.lock").read_bytes() == locked


def test_explicit_environment_installs_locked_members_without_live_selection(locked_project, tmp_path, monkeypatch):
    from pm.environment import PythonEnvironment
    import pm.paths
    import pm.workspace

    source, uv, env = locked_project
    before_lock = (source / "uv.lock").read_bytes()
    before_process = dict(os.environ)
    before_env = dict(env)
    # Supplying a prepared workspace must bypass live roots and plugin discovery.
    monkeypatch.setattr(pm.paths, "repo_root", lambda: pytest.fail("implicit source lookup"))
    monkeypatch.setattr(pm.workspace, "enabled_member_dirs", lambda: pytest.fail("profile discovery"))
    environment = PythonEnvironment(
        uv=uv, python=Path(sys.executable), destination=tmp_path / "candidate",
        cache=tmp_path / "cache", env=env,
    )
    environment.create()
    environment.sync(source, extras=["chosen"])
    environment.check()
    result = json.loads(_run(
        [str(environment.executable), "-I", "-c",
         "import sys, json, base_dep, member_dep, chosen_dep, importlib.util; "
         "print(json.dumps([sys.base_prefix, member_dep.__version__, "
         "importlib.util.find_spec('other_dep') is None]))"], cwd=tmp_path, env=env,
    ))
    assert result == [sys.base_prefix, "1.0", True]
    assert (source / "uv.lock").read_bytes() == before_lock
    assert not (source / ".venv").exists()
    assert not (Path(env["HERMES_HOME"])).exists()
    assert env == before_env
    assert dict(os.environ) == before_process


def test_explicit_workspace_preserves_seed_and_replays_copied_members(locked_project, tmp_path, monkeypatch):
    from pm.environment import PythonEnvironment
    import pm.workspace as workspace

    source, uv, env = locked_project
    project = source / "pyproject.toml"
    project.write_text(project.read_text().replace(
        '[tool.uv.workspace]\nmembers=["member"]\n', "",
    ).replace('base-dep==1.0', 'base-dep>=1,<2'))
    _run([str(uv), "lock", "--python", sys.executable], cwd=source, env=env)
    seed_bytes = (source / "uv.lock").read_bytes()
    _wheel(tmp_path / "wheels", "base_dep", "1.1")
    original_member = source / "member"
    before_member = (original_member / "pyproject.toml").read_bytes()
    monkeypatch.setattr(workspace.paths, "repo_root", lambda: pytest.fail("implicit source discovery"))
    monkeypatch.setattr(workspace, "enabled_member_dirs", lambda: pytest.fail("profile discovery"))

    first = PythonEnvironment(uv=uv, python=Path(sys.executable), destination=tmp_path / "first" / "venv",
                              cache=tmp_path / "cache", env=env, offline=True)
    first.create()
    workspace.lock_and_sync(
        [original_member], ["chosen"], source=source, root=tmp_path / "first" / "workspace",
        venv_dir=first.destination, environment=first, seed_lock=source / "uv.lock",
    )
    first.check()
    assert _run([str(first.executable), "-I", "-c", "import base_dep, member_dep; print(base_dep.__version__)"],
                cwd=tmp_path, env=env) == "1.0", "compatible seed must not be upgraded"
    assert (source / "uv.lock").read_bytes() == seed_bytes
    assert (original_member / "pyproject.toml").read_bytes() == before_member
    recorded = tmp_path / "first" / "workspace"
    recorded_lock = (recorded / "uv.lock").read_bytes()

    # Repair replays recorded inputs, not today's edited source/plugins.
    (original_member / "pyproject.toml").write_text("broken plugin TOML")
    project.write_text("broken source TOML")
    second = PythonEnvironment(uv=uv, python=Path(sys.executable), destination=tmp_path / "second" / "venv",
                               cache=tmp_path / "cache", env=env, offline=True)
    second.create()
    workspace.lock_and_sync(
        [], ["chosen"], source=source, root=tmp_path / "second" / "workspace",
        venv_dir=second.destination, environment=second, replay=recorded,
    )
    second.check()
    assert _run([str(second.executable), "-I", "-c", "import member_dep, chosen_dep; print(member_dep.__version__)"],
                cwd=tmp_path, env=env) == "1.0"
    assert (tmp_path / "second" / "workspace" / "uv.lock").read_bytes() == recorded_lock
    assert (recorded / "uv.lock").read_bytes() == recorded_lock


def test_live_apply_keeps_selection_on_failed_union(locked_project, tmp_path, monkeypatch):
    from hermes_cli.runtime_paths import runtime_facts_path, selected_venv
    from pm.lock import Facts
    from pm.packages import Venv
    import pm.paths
    from pm.workspace import ResolutionConflict

    source, uv, env = locked_project
    metadata = source / "pyproject.toml"
    metadata.write_text(metadata.read_text().replace('[tool.uv.workspace]\nmembers=["member"]\n', ""))
    _run([str(uv), "lock", "--python", sys.executable], cwd=source, env=env)
    source_lock = (source / "uv.lock").read_bytes()
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "isolated-home")
    monkeypatch.setenv("HERMES_HOME", env["HERMES_HOME"])
    monkeypatch.setattr(pm.paths, "repo_root", lambda: source)
    # Tool acquisition is the adapter's job; the same prepared env is not mutated.
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (uv, Path(sys.executable)))
    original_env = dict(env)
    prepared = Venv().apply(["chosen"], plugin_dirs=[source / "member"])
    assert env == original_env
    assert selected_venv(source) == source / "venv", "preparation must not select the candidate"
    assert (prepared["environment"].parent / ".lease-managed").is_file()
    facts = Facts(runtime_facts_path(source))
    facts.record_state("venv", "fixture-stamp", ["chosen"], **prepared)
    prior_facts = facts.path.read_bytes()
    generations = prepared["environment"].parent.parent
    prior_generations = set(generations.iterdir())
    plugin = source / "member" / "pyproject.toml"
    plugin.write_text(plugin.read_text().replace('member-dep==1.0', 'member-dep==2.0'))
    with pytest.raises(ResolutionConflict):
        Venv().apply(["chosen"], plugin_dirs=[source / "member"])
    assert selected_venv(source) == prepared["environment"]
    assert facts.path.read_bytes() == prior_facts
    assert set(generations.iterdir()) == prior_generations
    assert (source / "uv.lock").read_bytes() == source_lock
    assert env == original_env
