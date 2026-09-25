"""Real, offline uv construction below PM selection and distribution packaging."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from pm.plugin_inputs import Members
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
    (source / "root_app.py").write_text("VALUE = 'installed from the explicit source'\n", encoding="utf-8")
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
    monkeypatch.setenv("UV_PYTHON", "/not-the-interpreter")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(tmp_path / "wrong-environment"))
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "wrong-environment"))
    before = dict(os.environ)
    locked = (source / "uv.lock").read_bytes()
    executable = build_environment(explicit=True,
        source=source, python=Path(sys.executable), out=tmp_path / "native environment",
        cache=tmp_path / "cache", env=env, all_extras=True, offline=True,
        sealed=sealed,
    )
    from pm.environments import site_packages

    site = site_packages(executable.parent.parent)
    # Bytecode is written at build time, not by the first user request's import (#100461);
    # checked before anything imports from the environment.
    assert list(site.glob("*/__pycache__/*.pyc")) or list(site.glob("__pycache__/*.pyc")), \
        "dependencies were installed without bytecode"
    assert _run([str(executable), "-I", "-c",
                 "import root_app, member_dep, chosen_dep, other_dep; print(root_app.VALUE)"],
                cwd=tmp_path, env=env) == "installed from the explicit source"
    assert executable.parent.parent == tmp_path / "native environment"
    assert not (tmp_path / "wrong-environment").exists()
    assert "root_app" not in sys.modules
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


def test_all_extras_build_leaves_out_opt_in_extras(installable_project, tmp_path):
    source, uv, env = installable_project
    manifest = source / "pyproject.toml"
    manifest.write_text(manifest.read_text() + '\n[tool.hermes]\nopt-in-extras=["other"]\n', encoding="utf-8")
    from pm import build_environment

    probe = ("import json, importlib.util; print(json.dumps([importlib.util.find_spec(n) is not None "
             "for n in ('chosen_dep', 'other_dep')]))")
    bundle = build_environment(explicit=True, source=source, python=Path(sys.executable),
                               out=tmp_path / "bundle env", cache=tmp_path / "cache", env=env,
                               no_install_project=True, offline=True, all_extras=True)
    assert json.loads(_run([str(bundle), "-I", "-c", probe], cwd=tmp_path, env=env)) == [True, False]
    chosen = build_environment(explicit=True, source=source, python=Path(sys.executable),
                               out=tmp_path / "chosen env", cache=tmp_path / "cache", env=env,
                               no_install_project=True, offline=True, extras=["other"])
    assert json.loads(_run([str(chosen), "-I", "-c", probe], cwd=tmp_path, env=env)) == [False, True]



@pytest.mark.parametrize("lazy", [False, True])
def test_first_bundle_extension_preserves_shipped_extras(locked_project, build_worker, tmp_path, monkeypatch, lazy):
    import pm
    from pm.environments import selected_venv, runtime_facts_path
    from pm import paths
    from pm.features import write_features
    from pm.lock import Facts

    source, _, env = locked_project
    manifest = source / "pyproject.toml"
    manifest.write_text(manifest.read_text(encoding="utf-8-sig").replace('[tool.uv.workspace]\nmembers=["member"]\n', ""), encoding="utf-8")
    monkeypatch.setattr(paths, "repo_root", lambda: source)
    pm.lock_project(source, offline=True, explicit=True)
    base = tmp_path / "shipped"
    pm.build_environment(source=source, out=base, extras=["chosen"], env=env,
                         cache=tmp_path / "cache", offline=True, explicit=True)
    (tmp_path / "manifest.json").write_text(json.dumps({"repo": source.name, "venv": base.name}), encoding="utf-8")
    write_features(["chosen"], tmp_path)
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(f"security:\n  allow_lazy_installs: {str(lazy).lower()}\n", encoding="utf-8")
    assert selected_venv(source) == base
    assert Facts(runtime_facts_path(source)).get("venv") is None
    python_relative = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    assert _run([str(base / python_relative), "-I", "-c",
                 "import chosen_dep; print(chosen_dep.__version__)"], cwd=tmp_path, env=env) == "1.0"
    locked = (source / "uv.lock").read_bytes()

    # Admission adds only a plugin, not a list of the bundle's optional extras.
    pm.sync_venv(explicit=True, plugins=Members([source / "member"]))
    first = selected_venv(source)
    assert first != base
    executable = first / python_relative
    assert _run([str(executable), "-I", "-c",
                 "import base_dep, chosen_dep, member_dep; print(chosen_dep.__version__)"],
                cwd=tmp_path, env=env) == "1.0"
    first_fact = Facts(runtime_facts_path(source)).get("venv")
    assert first_fact is not None and first_fact["extras"] == ["chosen"]

    # Once recorded, the selection owns the baseline, even if inventory changes.
    write_features(["other"], tmp_path)
    _wheel(tmp_path / "wheels", "member_dep", "1.1")
    member = source / "member" / "pyproject.toml"
    member.write_text(member.read_text(encoding="utf-8-sig").replace("member-dep==1.0", "member-dep==1.1"), encoding="utf-8")
    pm.sync_venv(explicit=True, plugins=Members([source / "member"]))
    second = selected_venv(source)
    assert second != first
    second_fact = Facts(runtime_facts_path(source)).get("venv")
    assert second_fact is not None and second_fact["extras"] == ["chosen"]
    assert _run([str(second / executable.relative_to(first)), "-I", "-c",
                 "import chosen_dep, member_dep, importlib.util; "
                 "assert importlib.util.find_spec('other_dep') is None; print(member_dep.__version__)"],
                cwd=tmp_path, env=env) == "1.1"
    assert (source / "uv.lock").read_bytes() == locked


def test_retired_recorded_extra_does_not_break_two_profile_plugin_sync(locked_project, build_worker, tmp_path, monkeypatch):
    import pm
    from pm import paths
    from pm.environments import runtime_facts_path, selected_venv
    from pm.lock import Facts
    from pm.package import InstallError

    source, _, env = locked_project
    manifest = source / "pyproject.toml"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace(
        '[tool.uv.workspace]\nmembers=["member"]\n', ""), encoding="utf-8")
    monkeypatch.setattr(paths, "repo_root", lambda: source)
    pm.lock_project(source, offline=True, explicit=True)
    copies = []
    for profile in ("default", "named"):
        plugin = tmp_path / profile / "plugins" / "hindsight"
        plugin.mkdir(parents=True)
        (plugin / "pyproject.toml").write_text(
            '[project]\nname="hermes-plugin-hindsight"\nversion="1.0.1"\n'
            'requires-python=">=3.11"\ndependencies=["member-dep==1.0"]\n'
            '[tool.uv]\npackage=false\n', encoding="utf-8")
        copies.append(plugin)

    pm.sync_venv(["chosen"], explicit=True, plugins=Members(copies))
    assert Facts(runtime_facts_path(source)).get("venv")["extras"] == ["chosen"]
    manifest.write_text(manifest.read_text(encoding="utf-8").replace(
        'chosen=["chosen-dep==1.0"]\n', ""), encoding="utf-8")
    pm.lock_project(source, offline=True, explicit=True)
    with pytest.raises(InstallError, match="unknown extras requested"):
        pm.sync_venv(["chosen"], explicit=True, plugins=Members(copies))

    pm.sync_venv(explicit=True, plugins=Members(copies))
    assert Facts(runtime_facts_path(source)).get("venv")["extras"] == []
    environment = selected_venv(source)
    executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    assert _run([str(executable), "-I", "-c", "import base_dep, member_dep; print(member_dep.__version__)"],
                cwd=tmp_path, env=env) == "1.0"


def test_worker_sync_reuses_unions_and_reports_real_lock_drift(locked_project, build_worker, tmp_path, monkeypatch):
    import pm
    from pm.environments import selected_venv, runtime_facts_path
    from pm.lock import Facts, Lockfile
    from pm import paths

    source, _, env = locked_project
    manifest = source / "pyproject.toml"
    manifest.write_text(manifest.read_text(encoding="utf-8-sig").replace('[tool.uv.workspace]\nmembers=["member"]\n', ""), encoding="utf-8")
    monkeypatch.setattr(paths, "repo_root", lambda: source)
    pm.lock_project(source, offline=True, explicit=True)
    assert pm.check() == []
    parent_path, parent_env = list(sys.path), dict(os.environ)
    # The worker imports its own class; this trap affects only inline installs.
    monkeypatch.setattr("pm.packages.Venv.apply", lambda *a, **kw: pytest.fail("venv apply ran in caller"))

    pm.sync_venv(["chosen"], explicit=True, plugins=Members([]))
    first = selected_venv(source)
    first_fact = Facts(runtime_facts_path(source)).get("venv")
    pm.sync_venv(["chosen"], explicit=True, plugins=Members([]))
    assert selected_venv(source) == first
    assert Facts(runtime_facts_path(source)).get("venv") == first_fact
    assert _run([str(first / ("Scripts/python.exe" if os.name == "nt" else "bin/python")), "-I", "-c",
                 "import base_dep, chosen_dep, importlib.util; assert importlib.util.find_spec('other_dep') is None; print(chosen_dep.__version__)"],
                cwd=tmp_path, env=env) == "1.0"

    pm.sync_venv(["other"], explicit=True, plugins=Members([]))
    second = selected_venv(source)
    assert second != first
    assert Facts(runtime_facts_path(source)).get("venv")["extras"] == ["chosen", "other"]
    pm.sync_venv(["chosen"], explicit=True, plugins=Members([]))
    assert selected_venv(source) == second
    assert _run([str(second / ("Scripts/python.exe" if os.name == "nt" else "bin/python")), "-I", "-c",
                 "import chosen_dep, other_dep; print(chosen_dep.__version__, other_dep.__version__)"],
                cwd=tmp_path, env=env) == "1.0 1.0"
    assert pm.check() == []
    _wheel(tmp_path / "wheels", "base_dep", "1.1")
    manifest.write_text(manifest.read_text(encoding="utf-8-sig").replace("base-dep==1.0", "base-dep==1.1"), encoding="utf-8")
    pm.lock_project(source, offline=True, explicit=True)
    assert pm.check() == ["venv: out of sync with uv.lock"]
    lock = Lockfile(paths.lockfile_path())
    lock.set_pin("node", "fixture", {"any": {"url": "https://example.invalid/node", "sha256": "0" * 64}})
    lock.save()
    assert "node: not installed or outdated" in pm.check()
    assert selected_venv(source) == second
    assert list(sys.path) == parent_path and dict(os.environ) == parent_env
    assert not {"base_dep", "chosen_dep", "other_dep"} & sys.modules.keys()


@pytest.mark.parametrize("operation", ["sync", "requirements", "application"])
def test_build_backend_output_is_streamed_before_build_finishes(installable_project, tmp_path, monkeypatch, operation):
    import io
    from pm.environment import PythonEnvironment

    monkeypatch.setenv("HERMES_VERBOSE", "1")  # live backend output is the streamed (CI) view's contract
    source, uv, env = installable_project
    release = tmp_path / "release-build"
    stdout_marker = "construction-root: backend stdout"
    stderr_marker = "construction-root: backend stderr"
    backend = source / "local_backend.py"
    backend.write_text(backend.read_text() + f'''
import sys
import time

original_build = build_wheel

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    print({stdout_marker!r}, flush=True)
    print({stderr_marker!r}, file=sys.stderr, flush=True)
    release = Path({str(release)!r})
    deadline = time.monotonic() + 10
    while not release.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert release.exists(), "backend output was hidden until build exit"
    return original_build(wheel_directory, config_settings, metadata_directory)

build_editable = build_wheel
''', encoding="utf-8")

    class AcknowledgingLog(io.StringIO):
        def write(self, text):
            written = super().write(text)
            if stdout_marker in self.getvalue() and stderr_marker in self.getvalue():
                release.touch()
            return written

    output = AcknowledgingLog()
    environment = PythonEnvironment(
        uv=uv, python=Path(sys.executable), destination=tmp_path / "built",
        cache=tmp_path / "build-cache", offline=True, output=output,
        env=dict(env, UV_NO_INDEX="1", UV_FIND_LINKS=str(tmp_path / "wheels")),
    )
    if operation == "application":
        from pm.packages import Venv

        monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (uv, Path(sys.executable)))
        monkeypatch.setattr(sys, "stderr", output)
        result = Venv(source).apply([], plugin_dirs=[], explicit=True)
        executable = result["environment"] / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    else:
        environment.create()
        if operation == "sync":
            environment.sync(source, timeout=30)
        else:
            environment.install_requirements([source.as_uri()])
        environment.check()
        executable = environment.executable
    assert release.is_file(), "both backend streams must arrive during the build"
    assert _run([str(executable), "-I", "-c", "import root_app; print(root_app.VALUE)"],
                cwd=tmp_path, env=env) == "installed from the explicit source"


@pytest.mark.parametrize("diagnostic", ["No solution found", "Connection timed out", "Failed to build wheel"])
def test_streaming_bounds_memory_without_losing_failure_class(tmp_path, diagnostic):
    import tracemalloc
    from pm.environment import PythonEnvironment
    from pm.workspace import classify_uv_failure

    # Separate writes split the resolver marker across pipe reads; subsequent
    # verbose output evicts it from the retained tail without changing its class.
    script = (
        f"import os, time; os.write(2, {diagnostic[:4].encode()!r}); time.sleep(.15); "
        f"os.write(2, {diagnostic[4:].encode()!r}); "
        "[os.write(2, b'x' * 65536) for _ in range(128)]; "
        "os.write(2, b'final diagnostic'); raise SystemExit(1)"
    )
    with open(os.devnull, "w", encoding="utf-8") as output:
        environment = PythonEnvironment(uv=Path(sys.executable), python=Path(sys.executable),
            destination=tmp_path / "venv", cache=tmp_path / "cache", env=dict(os.environ), output=output)
        tracemalloc.start()
        try:
            result = environment._run(["-c", script], cwd=tmp_path, timeout=30)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    assert result.returncode == 1
    assert len(result.stderr) <= 2000
    assert peak < 2 * 1024 * 1024, f"stream capture retained {peak} bytes"
    actual = classify_uv_failure("sync", result.returncode, result.stderr)
    assert type(actual) is type(classify_uv_failure("sync", 1, diagnostic))
    assert "final diagnostic" in str(actual)


def test_child_output_is_live_and_keeps_explicit_index_credentials(tmp_path, monkeypatch):
    import io
    from pm.environment import PythonEnvironment

    monkeypatch.setenv("HERMES_VERBOSE", "1")  # CI's streamed log, not the contained view

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
def streaming_runner(request, tmp_path, monkeypatch):
    import contextlib
    import io
    from pm.cli import _run_live
    from pm.environment import PythonEnvironment

    monkeypatch.setenv("HERMES_VERBOSE", "1")  # CI's streamed log, not the contained view

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

    from pm.package import InstallError

    return run, output, RuntimeError if request.param == "cli" else InstallError


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
        backend.write_text(backend.read_text(encoding="utf-8-sig").replace("base-dep==1.0", "base-dep==2.0"), encoding="utf-8")
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
    manifest.write_text(manifest.read_text(encoding="utf-8-sig").replace('base-dep==1.0', 'base-dep>=1,<2') +
                        '\n[dependency-groups]\nqa=["other-dep==1.0"]\n')
    environment = PythonEnvironment(uv=uv, python=Path(sys.executable), destination=tmp_path / "candidate",
                                    cache=tmp_path / "cache", env=env, offline=True)
    environment.lock(source, timeout=60)
    _wheel(tmp_path / "wheels", "base_dep", "1.1")
    environment.lock(source, timeout=60)
    packages = tomllib.loads((source / "uv.lock").read_text(encoding="utf-8-sig"))["package"]
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
    project.write_text(project.read_text(encoding="utf-8-sig").replace(
        '[tool.uv.workspace]\nmembers=["member"]\n', "",
    ).replace('base-dep==1.0', 'base-dep>=1,<2'))
    _run([str(uv), "lock", "--python", sys.executable], cwd=source, env=env)
    seed_bytes = (source / "uv.lock").read_bytes()
    _wheel(tmp_path / "wheels", "base_dep", "1.1")
    original_member = source / "member"
    before_member = (original_member / "pyproject.toml").read_bytes()
    stamp = workspace.members_stamp([original_member])
    assert stamp != workspace.members_stamp([])
    assert workspace.members_stamp([original_member, original_member]) == stamp
    monkeypatch.setattr(workspace.paths, "repo_root", lambda: pytest.fail("implicit source discovery"))
    monkeypatch.setattr(workspace, "enabled_member_dirs", lambda: pytest.fail("profile discovery"))

    first = PythonEnvironment(uv=uv, python=Path(sys.executable), destination=tmp_path / "first" / "venv",
                              cache=tmp_path / "cache", env=env, offline=True)
    first.create()
    workspace.lock_and_sync(
        [original_member], ["chosen"], source=source, root=tmp_path / "first" / "workspace",
        environment=first, seed_lock=source / "uv.lock",
    )
    first.check()
    assert _run([str(first.executable), "-I", "-c", "import base_dep, member_dep; print(base_dep.__version__)"],
                cwd=tmp_path, env=env) == "1.0", "compatible seed must not be upgraded"
    assert (source / "uv.lock").read_bytes() == seed_bytes
    assert (original_member / "pyproject.toml").read_bytes() == before_member
    recorded = tmp_path / "first" / "workspace"
    recorded_lock = (recorded / "uv.lock").read_bytes()
    import tomllib
    document = tomllib.loads((recorded / "pyproject.toml").read_text(encoding="utf-8-sig"))
    assert document["project"] == tomllib.loads(project.read_text(encoding="utf-8-sig"))["project"]
    [relative] = document["tool"]["uv"]["workspace"]["members"]
    copied = recorded / relative / "pyproject.toml"
    copied_data = tomllib.loads(copied.read_text(encoding="utf-8-sig"))
    original_data = tomllib.loads(before_member.decode("utf-8-sig"))
    assert copied_data["project"].pop("name") == original_data["project"].pop("name")
    assert copied_data == original_data
    assert copied.resolve().is_relative_to(recorded.resolve())
    (original_member / "pyproject.toml").unlink()
    assert workspace.members_stamp([original_member]) != stamp

    # Repair replays recorded inputs, not today's edited source/plugins.
    (original_member / "pyproject.toml").write_text("broken plugin TOML", encoding="utf-8")
    project.write_text("broken source TOML", encoding="utf-8")
    second = PythonEnvironment(uv=uv, python=Path(sys.executable), destination=tmp_path / "second" / "venv",
                               cache=tmp_path / "cache", env=env, offline=True)
    second.create()
    workspace.lock_and_sync(
        [], ["chosen"], source=source, root=tmp_path / "second" / "workspace",
        environment=second, seed_lock=None, replay=recorded,
    )
    second.check()
    assert _run([str(second.executable), "-I", "-c", "import member_dep, chosen_dep; print(member_dep.__version__)"],
                cwd=tmp_path, env=env) == "1.0"
    assert (tmp_path / "second" / "workspace" / "uv.lock").read_bytes() == recorded_lock
    assert (recorded / "uv.lock").read_bytes() == recorded_lock


@pytest.mark.parametrize("failure", ["facts", "missing-cfg", "restart"])
def test_real_sync_retains_selection_until_commit(locked_project, tmp_path, monkeypatch, failure):
    import importlib
    import pm.extras as extras
    from pm import paths
    from pm.lock import Facts
    from pm.environments import selected_venv

    source, uv, env = locked_project
    monkeypatch.setattr(paths, "repo_root", lambda: source)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kw: (uv, Path(sys.executable)))
    engine = importlib.import_module("pm.install")
    monkeypatch.setattr(engine, "lazy_installs_allowed", lambda: True)
    engine.sync_venv(["chosen"], plugins=Members([]), explicit=True)
    old = selected_venv(source)
    facts = paths.runtime_facts_path().read_bytes()
    home = Path(os.environ["HERMES_HOME"])
    config = home / "config.yaml"
    config.write_text("plugins: {enabled: []}\nsecurity: {allow_lazy_installs: true}\n", encoding="utf-8")
    config_bytes = config.read_bytes()
    if failure == "facts":
        def refuse(*args, **kwargs):
            raise OSError("facts disk full")
        with monkeypatch.context() as fault:
            fault.setattr(Facts, "record_state", refuse)
            with pytest.raises(OSError, match="facts disk full"):
                engine.sync_venv(["other"], plugins=Members([]), explicit=True)
        assert selected_venv(source) == old
        assert paths.runtime_facts_path().read_bytes() == facts
    elif failure == "missing-cfg":
        (old / "pyvenv.cfg").unlink()
        engine.sync_venv(["chosen"], plugins=Members([]), explicit=True)
        assert selected_venv(source) != old
        assert (selected_venv(source) / "pyvenv.cfg").is_file()
    else:
        monkeypatch.setattr("pm.client.sync_venv", engine.sync_venv)
        monkeypatch.setattr(extras, "available", lambda _: False)
        with pytest.raises(engine.InstallError, match="restart"):
            extras.ensure_import("other")
        assert selected_venv(source) != old
        assert "other_dep" not in sys.modules
    assert old.is_dir()
    assert config.read_bytes() == config_bytes


def test_live_apply_keeps_selection_on_failed_union(locked_project, tmp_path, monkeypatch):
    from pm.environments import runtime_facts_path, selected_venv
    from pm.lock import Facts
    from pm.packages import Venv
    import pm.paths
    from pm.workspace import ResolutionConflict

    source, uv, env = locked_project
    metadata = source / "pyproject.toml"
    metadata.write_text(metadata.read_text(encoding="utf-8-sig").replace('[tool.uv.workspace]\nmembers=["member"]\n', ""), encoding="utf-8")
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
    plugin.write_text(plugin.read_text(encoding="utf-8-sig").replace('member-dep==1.0', 'member-dep==2.0'), encoding="utf-8")
    with pytest.raises(ResolutionConflict):
        Venv().apply(["chosen"], plugin_dirs=[source / "member"])
    assert selected_venv(source) == prepared["environment"]
    assert facts.path.read_bytes() == prior_facts
    assert set(generations.iterdir()) == prior_generations
    assert (source / "uv.lock").read_bytes() == source_lock
    assert env == original_env
