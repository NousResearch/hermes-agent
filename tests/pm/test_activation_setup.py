"""Bash activation crosses real setup, PM install, and uv dependency publication.

Only the downloaded tool payloads are fixtures: bootstrap uv locates the test
interpreter, then delegates every dependency operation to real, offline uv.
PM/activation/setup code is copied unmodified; all state is disposable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys

import pytest

from pm.lock import Lockfile
from pm.store import current_target
from tests.pm.test_pm_core import make_tar, served  # noqa: F401 -- shared HTTP fixture
from tests.pm.test_workspace_build_inputs import _wheel


pytestmark = pytest.mark.platforms("posix")
REPO = Path(__file__).resolve().parents[2]


def _snapshot(paths):
    """Notice creation, rewriting, chmod, and new children in protected locations."""
    result = {}
    for root in paths:
        for path in [root, *sorted(root.rglob("*"))]:
            if path.exists():
                info = path.stat()
                result[str(path)] = (
                    stat.S_IMODE(info.st_mode), info.st_mtime_ns,
                    path.read_bytes() if path.is_file() else None,
                )
    return result


def test_activation_real_setup_pm_lifecycle(tmp_path, served):
    bash, uv = shutil.which("bash"), shutil.which("uv")
    assert bash and uv, "this integration contract requires Bash and real uv"
    interpreter = str(Path(sys._base_executable).resolve())
    core = tmp_path / "checkout with spaces"
    core.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    hermes_home = home / ".hermes"
    runtime = tmp_path / "runtime"
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    env = {
        "PATH": os.environ["PATH"], "HOME": str(home), "HERMES_HOME": str(hermes_home),
        "HERMES_RUNTIME_DIR": str(runtime), "SHELL": bash, "TMPDIR": str(scratch),
        "XDG_CONFIG_HOME": str(home / ".config"), "XDG_CACHE_HOME": str(home / ".cache"),
        "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1", "UV_OFFLINE": "1",
        "UV_CACHE_DIR": str(home / ".cache" / "uv"), "UV_PYTHON_DOWNLOADS": "never",
    }
    for name in ("activate", "setup-hermes.sh", "hermes_constants.py", "utils.py"):
        shutil.copy2(REPO / name, core / name)
    for name in ("pm", "hermes_cli"):
        shutil.copytree(REPO / name, core / name, ignore=shutil.ignore_patterns("__pycache__"))
    # Plugin selection imports the real CLI config reader even with no plugins.
    # Supply its installed YAML dependency, not a stub parser or config module.
    import yaml
    shutil.copytree(Path(yaml.__file__).parent, core / "yaml", ignore=shutil.ignore_patterns("__pycache__"))

    # Never copy real user files: these are deliberately public fixture sentinels.
    protected = [core / ".env", home / ".local" / "bin", hermes_home / "skills",
                 home / ".bashrc", home / ".bash_profile", home / ".zshrc"]
    for path in (home / ".local" / "bin" / "hermes", hermes_home / "skills" / "keep.md",
                 home / ".bashrc", home / ".bash_profile", home / ".zshrc"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("user-owned fixture\n", encoding="utf-8")
    (core / ".env.example").write_text("FIXTURE_ONLY=example\n", encoding="utf-8")
    (core / "skills").mkdir()
    (core / "skills" / "bundled.md").write_text("must not be seeded\n", encoding="utf-8")
    untouched = _snapshot(protected)

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _wheel(wheels, "activation_dep", "1.0")
    (core / "pyproject.toml").write_text(
        '[project]\nname="activation-proof"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=["activation-dep==1.0"]\n[project.optional-dependencies]\nall=[]\n'
        '[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding="utf-8",
    )
    locked = subprocess.run(
        [uv, "lock", "--offline", "--python", interpreter], cwd=core, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert locked.returncode == 0, locked.stdout + locked.stderr
    dependency_lock = (core / "uv.lock").read_bytes()

    docroot, base_url = served
    # Both the shell bootstrap and PM's fallback downloader stay on loopback.
    (core / "pm" / "artifact-mirror.json").write_text(
        json.dumps({"origin": base_url, "prefix": "mirror/"}, indent=2) + "\n",
        encoding="utf-8",
    )
    calls = tmp_path / "uv-calls"
    uv_script = (
        f"#!{bash}\n"
        f"printf '%s\\n' \"$*\" >> {shlex.quote(str(calls))}\n"
        'if [ "$1 $2" = "python install" ]; then exit 0; fi\n'
        f'if [ "$1 $2" = "python find" ]; then printf \'%s\\n\' {shlex.quote(interpreter)}; exit 0; fi\n'
        f"exec {shlex.quote(uv)} --offline \"$@\"\n"
    )
    _, uv_digest = make_tar(docroot, "uv.tar.gz", {"uv-fixture/uv": uv_script})
    python_script = f'#!{bash}\nexec {shlex.quote(interpreter)} "$@"\n'
    lock = Lockfile(core / "pm" / "lock.json")
    # Replace shipped pins, not PM implementations or its package registry.
    lock.path.unlink()
    lock = Lockfile(lock.path)
    target = current_target()
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}+fixture"

    def pin_python(revision):
        filename = f"python-{revision}.tar.gz"
        _, digest = make_tar(docroot, filename, {
            "python/bin/python3": python_script,
            "python/revision": revision,
        })
        lock.set_pin("python", version, {target: {"url": f"{base_url}/{filename}", "sha256": digest}})
        lock.save()
        return digest

    lock.set_pin("uv", "fixture", {target: {"url": f"{base_url}/uv.tar.gz", "sha256": uv_digest}})
    first_digest = pin_python("first")

    def activate(*, succeeds=True):
        # No `set -e`: inspect source's status and prove the caller survives.
        script = '''
prior_path=$PATH
prior_pythonpath=${PYTHONPATH-}
source "$1/activate"
status=$?
if [ "$status" != 0 ]; then
    test "$PATH" = "$prior_path" || exit 91
    test "${PYTHONPATH-}" = "$prior_pythonpath" || exit 92
    test -z "${__HERMES_ACTIVATED-}" || exit 93
    printf 'CALLER_SURVIVED:%s\\n' "$status"
    exit "$status"
fi
python3 -c 'import activation_dep, json, os; print(json.dumps({"version": activation_dep.__version__, "module": activation_dep.__file__, "pythonpath": os.environ["PYTHONPATH"]}))' || exit 94
deactivate
test "$PATH" = "$prior_path" || exit 95
test "${PYTHONPATH-}" = "$prior_pythonpath" || exit 96
'''
        result = subprocess.run(
            [bash, "--noprofile", "--norc", "-c", script, "activation-test", str(core)],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90,
        )
        assert _snapshot(protected) == untouched
        assert (result.returncode == 0) is succeeds, result.stdout + result.stderr
        return result

    def selection():
        records = list(hermes_home.glob("installs/*/facts.json"))
        assert len(records) == 1
        return json.loads(records[0].read_text())["packages"]["venv"]

    def operations(name):
        return [line for line in calls.read_text().splitlines() if line.split()[0] == name]

    cold = activate()
    first = selection()
    probe = json.loads(cold.stdout)
    assert probe["version"] == "1.0"
    assert Path(probe["module"]).is_relative_to(Path(first["environment"]))
    assert probe["pythonpath"].split(os.pathsep)[0] == str(core)
    assert len(operations("venv")) == len(operations("sync")) == 1
    facts = json.loads((runtime / "facts.json").read_text())["packages"]
    assert facts["python"]["artifacts"] == [first_digest]
    assert facts["uv"]["artifacts"] == [uv_digest]

    # Also catch chmod of an existing .env, not just unwanted first creation.
    (core / ".env").write_text("FIXTURE_ONLY=existing\n", encoding="utf-8")
    (core / ".env").chmod(0o644)
    untouched = _snapshot(protected)
    activate()
    assert selection() == first
    assert len(operations("venv")) == len(operations("sync")) == 1
    assert len([line for line in operations("python") if line.startswith("python install ")]) == 2

    second_digest = pin_python("second")
    activate()
    second = selection()
    assert second["stamp"] != first["stamp"]
    assert second["environment"] != first["environment"]
    assert Path(first["environment"]).is_dir()
    assert len(operations("venv")) == len(operations("sync")) == 2
    facts = json.loads((runtime / "facts.json").read_text())["packages"]
    assert facts["python"]["artifacts"] == [second_digest]
    assert (core / "uv.lock").read_bytes() == dependency_lock

    # A real uv failure must cross PM -> setup -> source without selection or
    # caller mutation. An invalid local lock fails deterministically offline.
    (core / "uv.lock").write_text("not valid TOML [\n", encoding="utf-8")
    failed = activate(succeeds=False)
    assert "pm install failed" in failed.stderr
    assert "setup failed" in failed.stderr
    assert "CALLER_SURVIVED:" in failed.stdout
    assert selection() == second
    assert len(operations("sync")) == 3
    assert len([line for line in operations("python") if line.startswith("python install ")]) == 4
