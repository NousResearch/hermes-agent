"""Bash activation crosses real setup, PM install, and uv dependency publication.

Only the downloaded tool payloads are fixtures: bootstrap uv locates the test
interpreter, then delegates every dependency operation to real, offline uv.
The real PM stage builder seeds its locked dependencies online before activation.
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
from tests.pm._fixtures import _wheel, make_tar, served


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
    core, hermes_home, runtime, env, protected, dep_lock, srv = _setup_activation_lifecycle(tmp_path, served, bash, uv)
    assert not (hermes_home / "installs").exists()
    untouched = _snapshot(protected)
    activate = lambda succeeds=True: _run_activation(bash, core, tmp_path, env, protected, untouched, succeeds=succeeds)
    selection = lambda: json.loads(list(hermes_home.glob("installs/*/facts.json"))[0].read_text())["packages"]["venv"]
    operations = lambda name: [l for l in srv["calls"].read_text().splitlines() if l.split()[0] == name]
    app_syncs = lambda: [l for l in operations("sync") if "--frozen --all-packages" in l and "--group dev" not in l]

    cold, first = activate(), selection()
    probe = json.loads(cold.stdout)
    assert probe["version"] == "1.0" and Path(probe["module"]).is_relative_to(Path(first["environment"]))
    assert probe["pythonpath"].split(os.pathsep)[0] == str(core)
    assert "Preparing the isolated Hermes runtime" in cold.stderr and len(app_syncs()) == 1
    facts = json.loads((runtime / "facts.json").read_text())["packages"]
    assert facts["python"]["artifacts"] == [srv["first_digest"]] and facts["uv"]["artifacts"] == [srv["uv_digest"]]

    (core / ".env").write_text("FIXTURE_ONLY=existing\n", encoding="utf-8")
    (core / ".env").chmod(0o644)
    untouched = _snapshot(protected)
    activate()
    assert selection() == first and len(app_syncs()) == 1
    prepared = operations("sync")
    activate()
    assert selection() == first and operations("sync") == prepared
    assert len([line for line in operations("python") if line.startswith("python install ")]) == 3

    second_digest = srv["pin_python"]("second")
    activate()
    second = selection()
    assert second["stamp"] != first["stamp"] and second["environment"] != first["environment"]
    assert Path(first["environment"]).is_dir() and len(app_syncs()) == 2
    facts = json.loads((runtime / "facts.json").read_text())["packages"]
    assert facts["python"]["artifacts"] == [second_digest]
    assert (core / "uv.lock").read_bytes() == dep_lock

    (core / "uv.lock").write_text("not valid TOML [\n", encoding="utf-8")
    failed = activate(succeeds=False)
    assert "pm install failed" in failed.stderr and "setup failed" in failed.stderr
    assert "CALLER_SURVIVED:" in failed.stdout and selection() == second
    assert len(app_syncs()) == 3
    assert len([line for line in operations("python") if line.startswith("python install ")]) == 5


def _setup_activation_lifecycle(tmp_path, srv, bash, uv):
    interpreter = str(Path(sys._base_executable).resolve())
    core, home, hermes_home, runtime, env, protected = _prepare_checkout_and_env(tmp_path, bash, uv, interpreter)
    dep_lock = _prepare_wheels_and_lock(tmp_path, core, uv, interpreter, env)
    server = _prepare_mock_server_pins(core, tmp_path, srv, bash, uv, interpreter)
    return core, hermes_home, runtime, env, protected, dep_lock, server


def _prepare_checkout_and_env(tmp_path, bash, uv, interpreter):
    core, home = tmp_path / "checkout with spaces", tmp_path / "home"
    core.mkdir()
    home.mkdir()
    hermes_home, runtime = home / ".hermes", tmp_path / "runtime"
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    env = {
        "PATH": os.environ["PATH"], "HOME": str(home), "HERMES_HOME": str(hermes_home),
        "HERMES_RUNTIME_DIR": str(runtime), "SHELL": bash, "TMPDIR": str(scratch),
        "XDG_CONFIG_HOME": str(home / ".config"), "XDG_CACHE_HOME": str(home / ".cache"),
        "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1", "UV_OFFLINE": "1",
        "UV_CACHE_DIR": str(home / ".cache" / "uv"), "UV_PYTHON_DOWNLOADS": "never",
    }
    for name in ("activate", "setup-hermes.sh", "hermes_constants.py", "hermes_yaml.py", "utils.py",
                 "scripts/_activation.sh"):
        (core / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, core / name)
    for name in ("pm", "hermes_cli"):
        shutil.copytree(REPO / name, core / name, ignore=shutil.ignore_patterns("__pycache__"))
    import ruamel.yaml
    shutil.copytree(Path(ruamel.yaml.__file__).parent, core / "ruamel" / "yaml", ignore=shutil.ignore_patterns("__pycache__"))

    protected = [core / ".env", home / ".local" / "bin", hermes_home / "skills",
                 home / ".bashrc", home / ".bash_profile", home / ".zshrc"]
    for path in (home / ".local" / "bin" / "hermes", hermes_home / "skills" / "keep.md",
                 home / ".bashrc", home / ".bash_profile", home / ".zshrc"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("user-owned fixture\n", encoding="utf-8")
    (hermes_home / "skills").chmod(0o755)
    (core / ".env.example").write_text("FIXTURE_ONLY=example\n", encoding="utf-8")
    (core / "skills").mkdir()
    (core / "skills" / "bundled.md").write_text("must not be seeded\n", encoding="utf-8")
    return core, home, hermes_home, runtime, env, protected


def _prepare_wheels_and_lock(tmp_path, core, uv, interpreter, env):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _wheel(wheels, "activation_dep", "1.0")
    _wheel(wheels, "dev_fixture", "1.0")
    _wheel(wheels, "test_fixture", "1.0")
    (core / "pyproject.toml").write_text(
        '[project]\nname="activation-proof"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=["activation-dep==1.0"]\n[project.optional-dependencies]\nall=[]\n'
        '[dependency-groups]\ndev=["dev-fixture==1.0"]\ntest=["test-fixture==1.0"]\n'
        '[tool.uv]\npackage=false\nno-index=true\ndefault-groups=[]\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding="utf-8",
    )
    locked = subprocess.run(
        [uv, "lock", "--offline", "--python", interpreter], cwd=core, env=env,
        capture_output=True, text=True, timeout=60,
        check=False,
    )
    if locked.returncode != 0:
        raise RuntimeError(locked.stdout + locked.stderr)
    dependency_lock = (core / "uv.lock").read_bytes()

    seed = tmp_path / "pm-seed"
    seed_env = {key: value for key, value in env.items() if key != "UV_OFFLINE"}
    for key in ("SSL_CERT_FILE", "SSL_CERT_DIR", "NIX_SSL_CERT_FILE"):
        if key in os.environ:
            seed_env[key] = os.environ[key]
    seeded = subprocess.run(
        [interpreter, "-I", "-B", "-c",
         "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
         "from pm.runtime_stage import stage_runtime; "
         "stage_runtime(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), "
         "project=Path(sys.argv[1]) / 'pm', offline=False)",
         str(core), uv, interpreter, str(seed)],
        cwd=tmp_path, env=seed_env, capture_output=True, text=True, timeout=180,
        check=False,
    )
    if seeded.returncode != 0:
        raise RuntimeError(seeded.stdout + seeded.stderr)
    shutil.rmtree(seed)
    return dependency_lock


def _pin_mock_python(docroot, base_url, target, version, python_script, lock, revision):
    filename = f"python-{revision}.tar.gz"
    _, digest = make_tar(docroot, filename, {
        "python/bin/python3": python_script,
        "python/revision": revision,
    })
    lock.set_pin("python", version, {target: {"url": f"{base_url}/{filename}", "sha256": digest}})
    lock.save()
    return digest


def _prepare_mock_server_pins(core, tmp_path, srv, bash, uv, interpreter):
    docroot, base_url = srv
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
    lock.path.unlink()
    lock = Lockfile(lock.path)
    target = current_target()
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}+fixture"
    lock.set_pin("uv", "fixture", {target: {"url": f"{base_url}/uv.tar.gz", "sha256": uv_digest}})
    first_digest = _pin_mock_python(docroot, base_url, target, version, python_script, lock, "first")
    pin = lambda rev: _pin_mock_python(docroot, base_url, target, version, python_script, lock, rev)
    return {"calls": calls, "uv_digest": uv_digest, "first_digest": first_digest, "pin_python": pin}


def _run_activation(bash, core, tmp_path, env, protected, untouched, *, succeeds=True):
    script = '''
prior_path=$PATH
prior_pythonpath=${PYTHONPATH-}
source "$1/activate" --test-extras=all
status=$?
if [ "$status" != 0 ]; then
    test "$PATH" = "$prior_path" || exit 91
    test "${PYTHONPATH-}" = "$prior_pythonpath" || exit 92
    test -z "${__HERMES_ACTIVATED-}" || exit 93
    printf 'CALLER_SURVIVED:%s\\n' "$status"
    exit "$status"
fi
python3 -c 'import activation_dep, json, os; print(json.dumps({"version": activation_dep.__version__, "module": activation_dep.__file__, "pythonpath": os.environ["PYTHONPATH"]}))' || exit 94
"$__HERMES_TEST_PYTHON" -c 'import dev_fixture, test_fixture' || exit 97
python3 -c 'import importlib.util; assert importlib.util.find_spec("dev_fixture") is None' || exit 98
deactivate
test "$PATH" = "$prior_path" || exit 95
test "${PYTHONPATH-}" = "$prior_pythonpath" || exit 96
'''
    result = subprocess.run(
        [bash, "--noprofile", "--norc", "-c", script, "activation-test", str(core)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90, check=False,
    )
    if _snapshot(protected) != untouched:
        raise AssertionError(f"Protected paths mutated: {result.stdout}\n{result.stderr}")
    if (result.returncode == 0) != succeeds:
        raise AssertionError(f"Unexpected returncode {result.returncode} (expected succeeds={succeeds}):\n{result.stdout}\n{result.stderr}")
    return result
