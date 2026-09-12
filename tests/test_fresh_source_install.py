"""Whole current installer, real PM worker and local application dependency.

The bootstrap toolchain is prepared from this host's real Python/uv; this is
not a zero-Python download test. PM acquires real tool archives on loopback,
builds its unchanged runtime recipe and the tiny app, then publishes launchers.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tomllib

import pytest

from pm.store import current_target
from tests.pm._fixtures import _wheel, served as served

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("fault", [None, "missing-wheel", "bad-hash"])
def test_current_installer_publishes_real_dependencies_and_warm_path(tmp_path, served, fault):
    uv = shutil.which("uv")
    assert uv, "fresh-install acceptance requires real uv"
    python = Path(sys._base_executable).resolve()
    version = ".".join(map(str, sys.version_info[:3]))
    minor = ".".join(map(str, sys.version_info[:2]))
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "bootstrap-python"
    arch = {"x64": "x86_64", "arm64": "aarch64"}[current_target().split("-")[1]]
    bootstrap = managed / f"cpython-{version}-linux-{arch}-gnu"
    (bootstrap / "bin").mkdir(parents=True)
    shutil.copytree(sysconfig.get_path("stdlib"), bootstrap / f"lib/python{minor}",
                    ignore=shutil.ignore_patterns("site-packages", "__pycache__"))
    for path in bootstrap.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    shutil.copy2(python, bootstrap / f"bin/python{minor}")
    (bootstrap / "bin/python3").symlink_to(f"python{minor}")
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "LANG": "C.UTF-8",
           "HERMES_HOME": str(home / ".hermes"), "UV_PYTHON_INSTALL_DIR": str(managed),
           "UV_PYTHON_DOWNLOADS": "never", "UV_CACHE_DIR": str(tmp_path / "cache")}
    for key in ("SSL_CERT_FILE", "SSL_CERT_DIR", "NIX_SSL_CERT_FILE"):
        if key in os.environ:
            env[key] = os.environ[key]

    def run(argv, *, cwd=tmp_path, expected=0):
        result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=180)
        assert result.returncode == expected, result.stdout + result.stderr
        return result

    # Make the bootstrap layout complete using real uv, including its aliases.
    run([uv, "python", "install", "--no-bin", minor])
    source = tmp_path / "fixture source"
    source.mkdir()
    for name in ("pm", "hermes_cli"):
        shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("utils.py", "hermes_constants.py", "hermes_yaml.py", "hermes_bootstrap.py", "setup-hermes.sh"):
        shutil.copy2(ROOT / name, source / name)
    wheels = source / "wheels"
    wheels.mkdir()
    _wheel(wheels, "installer_probe", "1.0")
    recipe = tomllib.loads((source / "pm/pyproject.toml").read_text())
    yaml_dep = next(d for d in recipe["project"]["dependencies"] if d.startswith("ruamel.yaml"))
    (source / "pyproject.toml").write_text(
        '[project]\nname="installer-fixture"\nversion="1"\nrequires-python=">=3.14"\n'
        f'dependencies=["installer-probe==1.0",{json.dumps(yaml_dep)}]\n'
        '[project.optional-dependencies]\nall=[]\n[tool.uv]\npackage=false\n'
        '[tool.uv.sources]\ninstaller-probe={path="wheels/installer_probe-1.0-py3-none-any.whl"}\n',
        encoding="utf-8")
    run([uv, "lock", "--python", str(python)], cwd=source)
    # Only the application is a fixture; the shell, PM, bootstrap and writer run unchanged.
    (source / "hermes_cli/main.py").write_text(
        "import installer_probe, json, sys\n"
        "def main():\n print(json.dumps({'module':installer_probe.__file__, 'argv':sys.argv[1:]}))\n"
        "if __name__ == '__main__': main()\n", encoding="utf-8")
    docroot, url = served
    (source / "pm/artifact-mirror.json").write_text(
        json.dumps({"origin": url, "prefix": "mirror/"}, indent=2), encoding="utf-8")
    pins = {}
    for name, files in {
        "python": [(bootstrap, "python")],
        "uv": [(Path(uv).resolve(), "uv-dist/uv"), (Path(uv).resolve().with_name("uvx"), "uv-dist/uvx")],
    }.items():
        archive = docroot / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=1, dereference=True) as tar:
            for path, destination in files:
                tar.add(path, arcname=destination)
        pins[name] = {"version": version if name == "python" else run([uv, "--version"]).stdout.split()[1],
                      "artifacts": {current_target(): {"url": f"{url}/{archive.name}",
                      "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}}}
    if fault == "bad-hash":
        pins["python"]["artifacts"][current_target()]["sha256"] = "0" * 64
    if fault == "missing-wheel":
        next(wheels.glob("*.whl")).unlink()
    (source / "pm/lock.json").write_text(json.dumps({"schema": 1, "packages": pins}, indent=2), encoding="utf-8")
    run(["git", "init", "-b", "fixture"], cwd=source)
    run(["git", "add", "."], cwd=source)
    run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
         "commit", "-m", "fixture"], cwd=source)
    commit = run(["git", "rev-parse", "HEAD"], cwd=source).stdout.strip()
    install = tmp_path / "installed source"
    env["HERMES_REPO_URL"] = str(source)
    command = ["bash", str(ROOT / "scripts/install.sh"), "--dir", str(install),
               "--branch", "fixture", "--commit", commit, "--non-interactive", "--json"]
    result = run(command, expected=1 if fault else 0)
    if fault:
        assert not (install / ".hermes-bootstrap-complete").exists()
        assert not (home / ".local/bin/hermes").exists()
        for facts in (home / ".hermes/installs").glob("*/facts.json"):
            assert "venv" not in json.loads(facts.read_text())["packages"]
        assert '"stage":"python-deps"' in result.stdout
        return
    assert '"stage":"python-deps"' in result.stdout
    assert json.loads((install / ".hermes-bootstrap-complete").read_text())["pinnedCommit"] == commit
    facts = next((home / ".hermes/installs").glob("*/facts.json"))
    selection = json.loads(facts.read_text())["packages"]["venv"]
    launcher = home / ".local/bin/hermes"
    child = json.loads(run([str(launcher), "from elsewhere"]).stdout)
    assert child["argv"] == ["from elsewhere"]
    assert Path(child["module"]).is_relative_to(Path(selection["environment"]))
    assert not (install / "venv").exists()
    # The developer setup path publishes through the same writer after real PM.
    launcher.unlink()
    run(["bash", str(install / "setup-hermes.sh")])
    assert json.loads(run([str(launcher), "from setup"]).stdout)["argv"] == ["from setup"]
    # Warm path publication must work with all acquisition inputs unavailable.
    shutil.rmtree(docroot)
    shutil.rmtree(install / "wheels")
    env["UV_OFFLINE"] = "1"
    # A completed bootstrap can be read-only. Finding it must not reinstall it.
    marker = bootstrap / f"lib/python{minor}/EXTERNALLY-MANAGED"
    marker.chmod(0o444)
    before = facts.read_bytes()
    try:
        run([*command, "--stage", "path"])
    finally:
        marker.chmod(0o644)
    assert facts.read_bytes() == before
    assert json.loads(run([str(launcher), "offline"]).stdout)["argv"] == ["offline"]