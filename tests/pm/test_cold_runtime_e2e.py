"""Cold CLI provisioning and repair, with real tools and no mocked child code.

Only uv/Python are offered by the fixture's loopback archive server. The copied
PM recipe is unchanged; its small locked runtime is fetched from PyPI. The app
recipe is deliberately tiny so this cannot install the production dependency
set. Run through scripts/run_tests.sh with uv/uvx available on PATH.
"""
from __future__ import annotations

from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import tomllib

import pytest


class _ArchiveHandler(SimpleHTTPRequestHandler):
    def copyfile(self, source, outputfile):
        # The downloader closes its size probe without consuming the archive.
        try:
            super().copyfile(source, outputfile)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _run(command, *, cwd, env, expected=0, timeout=240):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                            text=True, timeout=timeout)
    assert result.returncode == expected, (
        f"{command!r}\nexit={result.returncode}\n{result.stdout}\n{result.stderr}"
    )
    return result


def _bare(python, repo, code, *, env, expected=0):
    return _run(
        [str(python), "-I", "-S", "-B", "-c",
         f"import sys; sys.path.insert(0, {str(repo)!r});\n" + code],
        cwd=repo.parent, env=env, expected=expected,
    )


@pytest.mark.platforms("linux")
def test_cold_cli_builds_own_runtime_discovers_plugins_and_repairs_app(tmp_path):
    from pm.packages import Python, Uv
    from pm.store import current_target, tree_digest

    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("real uv and uvx must be on PATH")
    uv = Path(uv).resolve()
    uvx = uv.with_name("uvx")
    assert uvx.is_file(), "the real uv distribution must include uvx"
    python = Path(sys._base_executable).resolve()
    if sys.version_info[:2] != (3, 14):
        pytest.skip("the checked-in PM runtime currently requires Python 3.14")

    source = Path(__file__).resolve().parents[2]
    repo = tmp_path / "source"
    repo.mkdir()
    for name in ("pm", "hermes_cli"):
        shutil.copytree(source / name, repo / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("utils.py", "hermes_constants.py", "hermes_yaml.py",
                 "hermes_bootstrap.py"):
        shutil.copy2(source / name, repo / name)
    # No production application lock or metadata enters this source snapshot.
    recipe = tomllib.loads((repo / "pm" / "pyproject.toml").read_text())
    yaml_requirement = next(dep for dep in recipe["project"]["dependencies"]
                            if dep.startswith("ruamel.yaml"))
    (repo / "pyproject.toml").write_text(
        '[project]\nname="cold-pm-app"\nversion="0.0.0"\n'
        'requires-python=">=3.14,<3.15"\n'
        f'dependencies=[{json.dumps(yaml_requirement)}]\n'
        '[project.optional-dependencies]\nall=[]\n'
        '[tool.uv]\npackage=false\n', encoding="utf-8",
    )
    home = tmp_path / "home"
    hermes_home = home / ".hermes"
    plugin = hermes_home / "plugins" / "cold-proof"
    plugin.mkdir(parents=True)
    config = hermes_home / "config.yaml"
    config.write_text("plugins:\n  enabled: [cold-proof]\n", encoding="utf-8")
    (plugin / "plugin.yaml").write_text(
        "name: cold-proof\nversion: 1.0.0\npip_dependencies: [idna==3.10]\n",
        encoding="utf-8",
    )
    store = hermes_home / "tools"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    env = {
        "HOME": str(home), "HERMES_HOME": str(hermes_home),
        "HERMES_RUNTIME_DIR": str(store), "PATH": os.defpath,
        "TMPDIR": str(scratch), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "UV_PYTHON_DOWNLOADS": "never", "UV_NO_CONFIG": "1",
        "UV_CACHE_DIR": str(tmp_path / "seed-cache"),
    }
    # Keep TLS functional on Nix without inheriting any application settings.
    for key in ("SSL_CERT_FILE", "SSL_CERT_DIR", "NIX_SSL_CERT_FILE"):
        if key in os.environ:
            env[key] = os.environ[key]
    _run([str(uv), "lock", "--project", str(repo), "--python", str(python)],
         cwd=tmp_path, env=env)

    archives = tmp_path / "archives"
    archives.mkdir()
    target = current_target()
    rows = {}
    for package, files in (
        (Python(), [(python, "python/bin/python3")]),
        (Uv(), [(uv, "uv-dist/uv"), (uvx, "uv-dist/uvx")]),
    ):
        archive = archives / f"{package.name}.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
            for binary, name in files:
                tar.add(binary, arcname=name)
        version = _run([str(files[0][0]), "--version"], cwd=tmp_path,
                       env=env).stdout.split()[1]
        rows[package.name] = {
            "version": version,
            "artifacts": {target: {"sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}},
        }
    server = ThreadingHTTPServer(("127.0.0.1", 0),
                                 partial(_ArchiveHandler, directory=str(archives)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for name, row in rows.items():
        row["artifacts"][target]["url"] = f"http://127.0.0.1:{server.server_port}/{name}.tar.gz"
    (repo / "pm" / "lock.json").write_text(
        json.dumps({"schema": 1, "packages": rows}), encoding="utf-8",
    )

    bootstrap = """
import importlib.util
assert importlib.util.find_spec('ruamel') is None
assert importlib.util.find_spec('yaml') is None
assert importlib.util.find_spec('packaging') is None
assert importlib.util.find_spec('idna') is None
"""
    cli = "\nsys.argv = ['hermes', 'pm', {action!r}]; import hermes_cli.main\n"
    try:
        assert not store.exists()
        assert not (hermes_home / "installs").exists()
        result = _bare(python, repo, bootstrap + cli.format(action="install"), env=env)
        assert "✓ venv" in result.stdout
        assert "Preparing the isolated PM runtime" in result.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)

    report_code = """
import json
from pathlib import Path
from hermes_cli.runtime_paths import install_state_dir, selected_venv, runtime_facts_path
root = Path(sys.path[0])
state = install_state_dir(root)
print(json.dumps({'state': str(state), 'app': str(selected_venv(root)),
                  'facts': json.loads(runtime_facts_path(root).read_text())}))
"""
    report = json.loads(_bare(python, repo, report_code, env=env).stdout)
    state, app = Path(report["state"]), Path(report["app"])
    assert state.is_relative_to(hermes_home)
    pm_root = state / "pm-runtime"
    selection = (pm_root / "selected.json").read_bytes()
    runtime = pm_root / json.loads(selection)["generation"]
    pm_python = runtime / "bin/python"
    assert runtime != app
    facts = json.loads((store / "facts.json").read_text())["packages"]
    assert set(facts) == {"python", "uv"}
    for name, fact in facts.items():
        assert fact["version"] == rows[name]["version"]
        assert fact["artifacts"] == [rows[name]["artifacts"][target]["sha256"]]
        assert tree_digest(store / fact["entry"]) == fact["digest"]

    probe = _run([str(pm_python), "-I", "-B", "-c", f"""
import importlib.util, json, sys
from pathlib import Path
sys.path.insert(0, {str(repo)!r})
from pm.workspace import enabled_member_dirs
from ruamel.yaml import YAML
import ruamel.yaml
assert enabled_member_dirs() == [Path({str(plugin)!r})]
assert importlib.util.find_spec('idna') is None
assert importlib.util.find_spec('openai') is None
print(json.dumps({{'yaml': ruamel.yaml.__file__, 'prefix': sys.prefix}}))
"""], cwd=tmp_path, env=env)
    pm_report = json.loads(probe.stdout)
    assert Path(pm_report["yaml"]).is_relative_to(runtime)
    app_code = """
import json
from pathlib import Path
from hermes_cli.runtime_paths import activate_dependencies, selected_venv
root = Path(sys.path[0])
activate_dependencies(root)
import ruamel.yaml, idna
from ruamel.yaml import YAML
assert idna.encode('bücher.example') == b'xn--bcher-kva.example'
assert YAML(typ='safe').load('proof: true')['proof'] is True
print(json.dumps({'yaml': ruamel.yaml.__file__, 'idna': idna.__file__,
                  'app': str(selected_venv(root))}))
"""
    app_report = json.loads(_bare(python, repo, app_code, env=env).stdout)
    assert Path(app_report["yaml"]).is_relative_to(app)
    assert Path(app_report["idna"]).is_relative_to(app)
    initial_fact = report["facts"]["packages"]["venv"]
    assert initial_fact["extras"] == ["all"]
    lock_before = Path(initial_fact["resolved_lock"]).read_bytes()
    assert b'idna' in lock_before

    # Real damage: the app no longer imports, while PM retains its independent YAML.
    shutil.rmtree(Path(app_report["yaml"]).parent)
    broken = _bare(python, repo, app_code, env=env, expected=1)
    assert "ModuleNotFoundError" in broken.stderr
    assert "ruamel.yaml" in broken.stderr
    repaired = _bare(python, repo, bootstrap + cli.format(action="repair"), env=env)
    assert "Restart Hermes" in repaired.stdout
    restored = json.loads(_bare(python, repo, app_code, env=env).stdout)
    repaired_app = Path(restored["app"])
    assert repaired_app != app
    assert Path(restored["yaml"]).is_relative_to(repaired_app)
    assert Path(restored["idna"]).is_relative_to(repaired_app)
    assert (pm_root / "selected.json").read_bytes() == selection
    after = json.loads(_bare(python, repo, report_code, env=env).stdout)
    repaired_fact = after["facts"]["packages"]["venv"]
    assert repaired_fact["extras"] == initial_fact["extras"]
    assert repaired_fact["stamp"] == initial_fact["stamp"]
    assert Path(repaired_fact["resolved_lock"]).read_bytes() == lock_before
    assert config.read_text() == "plugins:\n  enabled: [cold-proof]\n"
    assert not (state / ".repair-incomplete").exists()
    assert not (repo / "venv").exists()
    assert not (repo / ".venv").exists()
    print(f"cold PM proof: runtime={runtime}; first_app={app}; repaired_app={repaired_app}")
