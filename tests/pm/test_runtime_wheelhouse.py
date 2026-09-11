"""Exercise offline PM staging with real wheels, not the application's environment."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import urllib.request

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
import pytest

from pm.runtime import runtime_environment
from pm.runtime_stage import stage_runtime
from scripts.bundles.payload import seal_pm_runtime


@pytest.fixture(scope="module")
def locked_wheelhouse(tmp_path_factory):
    """Download host wheels first; only the subsequent stage runs offline."""
    wheelhouse = tmp_path_factory.mktemp("pm-wheelhouse")
    project = Path(__file__).resolve().parents[2] / "pm"
    lock = tomllib.loads((project / "uv.lock").read_text(encoding="utf-8"))
    tags = set(sys_tags())
    versions = {}
    for package in lock["package"]:
        if "registry" not in package.get("source", {}):
            continue
        choices = [wheel for wheel in package["wheels"]
                   if parse_wheel_filename(wheel["url"].rsplit("/", 1)[1])[3] & tags]
        assert choices, f"no host wheel for {package['name']}"
        wheel = choices[0]
        with urllib.request.urlopen(wheel["url"], timeout=60) as response:
            data = response.read()
        assert "sha256:" + hashlib.sha256(data).hexdigest() == wheel["hash"]
        (wheelhouse / wheel["url"].rsplit("/", 1)[1]).write_bytes(data)
        versions[package["name"]] = package["version"]
    return wheelhouse, versions


@pytest.fixture
def isolated_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.chdir(tmp_path)
    # Neither the caller's config nor a populated uv cache may supply this graph.
    (tmp_path / "uv.toml").write_text('required-version = "<0.1"\n', encoding="utf-8")
    from pm.packages import uv_cache_dir
    assert not any(entry.name != ".seeded" for entry in uv_cache_dir().iterdir())
    uv = shutil.which("uv")
    assert uv, "the wheelhouse staging test requires uv"
    return Path(uv)


@pytest.mark.platforms("linux")
def test_offline_wheelhouse_runtime_survives_sealing_and_move(
    tmp_path, isolated_builder, locked_wheelhouse,
):
    wheelhouse, versions = locked_wheelhouse
    root = tmp_path / "payload"
    python = root / "tools/python/bin/python"
    python.parent.mkdir(parents=True)
    shutil.copy2(Path(sys._base_executable).resolve(), python)
    executable = stage_runtime(isolated_builder, python, root / "pm-runtime",
                               wheelhouse=wheelhouse, offline=True)
    assert executable.is_file()
    seal_pm_runtime(root, python)
    moved = tmp_path / "installed elsewhere"
    root.rename(moved)
    runtime = moved / "pm-runtime"
    marker = json.loads((runtime / "pm-runtime.json").read_text(encoding="utf-8"))
    probe = """
import importlib.metadata, importlib.util, json, sys
sys.path.insert(0, sys.argv[1])
from packaging.utils import canonicalize_name
from ruamel.yaml import YAML
import packaging, tomli_w, _ruamel_yaml
assert YAML(typ='safe').load('isolated: true') == {'isolated': True}
assert importlib.util.find_spec('openai') is None
assert importlib.util.find_spec('yaml') is None
print(json.dumps({canonicalize_name(d.metadata['Name']): d.version
                  for d in importlib.metadata.distributions(path=[sys.argv[1]])}))
"""
    result = subprocess.run(
        [str(runtime / marker["python"]), "-I", "-S", "-B", "-c", probe,
         str(runtime / marker["sitePackages"])],
        cwd=tmp_path, env=runtime_environment(), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == versions


@pytest.mark.platforms("linux")
def test_offline_wheelhouse_rejects_missing_transitive_wheel(
    tmp_path, isolated_builder, locked_wheelhouse, capfd,
):
    from pm.package import InstallError

    wheelhouse, _ = locked_wheelhouse
    incomplete = tmp_path / "incomplete wheelhouse"
    shutil.copytree(wheelhouse, incomplete)
    native_wheel, = incomplete.glob("ruamel_yaml_clib-*.whl")
    native_wheel.unlink()
    destination = tmp_path / "pm-runtime"
    with pytest.raises(InstallError, match="pip exited"):
        stage_runtime(isolated_builder, Path(sys.executable), destination,
                      wheelhouse=incomplete, offline=True)
    assert "ruamel-yaml-clib" in capfd.readouterr().err
    assert not (destination / "pm-runtime.json").exists()
