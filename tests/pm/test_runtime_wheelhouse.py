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
from scripts.bundles.payload import seal_pm_runtime
from tests.pm._fixtures import stage_host_python


@pytest.fixture(scope="module")
def locked_wheelhouse(tmp_path_factory):
    """Download host wheels first; only the subsequent stage runs offline."""
    wheelhouse = tmp_path_factory.mktemp("pm-wheelhouse")
    project = Path(__file__).resolve().parents[2] / "pm"
    lock = tomllib.loads((project / "uv.lock").read_text(encoding="utf-8-sig"))
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
    tmp_path, isolated_builder, locked_wheelhouse, monkeypatch,
):
    wheelhouse, versions = locked_wheelhouse
    root = tmp_path / "payload"
    python = stage_host_python(root / "tools/python/bin/python")
    from pm import stage_manager_runtime
    from pm.lock import _write

    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (isolated_builder, python))
    executable = stage_manager_runtime(python=python, destination=root / "pm-runtime",
                                       wheelhouse=wheelhouse, offline=True)
    assert executable.is_file()
    marker_path = root / "pm-runtime/pm-runtime.json"
    assert marker_path.stat().st_mode & 0o777 == 0o600
    assert (root / "pm-runtime/.lock").is_file()
    seal_pm_runtime(root, python)
    assert marker_path.stat().st_mode & 0o777 == 0o644
    assert not (root / "pm-runtime/.lock").exists()
    private = tmp_path / "mutable/selected.json"
    _write(private, {"runtime": "private"})
    assert private.stat().st_mode & 0o777 == 0o600
    moved = tmp_path / "installed elsewhere"
    root.rename(moved)
    runtime = moved / "pm-runtime"
    marker = json.loads((runtime / "pm-runtime.json").read_text(encoding="utf-8-sig"))
    probe = """
import importlib.metadata, importlib.util, json, sys
sys.path.insert(0, sys.argv[1])
from packaging.utils import canonicalize_name
from ruamel.yaml import YAML
import packaging, tomli_w
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
    from pm import paths
    from pm.runtime import runtime_command
    repo = moved / "hermes-agent"
    repo.mkdir()
    (moved / "manifest.json").write_text('{"repo":"hermes-agent"}', encoding="utf-8")
    script = repo / "probe.py"
    script.write_text("import sys,json; print(json.dumps(sys.path))", encoding="utf-8")
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(paths, "repo_root", lambda: repo)
        child = subprocess.run(runtime_command(script), cwd=tmp_path, env=runtime_environment(),
                               capture_output=True, text=True, check=True, timeout=30)
    entries = json.loads(child.stdout)
    assert str((runtime / marker["sitePackages"]).resolve()) in entries
    recorded_site = (runtime / marker["sitePackages"]).resolve()
    assert not any(Path(entry).name in {"site-packages", "dist-packages"}
                   and Path(entry).resolve() != recorded_site for entry in entries)


@pytest.mark.platforms("linux")
def test_offline_wheelhouse_rejects_missing_locked_wheel(
    tmp_path, isolated_builder, locked_wheelhouse, capfd, monkeypatch,
):
    from pm.package import InstallError

    wheelhouse, _ = locked_wheelhouse
    incomplete = tmp_path / "incomplete wheelhouse"
    shutil.copytree(wheelhouse, incomplete)
    yaml_wheel = next(path for path in incomplete.glob("*.whl")
                      if parse_wheel_filename(path.name)[0] == "ruamel-yaml")
    yaml_wheel.unlink()
    destination = tmp_path / "pm-runtime"
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (isolated_builder, Path(sys.executable)))
    with pytest.raises(InstallError, match="pip exited"):
        from pm import stage_manager_runtime
        stage_manager_runtime(python=Path(sys.executable), destination=destination,
                              wheelhouse=incomplete, offline=True)
    assert "ruamel-yaml" in capfd.readouterr().err
    assert not destination.exists()
