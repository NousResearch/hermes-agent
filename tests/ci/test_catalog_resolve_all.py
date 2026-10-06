"""The catalog gate must reject every unresolved plugin, not just new failures."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.pm._fixtures import _wheel
from tests.pm.test_environment_build import locked_project as locked_project


@pytest.fixture
def catalog_gate(locked_project, tmp_path, monkeypatch):
    core, uv, env = locked_project
    _wheel(tmp_path / "wheels", "member_dep", "2.0")
    manifest = core / "pyproject.toml"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace(
        '[tool.uv.workspace]\nmembers=["member"]\n', ""), encoding="utf-8")
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (uv, Path(sys.executable)))
    for key in ("UV_CACHE_DIR", "UV_OFFLINE"):
        monkeypatch.setenv(key, env[key])
    script = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "catalog_resolve_all.py"
    spec = importlib.util.spec_from_file_location("catalog_resolve_all", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    catalog = tmp_path / "catalog"
    cache = tmp_path / "clones"
    catalog.mkdir()
    cache.mkdir()

    def run(requirements, *, baseline=False):
        for name, dependency in requirements.items():
            plugin = cache / name
            plugin.mkdir()
            (plugin / "pyproject.toml").write_text(
                f'[project]\nname="{name}"\nversion="1"\nrequires-python=">=3.11"\n'
                f'dependencies=["{dependency}"]\n[tool.uv]\npackage=false\n', encoding="utf-8")
            (plugin / "plugin.yaml").write_text(f"name: {name}\n", encoding="utf-8")
            git = ["git", "-C", str(plugin)]
            subprocess.run([*git, "init", "-q"], check=True, stdin=subprocess.DEVNULL)
            subprocess.run([*git, "add", "."], check=True, stdin=subprocess.DEVNULL)
            subprocess.run([*git, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                            "-c", "commit.gpgsign=false", "commit", "-qm", "plugin"],
                           check=True, stdin=subprocess.DEVNULL)
            sha = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True,
                                 text=True, stdin=subprocess.DEVNULL).stdout.strip()
            (catalog / f"{name}.yaml").write_text(
                f"name: {name}\nrepo: https://example.invalid/{name}\nsha: {sha}\n"
                "description: resolver fixture\nmaintainer: fixture\n", encoding="utf-8")
        report = tmp_path / "report.json"
        argv = [str(script), "--source", str(core), "--catalog", str(catalog),
                "--cache", str(cache), "--report", str(report), "--jobs", "1"]
        if baseline:
            known = tmp_path / "base.json"
            known.write_text(json.dumps({"failures": {name: "already broken" for name in requirements}}),
                             encoding="utf-8")
            argv += ["--baseline", str(known)]
        monkeypatch.setattr(sys, "argv", argv)
        return module.main(), report

    return run


@pytest.mark.parametrize("requirements,failing", [
    ({"healthy": "member-dep==1.0"}, False),
    ({"broken": "base-dep==9.9"}, True),
    ({"a": "base-dep>=1.0", "b": "base-dep<=1.0"}, False),
    ({"a": "member-dep<2.0", "b": "member-dep>=2.0"}, True),
])
def test_catalog_exit_status_matches_real_resolution(catalog_gate, capsys, requirements, failing):
    code, report = catalog_gate(requirements)
    result = json.loads(report.read_text(encoding="utf-8"))
    assert code == int(failing)
    assert bool(result["failures"]) == failing
    assert len(result["locked_together"]) + len(result["failures"]) == len(requirements)
    output = capsys.readouterr().out
    if failing:
        assert "::error" in output
        assert "::warning" not in output


def test_baseline_cannot_exempt_an_already_broken_plugin(catalog_gate):
    with pytest.raises(SystemExit) as error:
        catalog_gate({"broken": "base-dep==9.9"}, baseline=True)
    assert error.value.code == 2
