"""Dependency activation never crosses a CPython minor-version boundary."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("minor_delta", [-1, 1])
def test_foreign_python_generation_never_enters_the_running_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, minor_delta: int,
) -> None:
    """A lower or higher CPython generation stays out, even after repeated activation."""
    from pm import environments

    root = tmp_path / "repo"
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    state = environments.install_state_dir(root)
    generation = state / "environments" / "foreign"
    venv = generation / "venv"
    foreign = (sys.version_info.major, sys.version_info.minor + minor_delta)
    site = venv / (
        "Lib/site-packages"
        if os.name == "nt"
        else f"lib/python{foreign[0]}.{foreign[1]}/site-packages"
    )
    site.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        f"version_info = {foreign[0]}.{foreign[1]}.0\n", encoding="utf-8"
    )
    (site / "foreign_abi_only.py").write_text("ORIGIN = 'foreign'\n", encoding="utf-8")
    (state / "facts.json").write_text(
        json.dumps({"schema": 1, "packages": {"venv": {"environment": str(venv)}}}),
        encoding="utf-8",
    )

    code = """
import importlib.util
import os
import sys
from pathlib import Path

repo, root, foreign_site = map(Path, sys.argv[1:])
sys.path.insert(0, str(repo))
from pm.environments import activate_dependencies

activate_dependencies(root)
activate_dependencies(root)
paths = [Path(entry).resolve() for entry in sys.path if entry]
print(foreign_site.resolve() in paths)
print(importlib.util.find_spec("foreign_abi_only") is None)
print(str(foreign_site.resolve()) not in os.environ.get("PYTHONPATH", "").split(os.pathsep))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(Path(__file__).resolve().parents[2]),
            str(root),
            str(site),
        ],
        cwd=root.parent,
        env=dict(os.environ),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["False", "True", "True"]
