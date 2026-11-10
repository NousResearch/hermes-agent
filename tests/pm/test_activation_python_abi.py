"""Dependency activation never crosses a CPython minor-version boundary."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("minor_delta", "site_exists", "stderr_none", "expected", "warns", "error"),
    [
        (-1, True, False, ["False", "True", "True", "True"], True, None),
        (0, True, False, ["True", "False", "False", "False"], False, None),
        (1, True, False, ["False", "True", "True", "True"], True, None),
        (1, False, False, ["False", "True", "True", "True"], True, None),
        (-1, True, True, ["False", "True", "True", "True"], True, None),
        (0, False, False, None, False, "has no site-packages"),
    ],
)
def test_dependency_generation_activation_respects_the_running_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    minor_delta: int,
    site_exists: bool,
    stderr_none: bool,
    expected: list[str] | None,
    warns: bool,
    error: str | None,
) -> None:
    """Matching activates; either foreign direction stays out, even when incomplete or repeated."""
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
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        f"version_info = {foreign[0]}.{foreign[1]}.0\n", encoding="utf-8"
    )
    if site_exists:
        site.mkdir(parents=True)
        (site / "foreign_abi_only.py").write_text("ORIGIN = 'foreign'\n", encoding="utf-8")
    (generation / ".lease-managed").touch()
    (state / "facts.json").write_text(
        json.dumps({"schema": 1, "packages": {"venv": {"environment": str(venv)}}}),
        encoding="utf-8",
    )

    code = """
import importlib.util
import os
import sys
from pathlib import Path

repo, root, foreign_site = map(Path, sys.argv[1:4])
sys.path.insert(0, str(repo))
from pm.environments import activate_dependencies

if sys.argv[4] == "stderr-none":
    sys.stderr = None
try:
    activate_dependencies(root)
    activate_dependencies(root)
except RuntimeError as exc:
    print(f"ERROR: {exc}")
    raise SystemExit(0)
paths = [Path(entry).resolve() for entry in sys.path if entry]
print(foreign_site.resolve() in paths)
print(importlib.util.find_spec("foreign_abi_only") is None)
print(str(foreign_site.resolve()) not in os.environ.get("PYTHONPATH", "").split(os.pathsep))
leases = foreign_site.parents[3] / ".leases"
print(not leases.is_dir() or not any(leases.iterdir()))
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
            "stderr-none" if stderr_none else "stderr-open",
        ],
        cwd=root.parent,
        env=dict(os.environ),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    if error is not None:
        assert result.stdout.startswith("ERROR: ") and error in result.stdout
    else:
        assert result.stdout.splitlines() == expected
    assert ("keeping the Python" in result.stderr) is warns
