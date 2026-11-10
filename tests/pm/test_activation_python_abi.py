"""Dependency activation never crosses a CPython minor-version boundary."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    (
        "minor_delta",
        "site_exists",
        "selection_mode",
        "stderr_mode",
        "expected",
        "warns",
        "error",
    ),
    [
        (-1, True, "facts", "open", ["False", "True", "True", "True"], True, None),
        (0, True, "facts", "open", ["True", "False", "False", "False"], False, None),
        (1, True, "facts", "open", ["False", "True", "True", "True"], True, None),
        (1, False, "facts", "open", ["False", "True", "True", "True"], True, None),
        (-1, True, "facts", "none", ["False", "True", "True", "True"], True, None),
        (-1, True, "facts", "closed", ["False", "True", "True", "True"], False, None),
        (0, False, "facts", "open", None, False, "has no site-packages"),
        (0, False, "direct", "open", ["False", "True", "True", "True"], False, None),
    ],
)
def test_dependency_generation_activation_respects_the_running_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    minor_delta: int,
    site_exists: bool,
    selection_mode: str,
    stderr_mode: str,
    expected: list[str] | None,
    warns: bool,
    error: str | None,
) -> None:
    """Only matching activates; every rejected selection releases its generation."""
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
    if selection_mode == "facts":
        (state / "facts.json").write_text(
            json.dumps({"schema": 1, "packages": {"venv": {"environment": str(venv)}}}),
            encoding="utf-8",
        )

    code = """
import importlib.util
import os
import sys
from pathlib import Path

repo, root, foreign_site, generation = map(Path, sys.argv[1:5])
sys.path.insert(0, str(repo))
from pm import environments

if sys.argv[6] == "direct":
    environments.committed_venv = lambda _root: generation / "venv"
if sys.argv[5] == "none":
    sys.stderr = None
elif sys.argv[5] == "closed":
    sys.stderr.close()
leases = generation / ".leases"
try:
    environments.activate_dependencies(root)
    environments.activate_dependencies(root)
except RuntimeError as exc:
    print(f"ERROR: {exc}")
    print(not leases.is_dir() or not any(leases.iterdir()))
    raise SystemExit(0)
paths = [Path(entry).resolve() for entry in sys.path if entry]
print(foreign_site.resolve() in paths)
print(importlib.util.find_spec("foreign_abi_only") is None)
print(str(foreign_site.resolve()) not in os.environ.get("PYTHONPATH", "").split(os.pathsep))
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
            str(generation),
            stderr_mode,
            selection_mode,
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
        lines = result.stdout.splitlines()
        assert lines[0].startswith("ERROR: ") and error in lines[0]
        assert lines[1:] == ["True"]
    else:
        assert result.stdout.splitlines() == expected
    assert ("keeping the Python" in result.stderr) is warns
