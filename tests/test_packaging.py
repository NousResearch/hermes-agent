import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_hermes_state_modules_are_packaged():
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)

    packaged_modules = set(pyproject["tool"]["setuptools"]["py-modules"])

    state_modules = {
        path.stem
        for path in REPO_ROOT.glob("hermes_state*.py")
    }

    missing = state_modules - packaged_modules

    assert not missing, (
        "Top-level hermes_state modules missing from "
        f"[tool.setuptools].py-modules: {sorted(missing)}"
    )
