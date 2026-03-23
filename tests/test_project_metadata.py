"""Regression tests for packaging metadata in pyproject.toml."""

from pathlib import Path
import tomllib


def _load_optional_dependencies():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return project["optional-dependencies"]


def test_all_extra_includes_matrix_dependency():
    optional_dependencies = _load_optional_dependencies()

    assert "matrix" in optional_dependencies
    assert "hermes-agent[matrix]" in optional_dependencies["all"]
