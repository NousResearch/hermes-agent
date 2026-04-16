"""Regression tests for packaging metadata in pyproject.toml."""

from pathlib import Path
import tomllib


def _load_optional_dependencies():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return project["optional-dependencies"]


def test_matrix_extra_exists_but_excluded_from_all():
    """matrix-nio[e2e] depends on python-olm which is upstream-broken on modern
    macOS (archived libolm, C++ errors with Clang 21+).  The [matrix] extra is
    kept for opt-in install but deliberately excluded from [all] so one broken
    upstream dep doesn't nuke every other extra during ``hermes update``."""
    optional_dependencies = _load_optional_dependencies()

    assert "matrix" in optional_dependencies
    assert "hermes-agent[matrix]" not in optional_dependencies["all"]
