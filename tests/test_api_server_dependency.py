from pathlib import Path
import tomllib


def _project_metadata() -> dict:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        return tomllib.load(handle)


def _package_names(requirements: list[str]) -> set[str]:
    names = set()
    for requirement in requirements:
        name = requirement.split(";", 1)[0].split("[", 1)[0].split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0].strip().lower()
        if name:
            names.add(name)
    return names


def test_api_server_runtime_dependencies_are_core_dependencies():
    """API_SERVER_ENABLED must work without installing the messaging extra."""
    project = _project_metadata()["project"]
    base_deps = _package_names(project["dependencies"])

    assert "aiohttp" in base_deps
