"""Independent core/optional dependency and reviewed CVE policies."""
import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]


def _normalized(name: str) -> str:
    """PEP 503 form: the exemption table keys and uv agree on this spelling."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _exact_pins_by_table() -> dict[str, list[Requirement]]:
    """``{table: [Requirement, ...]}`` for every ``==`` pin across all dependency tables."""
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    tables = {"dependencies": project["dependencies"]}
    tables.update(
        {f"optional-dependencies.{extra}": specs
         for extra, specs in project["optional-dependencies"].items()})
    build = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["build-system"]
    tables["build-system.requires"] = build["requires"]
    pinned = {}
    for table, specs in tables.items():
        for spec in specs:
            requirement = Requirement(spec)
            pins = list(requirement.specifier)
            if len(pins) == 1 and pins[0].operator == "==":
                pinned.setdefault(table, []).append(requirement)
    return pinned


def test_test_dependencies_are_group_only_in_manifest_and_lock():
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    hermes = next(package for package in lock["package"] if package["name"] == manifest["project"]["name"])
    assert manifest["tool"]["uv"]["default-groups"] == []
    assert "dev" in manifest["dependency-groups"]
    assert "dev" not in manifest["project"]["optional-dependencies"]
    assert "dev" in hermes["dev-dependencies"]
    assert "dev" not in hermes.get("optional-dependencies", {})


def test_core_and_optional_speech_dependencies():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    core = {Requirement(dep).name for dep in project["dependencies"]}
    assert "packaging" in core  # Runtime code imports it directly, not transitively.
    assert "faster-whisper" not in core
    assert "faster-whisper" in {
        Requirement(dep).name for dep in project["optional-dependencies"]["stt-whisper"]
    }


def test_starlette_server_pins_and_lock_exclude_cve_2026_48710():
    # BadHost's reviewed fixed boundary is independent of today's exact pin.
    floor = Version("1.0.1")
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    found = set()
    for extra, specs in metadata["project"]["optional-dependencies"].items():
        for requirement in map(Requirement, specs):
            if requirement.name != "starlette":
                continue
            pins = list(requirement.specifier)
            assert len(pins) == 1 and pins[0].operator == "==", (extra, requirement)
            assert Version(pins[0].version) >= floor, (extra, requirement)
            found.add(extra)
    assert {"web", "mcp", "computer-use"} <= found
    dev = [req for req in map(Requirement, metadata["dependency-groups"]["dev"])
           if req.name == "starlette"]
    assert len(dev) == 1
    pins = list(dev[0].specifier)
    assert len(pins) == 1 and pins[0].operator == "==" and Version(pins[0].version) >= floor
    versions = [Version(row["version"]) for row in lock["package"] if row["name"] == "starlette"]
    assert versions and all(version >= floor for version in versions)

def test_build_system_requires_exempt_from_exclude_newer():
    """Guard cited by the ``exclude-newer`` comment block: an exact-pinned build requirement
    cannot move without a reviewed bump, so the rolling cutoff can only brick the build
    ("No solution found when resolving: setuptools==X.Y.Z", #78227 family)."""
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    exempt = {_normalized(name) for name in manifest["tool"]["uv"].get("exclude-newer-package", {})}
    for requirement in _exact_pins_by_table().get("build-system.requires", []):
        assert _normalized(requirement.name) in exempt, requirement.name


def test_exact_pinned_deps_exempt_from_exclude_newer():
    """Guard cited by the ``exclude-newer`` comment block: every exact pin is a reviewed
    version, so the rolling 14-day cutoff adds no supply-chain protection for it and only
    creates the release-day brick ("no version of X==Y" until the window passes)."""
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    exempt = {_normalized(name) for name in manifest["tool"]["uv"].get("exclude-newer-package", {})}
    missing = sorted(
        f"{table}: {requirement}"
        for table, requirements in _exact_pins_by_table().items()
        for requirement in requirements
        if _normalized(requirement.name) not in exempt
    )
    assert not missing, "\n".join(missing)
