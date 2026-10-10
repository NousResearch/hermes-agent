"""Independent core/optional dependency and reviewed CVE policies."""
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]


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


# Reviewed fixed boundaries (independent of today's exact pins). A lock
# regeneration or pin edit that slides any of these back below its floor
# reintroduces a published advisory.
_LOCK_SECURITY_FLOORS = {
    "pyjwt": "2.15.0",      # PYSEC-2026-4140..4152 (2.14.0), CVE-2026-101918 (2.15.0)
    "httpx2": "2.12.0",     # PYSEC-2026-3845..3849 (GHSA-7mj9-2mp8-4m2p family)
    "httpcore2": "2.10.0",  # PYSEC-2026-3844
    "urllib3": "2.8.0",     # PYSEC-2026-4175/4176/4177
    "tornado": "6.5.9",     # GHSA-chx6-46f5-w4vp, GHSA-c2m8-h5v5-343r, GHSA-3hv7-mjh2-fv65
}


def test_lock_excludes_published_advisories_for_http_and_jwt_stack():
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    for name, floor in _LOCK_SECURITY_FLOORS.items():
        versions = [Version(row["version"]) for row in lock["package"] if row["name"] == name]
        assert versions, name
        assert all(version >= Version(floor) for version in versions), (name, versions)


def test_manifest_pins_respect_http_and_jwt_security_floors():
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    groups = dict(metadata["project"]["optional-dependencies"])
    groups["<core>"] = metadata["project"]["dependencies"]
    groups["<dev>"] = metadata["dependency-groups"]["dev"]
    seen = set()
    for group, specs in groups.items():
        for requirement in map(Requirement, (s for s in specs if isinstance(s, str))):
            floor = _LOCK_SECURITY_FLOORS.get(requirement.name.lower())
            if floor is None:
                continue
            seen.add(requirement.name.lower())
            # Every allowed version must be at or above the fixed boundary.
            lower = [s for s in requirement.specifier if s.operator in ("==", ">=")]
            assert lower, (group, str(requirement))
            assert all(Version(s.version) >= Version(floor) for s in lower), (group, str(requirement))
    assert {"pyjwt", "httpx2", "urllib3"} <= seen
