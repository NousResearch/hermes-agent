"""Independent core/optional dependency and reviewed CVE policies."""
import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]


def _canonical(name):
    # PEP 503 normalisation, matching how uv keys exclude-newer-package.
    return re.sub(r"[-_.]+", "-", name.lower())


def _exact_pinned_names(requirements):
    pinned = set()
    for requirement in map(Requirement, requirements):
        specs = list(requirement.specifier)
        if any(spec.operator == "==" for spec in specs):
            pinned.add(_canonical(requirement.name))
    return pinned


def test_build_system_requires_exempt_from_exclude_newer():
    # uv applies exclude-newer to build-system.requires too, so a resolver
    # that cannot see an upload date filters the pinned setuptools/wheel and
    # the package cannot even build (#78227, #75992, #76020, #96488).
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    exempt = {_canonical(name) for name in manifest["tool"]["uv"]["exclude-newer-package"]}
    assert _exact_pinned_names(manifest["build-system"]["requires"]) <= exempt


def test_exact_pinned_deps_exempt_from_exclude_newer():
    # An exact pin cannot float, so the exclude-newer quarantine adds zero
    # supply-chain protection for it — while indexes whose PEP 691 simple API
    # omits per-file upload-time (e.g. the Tsinghua mirror) treat the pin as
    # newer than the cutoff and brick resolution outright (#132558).
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    exempt = {_canonical(name) for name in manifest["tool"]["uv"]["exclude-newer-package"]}
    pinned = _exact_pinned_names(manifest["project"]["dependencies"])
    for specs in manifest["project"].get("optional-dependencies", {}).values():
        pinned |= _exact_pinned_names(specs)
    for specs in manifest.get("dependency-groups", {}).values():
        pinned |= _exact_pinned_names(specs)
    assert pinned <= exempt


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
