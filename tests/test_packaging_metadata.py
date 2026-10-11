"""Independent core/optional dependency and reviewed CVE policies."""
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
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


def _exclude_newer_exempt(metadata):
    return {canonicalize_name(name) for name in metadata["tool"]["uv"]["exclude-newer-package"]}


def _is_exact_pin(spec):
    # ``==1.2.*`` shares the operator but still floats, so it keeps the quarantine.
    return spec.operator == "==" and not spec.version.endswith(".*")


def _exact_pins(specs):
    for requirement in map(Requirement, specs):
        if any(_is_exact_pin(spec) for spec in requirement.specifier):
            yield requirement


def test_exact_pins_skip_wildcard_equality():
    pins = {req.name for req in _exact_pins(
        ["exact==1.2.3", "prefix==1.2.*", "ranged>=1,<2", "mixed>=1,==1.4.0"])}
    assert pins == {"exact", "mixed"}


def test_exact_pinned_deps_exempt_from_exclude_newer():
    # An exact pin cannot float, so the 14-day cutoff only bricks it: a venv
    # that predates a release cannot resolve a pin published days before it.
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    specs = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        specs += extra
    for group in metadata["dependency-groups"].values():
        specs += [spec for spec in group if isinstance(spec, str)]
    exempt = _exclude_newer_exempt(metadata)
    missing = sorted({str(req) for req in _exact_pins(specs)
                      if canonicalize_name(req.name) not in exempt})
    assert not missing, missing


def test_build_system_requires_exempt_from_exclude_newer():
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    exempt = _exclude_newer_exempt(metadata)
    missing = [str(req) for req in _exact_pins(metadata["build-system"]["requires"])
               if canonicalize_name(req.name) not in exempt]
    assert not missing, missing


def test_exempt_cryptography_runtime_deps_are_exempt_too():
    # cryptography's cffi floor can sit inside the window (#135483); exempting
    # cryptography alone leaves the resolver with no cffi to pick.
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    exempt = _exclude_newer_exempt(metadata)
    assert "cryptography" in exempt
    assert {"cffi", "pycparser"} <= exempt
