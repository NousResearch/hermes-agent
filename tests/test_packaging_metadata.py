"""Independent core/optional dependency and reviewed CVE policies."""
import re
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


def _pep503(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def test_old_exact_pins_are_exempt_from_exclude_newer():
    # Exact pins whose locked release predates 2024 point at frozen-era
    # indexes whose upload dates uv frequently cannot see. Under the rolling
    # exclude-newer a cold-cache resolve then errors out instead of solving
    # (pilk==0.2.4, #136868; the defusedxml family, #80387), so every such pin
    # must carry a tool.uv.exclude-newer-package exemption. The pin cannot
    # float, so the exemption adds no supply-chain exposure.
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    published = {}
    for package in lock["package"]:
        stamps = [wheel["upload-time"] for wheel in package.get("wheels", []) if "upload-time" in wheel]
        sdist = package.get("sdist")
        if isinstance(sdist, dict) and "upload-time" in sdist:
            stamps.append(sdist["upload-time"])
        if stamps:
            published[_pep503(package["name"])] = min(stamps)
    exemptions = {_pep503(name) for name in manifest["tool"]["uv"]["exclude-newer-package"]}

    requirements = list(manifest["project"]["dependencies"])
    for specs in manifest["project"]["optional-dependencies"].values():
        requirements.extend(specs)
    checked = set()
    for requirement in map(Requirement, requirements):
        if len(requirement.specifier) != 1:
            continue
        pin = next(iter(requirement.specifier))
        if pin.operator != "==":
            continue
        name = _pep503(requirement.name)
        release = published.get(name)
        if release and release < "2024-01-01T00:00:00Z":
            checked.add((name, pin.version, release))
    assert checked, "expected at least one pre-2024 exact pin to be checked"
    unexempted = [pin for pin in checked if pin[0] not in exemptions]
    assert not unexempted, f"pre-2024 exact pins missing exclude-newer exemption: {unexempted}"
