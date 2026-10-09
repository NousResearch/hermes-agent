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


def test_exact_pinned_deps_exempt_from_exclude_newer():
    """The workspace quarantine exempts every declaration-layer exact pin (#135589).

    pyproject.toml documents exact pins as zero-float ("Exempting exact pins is pure
    brick-risk removal at no supply-chain cost") and references this test by name.
    Cross-checks with packaging's own parser so the stdlib regex in pm.workspace
    cannot silently drift from PEP 508.
    """
    import re

    from pm.workspace import _core_release_quarantine

    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    cutoff = metadata["tool"]["uv"]["exclude-newer"]
    _core_release_quarantine(metadata, REPO_ROOT / "uv.lock")
    policy = metadata["tool"]["uv"]["exclude-newer-package"]

    def normalized(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    declarations = list(metadata["project"]["dependencies"])
    for requirements in metadata["project"]["optional-dependencies"].values():
        declarations.extend(requirements)
    for requirements in metadata["dependency-groups"].values():
        declarations.extend(requirements)
    declarations.extend(metadata["build-system"]["requires"])
    declarations.extend(metadata["tool"]["uv"]["override-dependencies"])

    pinned = set()
    for requirement in map(Requirement, declarations):
        specifiers = list(requirement.specifier)
        if len(specifiers) == 1 and specifiers[0].operator in ("==", "==="):
            pinned.add(normalized(requirement.name))

    # The reporter's brick list plus packaging: none were on the hand-written list.
    assert {"resvg-py", "tomli-w", "truststore", "ddgs", "distlib", "packaging"} <= pinned
    for name in pinned:
        # A hand-written dated entry still wins; the pin must never fall to the cutoff.
        assert policy[normalized(name)] != cutoff, name

    # A floating dependency keeps the cutoff: the quarantine only skips what cannot move.
    assert "urllib3" not in pinned and policy[normalized("urllib3")] == cutoff
