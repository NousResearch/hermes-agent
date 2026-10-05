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


def test_excluded_override_keeps_its_thinc_branch_off_spacy_3():
    """An override-dependency is dropped by its marker, but the resolver still
    walks the excluded package's own requirements while backtracking. Left
    unpinned, `spacy-curated-transformers` floats to a 2.x release that demands
    `thinc>=9`, which no spacy 3.x can satisfy; the resolver then walks spacy
    itself down the whole 3.x line to 2.0.17, whose setup.py imports
    `msvccompiler` from the distutils that setuptools no longer vendors, so
    `uv lock --upgrade` dies building a 2017 sdist instead of resolving the
    3.8.16 that is already locked. Pinning the override keeps the excluded
    branch at the one release whose thinc range spacy 3.x can share."""
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))

    excluded = next(
        requirement for requirement in map(Requirement, metadata["tool"]["uv"]["override-dependencies"])
        if requirement.name == "spacy-curated-transformers"
    )
    assert list(excluded.specifier), "an unpinned override re-opens the spacy 2.x backtrack"

    locked = next(
        row for row in lock["manifest"]["overrides"]
        if row["name"] == "spacy-curated-transformers"
    )
    assert locked.get("specifier") == "==0.3.1", locked

    spacy = next(row for row in lock["package"] if row["name"] == "spacy")
    assert Version(spacy["version"]) >= Version("3.8"), spacy["version"]
    thinc = next(row for row in lock["package"] if row["name"] == "thinc")
    assert Version(thinc["version"]) < Version("9"), thinc["version"]


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
