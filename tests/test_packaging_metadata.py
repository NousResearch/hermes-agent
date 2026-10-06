"""Independent core/optional dependency and reviewed CVE policies."""
import tomllib
from pathlib import Path

import pytest
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


# ---------------------------------------------------------------------------
# Transitive-pin consistency: the locked `mcp` must satisfy what the locked
# `claude-agent-sdk` itself requires.
#
# Direct-pin checks compare `name==version` strings across extras and cannot
# see a TRANSITIVE contradiction: `claude-agent-sdk` declares its own `mcp`
# range, and a lock that pins `mcp` outside that range is unreproducible
# (relocking refuses to regenerate it) while a lock check still passes because
# nothing re-resolves. That is exactly how the `[claude-agent-sdk]` extra
# shipped with an SDK that pinned `mcp<2` next to the `mcp==2.0.0` the stdio
# server needs (#65982). The SDK's requirement is read from the installed
# distribution, so this test is hermetic and skips where the extra is not
# installed.
# ---------------------------------------------------------------------------


def _locked_versions(package: str) -> set[str]:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {row["version"] for row in lock["package"]
            if canonicalize_name(row["name"]) == canonicalize_name(package)}


def test_locked_mcp_satisfies_claude_agent_sdk_requirement():
    from importlib import metadata

    try:
        requires = metadata.requires("claude-agent-sdk") or []
    except metadata.PackageNotFoundError:
        pytest.skip("claude-agent-sdk extra is not installed in this environment")

    mcp_req = next(
        (Requirement(r) for r in requires if canonicalize_name(Requirement(r).name) == "mcp"),
        None,
    )
    assert mcp_req is not None, "claude-agent-sdk no longer declares an mcp requirement"

    locked = _locked_versions("mcp")
    assert len(locked) == 1, f"uv.lock must pin exactly one mcp, found {sorted(locked)}"
    locked_version = next(iter(locked))
    assert mcp_req.specifier.contains(locked_version, prereleases=True), (
        f"uv.lock pins mcp=={locked_version} but the locked claude-agent-sdk "
        f"requires mcp{mcp_req.specifier} — the lock is unreproducible; bump the "
        f"claude-agent-sdk pin (>=0.2.140 accepts mcp 2.x) and run `hermes pm lock`."
    )


def test_claude_agent_sdk_pin_accepts_the_mcp_major_the_extra_pins():
    """Hermetic floor for the transitive check above.

    The general test environment may omit `[claude-agent-sdk]`; the dedicated
    SDK packaging CI job installs it and runs the importlib-based check above.
    This guard also covers environments without the extra. Encode the fact that
    makes it resolvable at all: `claude-agent-sdk` first admitted `mcp` 2.x in
    0.2.140 (`mcp>=1.23.0,<3.0.0`; 0.2.120–0.2.139 pinned `mcp<2`). If the
    extra pins an `mcp` 2.x alongside an SDK older than that floor, the lock
    cannot resolve it and installing the extra breaks the hermes-tools stdio
    server (#65982).
    """
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pins: dict[str, set[str]] = {}
    for requirement in map(Requirement, data["project"]["optional-dependencies"]["claude-agent-sdk"]):
        exact = {spec.version for spec in requirement.specifier if spec.operator == "=="}
        pins.setdefault(canonicalize_name(requirement.name), set()).update(exact)
    sdk = pins.get("claude-agent-sdk")
    mcp = pins.get("mcp")
    assert sdk and len(sdk) == 1, "claude-agent-sdk must be exact-pinned in its extra"
    assert mcp and len(mcp) == 1, "the claude-agent-sdk extra must exact-pin mcp"
    sdk_version = Version(next(iter(sdk)))
    mcp_version = Version(next(iter(mcp)))
    if mcp_version.major >= 2:
        assert sdk_version >= Version("0.2.140"), (
            f"claude-agent-sdk=={sdk_version} pins mcp<2, but the extra pins "
            f"mcp=={mcp_version}; the first SDK admitting mcp 2.x is 0.2.140."
        )
