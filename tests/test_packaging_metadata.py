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


# The lock's `tool.uv.environments` entry pins python only, so uv resolves a
# split for every sys_platform marker it knows. Any sys_platform restriction
# added there must revisit this list.
# ponytail: one representative machine per OS rather than a full arch matrix —
# the question here is "does a distribution exist for this platform at all",
# and no playwright-shaped failure is arch-conditional. A package counts as
# covered when ANY locked version has an sdist or a matching wheel; the
# resolver is free to pick that version for the split, and keying on one
# version would make the assertion a lock snapshot.
LOCK_PLATFORMS = {
    "linux": {"platform_system": "Linux", "platform_machine": "x86_64"},
    "darwin": {"platform_system": "Darwin", "platform_machine": "arm64"},
    "win32": {"platform_system": "Windows", "platform_machine": "AMD64"},
    "android": {"platform_system": "Linux", "platform_machine": "aarch64"},
}


def _wheel_covers(platform_tag: str, system: str) -> bool:
    """Whether a wheel platform tag (the last ``-`` field) installs on `system`."""
    base = platform_tag.split(".")[0]  # manylinux_2_17_x86_64 -> manylinux_2_17_x86_64
    if base == "any":
        return True
    if base.startswith(("linux", "manylinux", "musllinux")):
        return system == "linux"
    if base.startswith("android"):
        return system == "android"
    if base.startswith("win"):
        return system == "win32"
    return system == "darwin" and base.startswith(("macosx", "darwin"))


def test_every_extra_dependency_is_installable_wherever_the_extra_is_supported():
    """A platform gate must exist wherever the lock cannot satisfy the extra.

    A registry release that ships no sdist can only be installed from a wheel
    matching the target, so a locked dependency with no wheel for a platform
    makes that platform's split unresolvable the moment the extra reaches it.
    ``playwright==1.62.0`` (``google-meet``) publishes eight macos/linux/windows
    wheels and no sdist at all, so with the extra ungated uv must resolve a
    ``sys_platform == 'android'`` split it can never install — and on an index
    that serves no PEP 658 metadata it cannot even read the version's metadata
    for that split, which fails the whole lock. Regression for #126194.
    """
    from packaging.markers import default_environment

    from pm import extras

    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))

    installable: dict[str, set[str]] = {}
    for package in lock["package"]:
        if not package.get("source", {}).get("registry"):
            continue  # git/URL members carry no distribution to match against
        tags = [wheel["url"].rsplit("/", 1)[-1].split("-")[-1].removesuffix(".whl")
                for wheel in package.get("wheels", ())]
        covered = {system for system in LOCK_PLATFORMS
                   if "sdist" in package or any(_wheel_covers(tag, system) for tag in tags)}
        installable.setdefault(package["name"].lower().replace("_", "-"), set()).update(covered)

    project = manifest["project"]["name"].lower().replace("_", "-")
    missing = [
        (extra, requirement.name, system)
        for extra, specs in manifest["project"]["optional-dependencies"].items()
        for spec in specs
        for requirement in (Requirement(spec),)
        if requirement.name.lower().replace("_", "-") in installable
        and requirement.name.lower().replace("_", "-") != project
        for system, traits in LOCK_PLATFORMS.items()
        if system not in installable[requirement.name.lower().replace("_", "-")]
        and (requirement.marker is None
             or requirement.marker.evaluate({**default_environment(), "sys_platform": system, **traits}))
        and extras.extra_supported(extra,
                                  environment={**default_environment(), "sys_platform": system, **traits},
                                  importable=lambda _: False)
    ]
    assert not missing, f"gated-off platforms needed, or extras need a distribution: {missing}"
