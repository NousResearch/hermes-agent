"""The stable release lanes, defined once.

A stable release is a set of lanes. A build lane produces something users get
(a package or an image); a test lane is a suite that gates the release. The
claim records the set, ``admit`` turns it into one flag per lane and one run
flag per workflow job, and every gate judges each job against that same set.

``release.py release`` starts from DEFAULT (every lane except macOS Intel and
Termux) and edits it with ``--only``, ``--with`` and ``--without``.

Lanes are independent switches, but a job runs only when every lane it needs
is on (JOB_LANES), so a test lane for something that is not built is inert:
``upgrade`` with ``macos-arm64`` off runs no macOS upgrade arm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Lane:
    name: str
    default: bool
    doc: str


LANES = (
    Lane("docker", True, "build and publish the Docker image"),
    Lane("windows", True, "Windows x64 + arm64 MSIX, the universal bundle and the Microsoft Store bundle"),
    Lane("macos-arm64", True, "macOS Apple Silicon DMG + ZIP"),
    Lane("macos-intel", False, "macOS Intel DMG + ZIP (unsupported on main; its runner adds over an hour)"),
    Lane("termux", False, "Termux (Android aarch64) APT package"),
    Lane("ci", True, "the full CI pipeline"),
    Lane("nix", True, "Nix checks"),
    Lane("docker-tests", True, "tests of the Docker image before it is published"),
    Lane("pm-bundle", True, "PM bundle acceptance"),
    Lane("windows-live", True, "Windows live venv E2E"),
    Lane("install-e2e", True, "install and update E2E"),
    Lane("bootstrap-version", True, "bootstrap installer release identity check"),
    Lane("termux-checks", True, "Termux verification (runs only with termux)"),
    Lane("smoke", True, "native install + chat smoke of each built desktop package"),
    Lane("upgrade", True, "signed-package upgrade acceptance (OLD to NEW) of each built desktop package"),
)
NAMES = tuple(lane.name for lane in LANES)
DEFAULT = frozenset(lane.name for lane in LANES if lane.default)

DESKTOP = ("windows", "macos-arm64", "macos-intel")
BUNDLES = (*DESKTOP, "termux")
# The protected desktop channel, the update feeds and the Termux index move only for a release that
# built every desktop package the channel serves. A partial set still stages, verifies and (for
# Windows) submits its packages, but leaves the channels on the previous release.
CHANNEL_DESKTOP = ("windows", "macos-arm64")
DERIVED = ("bundles", "desktop-channel")

# Each stable-release.yml job and the lanes that must all be on for it to run. A job named here
# reports `skipped` when one is off and `success` otherwise; a job not named never skips. DERIVED
# names are computed from the lanes: "bundles" is any build lane except docker, "desktop-channel"
# is the full desktop set the protected channel serves.
JOB_LANES: dict[str, frozenset[str]] = {name: frozenset(needs) for name, needs in {
    "ci": ("ci",),
    "nix": ("nix",),
    "docker": ("docker",),
    "publish-docker": ("docker",),
    "pm-bundle": ("pm-bundle",),
    "termux-checks": ("termux-checks", "termux"),
    "windows-live": ("windows-live",),
    "install-e2e": ("install-e2e",),
    "bootstrap-version": ("bootstrap-version",),
    "candidates-darwin-arm64": ("macos-arm64",),
    "candidates-darwin-x64": ("macos-intel",),
    "candidates-win32-arm64": ("windows",),
    "candidates-win32-x64": ("windows",),
    "candidates-win32-bundle": ("windows",),
    "candidates-termux": ("termux",),
    "candidate-manifest": ("bundles",),
    "publish-bundles": ("bundles",),
    "transitions-darwin-arm64": ("macos-arm64", "upgrade"),
    "transitions-darwin-x64": ("macos-intel", "upgrade"),
    "transitions-win32": ("windows", "upgrade"),
    "macos-packaged-arm64": ("macos-arm64", "upgrade"),
    "macos-packaged-x64": ("macos-intel", "upgrade"),
    "windows-packaged": ("windows", "upgrade"),
}.items()}

CANDIDATE_CALLS = tuple(job for job in JOB_LANES if job.startswith("candidates-"))

# Candidate handoff group and manifest target, each with the lane that builds it.
HANDOFF_LANE = {"win32-x64": "windows", "win32-arm64": "windows", "windows-universal": "windows",
                "darwin-arm64": "macos-arm64", "darwin-x64": "macos-intel", "termux": "termux"}
TARGET_LANE = {"windows/x64": "windows", "windows/arm64": "windows", "macos/arm64": "macos-arm64",
               "macos/x64": "macos-intel", "termux/aarch64": "termux"}
LEFT_OUT = {
    "docker": "the Docker image keeps its previous stable tag",
    "windows": "no Windows packages and no Microsoft Store submission",
    "macos-arm64": "no macOS Apple Silicon package",
    "macos-intel": "Intel Macs get no update: the shared macOS stable feed lists arm64 only",
    "termux": "the Termux APT channel stays on its previous release",
}


def select(*, only: Iterable[str] | None = None, with_: Iterable[str] = (),
           without: Iterable[str] = ()) -> frozenset[str]:
    """The lane set a ``release`` command asks for. ``only`` replaces DEFAULT as the base."""
    base, added, removed = (_known(group) for group in (DEFAULT if only is None else only, with_, without))
    if added & removed:
        raise ValueError(f"--with and --without both name {', '.join(sorted(added & removed))}")
    return frozenset((base | added) - removed)


def _known(names: Iterable[str]) -> frozenset[str]:
    names = frozenset(names)
    unknown = names - set(NAMES)
    if unknown:
        raise ValueError(f"unknown lane {', '.join(sorted(unknown))}; lanes are {', '.join(NAMES)}")
    return names


def parse_list(values: Iterable[str] | None) -> list[str] | None:
    """Flatten repeated comma-separated option values; None when the option was not given."""
    if values is None:
        return None
    return [name.strip() for value in values for name in value.split(",") if name.strip()]


def effective(lanes: Iterable[str]) -> frozenset[str]:
    lanes = frozenset(lanes)
    return lanes | {name for name, derived in (("bundles", has_bundles(lanes)),
                                               ("desktop-channel", advances_desktop(lanes))) if derived}


def flags(lanes: Iterable[str]) -> dict[str, bool]:
    """One flag per lane plus the derived names: what ``admit`` hands the workflow."""
    on = effective(lanes)
    return {name: name in on for name in (*NAMES, *DERIVED)}


def from_flags(raw: object) -> frozenset[str]:
    """The lane set of a ``flags`` mapping; refuses anything but a boolean for every lane."""
    if not isinstance(raw, Mapping) or set(raw) != {*NAMES, *DERIVED} \
            or any(not isinstance(value, bool) for value in raw.values()):
        raise ValueError("lanes must be a boolean for every lane")
    return frozenset(name for name in NAMES if raw[name])


def jobs_to_run(lanes: Iterable[str]) -> dict[str, bool]:
    """Whether each lane-controlled job runs. Every ``if`` in the workflow reads this."""
    on = effective(lanes)
    return {job: needs <= on for job, needs in JOB_LANES.items()}


def handoffs(lanes: Iterable[str], names: Iterable[str]) -> tuple[str, ...]:
    """``names`` without the candidate handoff groups the lanes did not build."""
    on = frozenset(lanes)
    return tuple(name for name in names if HANDOFF_LANE[name] in on)


def targets(lanes: Iterable[str]) -> frozenset[str]:
    """The manifest platform/arch targets the lanes build."""
    on = frozenset(lanes)
    return frozenset(target for target, lane in TARGET_LANE.items() if lane in on)


def has_bundles(lanes: Iterable[str]) -> bool:
    return bool(frozenset(lanes) & set(BUNDLES))


def runs_smokes(lanes: Iterable[str]) -> bool:
    """A smoke result exists only for a built desktop package."""
    on = frozenset(lanes)
    return "smoke" in on and bool(on & set(DESKTOP))


def advances_desktop(lanes: Iterable[str]) -> bool:
    return frozenset(lanes) >= set(CHANNEL_DESKTOP)


def moves_channels(lanes: Iterable[str]) -> bool:
    """Whether publication moves a channel head. A release that moves none is finished once public."""
    on = frozenset(lanes)
    return "docker" in on or advances_desktop(on)


_LEGACY_OPTIONAL = ("includeMacosIntel", "includeTermux")
_LEGACY_BUNDLE_LANES = frozenset({*BUNDLES, "pm-bundle", "upgrade", "smoke"})
_TEST_LANES = frozenset({"ci", "nix", "docker-tests", "pm-bundle", "windows-live", "install-e2e",
                         "bootstrap-version", "termux-checks", "smoke", "upgrade"})


def claim_lanes(metadata: Mapping) -> frozenset[str]:
    """The lanes a claim message records.

    Claims cut before lanes existed carry ``skipBundles``, ``skipTests`` and
    (later) ``includeMacosIntel`` / ``includeTermux`` instead of ``lanes``. They
    stay readable for as long as their tags do: each translates to the lane set
    that gated the same jobs. A missing include key reads as false.
    """
    if "lanes" in metadata:
        lanes = metadata["lanes"]
        if (not isinstance(lanes, list) or any(not isinstance(name, str) for name in lanes)
                or lanes != sorted(set(lanes))):
            raise ValueError("Stable claim metadata is invalid")
        try:
            return _known(lanes)
        except ValueError as error:
            raise ValueError("Stable claim metadata is invalid") from error
    if any(not isinstance(metadata.get(key), bool)
           for key in ("skipBundles", "skipTests", *(k for k in _LEGACY_OPTIONAL if k in metadata))):
        raise ValueError("Stable claim metadata is invalid")
    lanes = set(NAMES) - {"macos-intel", "termux"}
    lanes |= {lane for key, lane in zip(_LEGACY_OPTIONAL, ("macos-intel", "termux")) if metadata.get(key)}
    if metadata["skipBundles"]:
        lanes -= _LEGACY_BUNDLE_LANES
    if metadata["skipTests"]:
        lanes -= _TEST_LANES
    return frozenset(lanes)


def describe(lanes: Iterable[str]) -> str:
    on = frozenset(lanes)
    off = [name for name in NAMES if name not in on]
    return f"on: {', '.join(name for name in NAMES if name in on) or 'none'}; off: {', '.join(off) or 'none'}"
