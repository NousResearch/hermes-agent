"""The opt-in stable release platforms, defined once.

A stable release builds macOS Intel and Termux only when its claim asks for
them. Each row ties a claim flag to everything that flag controls: the
release.py option, the admit output, the gate's removal reason, the jobs it
removes, the candidate handoff, the manifest target, and the candidate call
and smoke job. A claim cut before these flags existed lacks the keys, and a
missing key reads as false.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class OptionalPlatform:
    label: str
    claim_flag: str  # key in the claim message
    key: str  # normalized claim field, keyword argument, and admit output (with dashes)
    option: str  # release.py release option
    exclusion: str  # gate removal reason when the platform is left out
    jobs: tuple[str, ...]  # stable-release.yml jobs the exclusion removes
    handoff: str  # candidate handoff group
    target: str  # candidate manifest platform/arch
    left_out: str  # what users of the platform get from a release that leaves it out
    call: str | None = None  # candidate call whose result records the smoke
    smoke: str | None = None


PLATFORMS = (
    OptionalPlatform("macOS Intel", "includeMacosIntel", "include_macos_intel", "--include-macos-intel",
                     "excludeMacosIntel",
                     ("candidates-darwin-x64", "transitions-darwin-x64", "macos-packaged-x64"),
                     "darwin-x64", "macos/x64",
                     # promote rewrites the one shared stable-mac.yml feed with the arm64 leg alone.
                     "Intel Macs get no update: the shared macOS stable feed now lists arm64 only",
                     "candidates-darwin-x64", "smoke-darwin-x64"),
    OptionalPlatform("Termux", "includeTermux", "include_termux", "--include-termux", "excludeTermux",
                     ("candidates-termux", "termux-checks"), "termux", "termux/aarch64",
                     "the Termux APT channel stays on its previous release"),
)
CLAIM_FLAGS = tuple(p.claim_flag for p in PLATFORMS)
TARGETS = tuple(p.target for p in PLATFORMS)
# The opt-in handoffs and the manifest target each one produces.
HANDOFFS = {p.handoff: p.target for p in PLATFORMS}


def claim_flags(metadata: dict) -> dict:
    """The claim's opt-in flags, each false when absent; raises on a non-bool."""
    if any(not isinstance(metadata[flag], bool) for flag in CLAIM_FLAGS if flag in metadata):
        raise ValueError("Stable claim metadata is invalid")
    return {flag: metadata.get(flag, False) for flag in CLAIM_FLAGS}


def excluded(flags: Mapping[str, bool]) -> tuple[OptionalPlatform, ...]:
    """The platforms a claim leaves out. ``flags`` maps each ``key`` to its value."""
    return tuple(p for p in PLATFORMS if not flags.get(p.key))


def claim_handoffs(handoffs: tuple[str, ...], flags: Mapping[str, bool]) -> tuple[str, ...]:
    """``handoffs`` without the groups the claim did not opt into."""
    omitted = {p.handoff for p in excluded(flags)}
    return tuple(name for name in handoffs if name not in omitted)


def require_targets_match_claim(manifest: dict, *, include_macos_intel: bool,
                                include_termux: bool) -> None:
    """The opt-in targets are in the manifest exactly when the claim asked for them."""
    flags = {"include_macos_intel": include_macos_intel, "include_termux": include_termux}
    present = {f"{row.get('platform')}/{row.get('arch')}" for row in manifest.get("packages", [])}
    for p in PLATFORMS:
        wanted = flags[p.key]
        if (p.target in present) != wanted:
            raise ValueError(f"Candidate manifest {'lacks' if wanted else 'carries'} {p.target}, "
                             "which differs from the claim")
