"""Guard the Nix expressions against deprecated stdenv platform predicates.

nixpkgs deprecated the ``stdenv.is<Platform>`` predicates in favor of
``stdenv.hostPlatform.is<Platform>``; evaluating the flake on current nixpkgs
emits one warning per surviving call site (#109322). Keep them from coming
back now that the last call sites have been migrated.
"""

from __future__ import annotations

from pathlib import Path

NIX_DIR = Path(__file__).resolve().parents[1] / "nix"
DEPRECATED_PREDICATES = ("stdenv.isLinux", "stdenv.isDarwin")


def test_nix_expressions_avoid_deprecated_stdenv_platform_predicates() -> None:
    offenders: list[str] = []
    for path in sorted(NIX_DIR.rglob("*.nix")):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            hits = [pred for pred in DEPRECATED_PREDICATES if pred in line]
            if hits:
                offenders.append(
                    f"{path.relative_to(NIX_DIR)}:{lineno}: {', '.join(hits)}"
                )
    assert not offenders, (
        "deprecated stdenv platform predicates found "
        "(use stdenv.hostPlatform.* instead):\n" + "\n".join(offenders)
    )
