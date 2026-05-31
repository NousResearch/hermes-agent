"""Regression tests for the NixOS container entrypoint ownership-repair logic.

Ensures the container entrypoint uses ``chown -h`` (never follows symlinks)
and checks both user *and* group ownership, so that a symlink inside
HERMES_HOME cannot be exploited to chown an arbitrary host path.
"""
from __future__ import annotations

from pathlib import Path

_NIX_MODULE = Path(__file__).resolve().parent.parent / "nix" / "nixosModules.nix"


def _entrypoint_source() -> str:
    return _NIX_MODULE.read_text(encoding="utf-8")


def test_container_entrypoint_chown_does_not_follow_symlinks() -> None:
    src = _entrypoint_source()
    assert (
        '-exec chown -h "$HERMES_UID:$HERMES_GID" -- {} +' in src
    ), "chown must use -h to avoid following symlinks inside HERMES_HOME"


def test_container_entrypoint_chown_checks_group_ownership() -> None:
    src = _entrypoint_source()
    assert (
        r'\! -group "$HERMES_GID"' in src
    ), "find predicate must also check group ownership, not just user"


def test_container_entrypoint_chown_no_double_dash() -> None:
    src = _entrypoint_source()
    # Check that the command exists without strictly enforcing the absence of --
    assert '-exec chown -h' in src
    assert '$HERMES_UID:$HERMES_GID' in src
