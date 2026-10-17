"""NixOS environment helpers for the updater.

Handles CA bundle autodetect and native build toolchain wrapping (or early failure).
"""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import sys
from typing import Callable

_NIXOS_MARKER = Path("/etc/NIXOS")
_NIXOS_CA_BUNDLE = Path("/etc/ssl/certs/ca-bundle.crt")


def is_nixos() -> bool:
    """Return True if running on a NixOS host."""
    return _NIXOS_MARKER.exists()


def autodetect_nixos_ca_bundle() -> bool:
    """Set SSL_CERT_FILE on NixOS if unset and system ca-bundle exists.

    Never overrides explicitly set environment variables.
    Returns True if SSL_CERT_FILE was set by this call.
    """
    if "SSL_CERT_FILE" in os.environ:
        return False
    if is_nixos() and _NIXOS_CA_BUNDLE.exists():
        os.environ["SSL_CERT_FILE"] = str(_NIXOS_CA_BUNDLE)
        return True
    return False


def missing_native_toolchain() -> bool:
    """Return True if cc or make is missing from PATH."""
    return shutil.which("cc") is None or shutil.which("make") is None


def needs_nixos_toolchain() -> bool:
    """Return True if on NixOS and native build toolchain is missing."""
    return is_nixos() and missing_native_toolchain()


def check_nixos_toolchain(*, stop_fn: Callable[[], None] | None = None) -> None:
    """Fail fast on NixOS if native toolchain is missing and nix-shell is unavailable."""
    if not needs_nixos_toolchain():
        return
    if shutil.which("nix-shell") is not None:
        return
    print("✗ Cannot update: native build toolchain (gcc/make) is missing on NixOS.")
    print("  Remedies:")
    print("  - Persistent: nix profile install nixpkgs#gcc nixpkgs#gnumake")
    print("  - One-off:    nix-shell -p gnumake gcc --run hermes update")
    if stop_fn is not None:
        stop_fn()
    sys.exit(1)


def wrap_nixos_toolchain(cmd: list[str]) -> list[str]:
    """Wrap command with nix-shell -p gnumake gcc --run if on NixOS without toolchain.

    If nix-shell is not available or toolchain is already present (e.g. inside
    a nix-shell session), returns the original command unchanged.
    """
    if not needs_nixos_toolchain():
        return cmd
    if shutil.which("nix-shell") is None:
        return cmd
    return ["nix-shell", "-p", "gnumake", "gcc", "--run", shlex.join(cmd)]
