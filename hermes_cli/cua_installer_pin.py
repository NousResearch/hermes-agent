"""Pinned upstream cua-driver installer identity.

Hermes installs the cua-driver binary by running the upstream trycua/cua
installer script. To avoid executing whatever happens to be on the
upstream ``main`` branch at install time, the pin file records the exact
Git ref and the SHA-256 of the installer script at that ref.

Bump ``scripts/cua_installer_pin.env`` deliberately, like any other
dependency.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_PIN_PATH = Path(__file__).resolve().parent.parent / "scripts" / "cua_installer_pin.env"


class CuaInstallerIntegrityError(ValueError):
    """Fetched installer bytes did not match the pinned SHA-256."""


def _load_pin_file(path: Path = _PIN_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    required = (
        "CUA_INSTALLER_REF",
        "CUA_INSTALLER_SHA256_SH",
        "CUA_INSTALLER_SHA256_PS1",
        "CUA_DRIVER_RS_VERSION",
    )
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError(f"incomplete cua installer pin file {path}: missing {missing}")
    return values


_PIN = _load_pin_file()

CUA_INSTALLER_REF: str = _PIN["CUA_INSTALLER_REF"]
CUA_INSTALLER_SHA256_SH: str = _PIN["CUA_INSTALLER_SHA256_SH"]
CUA_INSTALLER_SHA256_PS1: str = _PIN["CUA_INSTALLER_SHA256_PS1"]
CUA_DRIVER_RS_VERSION: str = _PIN["CUA_DRIVER_RS_VERSION"]


def cua_installer_url(*, is_windows: bool, ref: str | None = None) -> str:
    """Return the pinned upstream installer URL for the requested platform."""
    script = "install.ps1" if is_windows else "install.sh"
    return (
        f"https://raw.githubusercontent.com/trycua/cua/{ref or CUA_INSTALLER_REF}"
        f"/libs/cua-driver/scripts/{script}"
    )


def cua_installer_expected_sha256(*, is_windows: bool) -> str:
    return CUA_INSTALLER_SHA256_PS1 if is_windows else CUA_INSTALLER_SHA256_SH


def verify_cua_installer_digest(data: bytes, expected_hex: str) -> str:
    """Return the digest if it matches ``expected_hex``; otherwise raise."""
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_hex.lower():
        raise CuaInstallerIntegrityError(
            f"cua-driver installer sha256 mismatch: got {digest}, expected {expected_hex.lower()}"
        )
    return digest
