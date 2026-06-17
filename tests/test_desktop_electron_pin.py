"""Regression: the desktop Electron dependency must be an exact, consistent pin.

The Windows desktop install failed at "Building desktop app" because Electron
changed its install mechanism mid patch-series:

    electron 40.9.3 .. 40.10.2  -> @electron/get@^2 + extract-zip@^2  (pure JS)
    electron 40.10.3 / 40.10.4  -> @electron/get@^5 +
                                   @electron-internal/extract-zip@^1 (native napi)

``apps/desktop/package.json`` declared ``electronVersion: 40.9.3`` (the tested,
JS-extract build) but pinned the dependency loosely as ``electron: ^40.9.3``.
``npm ci`` then resolved 40.10.3/40.10.4 — the new *native* extract-zip whose
win32-x64 binding fails to ``dlopen`` on some Windows hosts
(``ERR_DLOPEN_FAILED loading index.win32-x64-msvc.node``).

These tests lock the contract that prevents that drift, without hard-coding the
specific version (which is allowed to move):

1. the Electron dependency is an *exact* version (Electron Builder needs the
   installed binary to match ``electronVersion`` / ``electronDist``), and
2. the dependency, ``build.electronVersion``, and the resolved lockfile entry
   all agree — so ``npm ci`` installs exactly what the build packages.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
DESKTOP_PKG = REPO_ROOT / "apps" / "desktop" / "package.json"
ROOT_LOCK = REPO_ROOT / "package-lock.json"

# An exact semver: digits.digits.digits with an optional prerelease/build tag,
# but NO range operators (^ ~ > < = * x || spaces || -range).
_EXACT_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def _desktop_pkg() -> dict:
    assert DESKTOP_PKG.is_file(), f"missing {DESKTOP_PKG}"
    return json.loads(DESKTOP_PKG.read_text(encoding="utf-8"))


def _electron_spec(pkg: dict) -> str:
    for section in ("dependencies", "devDependencies"):
        spec = pkg.get(section, {}).get("electron")
        if spec:
            return spec
    pytest.fail("electron is not listed in apps/desktop dependencies")


def test_electron_dependency_is_exactly_pinned():
    """A loose range lets npm drift onto an Electron with a different installer."""
    spec = _electron_spec(_desktop_pkg())
    assert _EXACT_SEMVER.match(spec), (
        f"electron must be pinned to an exact version, got {spec!r}. "
        "A range (^/~) lets npm ci resolve a newer Electron whose postinstall "
        "may differ from the one the build was validated against."
    )


def test_electron_dependency_matches_electron_version():
    """electron-builder packages build.electronVersion against the installed binary."""
    pkg = _desktop_pkg()
    spec = _electron_spec(pkg)
    builder_version = pkg.get("build", {}).get("electronVersion")
    assert builder_version, "build.electronVersion is missing"
    assert spec == builder_version, (
        f"electron dependency ({spec!r}) must equal build.electronVersion "
        f"({builder_version!r}); otherwise electron-builder packages a different "
        "version than npm installs into electronDist."
    )


def test_lockfile_resolves_the_pinned_electron():
    """npm ci installs from the lockfile, so it must agree with the pin."""
    if not ROOT_LOCK.is_file():
        pytest.skip("root package-lock.json not present")
    spec = _electron_spec(_desktop_pkg())
    lock = json.loads(ROOT_LOCK.read_text(encoding="utf-8"))
    packages = lock.get("packages", {})
    resolved = [
        meta.get("version")
        for path, meta in packages.items()
        if path.endswith("node_modules/electron") and meta.get("version")
    ]
    assert resolved, "no electron entry found in package-lock.json"
    assert all(v == spec for v in resolved), (
        f"package-lock.json resolves electron to {sorted(set(resolved))}, "
        f"but the pin is {spec!r}; run `npm install --package-lock-only` so "
        "`npm ci` stays consistent."
    )


def test_electron_dist_matches_lockfile_install_location():
    """build.electronDist must point at where the lockfile installs Electron.

    electron-builder copies the unpacked Electron from ``build.electronDist``
    (resolved relative to ``apps/desktop``). npm workspace hoisting is not
    deterministic across machines/npm versions: it may nest Electron under
    ``apps/desktop/node_modules/electron`` or hoist it to the repo root. If
    electronDist points at one location while the lockfile installs at the
    other, packaging fails with ``The specified electronDist does not exist`` —
    the "Building desktop app" failure reported after the June lockfile
    regeneration floated Electron and reshuffled the hoist. Lock the two
    together so a hoist change (root <-> nested) can't silently break the path
    again.
    """
    if not ROOT_LOCK.is_file():
        pytest.skip("root package-lock.json not present")
    electron_dist = _desktop_pkg().get("build", {}).get("electronDist")
    assert electron_dist, "build.electronDist is missing"

    lock = json.loads(ROOT_LOCK.read_text(encoding="utf-8"))
    electron_paths = [
        path
        for path in lock.get("packages", {})
        if path.endswith("node_modules/electron")
    ]
    assert electron_paths, "no electron entry found in package-lock.json"

    desktop_dir = REPO_ROOT / "apps" / "desktop"
    # electronDist is resolved relative to the apps/desktop project dir.
    configured = (desktop_dir / electron_dist).resolve()
    # Where the lockfile actually places Electron's unpacked dist.
    installed = {(REPO_ROOT / p / "dist").resolve() for p in electron_paths}
    assert configured in installed, (
        f"build.electronDist={electron_dist!r} resolves to {configured}, but the "
        f"lockfile installs Electron at {sorted(str(p) for p in installed)}. "
        "electron-builder will fail with 'electronDist does not exist'."
    )
