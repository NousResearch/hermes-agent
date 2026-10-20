"""Every standalone npm package must resolve its lockfile under the release-age gate.

npm reads project config only from the directory that owns the ``package.json``
being installed — the root ``.npmrc`` does not cascade into directories that
ship their own ``package.json`` + ``package-lock.json``. Each such package
needs its own ``.npmrc`` with the same ``min-release-age`` floor as the root,
or its lockfile is resolved with brand-new releases the root gate exists to
hold back (#134019).

This matters most where Hermes installs the package for the user: the WhatsApp
bridge (holds the session credentials and parses inbound messages from
arbitrary senders; ``_ensure_bridge_deps`` runs ``npm install`` in the bridge
dir) and the Photon sidecar mirror install exactly what the committed lockfile
pins, so users cannot apply the cooldown on their side.

Deliberately structural, not behavioral: no npm invocation, just the invariant
that would have caught the drift.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_ROOT_FLOOR_DAYS = "14"


def _standalone_lockfile_dirs() -> list[Path]:
    return sorted(
        lock.parent
        for lock in REPO_ROOT.rglob("package-lock.json")
        if "node_modules" not in lock.parts[1:-1]
    )


def _release_age_floor(npmrc: Path) -> str | None:
    for line in npmrc.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        key, sep, value = line.partition("=")
        if sep and key.strip() == "min-release-age":
            return value.strip()
    return None


class TestStandalonePackagesCarryTheGate:
    def test_every_lockfile_dir_has_the_release_age_floor(self):
        """A missing per-package .npmrc silently drops the root's age gate:
        with a lockfile present, `npm install` and `npm ci` install the pinned
        versions regardless of what any user-side config says."""
        ungated = []
        for pkg_dir in _standalone_lockfile_dirs():
            npmrc = pkg_dir / ".npmrc"
            if not npmrc.is_file() or _release_age_floor(npmrc) != _ROOT_FLOOR_DAYS:
                ungated.append(str(pkg_dir.relative_to(REPO_ROOT)))
        assert not ungated, (
            "Directories with a package-lock.json but no .npmrc carrying "
            f"min-release-age={_ROOT_FLOOR_DAYS}: npm only reads project config "
            "from the package's own directory, so these lockfiles resolve "
            f"without the root's release-age gate: {ungated}"
        )

    def test_photon_sidecar_mirror_carries_npmrc(self):
        """The read-only-tree fallback installs deps with npm inside
        HERMES_HOME/photon/sidecar; a .npmrc left out of the mirror whitelist
        would drop the gate exactly where the runtime install happens."""
        from plugins.platforms.photon.sidecar_paths import _MIRROR_FILES

        assert ".npmrc" in _MIRROR_FILES
