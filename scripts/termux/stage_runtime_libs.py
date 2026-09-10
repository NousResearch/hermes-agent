#!/usr/bin/env python3
"""Stage the termux runtime libs into <payload>/runtime-libs/lib/.

The sealed termux deb is self-contained by contract: the device's termux
tree may have NONE of the payload interpreters' runtime libs installed
(first real-device install: `import ctypes` dlopened libffi.so and died).
The pin table (runtime_libs.json) is derived from the suppliers' own
dependency metadata -- the termux-main python .deb's Depends line, uv's zstd,
nodejs's libc++/c-ares/libicu -- not from whichever lib happens to error
first.

Each .deb is downloaded + sha256-verified through pm's hardened downloader
and unpacked with pm's DebPackage (traversal/symlink-checked ar+tar, never
dpkg, never executing package content). Every *.so* from the package's
lib/ payload merges into ONE flat directory -- the trampolines then put a
single payload dir on the linker path.

Cache correctness: the merged directory is only trusted when a manifest
(runtime-libs/manifest.json) records it. The manifest binds the EXACT pin
table (its sha256) to every merged .so's sha256, and is usable only when
the staged set is nonempty, EXACTLY matches the manifest's file set, and
every hash matches. Anything else -- missing file, extra file, stale
bytes, stale table -- is a miss: the merged dir is rebuilt COMPLETELY
from digest-verified .deb downloads/extractions and the manifest is
emitted only after the complete build succeeds.

Usage: stage_runtime_libs.py <payload-dir>
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pm.downloader import Download  # noqa: E402
from pm.artifact_mirror import pinned_source  # noqa: E402
from pm.package import DebPackage  # noqa: E402

PREFIX_REL = DebPackage.prefix_rel
MANIFEST_NAME = "manifest.json"


class StageError(RuntimeError):
    """Runtime-lib staging failed; the merged dir may be incomplete."""


class _LibDeb(DebPackage):
    """Extraction-only view of a pinned runtime-lib .deb."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.deb_package = name

    def fetch_url(self, version: str, target: str) -> str:  # pragma: no cover
        raise NotImplementedError("pinned URLs live in runtime_libs.json")

    def verify(self, entry: Path, target: str) -> str:  # pragma: no cover
        return ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _table_sha256(table: dict) -> str:
    return hashlib.sha256(
        json.dumps(table, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _cache_valid(out: Path, manifest_path: Path, table: dict) -> bool:
    """The staged merged dir is usable only when a manifest exists that
    binds the exact current table to a nonempty, EXACT set of .so files
    whose bytes all hash to the recorded values."""
    if not out.is_dir() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
    except (ValueError, KeyError, OSError):
        return False
    if manifest.get("table_sha256") != _table_sha256(table):
        return False
    if not isinstance(files, dict) or not files:
        return False
    staged = {p.name: p for p in out.glob("*.so*")}
    if set(staged) != set(files):
        return False  # extra or missing files: do not trust the survivor
    for name, want in files.items():
        try:
            if _sha256_file(staged[name]) != want:
                return False
        except OSError:
            return False
    return True


def _work_dir(payload: Path) -> Path:
    return payload / ".work" / "runtime-libs"


def download_path(payload: Path, name: str) -> Path:
    """The verified input consumed by this stager and the CI archive preflight."""
    return _work_dir(payload) / "dl" / f"{name}.deb"


def _ensure_extracted(payload: Path, name: str, row: dict) -> Path:
    """Extract fresh bytes from a digest-verified archive on every cache miss."""
    work = _work_dir(payload)
    extract = work / "extract" / name
    archive = download_path(payload, name)
    scratch = archive.parent
    scratch.mkdir(parents=True, exist_ok=True)
    archive_ok = False
    if archive.exists():
        try:
            archive_ok = _sha256_file(archive) == row["sha256"]
        except OSError:
            archive_ok = False
    if not archive_ok:
        try:
            Download([pinned_source(row["url"], archive, row["sha256"])],
                     partials_dir=scratch).run()
        except Exception as exc:
            raise StageError(f"{name} {row['version']}: download failed from {row['url']}: {exc}") from exc
    if extract.exists():
        shutil.rmtree(extract)
    extract.mkdir(parents=True, exist_ok=True)
    _LibDeb(name).unpack(archive, extract, "linux-arm64-bionic")
    return extract


def stage(payload: Path, table: dict, licenses: dict | None = None) -> Path:
    """Stage every pinned runtime lib into <payload>/runtime-libs/lib/.

    Returns the merged output dir. Raises StageError on any failure; the
    manifest is written only after a complete, verified build.
    """
    payload = Path(payload).resolve()
    payload.mkdir(parents=True, exist_ok=True)
    out = payload / "runtime-libs" / "lib"
    manifest_path = payload / "runtime-libs" / MANIFEST_NAME

    identity = {"libs": table, "licenses": licenses}
    if _cache_valid(out, manifest_path, identity):
        print(f"runtime libs already staged (manifest-verified): "
              f"{len(list(out.glob('*.so*')))} .so* -> {out}")
        return out

    # A cache miss rebuilds from verified archives, never scratch extracts.
    if out.parent.exists():
        shutil.rmtree(out.parent)
    out.mkdir(parents=True, exist_ok=True)

    merged = 0
    for name, row in table.items():
        extract = _ensure_extracted(payload, name, row)
        lib_dir = extract / PREFIX_REL / "lib"
        sos = sorted(lib_dir.glob("*.so*"))
        if not sos:
            raise StageError(f"{name}: no shared objects in "
                             f"{PREFIX_REL}/lib of the package")
        n = 0
        for so in sos:
            dest = out / so.name
            if dest.exists():
                # Co-installed packages share sonames: only identical
                # bytes may collide.
                if _sha256_file(so) != _sha256_file(dest):
                    raise StageError(
                        f"soname collision with different bytes: "
                        f"{so.name} from {name} vs already-staged")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(so, dest)
            n += 1
        notices = extract / PREFIX_REL / "share/doc"
        if notices.is_dir():
            target = payload / "runtime-libs/share/doc"
            shutil.copytree(notices, target, dirs_exist_ok=True, symlinks=True)
        merged += n
        print(f"  {name} {row['version']}: {n} new .so* -> runtime-libs/lib", flush=True)

    if not any(out.glob("*.so*")):
        raise StageError("no shared objects staged")

    if licenses:
        extract = _ensure_extracted(payload, "termux-licenses", licenses)
        shutil.copytree(
            extract / PREFIX_REL / "share/LICENSES", payload / "runtime-libs/share/LICENSES",
            dirs_exist_ok=True, symlinks=True,
        )
    # Manifest last: its existence is the promise that the build above
    # completed for the exact table.
    manifest = {
        "table_sha256": _table_sha256(identity),
        "files": {p.name: _sha256_file(p) for p in sorted(out.glob("*.so*"))},
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"runtime libs staged: {len(table)} packages, {merged} new .so* "
          f"-> {out}")
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: stage_runtime_libs.py <payload-dir>", file=sys.stderr)
        return 2
    payload = Path(sys.argv[1]).resolve()
    table = json.loads((HERE / "runtime_libs.json").read_text(encoding="utf-8"))
    try:
        stage(payload, table["libs"], table.get("licenses"))
    except Exception as exc:  # noqa: BLE001 -- CLI boundary reports and exits
        print(f"runtime-lib staging failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
