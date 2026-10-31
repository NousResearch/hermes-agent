#!/usr/bin/env python3
"""Package a verified PM payload into a relocatable air-gap transfer archive.

A PM payload (``hermes pm bundle``) is already a complete, self-contained
install: repo snapshot, tool store, sealed ``pm-runtime``, a built venv, and a
lock-scoped ``uv-cache`` whose wheels let later dependency changes resolve with
no network. What it is not is *transferable* — it is a directory tree of loose
files with relative symlinks, which does not survive a copy to removable media
without something to check the copy against.

This script produces the two things an air-gapped handoff needs:

* one archive of the payload, and
* a manifest recording what was packed, for which target, from which revision,
  and the SHA-256 of the archive.

The digest is the point. The repository's own trust boundary is the build side:
``hermes pm bundle`` verifies the payload it produced, and the archive is a
byte-exact compressed copy of those verified bytes, so a SHA-256 of the archive
covers every payload file transitively. An operator hashes the archive on the
connected side and again on the air-gapped side and compares — without trusting
the transport in between.

    python3 scripts/offline/pack.py --payload <dir> --out <dir>

Targets are not portable between platforms: a payload pins the exact standalone
interpreter, compiled wheels and native tool binaries for the machine that built
it. The manifest records the target so a mismatched transfer is detectable;
``hermes pm bundle`` is run once per target platform.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACK_SCHEMA = 1
ARCHIVE_SUFFIX = ".tar.gz"
# Payload state that must travel: the repo snapshot, the interpreter/tool store,
# the sealed PM runtime, the built venv, and the lock-scoped wheel cache.
REQUIRED_ENTRIES = ("hermes-agent", "tools", "pm-runtime", "venv", "uv-cache", "manifest.json")


def sha256_file(path: Path, *, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _validate_payload(payload: Path) -> dict:
    """Refuse to package something that is not a complete payload.

    A half-built payload still packs into a plausible-looking archive; the
    failure would then surface on the air-gapped machine, where it cannot be
    repaired. Catch it here instead.
    """
    if not payload.is_dir():
        raise SystemExit(f"payload directory not found: {payload}")
    for name in REQUIRED_ENTRIES:
        if not (payload / name).exists():
            raise SystemExit(f"payload is incomplete: {name} is missing from {payload}")
    try:
        manifest = json.loads((payload / "manifest.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"payload manifest is unreadable: {exc}") from exc
    if manifest.get("schema") != 1:
        raise SystemExit(f"unsupported payload manifest schema: {manifest.get('schema')!r}")
    if not manifest.get("target"):
        raise SystemExit("payload manifest records no target; it cannot be matched to a machine")
    commands = manifest.get("runtime", {}).get("commands") or {}
    for launcher in manifest.get("launchers", []):
        command = commands.get(launcher)
        if not command or not (payload / command).exists():
            raise SystemExit(f"payload launcher is missing: {launcher} -> {command!r}")
    return manifest


def _archive(payload: Path, archive: Path) -> None:
    """Store the payload under its own directory name.

    The member prefix is ``payload.name`` so extraction yields a single
    relocatable directory instead of scattering the tree into the caller's
    working directory.
    """
    with tarfile.open(archive, "w:gz", compresslevel=1) as bundle:
        bundle.add(payload, arcname=payload.name, recursive=True)


def build_manifest(payload: Path, archive: Path) -> dict:
    payload_manifest = json.loads((payload / "manifest.json").read_text(encoding="utf-8-sig"))
    return {
        "schema": PACK_SCHEMA,
        "kind": "hermes-offline-payload",
        "target": payload_manifest.get("target"),
        "repo": payload_manifest.get("repo"),
        "ref": payload_manifest.get("ref"),
        "archive": archive.name,
        # Every archive member is stored under the payload directory's own name
        # (see _archive), so consumers need that prefix to address the tree.
        "payloadRoot": payload.name,
        "archiveSha256": sha256_file(archive),
        "archiveBytes": archive.stat().st_size,
        "launchers": payload_manifest.get("launchers", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--payload", type=Path, required=True, help="a payload produced by `hermes pm bundle --out`")
    parser.add_argument("--out", type=Path, required=True, help="directory to write the archive and manifest into")
    parser.add_argument("--name", help="archive base name (default: derived from the payload target)")
    args = parser.parse_args()

    payload = args.payload.resolve()
    manifest = _validate_payload(payload)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    name = args.name or f"hermes-offline-{manifest['target']}"
    archive = out / f"{name}{ARCHIVE_SUFFIX}"
    archive.unlink(missing_ok=True)

    print(f"packing {payload} ({manifest['target']}) -> {archive}", flush=True)
    _archive(payload, archive)

    packed = build_manifest(payload, archive)
    packed["archive"] = archive.name
    manifest_path = out / f"{name}.manifest.json"
    manifest_path.write_text(json.dumps(packed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_path = archive.with_name(archive.name + ".sha256")
    checksum_path.write_text(f"{packed['archiveSha256']}  {archive.name}\n", encoding="utf-8")

    megabytes = packed["archiveBytes"] / (1 << 20)
    print(f"✓ {archive.name} ({megabytes:.1f} MiB, target {packed['target']})")
    print(f"✓ {manifest_path.name}")
    print(f"✓ {checksum_path.name}")
    print("\nTransfer all three files to the air-gapped machine, then run:")
    print(f"  ./scripts/offline/install.sh --archive {archive.name} --manifest {manifest_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
