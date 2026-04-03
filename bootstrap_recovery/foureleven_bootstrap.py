from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_RECOVERY_FILES = (
    "config.yaml",
    "SOUL.md",
    "memory.db",
    "memories/MEMORY.md",
    "memories/USER.md",
)

CHAIN_POINTERS = {
    "shell_root": "memory/chain-of-shells/LATEST-shell-root",
    "retrieval_index": "memory/chain-of-shells/LATEST-retrieval-index",
    "recovery_head": "memory/chain-of-shells/LATEST-recovery-head",
    "pulse": "memory/chain-of-shells/LATEST-pulse",
    "bundle_manifest": "memory/chain-of-shells/LATEST-bundle-manifest",
    "context_index": "memory/chain-of-shells/context/LATEST-context-index",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _read_pointer(root: Path, relative_pointer: str) -> str | None:
    pointer = root / relative_pointer
    if not pointer.exists():
        return None
    value = pointer.read_text(encoding="utf-8", errors="ignore").strip()
    return value or None


def inspect_hermes_home(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    present = []
    missing = []
    for rel in REQUIRED_RECOVERY_FILES:
        if (root / rel).exists():
            present.append(rel)
        else:
            missing.append(rel)
    chain_latest = {}
    for name, rel in CHAIN_POINTERS.items():
        target = _read_pointer(root, rel)
        if target:
            chain_latest[name] = target
    ready = not missing and {"shell_root", "retrieval_index", "recovery_head", "pulse", "bundle_manifest"}.issubset(chain_latest)
    return {
        "root": str(root),
        "present": present,
        "missing": missing,
        "chain_latest": chain_latest,
        "ready": ready,
    }


def _gather_files(root: Path) -> list[Path]:
    files = [root / rel for rel in REQUIRED_RECOVERY_FILES if (root / rel).exists()]
    skill_dir = root / "skills"
    if skill_dir.exists():
        files.extend(p for p in skill_dir.rglob("*") if p.is_file())
    for rel in CHAIN_POINTERS.values():
        p = root / rel
        if p.exists():
            files.append(p)
            target = Path(p.read_text(encoding="utf-8", errors="ignore").strip())
            if target.exists() and target.is_file():
                files.append(target)
    unique = []
    seen = set()
    for f in files:
        try:
            resolved = f.resolve()
        except Exception:
            resolved = f
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)
    return unique


def create_recovery_bundle(source_root: Path | str, output_path: Path | str) -> dict[str, Any]:
    source_root = Path(source_root)
    output_path = Path(output_path)
    report = inspect_hermes_home(source_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = _gather_files(source_root)
    manifest_files = []
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            rel = path.relative_to(source_root)
            arc = Path("payload") / rel
            zf.write(path, arc.as_posix())
            manifest_files.append({
                "path": rel.as_posix(),
                "archive_path": arc.as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        manifest = {
            "format": "foureleven-recovery-bundle-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "ready": report["ready"],
            "missing": report["missing"],
            "chain_latest": report["chain_latest"],
            "files": manifest_files,
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    return {"success": True, "output": str(output_path), "manifest": manifest}


def restore_recovery_bundle(bundle_path: Path | str, target_root: Path | str) -> dict[str, Any]:
    bundle_path = Path(bundle_path)
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path) as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        for entry in manifest.get("files", []):
            arc = entry["archive_path"]
            data = zf.read(arc)
            digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
            if digest != entry["sha256"]:
                return {"success": False, "error": f"Hash mismatch for {entry['path']}"}
            dest = target_root / entry["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
    return {"success": True, "target_root": str(target_root), "file_count": len(manifest.get('files', []))}


def _default_output_name() -> Path:
    return Path.cwd() / "foureleven-recovery.zip"


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="foureleven stage-0 recovery bootstrap")
    sub = parser.add_subparsers(dest="action")

    p_status = sub.add_parser("status", help="Inspect a Hermes home for recoverability")
    p_status.add_argument("--source", default=os.getenv("HERMES_HOME", str(Path.home() / ".hermes")))

    p_bundle = sub.add_parser("bundle", help="Create a portable recovery bundle")
    p_bundle.add_argument("--source", default=os.getenv("HERMES_HOME", str(Path.home() / ".hermes")))
    p_bundle.add_argument("--output", default=str(_default_output_name()))

    p_restore = sub.add_parser("restore", help="Restore a portable recovery bundle into a target directory")
    p_restore.add_argument("--bundle", required=True)
    p_restore.add_argument("--target", required=True)

    args = parser.parse_args(argv)
    if args.action == "status":
        print(json.dumps(inspect_hermes_home(args.source), indent=2, ensure_ascii=False))
        return 0
    if args.action == "bundle":
        print(json.dumps(create_recovery_bundle(args.source, args.output), indent=2, ensure_ascii=False))
        return 0
    if args.action == "restore":
        result = restore_recovery_bundle(args.bundle, args.target)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("success") else 1
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(cli())
