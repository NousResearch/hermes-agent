#!/usr/bin/env python3
"""Checkpoint research and losslessly archive explicitly registered run artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile
from datetime import datetime, timezone

KIND = "deep-research-run-v1"
MANIFEST = "run_manifest.json"
DEFAULTS = {"quick": 4, "standard": 8, "deep": 12}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def safe_path(root, name):
    """Accept only portable relative file paths; never follow symlinks."""
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or "\\" in name or ":" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError(f"Unsafe artifact path: {name!r}")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink artifact refused: {name}")
    if path.exists() and not path.is_file():
        raise ValueError(f"Artifact must be a file: {name}")
    return path


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def save(root, data):
    target = safe_path(root, MANIFEST)
    data["updated_at"] = now()
    handle, temporary = tempfile.mkstemp(prefix=".manifest-", dir=root)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(root):
    if root.is_symlink():
        raise ValueError("Run directory must not be a symlink")
    data = json.loads(safe_path(root, MANIFEST).read_text(encoding="utf-8"))
    if data.get("kind") != KIND:
        raise ValueError("Not a managed research run")
    for name in data["artifacts"]:
        safe_path(root, name)
    safe_path(root, data["report"])
    return data


def init(args):
    root = Path(args.run).expanduser()
    # New, dedicated directory: never adopt a workspace or personal directory.
    root.mkdir(parents=True, exist_ok=False)
    data = {
        "kind": KIND, "created_at": now(), "status": "active",
        "question": args.question, "scope": args.scope, "mode": args.mode,
        "assumptions": args.assumption,
        "budgets": {"search_calls": args.search_budget or DEFAULTS[args.mode],
                    "storage_bytes": args.storage_mb * 1024 * 1024},
        "completed": [], "gaps": [], "next_action": "Frame questions and initialize evidence.json",
        "artifacts": [MANIFEST], "report": "report.md",
    }
    save(root, data)
    print(f"Created {root}")


def checkpoint(args):
    root = Path(args.run).expanduser()
    data = load(root)
    if data["status"] == "archived":
        raise ValueError("Restore the archive to a new directory before resuming")
    for name in args.artifact:
        path = safe_path(root, name)
        if not path.is_file():
            raise ValueError(f"Artifact does not exist: {name}")
        if name not in data["artifacts"]:
            data["artifacts"].append(name)
    if args.status:
        data["status"] = args.status
    if args.next_action is not None:
        data["next_action"] = args.next_action
    for key in ("completed", "gaps"):
        value = getattr(args, key)
        if value is not None:
            data[key] = value
    if args.clear_gaps:
        data["gaps"] = []
    if data["status"] == "complete":
        report = safe_path(root, data["report"])
        if not report.is_file() or not report.stat().st_size:
            raise ValueError("Completion requires a nonempty report.md")
        if data["report"] not in data["artifacts"]:
            data["artifacts"].append(data["report"])
        data["next_action"] = ""
    save(root, data)
    status(args)


def status(args):
    root = Path(args.run).expanduser()
    data = load(root)
    total = sum(safe_path(root, name).stat().st_size
                for name in data["artifacts"] if safe_path(root, name).exists())
    print(json.dumps({key: data[key] for key in
                     ("question", "scope", "mode", "status", "budgets", "completed", "gaps", "next_action")}, indent=2))
    print(f"Registered working artifacts: {total} bytes")
    if total > data["budgets"]["storage_bytes"]:
        print("WARN: storage budget exceeded; stop retaining raw pages and review registered artifacts.")


def verify(archive, expected):
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.namelist()) != len(expected) or set(bundle.namelist()) != set(expected):
            raise ValueError("Archive inventory mismatch")
        for name, checksum in expected.items():
            value = hashlib.sha256()
            with bundle.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(block)
            if value.hexdigest() != checksum:
                raise ValueError(f"Archive verification failed: {name}")


def archive(args):
    root = Path(args.run).expanduser()
    data = load(root)
    if data["status"] != "complete":
        raise ValueError("Only completed, inactive runs can be archived")
    report = safe_path(root, data["report"])
    if not report.is_file() or not report.stat().st_size:
        raise ValueError("A nonempty final report is required")
    names = sorted(set(data["artifacts"]))
    if MANIFEST not in names or data["report"] not in names:
        raise ValueError("Register manifest and report before archiving")
    target = safe_path(root, "artifacts.zip")
    if target.exists() or "artifacts.zip" in names:
        raise ValueError("Archive already exists or is registered as an input")
    files = {name: safe_path(root, name) for name in names}
    hashes = {name: digest(path) for name, path in files.items()}
    raw_bytes = sum(path.stat().st_size for path in files.values())
    handle, temporary = tempfile.mkstemp(prefix=".archive-", suffix=".zip", dir=root)
    os.close(handle)
    try:
        # Optional standard-library codecs may be absent in a custom Python build.
        if zipfile.lzma is not None:
            compression, method = zipfile.ZIP_LZMA, "lzma"
        elif zipfile.zlib is not None:
            compression, method = zipfile.ZIP_DEFLATED, "deflate"
        else:
            compression, method = zipfile.ZIP_STORED, "stored"
        with zipfile.ZipFile(temporary, "w", compression=compression) as bundle:
            for name, path in files.items():
                bundle.write(path, name)
        verify(temporary, hashes)
        for name, path in files.items():
            if digest(safe_path(root, name)) != hashes[name]:
                raise ValueError(f"Artifact changed during archive: {name}")
        # Exclusive publication prevents replacing an existing archive.
        with target.open("xb") as output, open(temporary, "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                output.write(block)
        verify(target, hashes)
    finally:
        os.unlink(temporary)
    data["status"] = "archived"
    data["archive"] = {"path": target.name, "sha256": digest(target), "members": hashes,
                       "raw_bytes": raw_bytes, "archive_bytes": target.stat().st_size,
                       "compression": method}
    save(root, data)
    removed = []
    if args.prune:
        for name in names:
            if name in {MANIFEST, data["report"]}:
                continue
            path = safe_path(root, name)
            if digest(path) != hashes[name]:
                raise ValueError(f"Changed artifact retained: {name}; verified archive is available")
            path.unlink()
            removed.append(name)
    print(f"Verified lossless archive ({method}): {raw_bytes} -> {target.stat().st_size} bytes ({target})")
    print(f"Removed {len(removed)} registered working files; report and manifest retained."
          if args.prune else "Working files retained; no disk reclamation requested.")
    if removed:
        print("Recover with restore into a new directory: " + ", ".join(removed))


def restore(args):
    root = Path(args.run).expanduser()
    data = load(root)
    record = data.get("archive")
    if not record:
        raise ValueError("Run has no archive record")
    bundle_path = safe_path(root, record["path"])
    if digest(bundle_path) != record["sha256"]:
        raise ValueError("Archive checksum mismatch")
    verify(bundle_path, record["members"])
    destination = Path(args.destination).expanduser()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Restore destination must be a new directory")
    for name in record["members"]:
        safe_path(destination, name)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(bundle_path) as bundle:
        for name in record["members"]:
            target = safe_path(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as source, target.open("xb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
    print(f"Restored completed run to {destination}; checkpoint --status active to resume.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    command = commands.add_parser("init")
    command.add_argument("run")
    command.add_argument("--question", required=True)
    command.add_argument("--scope", required=True)
    command.add_argument("--mode", choices=DEFAULTS, default="standard")
    command.add_argument("--assumption", action="append", default=[])
    command.add_argument("--search-budget", type=positive)
    command.add_argument("--storage-mb", type=positive, default=64)
    command.set_defaults(func=init)
    command = commands.add_parser("checkpoint")
    command.add_argument("run")
    command.add_argument("--status", choices=("active", "complete"))
    command.add_argument("--artifact", action="append", default=[])
    command.add_argument("--completed", action="append")
    command.add_argument("--gaps", action="append")
    command.add_argument("--clear-gaps", action="store_true")
    command.add_argument("--next-action")
    command.set_defaults(func=checkpoint)
    command = commands.add_parser("status")
    command.add_argument("run")
    command.set_defaults(func=status)
    command = commands.add_parser("archive")
    command.add_argument("run")
    command.add_argument("--prune", action="store_true", help="Remove only verified archived working files; keep report and manifest")
    command.set_defaults(func=archive)
    command = commands.add_parser("restore")
    command.add_argument("run")
    command.add_argument("destination")
    command.set_defaults(func=restore)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
