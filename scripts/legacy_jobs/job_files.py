#!/usr/bin/env python3
"""Read/write/list files in a legacy job's private data/ folder."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from .job_state import JobPathError, resolve_job_folder
except ImportError:  # script execution
    from job_state import JobPathError, resolve_job_folder

REPO_DIR = Path(__file__).resolve().parents[2]
DATA_SUBDIR = "data"
MAX_FILE_BYTES = int(os.environ.get("JOB_FILES_MAX_FILE_BYTES", 50 * 1024 * 1024))
MAX_DIR_BYTES = int(os.environ.get("JOB_FILES_MAX_DIR_BYTES", 200 * 1024 * 1024))


class JobFileError(ValueError):
    pass


def data_dir(job_path: str | Path, repo_dir: Path = REPO_DIR) -> Path:
    return resolve_job_folder(job_path, repo_dir) / DATA_SUBDIR


def resolve_data_path(job_path: str | Path, name: str | None = None, repo_dir: Path = REPO_DIR) -> Path:
    base = data_dir(job_path, repo_dir).resolve(strict=False)
    if name is None:
        return base
    rel = Path(name)
    if not name or name in {".", ".."}:
        raise JobFileError(f"invalid file name {name!r}")
    if rel.is_absolute():
        raise JobFileError(f"file name {name!r} must be relative to data/")
    target = (base / rel).resolve(strict=False)
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise JobFileError(f"file name {name!r} escapes the job data/ folder") from exc
    return target


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def write_file(job_path: str | Path, name: str, data: bytes, *, append: bool = False, repo_dir: Path = REPO_DIR) -> Path:
    base = data_dir(job_path, repo_dir)
    target = resolve_data_path(job_path, name, repo_dir)
    if len(data) > MAX_FILE_BYTES:
        raise JobFileError(f"input {len(data)} bytes exceeds per-file cap {MAX_FILE_BYTES}")
    old_target_size = target.stat().st_size if target.exists() else 0
    projected_file_size = old_target_size + len(data) if append else len(data)
    if projected_file_size > MAX_FILE_BYTES:
        raise JobFileError(f"file would be {projected_file_size} bytes, over cap {MAX_FILE_BYTES}")
    projected_dir_size = _dir_size(base) - (0 if append else old_target_size) + len(data)
    if projected_dir_size > MAX_DIR_BYTES:
        raise JobFileError(f"write would bring data/ to {projected_dir_size} bytes, over cap {MAX_DIR_BYTES}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if append:
        with target.open("ab") as fh:
            fh.write(data)
    else:
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
    return target


def read_file(job_path: str | Path, name: str, repo_dir: Path = REPO_DIR) -> bytes:
    target = resolve_data_path(job_path, name, repo_dir)
    if not target.exists():
        raise JobFileError(f"no such file: data/{name}")
    return target.read_bytes()


def list_files(job_path: str | Path, repo_dir: Path = REPO_DIR) -> list[tuple[str, int, str]]:
    base = data_dir(job_path, repo_dir)
    if not base.exists():
        return []
    out: list[tuple[str, int, str]] = []
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        st = path.stat()
        mtime = datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.append((path.relative_to(base).as_posix(), st.st_size, mtime))
    return out


def delete_file(job_path: str | Path, name: str, repo_dir: Path = REPO_DIR) -> bool:
    target = resolve_data_path(job_path, name, repo_dir)
    if target.exists():
        target.unlink()
        return True
    return False


def _die(exc: Exception) -> int:
    print(f"error: {exc}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("write")
    sp.add_argument("job")
    sp.add_argument("name")
    sp.add_argument("--append", action="store_true")
    sp = sub.add_parser("read")
    sp.add_argument("job")
    sp.add_argument("name")
    sp = sub.add_parser("list")
    sp.add_argument("job")
    sp = sub.add_parser("delete")
    sp.add_argument("job")
    sp.add_argument("name")
    sp = sub.add_parser("path")
    sp.add_argument("job")
    sp.add_argument("name", nargs="?")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "write":
            target = write_file(args.job, args.name, sys.stdin.buffer.read(), append=args.append)
            verb = "appended to" if args.append else "wrote"
            print(f"{verb} {target.relative_to(REPO_DIR).as_posix()} ({target.stat().st_size} bytes)")
        elif args.cmd == "read":
            sys.stdout.buffer.write(read_file(args.job, args.name))
        elif args.cmd == "list":
            for name, size, mtime in list_files(args.job):
                print(f"{size:>12}  {mtime}  {name}")
        elif args.cmd == "delete":
            deleted = delete_file(args.job, args.name)
            print(("deleted" if deleted else "nothing to delete:") + f" data/{args.name}")
        elif args.cmd == "path":
            print(resolve_data_path(args.job, args.name).as_posix())
    except (OSError, JobPathError, JobFileError) as exc:
        return _die(exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
