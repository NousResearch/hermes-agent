"""Prepare or consume the standalone PM payload, without desktop products."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def prepare(source: Path, work: Path, cache: Path, out: Path, ref: str) -> Path:
    from hermes_cli.runtime_state import _lock
    from scripts.bundles.desktop_prepare import git, require_source
    from scripts.bundles.desktop_toolchain import run_preparation
    from scripts.bundles.native_prepared import prepared_path
    from pm.lock import _write

    for path in (work, cache, out):
        if path.absolute() != path.resolve():
            raise ValueError("native build roots must not be symlinked")
    source, work, cache, out = (path.resolve() for path in (source, work, cache, out))
    revision = git(source, "rev-parse", "--verify", f"{ref}^{{commit}}")
    require_source(source, revision)
    for destination in (work, cache, out):
        if source.is_relative_to(destination):
            raise ValueError("native build output must not contain the source checkout")
    for left, right in ((work, cache), (out, cache), (work, out)):
        if left.is_relative_to(right) or right.is_relative_to(left):
            raise ValueError("native work, cache and output must be separate directories")
    owner = work / ".native-preparation"
    if work.exists():
        if not owner.is_file() or owner.read_text(encoding="utf-8-sig") != str(source):
            raise ValueError("native work directory is not owned by this checkout")
    else:
        work.mkdir(parents=True)
        owner.write_text(str(source), encoding="utf-8")
    with (work / ".lock").open("a+b") as lock:
        if not _lock(lock.fileno(), wait=False):
            raise ValueError("native preparation work directory is already in use")
        result = prepared_path(out)
        result.unlink(missing_ok=True)
        request = work / "request.json"
        _write(request, {"source": str(source), "work": str(work), "cache": str(cache),
                         "out": str(out), "ref": revision})
        status = run_preparation(source, work, cache, request, worker=Path(__file__).resolve())
        if status:
            result.unlink(missing_ok=True)
            raise RuntimeError(f"native preparation failed (exit {status})")
        if not result.is_file():
            raise RuntimeError("native preparation did not publish its result")
        return result


def prepare_in_worker(request: dict) -> Path:
    from scripts.bundles.desktop_prepare import require_source
    from scripts.bundles.desktop_toolchain import prepare_tools
    from scripts.bundles.native import prepare_native

    source, work, cache, out = (Path(request[name]) for name in ("source", "work", "cache", "out"))
    require_source(source, request["ref"])
    _, _, env = prepare_tools(source, work, cache, os.environ)
    return prepare_native(out=out, ref=request["ref"], source=source,
                          cache=Path(env["UV_CACHE_DIR"]), tools=cache / "tools", env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ref")
    selection.add_argument("--commit")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            if not args.request:
                parser.error("worker requires --request")
            prepare_in_worker(json.loads(args.request.read_text(encoding="utf-8-sig")))
            return 0
        if args.request:
            parser.error("--request is worker-only")
        if args.prepared:
            if any((args.source, args.work, args.cache, args.out, args.ref, args.commit, args.prepare_only)):
                parser.error("--prepared supplies the complete native build request")
            prepared = args.prepared.absolute()
        else:
            if not all((args.source, args.work, args.cache, args.out)):
                parser.error("preparation requires --source, --work, --cache and --out")
            if args.commit:
                from scripts.releases.commit_build import require_commit
                require_commit(args.commit)
            prepared = prepare(args.source, args.work, args.cache, args.out, args.commit or args.ref or "HEAD")
            if args.prepare_only:
                print(prepared)
                return 0
        from scripts.bundles.native import finish_native
        return finish_native(prepared, {})
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"native build: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
