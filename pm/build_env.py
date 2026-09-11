"""Build and inspect explicit Python dependency environments through PM."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys


def main(argv: Sequence[str] | None = None) -> int:
    import pm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--extra", dest="extras", action="append", default=[])
    parser.add_argument("--group", dest="groups", action="append", default=[])
    parser.add_argument("--only-groups", action="store_true", help="install selected groups without the application")
    parser.add_argument("--all-extras", action="store_true")
    parser.add_argument("--no-install-project", action="store_true")
    parser.add_argument("--resolve", action="store_true", help="resolve the source lock before building")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--lock-only", action="store_true")
    operation.add_argument("--check-lock", action="store_true")
    operation.add_argument("--export-requirements", type=Path)
    operation.add_argument("--prune-cache", action="store_true")
    operation.add_argument("--manager-runtime", action="store_true")
    parser.add_argument("--upgrade", action="store_true")
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--sealed", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--explicit", action="store_true", help="CLI builds are always explicit requests")
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--requirements", type=Path)
    parser.add_argument("--requirement", action="append", default=[])
    args = parser.parse_args(argv)
    requirements = list(args.requirement)
    if args.requirements is not None:
        requirements.extend(line.strip() for line in args.requirements.read_text(encoding="utf-8").splitlines()
                            if line.strip() and not line.lstrip().startswith("#"))
    if args.prune_cache:
        if args.cache is None:
            parser.error("--prune-cache requires --cache")
    elif not requirements and args.source is None:
        parser.error("--source is required for project operations")
    build = not (args.lock_only or args.check_lock or args.export_requirements or args.prune_cache)
    if build and args.out is None:
        parser.error("--out is required when building an environment")
    if args.manager_runtime and args.python is None:
        parser.error("--manager-runtime requires the target --python")
    try:
        if args.prune_cache:
            pm.prune_cache(args.cache, ci=args.ci)
            return 0
        if args.check_lock:
            pm.check_project_lock(args.source, python=args.python, cache=args.cache,
                                  offline=args.offline, explicit=True)
            return 0
        if args.export_requirements:
            pm.export_requirements(args.source, args.export_requirements, extras=args.extras,
                                   python=args.python, cache=args.cache, explicit=True)
            return 0
        if args.lock_only:
            pm.lock_project(args.source, upgrade=args.upgrade, python=args.python,
                            cache=args.cache, offline=args.offline, explicit=True)
            return 0
        if args.manager_runtime:
            executable = pm.stage_manager_runtime(python=args.python, destination=args.out,
                                                  project=args.source, cache=args.cache,
                                                  offline=args.offline, wheelhouse=args.wheelhouse)
        elif requirements:
            executable = pm.build_requirements_environment(
                requirements, out=args.out, python=args.python, cache=args.cache,
                wheelhouse=args.wheelhouse, offline=args.offline, sealed=args.sealed, explicit=True,
            )
        else:
            executable = pm.build_environment(
                source=args.source, out=args.out, python=args.python, cache=args.cache,
                extras=args.extras, groups=args.groups, only_groups=args.only_groups, all_extras=args.all_extras,
                no_install_project=args.no_install_project, frozen=not args.resolve,
                sealed=args.sealed, offline=args.offline, explicit=True,
            )
    except (pm.InstallError, OSError, ValueError) as exc:
        print(f"python environment: {exc}", file=sys.stderr)
        return 1
    print(executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
