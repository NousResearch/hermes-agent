"""Build and inspect explicit Python dependency environments through PM."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys


def _record_selection(source: Path, extras: list[str]) -> None:
    """Publish the built extras as the store's venv selection, as a bundle build does.

    A sync that finds no runtime selection starts from this one, so the first opt-in
    extra installed on use extends the shipped environment instead of replacing it
    with a generation that carries only that extra. The frozen build installed exactly
    this list from the lock, so it is recorded as given; an anchor inventory
    (``installed_extras``) misses meta extras such as ``all`` and claims extras whose
    anchors other packages happen to provide.
    """
    from pm.lock import Facts
    from pm.packages import Venv
    from pm.paths import facts_path

    selected = sorted(set(extras))
    stamp = Venv(source.resolve()).expected_stamp(selected, plugin_dirs=[])
    Facts(facts_path()).record_state("venv", stamp, selected)


def _check_record_selection(parser: argparse.ArgumentParser, args: argparse.Namespace, *,
                            build: bool, requirements: list[str]) -> None:
    """Only a project build from an explicit --extra list has a selection to record."""
    if args.record_selection and (not build or requirements or args.manager_runtime
                                  or args.all_extras or args.groups or not args.extras):
        parser.error("--record-selection records the --extra list of a project environment build")


def main(argv: Sequence[str] | None = None) -> int:
    import pm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--extra", dest="extras", action="append", default=[])
    parser.add_argument("--group", dest="groups", action="append", default=[])
    parser.add_argument("--all-extras", action="store_true")
    parser.add_argument("--no-install-project", action="store_true")
    parser.add_argument("--resolve", action="store_true", help="resolve the source lock before building")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--lock-only", action="store_true")
    operation.add_argument("--check-lock", action="store_true")
    operation.add_argument("--export-requirements", type=Path)
    operation.add_argument("--prune-cache", action="store_true")
    operation.add_argument("--exact-lock", action="store_true",
                           help="prune to the project lock: entries the uv.lock cannot resolve are deleted")
    operation.add_argument("--manager-runtime", action="store_true")
    parser.add_argument("--lock-source", type=Path, default=None,
                        help="repo whose uv.lock selects the kept entries for --exact-lock (default: cwd)")
    parser.add_argument("--upgrade", action="store_true")
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--sealed", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--explicit", action="store_true", help="CLI builds are always explicit requests")
    parser.add_argument("--record-selection", action="store_true",
                        help="record the --extra list as this install's baseline dependency selection")
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--requirements", type=Path)
    parser.add_argument("--requirement", action="append", default=[])
    args = parser.parse_args(argv)
    requirements = list(args.requirement)
    if args.requirements is not None:
        requirements.extend(line.strip() for line in args.requirements.read_text(encoding="utf-8-sig").splitlines()
                            if line.strip() and not line.lstrip().startswith("#"))
    if args.exact_lock:
        if args.ci or args.prune_cache:
            parser.error("--exact-lock replaces --ci/--prune-cache: downloaded wheels the lock keeps must survive")
        if args.lock_source is None:
            parser.error("--exact-lock requires --lock-source")
        if args.cache is None:
            parser.error("--exact-lock requires --cache")
    elif args.prune_cache:
        if args.cache is None:
            parser.error("--prune-cache requires --cache")
    elif not requirements and args.source is None:
        parser.error("--source is required for project operations")
    build = not (args.lock_only or args.check_lock or args.export_requirements
                 or args.prune_cache or args.exact_lock)
    if build and args.out is None:
        parser.error("--out is required when building an environment")
    if args.manager_runtime and args.python is None:
        parser.error("--manager-runtime requires the target --python")
    _check_record_selection(parser, args, build=build, requirements=requirements)
    try:
        if args.exact_lock:
            from pm.uv_cache_prune import prune_uv_cache_to_lock

            pruned = prune_uv_cache_to_lock(args.cache, args.lock_source)
            print(f"pruned {pruned} cache entries outside the lock")
            return 0
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
                extras=args.extras, groups=args.groups, all_extras=args.all_extras,
                no_install_project=args.no_install_project, frozen=not args.resolve,
                sealed=args.sealed, offline=args.offline, explicit=True,
            )
            if args.record_selection:
                _record_selection(args.source, args.extras)
    except (pm.InstallError, OSError, ValueError) as exc:
        print(f"python environment: {exc}", file=sys.stderr)
        return 1
    print(executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
