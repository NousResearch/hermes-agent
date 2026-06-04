"""Data-driven prod/dev volume split, driven by the client registry.

Replaces the hardcoded profile lists that ad-hoc deploy scripts carry. Given
the declarative registry (see :mod:`hermes_cli.clients`), this derives which
profiles belong to ``prod`` vs ``dev``, mirrors the source data volume into
per-environment volumes, and prunes any profile *not* in the registry for
that environment.

Safety model
------------
* **The source volume is only ever read.** Pruning happens exclusively on the
  derived targets. Refuses to run if a target equals/contains/is contained by
  the source.
* **Dry-run by default.** Nothing mutates unless ``apply=True`` (``--apply`` on
  the CLI). The plan is always printed first.
* **Missing-profile guard.** If the registry names a prod profile that doesn't
  exist on the source, that's almost certainly a typo that would silently
  drop a client — so ``apply`` aborts unless ``allow_missing=True``.
* The ``default`` profile is the volume root (not under ``profiles/``), so it
  is never a prune candidate.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from hermes_cli.clients import Registry, load_registry

# Injection seam for tests: a callable that runs an argv and raises on failure.
Runner = Callable[[Sequence[str]], None]


def _default_runner(argv: Sequence[str]) -> None:
    subprocess.run(list(argv), check=True)


@dataclass(frozen=True)
class SplitPlan:
    env: str
    target: Path
    keep: tuple[str, ...]      # profiles to retain (present on source, in registry)
    prune: tuple[str, ...]     # profiles present but not in registry for this env
    missing: tuple[str, ...]   # registry profiles absent from the source volume


def discover_profiles(data_dir: str | os.PathLike[str]) -> set[str]:
    """Names of profile directories under ``<data_dir>/profiles/``."""
    profiles_root = Path(data_dir) / "profiles"
    if not profiles_root.is_dir():
        return set()
    return {p.name for p in profiles_root.iterdir() if p.is_dir()}


def compute_plan(
    registry: Registry,
    env: str,
    target: str | os.PathLike[str],
    present_profiles: set[str],
) -> SplitPlan:
    """Compute keep/prune/missing for one environment.

    ``present_profiles`` is the set of profile dirs found on the *source*
    volume (the basis for both copy and prune decisions). The ``default``
    profile lives at the root, so it never appears here and is never pruned.
    """
    wanted = {p for p in registry.profiles_for_env(env) if p != "default"}
    keep = sorted(wanted & present_profiles)
    prune = sorted(present_profiles - wanted)
    missing = sorted(wanted - present_profiles)
    return SplitPlan(
        env=env,
        target=Path(target),
        keep=tuple(keep),
        prune=tuple(prune),
        missing=tuple(missing),
    )


def _assert_safe_target(source: Path, target: Path) -> None:
    source = source.resolve()
    target = target.resolve()
    if source == target:
        raise ValueError(f"target {target} must differ from source {source}")
    # Disallow nesting in either direction — copying/pruning would corrupt one.
    if source in target.parents:
        raise ValueError(f"target {target} must not live inside source {source}")
    if target in source.parents:
        raise ValueError(f"source {source} must not live inside target {target}")


def _mirror(source: Path, target: Path, runner: Runner) -> None:
    """Mirror ``source`` → ``target`` (target becomes an exact copy).

    Prefers ``rsync -a --delete`` (preserves perms/ownership, handles an
    existing target); falls back to a fresh ``cp -a`` when rsync is absent.
    """
    if shutil.which("rsync"):
        # Trailing slash on source copies contents into target.
        runner(["rsync", "-a", "--delete", f"{source}/", f"{target}/"])
    else:
        if target.exists():
            runner(["rm", "-rf", str(target)])
        runner(["cp", "-a", str(source), str(target)])


def backup_source(source: Path, runner: Runner) -> Path:
    """Tar-snapshot the source volume next to it; returns the archive path."""
    import datetime

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = source.parent / f"{source.name}-backup-{stamp}.tgz"
    runner(["tar", "-C", str(source.parent), "-czf", str(archive), source.name])
    return archive


def run_split(
    registry: Registry,
    source: str | os.PathLike[str],
    targets: dict[str, str | os.PathLike[str]],
    *,
    apply: bool = False,
    backup: bool = False,
    allow_missing: bool = False,
    runner: Runner = _default_runner,
    out=sys.stdout,
) -> list[SplitPlan]:
    """Execute (or dry-run) the prod/dev split.

    ``targets`` maps env name -> target path, e.g.
    ``{"prod": "/opt/data-prod", "dev": "/opt/data-dev"}``.
    Returns the per-env plans. Mutates the filesystem only when ``apply``.
    """
    source = Path(source)
    if not source.is_dir():
        raise ValueError(f"source data dir not found: {source}")

    for target in targets.values():
        _assert_safe_target(source, Path(target))

    present = discover_profiles(source)
    plans = [
        compute_plan(registry, env, target, present)
        for env, target in targets.items()
    ]

    mode = "APPLY" if apply else "DRY-RUN"
    print(f"[{mode}] source={source} profiles_present={sorted(present) or '∅'}", file=out)

    blocking_missing = [p for plan in plans for p in plan.missing]
    for plan in plans:
        print(
            f"  {plan.env:<4} -> {plan.target}\n"
            f"        keep:   {list(plan.keep) or '∅'}\n"
            f"        prune:  {list(plan.prune) or '∅'}"
            + (f"\n        MISSING (in registry, not on source): {list(plan.missing)}"
               if plan.missing else ""),
            file=out,
        )

    if blocking_missing and not allow_missing:
        msg = (
            "registry names profiles that don't exist on the source volume: "
            f"{sorted(set(blocking_missing))}. This usually means a typo that "
            "would silently drop a client. Fix the registry or pass "
            "allow_missing=True / --allow-missing to proceed anyway."
        )
        if apply:
            raise ValueError(msg)
        print(f"  WARNING: {msg}", file=out)

    if not apply:
        print("  (dry-run: no changes made; pass --apply to execute)", file=out)
        return plans

    if backup:
        archive = backup_source(source, runner)
        print(f"  backed up source -> {archive}", file=out)

    for plan in plans:
        target = plan.target
        _mirror(source, target, runner)
        for name in plan.prune:
            victim = target / "profiles" / name
            print(f"  prune {plan.env}: removing {victim}", file=out)
            shutil.rmtree(victim, ignore_errors=True)
        print(f"  {plan.env} ready at {target} (kept {list(plan.keep) or '∅'})", file=out)

    return plans


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hermes-client-split",
        description="Split the Hermes data volume into prod/dev by client registry.",
    )
    parser.add_argument("--registry", help="path to clients.yaml (default: $HERMES_CLIENTS_REGISTRY)")
    parser.add_argument("--source", default="/opt/data", help="source data volume (read-only)")
    parser.add_argument("--prod-target", default="/opt/data-prod")
    parser.add_argument("--dev-target", default="/opt/data-dev")
    parser.add_argument("--apply", action="store_true", help="actually perform the split (default: dry-run)")
    parser.add_argument("--backup", action="store_true", help="tar-snapshot the source before applying")
    parser.add_argument("--allow-missing", action="store_true",
                        help="proceed even if the registry names profiles absent from source")
    args = parser.parse_args(argv)

    try:
        registry = load_registry(args.registry)
        run_split(
            registry,
            source=args.source,
            targets={"prod": args.prod_target, "dev": args.dev_target},
            apply=args.apply,
            backup=args.backup,
            allow_missing=args.allow_missing,
        )
    except Exception as exc:  # surface a clean one-line error to the operator
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
