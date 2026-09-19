"""hermes pm: lock / install / repair / env / doctor / gc / bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pm import termux_libs
from pm.install import _facts, _lockfile, _store, ensure, stage_only
from pm.operations import lock_project
from pm.package import InstallError
from pm.paths import repo_root
from pm.registry import get_package, source_install_packages
from pm.store import ALL_TARGETS, current_target, hash_url
from pm.update import Resolved, resolve_package, reuse_index_responses


@reuse_index_responses()
def cmd_lock(args) -> int:
    """--bump <name> <version>: resolve every target's archives, hash them,
    write. A target with one archive pins the object; several pin a list.
    Target-independent urls collapse to one "any" artifact."""
    lockfile = _lockfile()
    package = get_package(args.name)
    artifacts: dict[str, object] = {}

    def pin(url: str) -> dict:
        print(f"    {url}")
        digest = package.known_sha256(args.version, url) or hash_url(url)
        print(f"      sha256 {digest}")
        return {"url": url, "sha256": digest}

    urls = {
        target: package.fetch_urls(args.version, target)
        for target in ALL_TARGETS
        if package.missing_reason(target) is None
    }
    distinct = {tuple(u) for u in urls.values()}
    if len(distinct) == 1:
        print("  any:")
        pinned = [pin(url) for url in next(iter(urls.values()))]
        artifacts["any"] = pinned[0] if len(pinned) == 1 else pinned
    else:
        for target, target_urls in urls.items():
            print(f"  {target}:")
            pinned = [pin(url) for url in target_urls]
            artifacts[target] = pinned[0] if len(pinned) == 1 else pinned
    lockfile.set_pin(args.name, args.version, artifacts)
    lockfile.save()
    print(f"pinned {args.name} {args.version} ({len(artifacts)} targets)")
    return 0


def _fmt_bytes(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MiB"


def _progress_stream():
    """Stream for in-place progress, or None off a terminal. Prefer stdout;
    fall back to stderr because activate.ps1 pipes stdout through Out-Host
    while stderr stays on the console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.isatty():
                return stream
        except (AttributeError, ValueError):
            continue
    return None


def _interactive() -> bool:
    return _progress_stream() is not None


def _live_progress(name: str):
    """Per-package progress for ensure(): download as % + MiB, unpack as a
    phase line. On a terminal the line redraws in place (~10 Hz); in a piped
    (CI) log each tick prints its own line, throttled to ~4 MiB steps so a
    slow line proves it's moving without flooding the log (a 1 MiB tick on a
    1.5 GiB model would be ~1,500 lines). ``finish()`` returns the cursor to
    a clean line so the caller's status glyph starts on its own row."""
    last = 0
    last_time = 0.0
    stream = _progress_stream()

    def write(line: str) -> None:
        if stream is not None:
            stream.write("\r\x1b[2K" + line)
            stream.flush()
        else:
            print(line, flush=True)

    def finish() -> None:
        if stream is not None:
            stream.write("\r\x1b[2K")
            stream.flush()

    def report(stage: str, done: int, total: int, label: str) -> None:
        nonlocal last, last_time
        if stage == "unpack":
            last = 0
            last_time = 0.0
            write(f"  {name}: unpacking{(' ' + label) if label else ''}")
            return
        if total <= 0:
            return
        if done < total:
            if stream is not None:
                now = time.monotonic()
                if now - last_time < 0.1:
                    return
                last_time = now
            elif done - last < 4 * 1024 * 1024:
                return
        last = done
        write(f"  {name}: {done / total * 100:5.1f}%  {_fmt_bytes(done)} / {_fmt_bytes(total)}")

    report.finish = finish  # type: ignore[attr-defined]
    return report


def _install_names(names: list[str], target: str | None = None) -> int:
    from pm.install import _install_operation

    failed = 0
    with _install_operation() as operation:
        for name in names:
            progress = _live_progress(name)
            try:
                if target is not None:
                    # Cross-target staging: publish the entry, touch no facts.
                    entry = stage_only(name, target)
                    print(f"✓ {name} (staged for {target}: {entry.name})")
                else:
                    ensure(name, explicit=True, progress=progress, _operation=operation)
                    if name == "python":
                        from hermes_cli.venv_sync import publish_launchers

                        publish_launchers(repo_root(), create=False)
                    progress.finish()
                    print(f"✓ {name}", flush=True)
            except InstallError as e:
                progress.finish()
                print(f"✗ {e}", flush=True)
                failed += 1
    return failed




def cmd_install(args) -> int:
    cross_target = getattr(args, "target", None)
    if cross_target:
        if cross_target not in ALL_TARGETS:
            print(f"✗ unknown target {cross_target!r}; known: {', '.join(ALL_TARGETS)}")
            return 1
        if not args.names:
            print("✗ --target requires explicit package names")
            return 1
    # Source-install launchers require the store interpreter, even though
    # Python remains optional when provisioning individual tools.
    names = args.names or source_install_packages(_lockfile().names())
    failed = _install_names(names, target=cross_target)
    if not args.names:
        from pm.install import sync_venv

        try:
            # Default the venv to the [all] feature set — the same thing
            # `hermes update` force-syncs on every run (update_cmd.py) and
            # the installers' old `--extra all` did. sync_venv unions, so
            # any lazy extras already recorded survive this; it only makes
            # a fresh bootstrap match what the first update would do.
            sync_venv(["all"], explicit=True)
            print("✓ venv")
        except InstallError as e:
            print(f"✗ {e}")
            failed += 1
    return 1 if failed else 0


def cmd_env(args) -> int:
    from pm.install import env_for

    names = args.names or _lockfile().names()
    print(json.dumps(env_for(*names), indent=2, sort_keys=True))
    return 0


def cmd_doctor(args) -> int:
    from pm.install import _identity, _installed_location
    from pm.store import tree_digest

    lockfile = _lockfile()
    facts = _facts()
    store = _store()
    target = current_target()
    bad = 0
    for name in lockfile.names():
        package = get_package(name)
        reason = package.missing_reason(target)
        if reason is not None:
            print(f"- {name}: n/a on {target} ({reason})")
            continue
        facts, store = _installed_location(package, lockfile, target) or (_facts(), _store())
        fact = facts.get(name)
        soft = package.optional or package.internal
        identity = _identity(lockfile, name, target)
        if (
            fact is not None
            and identity is not None
            and ("target" not in fact or "artifacts" not in fact)
        ):
            # Legacy fact: pre-dates digest-bound identity; installed()
            # treats it as not installed and forces one reinstall.
            print(f"{'?' if soft else '✗'} {name}: legacy fact: no recorded identity, run `hermes pm install`")
            bad += 0 if soft else 1
            continue
        if not facts.installed(name, lockfile.version(name), store.root, identity):
            state = "not installed" if fact is None else "outdated"
            print(f"{'?' if soft else '✗'} {name}: {state}")
            bad += 0 if soft else 1
            continue
        entry = store.entry(fact["entry"])
        reason = package.verify(entry, target)
        if reason:
            print(f"✗ {name}: installed but failed verification: {reason}")
            bad += 1
            continue
        recorded = fact.get("digest")
        if recorded is not None and tree_digest(entry) != recorded:
            # Doctor is the expensive-path tool: re-hash the realized
            # bytes. Boot checks stay O(1) json compares.
            print(f"✗ {name}: realized bytes do not match recorded digest")
            bad += 1
            continue
        print(f"✓ {name} {fact['version']}")
    return 1 if bad else 0


def _gc_store(store, facts) -> tuple[int, int]:
    """The sweep core shared by `pm gc` and `pm bundle`.

    Removes every store entry nothing references: fetch-<sha> download-cache
    dirs (the raw archives — needed only at install time, dead weight in a
    staged payload or a CI cache), orphaned package versions from an older
    lock, and expired partials. Keeps live package entries (recorded in
    facts) and partials an in-flight download still owns. Returns
    (removed, kept).
    """
    from pm.download_state import collect_partials
    from pm import paths

    partials_dir = paths.partials_root()
    if not store.root.is_dir() and not partials_dir.is_dir():
        return (0, 0)
    removed = 0
    with store.install_lock():
        facts.reload()
        keep = facts.entries_in_use()
        collect_partials(partials_dir)
        for item in sorted(store.root.iterdir()):
            if not item.is_dir() or item.name.startswith("."):
                continue
            if item.name in keep:
                continue
            print(f"removing {item.name}")
            shutil.rmtree(item, ignore_errors=True)
            removed += 1
    return (removed, len(keep))


def cmd_gc(args) -> int:
    from pm.paths import writable_store_root
    from pm.lock import Facts
    from pm.store import Store
    store = Store(writable_store_root())
    facts = _facts() if store.root == _store().root else Facts(store.root / "facts.json")
    removed, kept = _gc_store(store, facts)
    from hermes_cli.runtime_state import collect_generations
    from pm.paths import repo_root
    generations = collect_generations(repo_root())
    print(f"gc: removed {removed}, kept {kept}; removed {len(generations)} dependency generations")
    return 0


def _run_live(cmd: list[str], *, cwd, env, timeout: int = 3600) -> tuple[int, str]:
    """Stream CI progress with the Python engine's bounded drain and diagnostic tail."""
    from pm.environment import _run_streaming

    try:
        result = _run_streaming(cmd, cwd=cwd, env=env, timeout=timeout, output=sys.stdout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{cmd[0]} timed out after {timeout}s") from exc
    return result.returncode, result.stderr


def cmd_update(args) -> int:
    """`hermes pm update [names...] [--check] [--target T] [--uv] [--npm] [--termux]`.

    Resolve each package's latest via its own latest_versions() hook,
    intersect across targets, and (real mode) re-pin the lockfile + install
    the changed ones. --check is dry-run: hits upstream indexes, writes
    nothing. --uv / --npm also refresh uv.lock (+sync venv) / package-lock.
    --termux is its own repair pass: it repins only the pool archives the
    rolling termux-main pool has retired under our pins.
    """
    if args.termux:
        ignored = [name for name, given in (("names", args.names), ("--target", args.target),
                                            ("--uv", args.uv), ("--npm", args.npm)) if given]
        if ignored:
            print(f"::warning::--termux repairs the termux pool pins only; ignoring {', '.join(ignored)}")
        return _termux_pass(check=args.check)
    lockfile = _lockfile()
    names = args.names or [n for n in lockfile.names() if not get_package(n).internal or n == "uv"]
    if args.target and not args.check:
        print("::warning::--target is a CHECK-only cross-resolution flag; ignoring it for apply (the lockfile pins every target)")
        args.target = None
    target = args.target or current_target()

    resolved = []
    failures = []
    for name in names:
        package = get_package(name)
        targets = [t for t in ALL_TARGETS if package.missing_reason(t) is None]
        if args.target:  # cross-target check: only the requested target matters
            targets = [t for t in targets if t == args.target]
        if not targets:
            continue
        try:
            decision = resolve_package(package, targets, lockfile.version(name),
                                       artifacts=lockfile.pinned_artifacts(name))
        except Exception as e:  # an upstream index outage must not kill the whole check
            decision = Resolved(name, lockfile.version(name), package.version_style, reason=f"resolve failed: {e}")
            failures.append(name)
        resolved.append(decision)

    # ── report ────────────────────────────────────────────────────────────
    changed = [d for d in resolved if d.changed]
    if not resolved:
        print("pm update: nothing to check (no resolvable packages)")
        return 0
    width = max(len(d.name) for d in resolved)
    for d in resolved:
        if d.version is None:
            print(f"{d.name:<{width}}  {d.reason or 'up to date'}")
        elif d.changed:
            per = ""
            if d.per_target and len(set(d.per_target.values())) > 1:
                per = " (" + ", ".join(f"{t}={v}" for t, v in sorted(d.per_target.items())) + ")"
            if d.version == d.locked and d.artifact_updates:
                print(f"{d.name:<{width}}  {d.version}: newer artifacts for {', '.join(sorted(d.artifact_updates))}")
            else:
                print(f"{d.name:<{width}}  {d.locked or '—'} → {d.version}{per}")
        else:
            print(f"{d.name:<{width}}  {d.locked} up to date")
    if failures:
        print(f"pm update: resolution failed for {', '.join(failures)}; no changes applied")
        return 1
    if args.check:
        if args.uv:
            print("uv deps: would run `uv lock --upgrade` + venv sync")
        if args.npm:
            print("npm deps: would run `npm update`")
        return 1 if changed else 0

    # ── apply ─────────────────────────────────────────────────────────────
    if changed:
        for d in changed:
            package = get_package(d.name)
            artifacts = _pin_artifacts(package, d, lockfile.pinned_artifacts(d.name))
            lockfile.set_pin(d.name, d.version, artifacts)
            print(f"✓ {d.name} pinned {d.locked or '—'} → {d.version}")
        lockfile.save()
        failed = _install_names([d.name for d in changed])
        if failed:
            return 1
        try:
            from pm.install import sync_venv
            sync_venv(explicit=True)
            print("✓ venv")
        except InstallError as e:
            print(f"✗ {e}")
            return 1
    else:
        print("pm update: nothing to update")

    if args.uv:
        try:
            lock_project(repo_root(), upgrade=True, explicit=True)
        except InstallError as exc:
            print(f"✗ Python lock refresh failed: {exc}")
            return 1
        print("✓ uv.lock refreshed")
        try:
            from pm.install import sync_venv
            sync_venv(explicit=True)
            print("✓ venv")
        except InstallError as e:
            print(f"✗ {e}")
            return 1
    if args.npm:
        from pm.install import env_for, installed_package
        from pm.packages import npm_env
        from pm.paths import writable_store_root

        npm = installed_package("npm")
        node = installed_package("node")
        if npm is None or npm.binary is None or node is None or node.binary is None:
            print("✗ npm or Node: not installed; run `hermes pm install`")
            return 1
        env = npm_env(writable_store_root() / ".npm-cache", env_for("npm"))
        code, tail = _run_live([str(npm.binary), "update"], cwd=str(repo_root()), env=env)
        if code != 0:
            print(f"✗ npm update failed:\n{tail}")
            return 1
        print("✓ package-lock.json refreshed")
    return 0


def _termux_pass(*, check: bool) -> int:
    """Repin the pool archives Termux has retired under our pins.

    The bionic lock rows and the runtime-lib table have no shared version axis
    to resolve (each pins what its own supplier ships), so this is a repair,
    not an update: only rows whose archive is gone are touched.
    """
    table = termux_libs.load_table()
    lockfile = _lockfile()
    total = len(termux_libs.pins(table, lockfile))
    stale = termux_libs.retired(table, lockfile, termux_libs.index())
    if not stale:
        print(f"termux pins: {total} rows still served by the pool")
        return 0
    unresolved = [entry for entry in stale if entry.replacement is None]
    width = max(len(entry.pin.name) for entry in stale)
    for entry in stale:
        if entry.replacement is None:
            print(f"{entry.pin.name:<{width}}  {entry.pin.version}: the pool no longer carries it")
        else:
            print(f"{entry.pin.name:<{width}}  {entry.pin.version} → {entry.replacement.version}")
    if check:
        return 1
    applied = termux_libs.repair(table, lockfile, stale)
    if applied:
        termux_libs.save_table(table)
        lockfile.save()
        print(f"✓ {applied} termux pins repinned; restage the bionic payload to pick them up")
    return 1 if unresolved else 0


@reuse_index_responses()
def _pin_artifacts(package, decision, current: dict) -> dict:
    """Retain unresolved targets and reuse hashes for unchanged artifact URLs."""
    per_target = decision.per_target or {t: decision.version for t in ALL_TARGETS}
    artifacts = dict(current)
    hashes: dict[str, str] = {}
    for target, version in per_target.items():
        if package.missing_reason(target) is not None:
            continue
        old = current.get(target, current.get("any", []))
        old = old if isinstance(old, list) else [old]
        known = {row["url"]: row["sha256"] for row in old}
        urls = decision.artifact_updates.get(target)
        if urls is None:
            urls = package.fetch_urls(version, target)
        if urls == [row["url"] for row in old]:
            continue
        pinned = []
        for url in urls:
            digest = known.get(url) or hashes.get(url)
            if not digest:
                digest = package.known_sha256(version, url) or hash_url(url)
                hashes[url] = digest
            pinned.append({"url": url, "sha256": digest})
        artifacts[target] = pinned[0] if len(pinned) == 1 else pinned
    return artifacts


def cmd_status(args) -> int:
    """Print the latest pm sync receipt — the reader surface for the
    CLI/TUI/desktop (same schema as update receipts; a failed venv
    rebuild or a plugin bisect is as reportable as a failed update)."""
    import json as _json

    from pm import receipt

    data = receipt.latest()
    if data is None:
        print("no pm sync receipt yet (no venv operation has run)")
        return 0
    print(_json.dumps(data, indent=2))
    return 0


def cmd_repair(args) -> int:
    from hermes_cli._early_recovery import recover_if_needed
    from pm.paths import repo_root

    if not recover_if_needed(repo_root(), explicit=True):
        return 1
    print("Restart Hermes to use the repaired dependency environment.")
    return 0


def cmd_bundle(args) -> int:
    from scripts.bundles.native import stage_native
    return stage_native(args)



def main(argv=None) -> int:
    # Windows consoles default to cp1252; pm prints ✓/✗. Never let the
    # status glyphs crash the command reporting them. line_buffering:
    # pm output must stream live in a piped (CI) log, not sit in a block
    # buffer and flush only at exit.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace", line_buffering=True)
        except (AttributeError, OSError):
            pass
    parser = argparse.ArgumentParser(prog="hermes pm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("lock", help="write versions+hashes into pm/lock.json")
    p.add_argument("--bump", dest="name", required=True)
    p.add_argument("version")
    p.set_defaults(func=cmd_lock)

    p = sub.add_parser("install", help="install packages (default: all required)")
    p.add_argument("names", nargs="*")
    p.add_argument(
        "--target",
        help="stage for a cross target (e.g. linux-arm64-bionic on a glibc "
        "CI host); requires explicit package names",
    )
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("env", help="print composed env of installed packages")
    p.add_argument("names", nargs="*")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("doctor", help="check installed state against the lockfile")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("repair", help="rebuild the recorded dependency environment without changing its graph")
    p.set_defaults(func=cmd_repair)

    p = sub.add_parser("gc", help="remove store entries nothing references")
    p.set_defaults(func=cmd_gc)

    p = sub.add_parser("bundle", help="stage a payload (repo+store+facts+relocatable venv) into --out")
    p.add_argument("--out", required=True)
    p.add_argument("--ref", help="git ref for the repo snapshot (default HEAD)")
    p.add_argument("--cache", type=Path, help="persistent build cache (default: UV_CACHE_DIR or PM's shared cache)")
    p.set_defaults(func=cmd_bundle)

    p = sub.add_parser("status", help="print the latest pm sync receipt (machine-readable)")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("update", help="resolve latest versions and re-pin the lockfile")
    p.add_argument("names", nargs="*", help="packages to check/update (default: all with a latest source)")
    p.add_argument("--check", action="store_true",
                   help="dry-run: print what would change, write nothing (exit 1 if updates exist)")
    p.add_argument("--target", help="resolve for a different target instead of this machine (e.g. win32-arm64)")
    p.add_argument("--uv", action="store_true", help="also refresh uv.lock + venv (uv update + sync)")
    p.add_argument("--npm", action="store_true", help="also refresh package-lock.json (npm update)")
    p.add_argument("--termux", action="store_true",
                   help="repin the termux pool archives the pool has retired (the runtime-lib "
                        "table + the bionic lock rows); --check reports without writing")
    p.set_defaults(func=cmd_update)

    args = parser.parse_args(argv)
    from pm.runtime import is_runtime, run_cli

    try:
        if not is_runtime():
            return run_cli(list(sys.argv[1:] if argv is None else argv))
        return args.func(args)
    except InstallError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
