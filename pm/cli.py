"""hermes pm: lock / install / repair / env / doctor / gc / bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from pm.ensure import _facts, _lockfile, _store, ensure, stage_only
from pm.ensure import uv as pm_uv
from pm.package import InstallError
from pm.paths import repo_root
from pm.registry import get_package
from pm.store import ALL_TARGETS, current_target, hash_url
from pm.update import Resolved, resolve_package


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


def _live_progress(name: str):
    """Per-package progress for ensure(): download as % + MiB, unpack as a
    phase line. Throttled to ~4 MiB steps — a slow line proves it's moving
    in a piped (CI) log without flooding it (a 1 MiB tick on a 1.5 GiB
    model would be ~1,500 lines)."""
    last = 0

    def report(stage: str, done: int, total: int, label: str) -> None:
        nonlocal last
        if stage == "unpack":
            last = 0
            print(f"  {name}: unpacking{(' ' + label) if label else ''}", flush=True)
            return
        if total <= 0:
            return
        if done >= total or done - last >= 4 * 1024 * 1024:
            last = done
            print(
                f"  {name}: {done / total * 100:5.1f}%  {_fmt_bytes(done)} / {_fmt_bytes(total)}",
                flush=True,
            )

    return report


def _install_names(names: list[str], target: str | None = None) -> int:
    failed = 0
    for name in names:
        try:
            if target is not None:
                # Cross-target staging: publish the entry, touch no facts.
                entry = stage_only(name, target)
                print(f"✓ {name} (staged for {target}: {entry.name})")
            else:
                ensure(name, explicit=True, progress=_live_progress(name))
                print(f"✓ {name}", flush=True)
        except InstallError as e:
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
    names = args.names or [
        n for n in _lockfile().names() if not get_package(n).optional or n == "python"
    ]
    failed = _install_names(names, target=cross_target)
    if not args.names:
        from pm.ensure import sync_venv

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
    from pm.ensure import env_for

    names = args.names or _lockfile().names()
    print(json.dumps(env_for(*names), indent=2, sort_keys=True))
    return 0


def cmd_doctor(args) -> int:
    from pm.ensure import _identity, _installed_location
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
    """Run cmd with its output streamed through our stdout — a long uv
    venv build must prove liveness in a piped (CI) log, not vanish until
    exit — while still capturing the tail for the failure message. A
    reader thread drains output so proc.wait(timeout) keeps the wall-clock
    kill the old subprocess.run(timeout=) had. Returns (returncode, last
    ~2k chars of combined output)."""
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace",
    )
    tail = ""
    lock = threading.Lock()

    def drain() -> None:
        nonlocal tail
        for line in proc.stdout:
            print(line, end="", flush=True)
            with lock:
                tail = (tail + line)[-2000:]

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise RuntimeError(f"{cmd[0]} timed out after {timeout}s")
    thread.join()
    with lock:
        return code, tail


def cmd_update(args) -> int:
    """`hermes pm update [names...] [--check] [--target T] [--uv] [--npm]`.

    Resolve each package's latest via its own latest_versions() hook,
    intersect across targets, and (real mode) re-pin the lockfile + install
    the changed ones. --check is dry-run: hits upstream indexes, writes
    nothing. --uv / --npm also refresh uv.lock (+sync venv) / package-lock.
    """
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
            from pm.ensure import sync_venv
            sync_venv(explicit=True)
            print("✓ venv")
        except InstallError as e:
            print(f"✗ {e}")
            return 1
    else:
        print("pm update: nothing to update")

    if args.uv:
        uv_bin, env = pm_uv(realize=False)
        if uv_bin is None:
            print("✗ uv: not installed")
            return 1
        code, tail = _run_live([uv_bin, "lock", "--upgrade"], cwd=str(repo_root()), env=env)
        if code != 0:
            print(f"✗ uv lock --upgrade failed:\n{tail}")
            return 1
        print("✓ uv.lock refreshed")
        try:
            from pm.ensure import sync_venv
            sync_venv(explicit=True)
            print("✓ venv")
        except InstallError as e:
            print(f"✗ {e}")
            return 1
    if args.npm:
        from pm.ensure import env_for, installed_package
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


def _pin_artifacts(package, decision, current: dict) -> dict:
    """Retain unresolved targets and reuse hashes for unchanged artifact URLs."""
    per_target = decision.per_target or {t: decision.version for t in ALL_TARGETS}
    artifacts = dict(current)
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
        pinned = [{"url": url, "sha256": known.get(url) or package.known_sha256(version, url) or hash_url(url)}
                  for url in urls]
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
    p.set_defaults(func=cmd_update)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
