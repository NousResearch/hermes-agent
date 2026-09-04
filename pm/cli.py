"""hermes pm: lock / install / env / doctor / gc / bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from pm.ensure import _facts, _lockfile, _store, ensure
from pm.ensure import uv as pm_uv
from pm.package import InstallError
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


def _install_names(names: list[str]) -> int:
    failed = 0
    for name in names:
        try:
            ensure(name, explicit=True, progress=_live_progress(name))
            print(f"✓ {name}", flush=True)
        except InstallError as e:
            print(f"✗ {e}", flush=True)
            failed += 1
    return failed



def _bundle_package_names() -> list[str]:
    names = [
        n
        for n in _lockfile().names()
        if not get_package(n).internal or n == "uv"
    ]
    if "python" not in names:
        names.append("python")
    return names


def cmd_install(args) -> int:
    names = args.names or [
        n for n in _lockfile().names() if not get_package(n).optional
    ]
    failed = _install_names(names)
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
    from pm.ensure import _identity
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


def _venv_site_packages(venv_dir: Path, win: bool) -> Optional[Path]:
    """The site-packages dir uv sync fills inside the venv (the venv stays
    a dependency target; it is never where `python` resolves)."""
    if win:
        candidate = venv_dir / "Lib" / "site-packages"
        return candidate if candidate.is_dir() else None
    for candidate in sorted(venv_dir.glob("lib/python3.*/site-packages")):
        if candidate.is_dir():
            return candidate
    return None


def _develop_env(names: list[str]) -> Optional[dict]:
    """The `pm develop` subshell environment. `python` resolves to the pm
    STORE python (its bin dir first on PATH); imports come from
    PYTHONPATH=<repo>;<venv>/site-packages (repo first — the venv stays a
    dependency target only). VIRTUAL_ENV and the venv bin dir are
    deliberately absent: nothing in the devshell boots through the venv
    (pm work item 3; pyvenv.cfg is inert dead config). Returns None when
    the store interpreter has not been materialized (`hermes pm install`)."""
    import os

    from pm import paths
    from pm.ensure import env_for

    facts = _facts()
    python_fact = facts.get("python")
    target = current_target()
    if not python_fact or "entry" not in python_fact:
        return None
    python_bin = get_package("python").binary(
        _store().entry(python_fact["entry"]), target
    )
    if python_bin is None or not python_bin.is_file():
        return None

    env = env_for(*names, base_env=dict(os.environ))
    repo = paths.repo_root()
    venv_dir = repo / (".venv" if (repo / ".venv").is_dir() else "venv")
    win = target.startswith("win32")
    venv_bin = venv_dir / ("Scripts" if win else "bin")
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    # The venv bin dir must never be where `python` resolves.
    path_entries = [
        p for p in env.get("PATH", "").split(os.pathsep) if p and Path(p) != venv_bin
    ]
    env["PATH"] = os.pathsep.join([str(python_bin.parent), *path_entries])
    site = _venv_site_packages(venv_dir, win)
    ours = str(repo) + (os.pathsep + str(site) if site else "")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ours + (os.pathsep + existing if existing else "")
    return env


def cmd_develop(args) -> int:
    """Install everything, sync the venv, then activate: spawn a subshell
    with every tool's env composed in and the pm STORE python resolving
    `python` (PYTHONPATH=repo;venv-site-packages for imports — never the
    venv interpreter). The devshell equivalent of nix develop. --print
    emits eval-able exports for the current shell."""
    import os
    import subprocess

    from pm.ensure import sync_venv

    install_names = [
        n for n in _lockfile().names()
        if not get_package(n).optional
        and get_package(n).missing_reason(current_target()) is None
    ]
    # The devshell boots the STORE python — make sure it is materialized.
    if (
        "python" not in install_names
        and get_package("python").missing_reason(current_target()) is None
    ):
        install_names.append("python")
    failed = _install_names(install_names)
    try:
        sync_venv(explicit=True)
        print("✓ venv")
    except InstallError as e:
        print(f"✗ {e}")
        failed += 1
    if failed:
        return 1

    env = _develop_env(_lockfile().names())
    if env is None:
        print("✗ develop: no pm store interpreter — run 'hermes pm install' first")
        return 1

    from pm import paths

    win = current_target().startswith("win32")
    if args.print_env:
        changed = {k: v for k, v in env.items() if os.environ.get(k) != v}
        for key, value in sorted(changed.items()):
            if win and os.environ.get("SHELL") is None:
                print(f'$env:{key} = "{value}"')
            else:
                escaped = value.replace("'", "'\\''")
                print(f"export {key}='{escaped}'")
        return 0

    shell = os.environ.get("SHELL") or os.environ.get("COMSPEC") or (
        "cmd.exe" if win else "/bin/sh"
    )
    print(f"pm develop: entering {shell} (exit to leave)")
    return subprocess.call([shell], env=env, cwd=paths.repo_root())


def _gc_store(store, facts) -> tuple[int, int]:
    """The sweep core shared by `pm gc` and `pm bundle`.

    Removes every store entry nothing references: fetch-<sha> download-cache
    dirs (the raw archives — needed only at install time, dead weight in a
    staged payload or a CI cache), orphaned package versions from an older
    lock, and expired partials. Keeps live package entries (recorded in
    facts) and partials an in-flight download still owns. Returns
    (removed, kept).
    """
    from pm.downloader import gc_protected_names
    from pm import paths

    partials_dir = paths.partials_root()
    if not store.root.is_dir() and not partials_dir.is_dir():
        return (0, 0)
    removed = 0
    with store.install_lock():
        facts.reload()
        keep = facts.entries_in_use()
        # Partials an in-flight (or recently interrupted) download still
        # owns must survive the sweep. They live in the writable partials
        # area, NOT the store root, so sweep that area directly.
        protected_partials = gc_protected_names(partials_dir)
        if partials_dir.is_dir():
            for child in sorted(partials_dir.iterdir()):
                if child.name in protected_partials:
                    continue
                print(f"removing partials/{child.name}")
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    try:
                        child.unlink()
                    except OSError:
                        pass
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
    facts = _facts()
    store = _store()
    removed, kept = _gc_store(store, facts)
    print(f"gc: removed {removed}, kept {kept}")
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
    for name in names:
        package = get_package(name)
        targets = [t for t in ALL_TARGETS if package.missing_reason(t) is None]
        if args.target:  # cross-target check: only the requested target matters
            targets = [t for t in targets if t == args.target]
        if not targets:
            continue
        try:
            decision = resolve_package(package, targets, lockfile.version(name))
        except Exception as e:  # an upstream index outage must not kill the whole check
            decision = Resolved(name, lockfile.version(name), package.version_style, reason=f"resolve failed: {e}")
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
            print(f"{d.name:<{width}}  {d.locked or '—'} → {d.version}{per}")
        else:
            print(f"{d.name:<{width}}  {d.locked} up to date")
    if args.check:
        if args.uv:
            print("uv deps: would run `uv update` + venv sync")
        if args.npm:
            print("npm deps: would run `npm update`")
        return 1 if changed else 0

    # ── apply ─────────────────────────────────────────────────────────────
    if changed:
        for d in changed:
            package = get_package(d.name)
            artifacts = _pin_artifacts(package, d)
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
        uv_bin, env = pm_uv()
        if uv_bin is None:
            print("✗ uv: not installed")
            return 1
        code, tail = _run_live([uv_bin, "update"], cwd=".", env=env)
        if code != 0:
            print(f"✗ uv update failed:\n{tail}")
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
        code, tail = _run_live(["npm", "update"], cwd=".", env=dict(os.environ))
        if code != 0:
            print(f"✗ npm update failed:\n{tail}")
            return 1
        print("✓ package-lock.json refreshed")
    return 0


def _pin_artifacts(package, decision) -> dict:
    """The lockfile artifacts dict for a resolved update, mirroring cmd_lock's
    per-target shape (identical single artifacts collapse to 'any'). For
    minor-style packages each target pins its OWN patch version. The
    lockfile always pins EVERY target the package serves — apply never
    narrows to the current machine."""
    per_target = decision.per_target or {t: decision.version for t in ALL_TARGETS}
    urls_by_target = {}
    for t, version in per_target.items():
        if package.missing_reason(t) is not None:
            continue
        urls_by_target[t] = [
            {"url": u, "sha256": package.known_sha256(version, u) or hash_url(u)}
            for u in package.fetch_urls(version, t)
        ]
    distinct = {tuple(u["url"] for u in v) for v in urls_by_target.values()}
    if len(distinct) == 1:
        first = next(iter(urls_by_target.values()))
        return {"any": first[0] if len(first) == 1 else first}
    return urls_by_target


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


def cmd_bundle(args) -> int:
    """Stage a complete payload for THIS machine's target into --out:
    repo snapshot + store + facts (via the normal install path, redirected)
    + a relocatable venv built on the staged interpreter and synced from
    uv.lock. Built natively per (os, arch); there is no cross-target
    staging."""
    import os

    from pm import paths

    out = Path(args.out).resolve()
    store_dir = out / "tools"
    store_dir.mkdir(parents=True, exist_ok=True)
    # A manifest from a previous run would make this payload look sealed
    # and refuse its own staging; it is rewritten at the end.
    (out / "manifest.json").unlink(missing_ok=True)

    repo_dir = out / "hermes-agent"
    ref = args.ref or "HEAD"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True)
    print(f"staging repo snapshot ({ref})…", flush=True)
    archive = subprocess.run(
        ["git", "archive", "--format=tar", ref],
        cwd=paths.repo_root(), capture_output=True, timeout=600,
    )
    if archive.returncode != 0:
        print(f"✗ repo: git archive {ref} failed: {archive.stderr.decode()[-500:]}")
        return 1
    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(repo_dir, filter="data")
    print(f"✓ repo ({ref})")

    os.environ["HERMES_RUNTIME_DIR"] = str(store_dir)
    paths._stamp.cache_clear()

    names = _bundle_package_names()
    failed = _install_names(
        [n for n in names if get_package(n).missing_reason(current_target()) is None]
    )

    # Prune the staged store BEFORE the venv sync and packaging: drop the
    # fetch-<sha> download-cache archives (needed only at install time — dead
    # weight in the shipped payload AND in the CI cache that restores this
    # dir) and any orphaned package versions left over from an older lock
    # the cache carried in. A lean staged store = a lean CI cache.
    removed, kept = _gc_store(_store(), _facts())
    print(f"✓ gc: pruned {removed} fetch/stale entries, kept {kept}")

    uv_bin, env = pm_uv()
    if uv_bin is None:
        print("✗ venv: uv did not stage")
        return 1

    python_fact = _facts().get("python")
    if python_fact is None:
        print("✗ venv: no staged interpreter to build on")
        return 1
    python_bin = get_package("python").binary(
        _store().entry(python_fact["entry"]), current_target()
    )

    # Build + sync INSIDE the staged repo: the editable project install
    # must point at the payload's own tree, not this checkout.
    venv_dir = out / "venv"
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    env["VIRTUAL_ENV"] = str(venv_dir)
    env.pop("UV_NO_CONFIG", None)
    if current_target().startswith("darwin"):
        # python-build-standalone bakes phantom toolchain paths (its build
        # dir's llvm-ar) into sysconfig; sdist builds then fail with
        # "No such file or directory: .../tools/llvm/bin/llvm-ar". Point
        # sdist builds at the machine's real toolchain.
        env.setdefault("AR", "/usr/bin/ar")
        env.setdefault("CC", "clang")
    for cmd in (
        [uv_bin, "venv", "--relocatable", "--python", str(python_bin), str(venv_dir)],
        [uv_bin, "sync", "--frozen", "--all-extras", "--active"],
    ):
        print(f"  venv: $ {' '.join(cmd)}", flush=True)
        code, tail = _run_live(cmd, cwd=repo_dir, env=env)
        if code != 0:
            print(f"✗ venv: {' '.join(cmd[1:3])} failed:\n{tail}")
            return 1
    print("✓ venv (relocatable, all extras, on the staged interpreter)")

    # The frozen feature set: the EXACT extras that installed on this
    # target (markers gate some off per-platform). This file is the
    # lazy-off contract — pm sync never deviates from it.
    from pm.features import installed_extras, write_features

    features = installed_extras(repo_dir, venv_dir)
    write_features(features, out)
    print(f"✓ enabled-features.json ({len(features)} extras recorded)")

    # Ship the uv cache: the staged venv sync just warmed the hermes-owned
    # cache with every wheel this payload needs. Copying it in makes a
    # mutable-venv rebuild from the bundle near-free (`uv sync --offline`
    # from a warm cache probed at 0.4s vs 1.2s cold) — the blow-away-on-
    # update contract depends on it.
    from pm.packages import uv_cache_dir as bundle_uv_cache_dir

    payload_cache = out / "uv-cache"
    if payload_cache.exists():
        shutil.rmtree(payload_cache, ignore_errors=True)
    src_cache = bundle_uv_cache_dir()
    if src_cache.is_dir():
        print(f"  uv-cache: copying {src_cache} → payload...", flush=True)
        shutil.copytree(src_cache, payload_cache)
        print("✓ uv-cache (staged — warm rebuilds for the mutable venv)")
    else:
        print("  uv-cache: none warm (first bundle on this machine?)")

    bad = _arch_guard(store_dir)
    for line in bad:
        print(f"✗ arch: {line}")
        failed += 1

    manifest = {
        "schema": 1,
        "target": current_target(),
        "ref": ref,
        "repo": "hermes-agent",
        "venv": "venv",
        "store": "tools",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"✓ manifest ({out / 'manifest.json'})")
    return 1 if failed else 0


def _arch_guard(store_dir: Path) -> list[str]:
    """Every staged binary must be built for this machine's target — a
    payload staged with a mismatched interpreter or PATH tool ships an
    artifact that cannot run. Reads facts, probes each entry binary."""
    from pm.lock import Facts
    from pm.package import machine_matches_binary

    facts = Facts(store_dir / "facts.json")
    problems = []
    target = current_target()
    for name in _lockfile().names():
        package = get_package(name)
        fact = facts.get(name)
        if fact is None or "entry" not in fact:
            continue
        binary = package.binary(store_dir / fact["entry"], target)
        if binary is None or not binary.is_file():
            continue
        verdict = machine_matches_binary(binary, target)
        # A package that declares this target as emulated (x64 binary run
        # under Windows ARM64 built-in emulation) is fine with the x64 PE.
        if verdict is False and target not in package.emulated_arch_targets:
            problems.append(f"{name}: {binary.name} is not a {target} binary")
    return problems


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
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("env", help="print composed env of installed packages")
    p.add_argument("names", nargs="*")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("doctor", help="check installed state against the lockfile")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("develop", help="install + sync, then activate a devshell with the composed env")
    p.add_argument("--print", dest="print_env", action="store_true",
                   help="print eval-able exports instead of spawning a shell")
    p.set_defaults(func=cmd_develop)

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
