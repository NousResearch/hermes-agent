"""Native payload staging through PM's existing package authority."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from pm.cli import _install_names, _run_live
from pm.ensure import _store, _facts, _lockfile, uv as pm_uv
from pm.lock import Facts
from pm.registry import get_package, walk
from pm.store import current_target

def _bundle_package_names() -> list[str]:
    names = [
        n
        for n in _lockfile().names()
        if not get_package(n).internal or n == "uv"
    ]
    if "python" not in names:
        names.append("python")
    return names


def _arch_guard(store_dir: Path) -> list[str]:
    """Every staged binary must be built for this machine's target — a
    payload staged with a mismatched interpreter or PATH tool ships an
    artifact that cannot run. Reads facts, probes each entry binary."""
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



def stage_uv_cache(source: Path, destination: Path) -> None:
    """Keep extracted wheels for offline installs, not unsigned build ZIPs.

    The macOS signer reaches extracted native code, but not code inside ZIPs.
    uv installs built wheels from their extracted cache entries.
    """
    shutil.copytree(source, destination)
    for bucket in destination.glob("sdists-v*"):
        for wheel in bucket.rglob("*.whl"):
            if wheel.is_file():
                wheel.unlink()


def stage_pm_runtime(root: Path, uv: Path, python: Path, repo: Path, *, offline: bool = False) -> None:
    """Publish the same PM dependency graph as source installs, ready offline."""
    from pm.runtime_stage import stage_runtime
    from scripts.bundles.payload import seal_pm_runtime

    destination = root / "pm-runtime"
    if destination.exists():
        shutil.rmtree(destination)
    stage_runtime(uv, python, destination, project=repo / "pm", offline=offline)
    seal_pm_runtime(root, python)


def stage_native(args) -> int:
    previous = os.environ.get("HERMES_RUNTIME_DIR")
    try:
        return _stage_native(args)
    finally:
        if previous is None:
            os.environ.pop("HERMES_RUNTIME_DIR", None)
        else:
            os.environ["HERMES_RUNTIME_DIR"] = previous


def _stage_native(args) -> int:
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
    from scripts.bundles.payload import snapshot, write_manifest, relativize_links
    print(f"staging repo snapshot ({ref})…", flush=True)
    snapshot(paths.repo_root(), ref, repo_dir)

    os.environ["HERMES_RUNTIME_DIR"] = str(store_dir)

    names = [
        n for n in _bundle_package_names()
        if get_package(n).missing_reason(current_target()) is None
    ]
    failed = _install_names(names)

    # Prune the staged store BEFORE the venv sync and packaging: drop the
    # fetch-<sha> download-cache archives (needed only at install time — dead
    # weight in the shipped payload AND in the CI cache that restores this
    # dir) and any orphaned package versions left over from an older lock
    # the cache carried in. A lean staged store = a lean CI cache.
    if failed:
        return 1
    # Only this build's store is ours to prune; machine-wide partials are not.
    # Cached facts may still name packages removed from the current selection.
    # Retain the dependency closure before using facts as the deletion roots.
    facts = Facts(store_dir / "facts.json", strict=True)
    facts.retain({package.name for package in walk(names)})
    keep = facts.entries_in_use()
    for entry in store_dir.iterdir():
        if entry.is_dir() and not entry.name.startswith(".") and entry.name not in keep:
            shutil.rmtree(entry)


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

    stage_pm_runtime(out, Path(uv_bin), python_bin, repo_dir)
    print("✓ pm-runtime (independent locked dependencies)", flush=True)

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

    # Inventory the staged interpreter before publishing the bundle contract.
    from pm.features import FeatureProbeError, installed_extras, write_features

    try:
        features = installed_extras(repo_dir, venv_dir, python_exe=python_bin)
    except FeatureProbeError as exc:
        print(f"✗ features: {exc}")
        return 1
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
        stage_uv_cache(src_cache, payload_cache)
        print("✓ uv-cache (staged — warm rebuilds for the mutable venv)")
    else:
        print("  uv-cache: none warm (first bundle on this machine?)")

    bad = _arch_guard(store_dir)
    for line in bad:
        print(f"✗ arch: {line}")
        failed += 1

    if failed:
        return 1
    relativize_links(out)
    from scripts.bundles.payload import record_tools
    recorded = {name: fact["entry"] for name in names if (fact := _facts().get(name)) and "entry" in fact}
    record_tools(out, paths.lockfile_path(), current_target(), recorded)
    write_manifest(out, target=current_target(), repo="hermes-agent", ref=ref)
    print(f"✓ manifest ({out / 'manifest.json'})")
    return 1 if failed else 0


