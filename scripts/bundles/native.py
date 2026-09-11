"""Native payload staging through PM's existing package authority."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import argparse
from pathlib import Path

from pm.cli import _install_names
from pm.ensure import _store, _facts, _lockfile
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


def stage_pm_runtime(root: Path, python: Path, repo: Path, *, offline: bool = False,
                     cache: Path | None = None) -> None:
    """Publish the same PM dependency graph as source installs, ready offline."""
    from pm import stage_manager_runtime
    from scripts.bundles.payload import seal_pm_runtime

    destination = root / "pm-runtime"
    if destination.exists():
        shutil.rmtree(destination)
    stage_manager_runtime(python=python, destination=destination, project=repo / "pm", offline=offline, cache=cache)
    seal_pm_runtime(root, python)


def stage_native(args) -> int:
    """Isolate HOME and PM state, but retain the provider's reusable build cache."""
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").unlink(missing_ok=True)
    root = Path(__file__).resolve().parents[2]
    cache = Path(getattr(args, "cache", None) or os.environ.get("UV_CACHE_DIR") or out.parent / ".uv-cache").resolve()
    base_env = dict(os.environ)
    if current_target() == "win32-arm64":
        from scripts.build.windows_deps import prepare_windows_environment

        base_env = prepare_windows_environment(source=root, state=out.parent / ".build-deps", env=base_env)
    with tempfile.TemporaryDirectory(prefix=".build-", dir=out) as work:
        env = {**base_env, "HOME": work, "USERPROFILE": work,
               "HERMES_HOME": str(Path(work) / ".hermes"),
               "HERMES_RUNTIME_DIR": str(out / "tools"),
               "HERMES_PYTHON_SRC_ROOT": str(root),
               "XDG_CACHE_HOME": str(Path(work) / "cache"),
               "XDG_CONFIG_HOME": str(Path(work) / "config"),
               "UV_CACHE_DIR": str(cache),
               "PYTHONPATH": os.pathsep.join([str(root), *filter(None, sys.path)])}
        command = [sys.executable, "-B", "-m", "scripts.bundles.native", "--out", str(out),
                   "--ref", args.ref or "HEAD", "--source", str(root)]
        for name, product in getattr(args, "frontends", {}).items():
            command += [f"--{name}", str(product)]
        return subprocess.run(command, cwd=root, env=env).returncode


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
    from scripts.bundles.payload import snapshot
    print(f"staging repo snapshot ({ref})…", flush=True)
    snapshot(getattr(args, "source", None) or paths.repo_root(), ref, repo_dir)
    # PM's provider code reads its adjacent lock. Never combine that tool graph
    # with a revision selecting different pins.
    if (repo_dir / "pm/lock.json").read_bytes() != paths.lockfile_path().read_bytes():
        raise ValueError("selected revision's PM lock differs from the builder; use a checkout at that revision")

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


    python_fact = _facts().get("python")
    if python_fact is None:
        print("✗ venv: no staged interpreter to build on")
        return 1
    python_bin = get_package("python").binary(
        _store().entry(python_fact["entry"]), current_target()
    )

    if python_bin is None:
        raise FileNotFoundError("staged Python executable is missing")
    cache = Path(os.environ["UV_CACHE_DIR"])
    stage_pm_runtime(out, python_bin, repo_dir, cache=cache)
    print("✓ pm-runtime (independent locked dependencies)", flush=True)

    # Build + sync INSIDE the staged repo: the editable project install
    # must point at the payload's own tree, not this checkout.
    venv_dir = out / "venv"
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    env = dict(os.environ)
    if current_target().startswith("darwin"):
        # python-build-standalone bakes phantom toolchain paths (its build
        # dir's llvm-ar) into sysconfig; sdist builds then fail with
        # "No such file or directory: .../tools/llvm/bin/llvm-ar". Point
        # sdist builds at the machine's real toolchain.
        env.setdefault("AR", "/usr/bin/ar")
        env.setdefault("CC", "clang")
    from pm import build_environment
    from pm.package import InstallError

    try:
        build_environment(source=repo_dir, python=python_bin, out=venv_dir,
                          env=env, cache=cache, all_extras=True, sealed=True, explicit=True)
    except InstallError as exc:
        print(f"✗ venv: {exc}")
        return 1
    print("✓ venv (all extras, on the staged interpreter)")

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
    payload_cache = out / "uv-cache"
    if payload_cache.exists():
        shutil.rmtree(payload_cache, ignore_errors=True)
    src_cache = cache
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
    from scripts.bundles.payload import record_tools
    recorded = {name: fact["entry"] for name in names if (fact := _facts().get(name)) and "entry" in fact}
    record_tools(out, paths.lockfile_path(), current_target(), recorded)
    from scripts.build.agent import assemble
    from scripts.build.inputs import AgentInputs, RESOURCE_ENV, dependency_site

    assemble(AgentInputs(
        project=repo_dir / "pyproject.toml", code=repo_dir, repo="hermes-agent",
        placement="contained", target=current_target(), python=python_bin,
        site_packages=dependency_site(venv_dir, python_fact["version"], current_target()), environment=venv_dir,
        tools=store_dir, pm_runtime=out / "pm-runtime", ref=ref,
        resources={name: repo_dir / name for name in RESOURCE_ENV},
        frontends=getattr(args, "frontends", {}), features=out / "enabled-features.json",
    ), out)
    print(f"✓ manifest ({out / 'manifest.json'})")
    return 1 if failed else 0




def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tui", type=Path)
    parser.add_argument("--web", type=Path)
    args = parser.parse_args()
    args.frontends = {name: path for name in ("tui", "web") if (path := getattr(args, name)) is not None}
    return _stage_native(args)


if __name__ == "__main__":
    raise SystemExit(main())
