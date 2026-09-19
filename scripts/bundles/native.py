"""Native payload staging through PM's existing package authority."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import argparse
from dataclasses import asdict
from pathlib import Path

from pm.install import _lockfile
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



from pm.cache_lock import lock_package_names, prune_uv_cache_to_lock

__all__ = ["prune_uv_cache_to_lock", "lock_package_names", "stage_uv_cache"]


def stage_uv_cache(source: Path, destination: Path) -> None:
    """Ship everything a per-install venv rebuild resolves from offline.

    sdist source trees (including Rust target/ outputs) are build-only bulk —
    offline rebuilds install the built wheel entries, never recompile from
    source — so they stay out. Built wheel ZIPs (including the copies uv keeps
    beside sdist metadata.msgpack shards) ship: without them an offline
    rebuild falls back to wanting the sdist download and fails closed.
    """
    def ignore(directory: str, names: list[str]) -> set[str]:
        path = Path(directory)
        parts = path.relative_to(source).parts
        if not parts or not parts[0].startswith("sdists-v"):
            return set()
        # Source trees live under a revision selected by a sibling pointer.
        if "src" in names and (path / "src").is_dir() and any(
            (path.parent / pointer).is_file() for pointer in ("revision.http", "revision.rev")
        ):
            return {"src"}
        return set()

    shutil.copytree(source, destination, ignore=ignore)


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
    from pm.packages import uv_cache_dir

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").unlink(missing_ok=True)
    root = Path(__file__).resolve().parents[2]
    # Resolve before HOME isolation so the build warms the cache CI saves.
    cache = Path(getattr(args, "cache", None) or os.environ.get("UV_CACHE_DIR") or uv_cache_dir()).resolve()
    base_env = dict(os.environ)
    if current_target() == "win32-arm64":
        from scripts.build.windows_deps import prepare_windows_environment

        base_env = prepare_windows_environment(source=root, state=out.parent / ".build-deps", env=base_env)
    # Rustup resolves its installed toolchain under HOME unless these are explicit.
    # Preserve the build host's compiler and Cargo cache before isolating app state.
    for key, directory in (("CARGO_HOME", ".cargo"), ("RUSTUP_HOME", ".rustup")):
        base_env.setdefault(key, str(Path.home() / directory))
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


def prune_staged_store(store_dir: Path, names: list[str]) -> None:
    """Prune this build's store, never the ambient user's tool store."""
    facts = Facts(store_dir / "facts.json", strict=True)
    facts.retain({package.name for package in walk(names)})
    keep = facts.entries_in_use()
    for entry in store_dir.iterdir():
        if entry.is_dir() and not entry.name.startswith(".") and entry.name not in keep:
            shutil.rmtree(entry)


def prepare_native(*, out: Path, ref: str, source: Path, cache: Path,
                   tools: Path | None = None, env: dict | None = None) -> Path:
    """Prepare final payload dependencies inside the caller's isolated PM process.

    The caller supplies its compiler environment; only the full standalone
    wrapper provisions machine prerequisites and isolates HOME.
    """
    from scripts.bundles.native_prepared import preparation_lock, prepared_path

    out = Path(out).absolute()
    if out != out.resolve():
        raise ValueError("symlinked native output")
    out.mkdir(parents=True, exist_ok=True)
    with preparation_lock(out):
        prepared_path(out).unlink(missing_ok=True)
        (out / "manifest.json").unlink(missing_ok=True)
        return _prepare_native(out=out, ref=ref, source=Path(source).resolve(),
                               cache=Path(cache).resolve(), tools=tools, env=env)


def _prepare_native(*, out: Path, ref: str, source: Path, cache: Path,
                    tools: Path | None, env: dict | None) -> Path:
    from pm import paths
    from pm.package import InstallError

    store_dir = out / "tools"
    store_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = out / "hermes-agent"
    from scripts.bundles.payload import INERT_SNAPSHOT_DIRS, snapshot
    print(f"staging repo snapshot ({ref})…", flush=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=source,
        text=True, encoding="utf-8").strip()
    snapshot(source, revision, repo_dir, exclude=INERT_SNAPSHOT_DIRS)
    # PM's provider code reads its adjacent lock. Never combine that tool graph
    # with a revision selecting different pins.
    if (repo_dir / "pm/lock.json").read_bytes() != paths.lockfile_path().read_bytes():
        raise ValueError("selected revision's PM lock differs from the builder; use a checkout at that revision")

    names = [
        n for n in _bundle_package_names()
        if get_package(n).missing_reason(current_target()) is None
    ]
    from pm import prepare_tools, stage_tools

    prepare_tools(names, out=Path(tools) if tools is not None else store_dir,
                  target=current_target(), cache=cache)
    if tools is not None:
        stage_tools(names, source_store=Path(tools), out=store_dir, target=current_target())

    # Prune the staged store BEFORE the venv sync and packaging: drop the
    # fetch-<sha> download-cache archives (needed only at install time — dead
    # weight in the shipped payload AND in the CI cache that restores this
    # dir) and any orphaned package versions left over from an older lock
    # the cache carried in. A lean staged store = a lean CI cache.

    # Only this build's store is ours to prune; machine-wide partials are not.
    # Cached facts may still name packages removed from the current selection.
    # Retain the dependency closure before using facts as the deletion roots.
    prune_staged_store(store_dir, names)


    facts = Facts(store_dir / "facts.json")
    python_fact = facts.get("python")
    if python_fact is None:
        raise InstallError("venv", "no staged interpreter to build on")
    python_bin = get_package("python").binary(
        store_dir / python_fact["entry"], current_target()
    )

    if python_bin is None:
        raise FileNotFoundError("staged Python executable is missing")
    stage_pm_runtime(out, python_bin, repo_dir, cache=cache)
    print("✓ pm-runtime (independent locked dependencies)", flush=True)

    # Build + sync INSIDE the staged repo: the editable project install
    # must point at the payload's own tree, not this checkout.
    venv_dir = out / "venv"
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    env = dict(os.environ if env is None else env)
    from pm import build_environment

    # Cold native wheels need a larger budget than interactive installs.
    build_environment(source=repo_dir, python=python_bin, out=venv_dir,
                      env=env, cache=cache, all_extras=True, sealed=True, explicit=True,
                      timeout=2 * 60 * 60)
    print("✓ venv (all extras, on the staged interpreter)")

    # Inventory the staged interpreter before publishing the bundle contract.
    from pm.features import installed_extras, write_features

    features = installed_extras(repo_dir, venv_dir, python_exe=python_bin)
    write_features(features, out)
    print(f"✓ enabled-features.json ({len(features)} extras recorded)")

    # Ship the full uv cache (build-only sdist sources and wheel ZIPs are
    # already excluded by stage_uv_cache). Any per-install venv rebuild —
    # plugin extras, feature changes — resolves entirely from the shipped
    # wheels with zero network access, on every profile of this install.
    payload_cache = out / "uv-cache"
    if payload_cache.exists():
        shutil.rmtree(payload_cache, ignore_errors=True)
    src_cache = cache
    if src_cache.is_dir():
        print(f"  uv-cache: copying {src_cache} → payload...", flush=True)
        stage_uv_cache(src_cache, payload_cache)
        pruned = prune_uv_cache_to_lock(payload_cache, repo_dir)
        print(f"✓ uv-cache (lock-scoped: pruned {pruned} stale entries; offline rebuilds resolve from shipped wheels)", flush=True)
    else:
        raise InstallError("uv-cache", "runtime dependency cache is missing")

    bad = _arch_guard(store_dir)
    for line in bad:
        print(f"✗ arch: {line}")
    if bad:
        raise InstallError("tools", "native architecture verification failed")
    from scripts.bundles.payload import record_tools
    recorded = {name: fact["entry"] for name in names if (fact := facts.get(name)) and "entry" in fact}
    record_tools(out, paths.lockfile_path(), current_target(), recorded)
    from scripts.build.inputs import AgentInputs, RESOURCE_ENV, dependency_site
    from scripts.bundles.native_prepared import publish_prepared

    site = dependency_site(venv_dir, python_fact["version"], current_target())
    (site / "hermes-agent.pth").write_text(
        Path(os.path.relpath(repo_dir, site)).as_posix() + "\n", encoding="utf-8")
    from scripts.bundles.payload import relativize_links
    relativize_links(out)
    from scripts.bundles.bytecode import bake_bytecode
    baked = bake_bytecode(out, python_bin)
    print(f"✓ baked bytecode ({baked['modules']} modules, unchecked-hash, read-only caches)")
    inputs = AgentInputs(
        project=repo_dir / "pyproject.toml", code=repo_dir, repo="hermes-agent",
        placement="contained", target=current_target(), python=python_bin,
        site_packages=site, environment=venv_dir,
        tools=store_dir, pm_runtime=out / "pm-runtime", ref=ref,
        resources={name: repo_dir / name for name in RESOURCE_ENV},
        features=out / "enabled-features.json",
    )
    return publish_prepared(out, source, revision, inputs)


def finish_native(prepared: Path, frontends: dict[str, Path]) -> int:
    """Consume verified job-local paths. No dependency acquisition or repair."""
    from scripts.build.agent import assemble
    from scripts.build.inputs import AgentInputs
    from scripts.bundles.native_prepared import load_prepared, preparation_lock

    prepared = Path(prepared).absolute()
    if not prepared.name.endswith(".prepared.json") or not prepared.is_file():
        raise ValueError("native preparation is missing or invalid; run preparation again")
    out = prepared.with_name(prepared.name.removesuffix(".prepared.json"))
    with preparation_lock(out):
        (out / "manifest.json").unlink(missing_ok=True)
        inputs = load_prepared(prepared)
        values = asdict(inputs)
        values["frontends"] = {name: Path(path).absolute() for name, path in frontends.items()}
        assemble(AgentInputs.from_dict(values), out)
    print(f"✓ manifest ({out / 'manifest.json'})")
    return 0


def _stage_native(args) -> int:
    from pm import paths
    from pm.features import FeatureProbeError
    from pm.package import InstallError

    try:
        prepared = prepare_native(
            out=Path(args.out), ref=args.ref or "HEAD",
            source=getattr(args, "source", None) or paths.repo_root(),
            cache=Path(getattr(args, "cache", None) or os.environ["UV_CACHE_DIR"]),
            tools=getattr(args, "tools", None),
        )
        return finish_native(prepared, getattr(args, "frontends", {}))
    except (InstallError, FeatureProbeError) as exc:
        print(f"✗ {exc}")
        return 1




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
