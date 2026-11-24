"""Exit 0 only when a tool store already holds this checkout's pins.

Usage: python -I -B scripts/tool_store_pins.py REPO STORE
(on success prints the pinned Python major.minor)

scripts/run_tests.sh asks this before sharing the account's tool store with a
faked HOME. facts.json selects ONE entry per package, not one per version:
provisioning a checkout with other pins re-points the account's own runtime at
them and pulls the selection from under a concurrent run on the store's pins.
So share only when every tool `pm install` would provision here (cmd_install's
roots, defaults and the venv's deps) passes PM's own installed-at-the-pin lookup
without an exec-bit heal, and records the env this checkout's package code
would record: a stale env makes `pm install` re-record the shared facts.json,
and a missing or non-executable binary (setup-hermes.sh's uv copy included)
sends setup or `pm install` into a re-stage whose unlocked remove + publish
races other runs on the shared store. Read-only: no lock, no install, no
bytecode; any unreadable record or error means "do not share".
"""
import os
import sys
from pathlib import Path


def _setup_uv(store: Path, target: str, version: str) -> Path:
    """setup-hermes.sh's exact uv path: $store/uv-$uv_version-$os-$arch/uv[.exe]."""
    os_arch = "-".join(target.split("-")[:2])
    return store / f"uv-{version}-{os_arch}" / ("uv.exe" if os_arch.startswith("win32") else "uv")


def main(repo: str, store: Path) -> int:
    sys.path.insert(0, repo)
    from pm.defaults import default_packages
    from pm.install import _installed_location
    from pm.lock import Facts, Lockfile
    from pm.package import StatePackage
    from pm.registry import tool_roots, walk
    from pm.store import current_target

    lock = Lockfile(Path(repo, "pm", "lock.json"))
    # Strict first: PM's lenient read would quarantine an unparsable file as .corrupt.
    Facts(store / "facts.json", strict=True)
    names, target = lock.names(), current_target()
    for package in walk(tool_roots(names) + default_packages(names) + ["venv"]):
        if isinstance(package, StatePackage):
            continue
        location = _installed_location(package, lock, target, roots=(store,), heal=False)
        if location is None:
            return 1
        facts, found = location
        entry = found.entry(facts.get(package.name)["entry"])
        if facts.env_for(package.name, found.root) != package.env(entry, target):
            return 1
    uv = _setup_uv(store, target, lock.version("uv"))
    if not uv.is_file() or not (os.name == "nt" or os.access(uv, os.X_OK)):
        return 1
    # The runner reuses uv's interpreter cache only when it already holds this minor.
    print(".".join(lock.version("python").split("+")[0].split(".")[:2]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], Path(sys.argv[2])))
