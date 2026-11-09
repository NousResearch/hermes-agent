"""run_tests.sh shares the account's tool store only when it can run from it as is.

A store whose facts match the checkout's pins must still not be shared when
running from it would write to it: a missing or non-executable binary sends
setup-hermes.sh (uv) or `pm install` into an unlocked re-stage, and an env that
differs from what this checkout's package code records makes `pm install`
re-record the shared facts.json.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pm.defaults import default_packages
from pm.lock import Facts, Lockfile
from pm.package import StatePackage
from pm.registry import get_package, tool_roots, walk
from pm.store import current_target

ROOT = Path(__file__).resolve().parents[2]


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def _matching_store(store: Path) -> Path:
    """A store recording every walked tool at the lock's pins with the env PM
    would record, each entry holding an executable binary, plus setup-hermes.sh's
    own uv copy. PM's entries are named so none is the setup path, which keeps
    the two uv checks apart. Returns setup-hermes.sh's uv path."""
    lock = Lockfile(ROOT / "pm" / "lock.json")
    names, target = lock.names(), current_target()
    facts = Facts(store / "facts.json")
    for package in walk(tool_roots(names) + default_packages(names) + ["venv"]):
        if isinstance(package, StatePackage):
            continue
        version = lock.version(package.name)
        entry = store / f"pm-{package.name}-{version}-{target}"
        entry.mkdir(parents=True)
        binary = package.binary(entry, target)
        _executable(binary if binary is not None else entry / "chrome")  # chromium's is discovered
        facts.record(package.name, version, entry.name, package.env(entry, target), store,
                     target=target, artifacts=[a["sha256"] for a in lock.artifacts(package.name, target)])
    # setup-hermes.sh: entry="$store/uv-$uv_version-$os-$arch"
    return _executable(store / f"uv-{lock.version('uv')}-{'-'.join(target.split('-')[:2])}" / "uv")


def _shares(store: Path) -> bool:
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(ROOT / "scripts/tool_store_pins.py"), str(ROOT), str(store)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    return result.returncode == 0


def _entry_binary(store: Path, name: str) -> Path:
    fact = Facts(store / "facts.json", strict=True).get(name) or {}
    binary = get_package(name).binary(store / fact["entry"], current_target())
    assert binary is not None
    return binary


def _stale_env(store: Path, name: str) -> None:
    data = json.loads((store / "facts.json").read_text())
    data["packages"][name]["env"] = {}
    (store / "facts.json").write_text(json.dumps(data))


BREAKAGES = {
    "setup_uv_missing": lambda store, uv: uv.unlink(),
    "setup_uv_not_executable": lambda store, uv: uv.chmod(0o644),
    "ripgrep_not_executable": lambda store, uv: _entry_binary(store, "ripgrep").chmod(0o644),
    "node_env_stale": lambda store, uv: _stale_env(store, "node"),
}


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("breakage", sorted(BREAKAGES))
def test_store_running_from_which_would_write_to_it_is_not_shared(tmp_path, breakage):
    store = tmp_path / "tools"
    uv = _matching_store(store)
    assert _shares(store), "control: a complete store at this checkout's pins must be shared"
    BREAKAGES[breakage](store, uv)
    before = sorted((p.relative_to(store), p.is_file() and p.read_bytes(), p.stat().st_mode)
                    for p in store.rglob("*"))
    assert not _shares(store)
    after = sorted((p.relative_to(store), p.is_file() and p.read_bytes(), p.stat().st_mode)
                   for p in store.rglob("*"))
    assert after == before, "the check must stay read-only (no heal, no re-record, no .corrupt)"
