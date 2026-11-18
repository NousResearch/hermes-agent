"""run_tests.sh shares the account's tool store only when it can run from it as is.

A store whose facts match the checkout's pins must still not be shared when
running from it would write to it: a missing or non-executable binary sends
setup-hermes.sh (uv) or `pm install` into an unlocked re-stage, and an env that
differs from what this checkout's package code records makes `pm install`
re-record the shared facts.json.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pm.defaults import default_packages
from pm.lock import Facts, Lockfile
from pm.package import StatePackage
from pm.registry import get_package, tool_roots, walk
from pm.store import current_target
from pm.testenv import developer_tools

ROOT = Path(__file__).resolve().parents[2]


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
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
    for package in walk(tool_roots(names) + developer_tools(names) + default_packages(names) + ["venv"]):
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
    data = json.loads((store / "facts.json").read_text(encoding="utf-8-sig"))
    data["packages"][name]["env"] = {}
    (store / "facts.json").write_text(json.dumps(data), encoding="utf-8")


BREAKAGES = {
    "setup_uv_missing": lambda store, uv: uv.unlink(),
    "setup_uv_not_executable": lambda store, uv: uv.chmod(0o644),
    "ripgrep_not_executable": lambda store, uv: _entry_binary(store, "ripgrep").chmod(0o644),
    "just_not_executable": lambda store, uv: _entry_binary(store, "just").chmod(0o644),
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


def _snapshot(store: Path) -> list:
    return sorted((p.relative_to(store), p.is_file() and p.read_bytes(), p.stat().st_mode) for p in store.rglob("*"))


def _run_runner(tmp_path: Path, account: Path, home: Path, naming: dict, *, bootstrap: bool = True):
    """Run the real runner in a repo-shaped tree holding the real runner, activation library, pin
    check and pm; return the run and the store its setup step was aimed at (None when setup never
    ran). Setup is replaced at its process boundary: it records the store pm resolves for this run
    and stops, so nothing is installed. ``getent`` answers with ``account`` as the OS account's
    home, which makes ``home`` a faked HOME. ``bootstrap=False`` leaves the checkout without the
    interpreter the runner could ask pm with."""
    root = tmp_path / "checkout"
    if not root.exists():
        (root / "scripts").mkdir(parents=True)
        for name in ("run_tests.sh", "run-in-hermes-env", "_activation.sh", "tool_store_pins.py"):
            shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
        for name in ("pm", "hermes_constants.py", "hermes_platform"):
            (root / name).symlink_to(ROOT / name)
        if bootstrap:
            _executable(root / ".venv" / "bin" / "python").write_text(
                f"#!/bin/sh\nexec '{sys.executable}' \"$@\"\n", encoding="utf-8")
        (root / "setup-hermes.sh").write_text(
            f"'{sys.executable}' -I -B -c 'import sys\nsys.path.insert(0, sys.argv[1])\n"
            "from pm.environments import store_root\nprint(store_root(sys.argv[1]), end=\"\")' "
            f"'{root}' > '{root}/provisioned'\nexit 1\n", encoding="utf-8")
        _executable(tmp_path / "bin" / "getent").write_text(
            f"#!/bin/sh\necho 'account:x:0:0::{account}:/bin/sh'\n", encoding="utf-8")
    home.mkdir(exist_ok=True)
    (root / "provisioned").unlink(missing_ok=True)
    run = subprocess.run(
        ["bash", str(root / "scripts" / "run_tests.sh"), "tests/nothing.py"], cwd=root,
        env={"PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}", "HOME": str(home),
             "PYTHONDONTWRITEBYTECODE": "1", **naming},
        capture_output=True, text=True, encoding="utf-8", timeout=120, check=False,
    )
    provisioned = root / "provisioned"
    return run, Path(provisioned.read_text(encoding="utf-8-sig")) if provisioned.is_file() else None


def _provisioned_store(tmp_path: Path, account: Path, home: Path, naming: dict) -> Path:
    run, store = _run_runner(tmp_path, account, home, naming)
    assert store is not None, run.stdout + run.stderr
    return store


def _store_at_other_pins(store: Path) -> Path:
    """An account store recording none of this checkout's pins."""
    store.mkdir(parents=True)
    (store / "facts.json").write_text('{"packages": {}}', encoding="utf-8")
    return store


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("named_by", ["HERMES_RUNTIME_DIR", "HERMES_HOME"])
def test_faked_home_run_naming_the_account_store_never_provisions_other_pins_into_it(tmp_path, named_by):
    """A lane that passes the account's store explicitly, or inherits the agent shell's HERMES_HOME,
    must not re-stage its checkout's pins over the ones the account's own Hermes runs on."""
    account, home = tmp_path / "account", tmp_path / "fakehome"
    store = account / ".hermes" / "tools"
    _matching_store(store)
    naming = {"HERMES_RUNTIME_DIR": str(store)} if named_by == "HERMES_RUNTIME_DIR" else {"HERMES_HOME": str(store.parent)}
    assert _provisioned_store(tmp_path, account, home, naming).samefile(store), \
        "control: a named store at this checkout's pins is used as is"

    data = json.loads((store / "facts.json").read_text(encoding="utf-8-sig"))
    data["packages"]["chromium"]["version"] = "0+another-checkouts-pin"
    (store / "facts.json").write_text(json.dumps(data), encoding="utf-8")
    before = _snapshot(store)
    assert _provisioned_store(tmp_path, account, home, naming) == (home / ".hermes" / "tools").resolve()
    assert _snapshot(store) == before


def _naming(named_by: str, store: Path) -> dict:
    return {"HERMES_RUNTIME_DIR": str(store)} if named_by == "HERMES_RUNTIME_DIR" else {"HERMES_HOME": str(store.parent)}


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("alias", [".hermes", ".hermes/tools"])
def test_faked_home_whose_own_store_aliases_the_account_store_is_refused(tmp_path, alias):
    """The faked home's store is where a run naming the account's store is sent instead; when that
    path is itself a link into the account's store, provisioning it re-stages the account's store."""
    account, home = tmp_path / "account", tmp_path / "fakehome"
    store = _store_at_other_pins(account / ".hermes" / "tools")
    link = home / alias
    link.parent.mkdir(parents=True)
    link.symlink_to(store if alias.endswith("tools") else store.parent)
    before = _snapshot(store)
    run, provisioned = _run_runner(tmp_path, account, home, {"HERMES_RUNTIME_DIR": str(store)})
    assert provisioned is None, f"setup reached {provisioned}\n{run.stdout}{run.stderr}"
    assert run.returncode != 0
    assert "account's tool store" in run.stderr
    assert _snapshot(store) == before


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("named_by", ["HERMES_RUNTIME_DIR", "HERMES_HOME"])
def test_suffixed_account_store_is_protected_like_the_default_one(tmp_path, named_by):
    """HERMES_DATA_DIR_SUFFIX selects the account's ~/.hermes<suffix> home (hermes_constants), so
    its store is the account's store as much as ~/.hermes/tools is."""
    account, home = tmp_path / "account", tmp_path / "fakehome"
    store = _store_at_other_pins(account / ".hermes-isolated" / "tools")
    naming = {"HERMES_DATA_DIR_SUFFIX": "-isolated", **_naming(named_by, store)}
    provisioned = _provisioned_store(tmp_path, account, home, naming)
    assert not provisioned.exists() or not provisioned.samefile(store)
    assert provisioned == (home / ".hermes-isolated" / "tools").resolve()


@pytest.mark.platforms("posix")
def test_store_that_cannot_be_checked_without_a_bootstrap_python_is_refused(tmp_path):
    """With no interpreter to ask pm which store an inherited HERMES_HOME selects, the run must not
    fall through to setup, which would fetch its own Python and provision that store."""
    account, home = tmp_path / "account", tmp_path / "fakehome"
    store = _store_at_other_pins(account / ".hermes" / "tools")
    before = _snapshot(store)
    run, provisioned = _run_runner(tmp_path, account, home, _naming("HERMES_HOME", store), bootstrap=False)
    assert provisioned is None, f"setup reached {provisioned}\n{run.stdout}{run.stderr}"
    assert run.returncode != 0
    assert "cannot check" in run.stderr
    assert _snapshot(store) == before


@pytest.mark.platforms("posix")
def test_explicit_runtime_dir_is_checked_without_a_bootstrap_python(tmp_path):
    """pm takes HERMES_RUNTIME_DIR verbatim, so naming the account's store that way is recognised
    without an interpreter, and the run provisions the faked home's store instead of refusing."""
    account, home = tmp_path / "account", tmp_path / "fakehome"
    store = _store_at_other_pins(account / ".hermes" / "tools")
    run, provisioned = _run_runner(tmp_path, account, home, _naming("HERMES_RUNTIME_DIR", store), bootstrap=False)
    assert provisioned == (home / ".hermes" / "tools").resolve(), run.stdout + run.stderr
