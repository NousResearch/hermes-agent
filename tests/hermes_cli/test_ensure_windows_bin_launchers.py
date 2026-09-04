"""``hermes`` must survive git operations on the checkout (launcher layout).

The Windows ``hermes`` command is a staged launcher whose canonical home is
the managed binary dir ``HERMES_HOME\\bin`` — OUTSIDE the git checkout —
because the earlier in-checkout home (``hermes-agent\\bin``) was swept by
``hermes update``'s autostash (``git stash push --include-untracked``) and,
with the desktop updater's ``--keep-stash``, never restored: ``hermes``
stopped resolving in every new terminal (``venv\\Scripts`` itself must stay
off PATH — it shadows the user's ``python``, #83797).

Under pm (no-boot-through-venv), ``ensure_windows_bin_launchers`` re-stages
launchers that boot the pm STORE python with ``PYTHONPATH=repo;site-packages``
— never ``venv\\Scripts\\python.exe`` (``pyvenv.cfg`` is inert dead config).
When the store interpreter has not materialized yet, a runtime-resolving
``.cmd`` delegator is staged that finds the store python at boot and fails
with a clear message until ``hermes pm install`` lands it; legacy copied
venv trampolines (detected by their embedded interpreter path) are replaced.
``migrate_windows_bin_path`` moves an existing install's PATH to the
canonical layout from the ``hermes update`` tail. Platform verdict, PATH
values, and registry I/O are injected parameters (same pattern as
``hermes_constants.venv_bin_dir``), so these tests are host-independent
input→output checks, not host fakes.
"""

from pathlib import Path

import json

import pytest

from hermes_cli._install_repair import (
    _WINDOWS_BIN_LAUNCHERS,
    _normalize_windows_path,
    ensure_windows_bin_launchers,
    migrate_windows_bin_path,
)


def _make_managed(tmp_path, monkeypatch):
    """Fake managed layout: HERMES_HOME/hermes-agent/venv (+ a fake venv
    python whose path legacy trampolines would embed)."""
    home = tmp_path / "hermes"
    root = home / "hermes-agent"
    (root / "venv" / "Scripts").mkdir(parents=True)
    (root / "venv" / "Scripts" / "python.exe").write_bytes(b"MZ fake venv python")
    (root / "venv" / "pyvenv.cfg").write_text("home = X\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home, root


def _make_store(tmp_path, monkeypatch, *, version="3.11.15+x20260807"):
    """A fake pm store with a materialized python package. The interpreter
    file's bytes don't matter — the heal stages launchers, it never runs
    them — but the LAYOUT must match what pm/paths.py + facts.json
    describe (HERMES_RUNTIME_DIR override keeps tests off the real store)."""
    store = tmp_path / "store"
    entry = store / f"python-{version}-win32-arm64"
    (entry / "bin").mkdir(parents=True)
    (entry / "bin" / "python3").write_bytes(b"MZ fake store python")
    # Windows layout: python.exe at the entry root.
    (entry / "python.exe").write_bytes(b"MZ fake store python")
    (store / "facts.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "packages": {
                    "python": {"entry": entry.name, "version": version}
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    return store, entry


def _launcher_files(bin_dir: Path, name: str) -> list[Path]:
    return [p for p in (bin_dir / name).parent.iterdir() if p.stem == name]


def _assert_boot_is_store_not_venv(launcher: Path, root: Path, store: Path):
    """Whatever form staged, it must resolve to the STORE interpreter and
    compose repo-first PYTHONPATH — and never reference the venv python."""
    venv_python = str(root / "venv" / "Scripts" / "python.exe")
    if launcher.suffix == ".exe":
        data = launcher.read_bytes()
        assert venv_python.encode("utf-8") not in data
        assert venv_python.encode("utf-16-le") not in data
        assert str(store).encode("utf-8") in data
    else:
        body = launcher.read_text(encoding="utf-8")
        assert venv_python not in body
        assert "python-*" in body  # store-python boot-time resolution
        assert "pm install" in body  # instructive failure until pm lands it


@pytest.fixture
def managed_install(tmp_path, monkeypatch):
    home, root = _make_managed(tmp_path, monkeypatch)
    # Keep every test off the real machine store by default.
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "empty-store"))
    return home, root


def test_no_store_python_yet_stages_runtime_resolving_cmd(managed_install):
    """A fresh install (store not materialized) still gets working launchers
    — they resolve the store python AT BOOT and say so when it is absent."""
    home, root = managed_install

    restored = ensure_windows_bin_launchers(root, windows=True, user_path_entries=[])

    assert len(restored) == len(_WINDOWS_BIN_LAUNCHERS)
    for name in _WINDOWS_BIN_LAUNCHERS:
        body = (home / "bin" / f"{name}.cmd").read_text(encoding="utf-8")
        assert "python-*" in body
        assert "pm install" in body
        # Never a venv interpreter anywhere in the boot path.
        assert "venv\\Scripts\\python" not in body
        assert str(root / "venv") not in body.split("PYTHONPATH")[0]


def test_store_python_launcher_boot_the_store_not_the_venv(tmp_path, monkeypatch):
    """With the store materialized, the staged launchers bind to the store
    interpreter and never to the venv python."""
    home, root = _make_managed(tmp_path, monkeypatch)
    store, entry = _make_store(tmp_path, monkeypatch)

    restored = ensure_windows_bin_launchers(root, windows=True, user_path_entries=[])

    assert len(restored) == len(_WINDOWS_BIN_LAUNCHERS)
    for name in _WINDOWS_BIN_LAUNCHERS:
        staged = [p for p in map(Path, restored) if p.stem == name]
        assert staged, name
        _assert_boot_is_store_not_venv(staged[0], root, store)


def test_legacy_venv_trampoline_is_replaced_by_store_launcher(
    tmp_path, monkeypatch
):
    """Pre-pm installs carry copied venv console-script trampolines — they
    boot through venv\\Scripts\\python.exe and must be replaced."""
    home, root = _make_managed(tmp_path, monkeypatch)
    store, _entry = _make_store(tmp_path, monkeypatch)
    bin_dir = home / "bin"
    bin_dir.mkdir()
    venv_python = str(root / "venv" / "Scripts" / "python.exe")
    for name in _WINDOWS_BIN_LAUNCHERS:
        (bin_dir / f"{name}.exe").write_bytes(
            b"MZ legacy trampoline " + venv_python.encode("utf-8")
        )

    restored = ensure_windows_bin_launchers(root, windows=True, user_path_entries=[])

    assert set(Path(p).stem for p in map(Path, restored)) == set(_WINDOWS_BIN_LAUNCHERS)
    for name in _WINDOWS_BIN_LAUNCHERS:
        _assert_boot_is_store_not_venv(bin_dir / f"{name}.exe", root, store)


def test_healthy_store_launcher_is_a_noop(tmp_path, monkeypatch):
    home, root = _make_managed(tmp_path, monkeypatch)
    store, _entry = _make_store(tmp_path, monkeypatch)
    bin_dir = home / "bin"
    bin_dir.mkdir()
    for name in _WINDOWS_BIN_LAUNCHERS:
        # A launcher that does NOT embed the venv interpreter counts as
        # present, whatever wrote it.
        (bin_dir / f"{name}.exe").write_bytes(b"MZ already-staged launcher")

    assert ensure_windows_bin_launchers(root, windows=True, user_path_entries=[]) == []
    for name in _WINDOWS_BIN_LAUNCHERS:
        assert (bin_dir / f"{name}.exe").read_bytes() == b"MZ already-staged launcher"


def test_healthy_canonical_layout_with_placeholder_cmds_gets_upgraded(
    tmp_path, monkeypatch
):
    """Placeholder .cmd delegators from the fresh install are upgraded to
    store-python launchers once `hermes pm install` materialized the store."""
    home, root = _make_managed(tmp_path, monkeypatch)
    bin_dir = home / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "empty-store"))
    ensure_windows_bin_launchers(root, windows=True, user_path_entries=[])
    store, _entry = _make_store(tmp_path, monkeypatch)

    ensure_windows_bin_launchers(root, windows=True, user_path_entries=[])

    for name in _WINDOWS_BIN_LAUNCHERS:
        _assert_boot_is_store_not_venv(bin_dir / f"{name}.exe", root, store)


def test_legacy_bin_restaged_only_while_on_user_path(managed_install):
    home, root = managed_install
    legacy = root / "bin"

    restored = ensure_windows_bin_launchers(
        root, windows=True, user_path_entries=[str(legacy)]
    )

    stems = {Path(p).stem for p in map(Path, restored)}
    assert set(_WINDOWS_BIN_LAUNCHERS) <= stems
    for name in _WINDOWS_BIN_LAUNCHERS:
        assert (legacy / f"{name}.cmd").is_file()          # legacy consent honored
        assert (home / "bin" / f"{name}.cmd").is_file()    # canonical healed too


def test_legacy_bin_not_restaged_without_path_consent(managed_install):
    home, root = managed_install

    ensure_windows_bin_launchers(root, windows=True, user_path_entries=[])

    assert not (root / "bin").exists()


def test_source_checkout_untouched(tmp_path, monkeypatch):
    """A checkout NOT under HERMES_HOME gains nothing anywhere."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "empty-store"))
    root = tmp_path / "src" / "hermes-agent"
    (root / "venv" / "Scripts").mkdir(parents=True)

    assert ensure_windows_bin_launchers(root, windows=True, user_path_entries=[]) == []
    assert not (home / "bin").exists()
    assert not (root / "bin").exists()


def test_noop_on_posix(managed_install):
    home, root = managed_install

    assert ensure_windows_bin_launchers(root, windows=False) == []
    assert not (home / "bin").exists()


def test_profile_session_still_heals_the_shared_bin(tmp_path, monkeypatch):
    """Under ``hermes -p <name>`` HERMES_HOME points inside profiles/<name>;
    the launcher dir is per-machine, so the heal must anchor on the default
    root and fire anyway — a habitual profile user gets the same repair."""
    home = tmp_path / "hermes"
    root = home / "hermes-agent"
    (root / "venv" / "Scripts").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home / "profiles" / "work"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "empty-store"))

    restored = ensure_windows_bin_launchers(root, windows=True, user_path_entries=[])

    assert len(restored) == len(_WINDOWS_BIN_LAUNCHERS)
    for name in _WINDOWS_BIN_LAUNCHERS:
        assert (home / "bin" / f"{name}.cmd").is_file()
    assert not (home / "profiles" / "work" / "bin").exists()


def test_no_staging_litter_left_behind(managed_install):
    home, root = managed_install

    ensure_windows_bin_launchers(root, windows=True, user_path_entries=[])

    leftovers = [
        p.name for p in (home / "bin").iterdir() if ".stage." in p.name or ".heal." in p.name
    ]
    assert leftovers == []


# ---------------------------------------------------------------------------
# migrate_windows_bin_path — the `hermes update` tail migration
# ---------------------------------------------------------------------------


def _fake_registry(initial: list[str]):
    """In-memory user-PATH store standing in for the HKCU registry value."""
    state = {"entries": list(initial), "kind": 2, "writes": 0}

    def read():
        return list(state["entries"]), state["kind"]

    def write(entries, kind):
        state["entries"] = list(entries)
        state["kind"] = kind
        state["writes"] += 1

    return state, read, write


def test_migration_moves_path_to_home_bin_and_strips_legacy(managed_install):
    home, root = managed_install
    legacy_bin = str(root / "bin")
    legacy_scripts = str(root / "venv" / "Scripts")
    state, read, write = _fake_registry(
        [legacy_bin, legacy_scripts, r"C:\Windows\system32"]
    )
    (root / "bin").mkdir()
    (root / "bin" / "hermes.cmd").write_text("@echo off\r\n", encoding="ascii")

    ok = migrate_windows_bin_path(
        root, windows=True, read_user_path=read, write_user_path=write
    )

    assert ok
    keys = [_normalize_windows_path(e) for e in state["entries"]]
    assert _normalize_windows_path(home / "bin") in keys
    assert _normalize_windows_path(legacy_bin) not in keys
    assert _normalize_windows_path(legacy_scripts) not in keys
    assert _normalize_windows_path(r"C:\Windows\system32") in keys  # untouched
    for name in _WINDOWS_BIN_LAUNCHERS:
        assert (home / "bin" / f"{name}.cmd").is_file()
    # Legacy FILES stay: editor/ACP configs holding absolute launcher paths
    # keep working. Only the PATH entry (the sweepable resolution route) goes.
    assert (root / "bin" / "hermes.cmd").exists()


def test_migration_works_when_only_legacy_copy_exists(tmp_path, monkeypatch):
    home, root = _make_managed(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "empty-store"))
    state, read, write = _fake_registry([str(root / "bin")])

    ok = migrate_windows_bin_path(
        root, windows=True, read_user_path=read, write_user_path=write
    )

    assert ok
    for name in _WINDOWS_BIN_LAUNCHERS:
        assert (home / "bin" / f"{name}.cmd").is_file()
    keys = [_normalize_windows_path(e) for e in state["entries"]]
    assert _normalize_windows_path(home / "bin") in keys


def test_migration_is_idempotent(managed_install):
    home, root = managed_install
    state, read, write = _fake_registry([str(home / "bin"), r"C:\Windows\system32"])

    assert migrate_windows_bin_path(
        root, windows=True, read_user_path=read, write_user_path=write
    )
    first_entries = list(state["entries"])
    first_writes = state["writes"]

    assert migrate_windows_bin_path(
        root, windows=True, read_user_path=read, write_user_path=write
    )
    assert state["entries"] == first_entries
    assert state["writes"] == first_writes  # no redundant registry write


def test_migration_works_with_empty_store(tmp_path, monkeypatch):
    """Staging no longer depends on venv console scripts: even with an EMPTY
    pm store (no python staged yet) the runtime-resolving delegators stage,
    so the PATH migration completes; the cmd boot path is what instructs
    the user to run `hermes pm install`."""
    home = tmp_path / "hermes"
    root = home / "hermes-agent"
    (root / "venv" / "Scripts").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (tmp_path / "empty-store").mkdir()
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "empty-store"))
    state, read, write = _fake_registry([str(root / "bin"), r"C:\Windows\system32"])

    ok = migrate_windows_bin_path(
        root, windows=True, read_user_path=read, write_user_path=write
    )

    assert ok
    for name in _WINDOWS_BIN_LAUNCHERS:
        body = (home / "bin" / f"{name}.cmd").read_text(encoding="utf-8")
        assert "pm install" in body
    assert state["writes"] == 1


def test_migration_skips_source_checkouts(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "empty-store"))
    root = tmp_path / "src" / "hermes-agent"
    (root / "venv" / "Scripts").mkdir(parents=True)
    state, read, write = _fake_registry([r"C:\Windows\system32"])

    assert not migrate_windows_bin_path(
        root, windows=True, read_user_path=read, write_user_path=write
    )
    assert state["writes"] == 0


def test_migration_noop_on_posix(managed_install):
    home, root = managed_install

    assert not migrate_windows_bin_path(root, windows=False)


def test_normalize_windows_path_equivalences():
    assert (
        _normalize_windows_path(r"C:\Users\Me\AppData\Local\hermes\bin")
        == _normalize_windows_path("c:/users/me/appdata/local/HERMES/BIN/")
    )


def test_repo_gitignores_the_legacy_bin_dir():
    """Transition safety: legacy in-checkout launchers must not be stash-swept.

    Until every install has migrated, pre-migration checkouts still carry
    launchers at ``<checkout>/bin``. ``hermes update`` autostashes with
    ``git stash push --include-untracked``; anything untracked and NOT
    ignored inside the checkout gets swept off disk. Exercises git's real
    ignore machinery rather than reading .gitignore text.
    """
    import os
    import shutil
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        pytest.skip("not running from a git checkout")

    # conftest blanks SystemRoot/ComSpec and PATH can resolve `git` to an
    # MSIX payload copy (WinError 5 outside its package context) — resolve
    # a launchable git and pass a complete child env, as the bootstrap
    # version-stamp tests do.
    git = shutil.which("git") or "git"
    if sys.platform == "win32":
        pf = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        for rel in (("Git", "cmd", "git.exe"), ("Git", "bin", "git.exe")):
            cand = pf.joinpath(*rel)
            if cand.exists():
                git = str(cand)
                break
    env = os.environ.copy()
    if sys.platform == "win32":
        env.setdefault("SystemRoot", r"C:\Windows")
        env.setdefault("ComSpec", r"C:\Windows\system32\cmd.exe")

    result = subprocess.run(
        [git, "-C", str(repo_root), "check-ignore", "-q", "bin/hermes.exe"],
        capture_output=True, env=env,
    )
    assert result.returncode == 0, (
        "bin/hermes.exe is not gitignored — hermes update's autostash "
        "(--include-untracked) would sweep pre-migration launchers off disk"
    )
