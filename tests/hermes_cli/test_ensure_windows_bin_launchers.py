"""``hermes`` must survive git operations on the checkout (launcher layout).

The Windows ``hermes`` command is a launcher derived from the venv console
script. Its canonical home is the managed binary dir ``HERMES_HOME\\bin`` —
OUTSIDE the git checkout — because the earlier in-checkout home
(``hermes-agent\\bin``) was swept by ``hermes update``'s autostash
(``git stash push --include-untracked``) and, with the desktop updater's
``--keep-stash``, never restored: ``hermes`` stopped resolving in every new
terminal (``venv\\Scripts`` itself must stay off PATH — it shadows the
user's ``python``, #83797).

``ensure_windows_bin_launchers`` re-stages missing launchers (canonical dir
always for the managed clone; legacy dir only while the user PATH still
points at it), choosing the form by venv kind: exe copy for normal venvs,
``.cmd`` delegator for relocatable venvs whose exe trampolines die when
copied out of ``venv\\Scripts``. ``migrate_windows_bin_path`` moves an
existing install's PATH to the canonical layout from the ``hermes update``
tail. Platform verdict, PATH values, and registry I/O are injected
parameters (same pattern as ``hermes_constants.venv_bin_dir``), so these
tests are host-independent input→output checks, not host fakes.
"""

from pathlib import Path

import pytest

from hermes_cli._install_repair import (
    _WINDOWS_BIN_LAUNCHERS,
    _normalize_windows_path,
    ensure_windows_bin_launchers,
    migrate_windows_bin_path,
)


def _make_managed(tmp_path, monkeypatch, *, relocatable: bool = False):
    """Fake managed layout: HERMES_HOME/hermes-agent/venv/Scripts + launchers."""
    home = tmp_path / "hermes"
    root = home / "hermes-agent"
    scripts = root / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    for name in _WINDOWS_BIN_LAUNCHERS:
        (scripts / f"{name}.exe").write_bytes(b"MZ console script: " + name.encode())
    cfg = "home = X\nversion_info = 3.11.15\n"
    if relocatable:
        cfg += "relocatable = true\n"
    (root / "venv" / "pyvenv.cfg").write_text(cfg, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home, root


@pytest.fixture
def managed_install(tmp_path, monkeypatch):
    return _make_managed(tmp_path, monkeypatch)


def test_managed_clone_heals_canonical_home_bin(managed_install):
    home, root = managed_install

    restored = ensure_windows_bin_launchers(root, windows=True, user_path_entries=[])

    assert len(restored) == len(_WINDOWS_BIN_LAUNCHERS)
    for name in _WINDOWS_BIN_LAUNCHERS:
        assert (home / "bin" / f"{name}.exe").read_bytes() == (
            root / "venv" / "Scripts" / f"{name}.exe"
        ).read_bytes()


def test_relocatable_venv_gets_cmd_delegators_not_exe_copies(tmp_path, monkeypatch):
    """A copied relocatable-venv trampoline dies ('uv trampoline failed to
    canonicalize script path') — the heal must emit .cmd delegators."""
    home, root = _make_managed(tmp_path, monkeypatch, relocatable=True)

    restored = ensure_windows_bin_launchers(root, windows=True, user_path_entries=[])

    assert {Path(p).suffix for p in restored} == {".cmd"}
    for name in _WINDOWS_BIN_LAUNCHERS:
        body = (home / "bin" / f"{name}.cmd").read_text(encoding="ascii")
        # Delegates to the in-venv exe by absolute path, forwarding args.
        assert str(root / "venv" / "Scripts" / f"{name}.exe") in body
        assert "%*" in body
        assert not (home / "bin" / f"{name}.exe").exists()


def test_existing_exe_counts_as_present_for_relocatable_venv(tmp_path, monkeypatch):
    """Exe copies staged before a venv rebuild embed the swapped-in-place
    venv's absolute path and keep working — never replaced with .cmd."""
    home, root = _make_managed(tmp_path, monkeypatch, relocatable=True)
    (home / "bin").mkdir()
    for name in _WINDOWS_BIN_LAUNCHERS:
        (home / "bin" / f"{name}.exe").write_bytes(b"pre-rebuild copy")

    assert ensure_windows_bin_launchers(root, windows=True, user_path_entries=[]) == []
    for name in _WINDOWS_BIN_LAUNCHERS:
        assert (home / "bin" / f"{name}.exe").read_bytes() == b"pre-rebuild copy"
        assert not (home / "bin" / f"{name}.cmd").exists()


def test_healthy_canonical_layout_is_a_noop(managed_install):
    home, root = managed_install
    (home / "bin").mkdir()
    for name in _WINDOWS_BIN_LAUNCHERS:
        (home / "bin" / f"{name}.exe").write_bytes(b"present")

    assert ensure_windows_bin_launchers(root, windows=True, user_path_entries=[]) == []


def test_legacy_bin_restaged_only_while_on_user_path(managed_install):
    home, root = managed_install
    legacy = root / "bin"

    restored = ensure_windows_bin_launchers(
        root, windows=True, user_path_entries=[str(legacy)]
    )

    stems = {Path(p).stem for p in restored}
    assert set(_WINDOWS_BIN_LAUNCHERS) <= stems
    for name in _WINDOWS_BIN_LAUNCHERS:
        assert (legacy / f"{name}.exe").is_file()        # legacy consent honored
        assert (home / "bin" / f"{name}.exe").is_file()  # canonical healed too


def test_legacy_bin_not_restaged_without_path_consent(managed_install):
    home, root = managed_install

    ensure_windows_bin_launchers(root, windows=True, user_path_entries=[])

    assert not (root / "bin").exists()


def test_source_checkout_untouched(tmp_path, monkeypatch):
    """A checkout NOT under HERMES_HOME gains nothing anywhere."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    root = tmp_path / "src" / "hermes-agent"
    scripts = root / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    for name in _WINDOWS_BIN_LAUNCHERS:
        (scripts / f"{name}.exe").write_bytes(b"MZ")

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
    scripts = root / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    for name in _WINDOWS_BIN_LAUNCHERS:
        (scripts / f"{name}.exe").write_bytes(b"MZ")
    (root / "venv" / "pyvenv.cfg").write_text("home = X\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home / "profiles" / "work"))

    restored = ensure_windows_bin_launchers(root, windows=True, user_path_entries=[])

    assert len(restored) == len(_WINDOWS_BIN_LAUNCHERS)
    for name in _WINDOWS_BIN_LAUNCHERS:
        assert (home / "bin" / f"{name}.exe").is_file()
    assert not (home / "profiles" / "work" / "bin").exists()


def test_noop_when_console_scripts_missing(tmp_path, monkeypatch):
    """A venv mid-repair has no console scripts — nothing to copy, no error."""
    home = tmp_path / "hermes"
    root = home / "hermes-agent"
    (root / "venv" / "Scripts").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert ensure_windows_bin_launchers(root, windows=True, user_path_entries=[]) == []


def test_no_staging_litter_left_behind(managed_install):
    home, root = managed_install

    ensure_windows_bin_launchers(root, windows=True, user_path_entries=[])

    leftovers = [p.name for p in (home / "bin").iterdir() if ".heal." in p.name]
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
    (root / "bin" / "hermes.exe").write_bytes(b"legacy copy")

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
        assert (home / "bin" / f"{name}.exe").is_file()
    # Legacy FILES stay: editor/ACP configs holding absolute launcher paths
    # keep working. Only the PATH entry (the sweepable resolution route) goes.
    assert (root / "bin" / "hermes.exe").read_bytes() == b"legacy copy"


def test_migration_works_for_relocatable_venv(tmp_path, monkeypatch):
    home, root = _make_managed(tmp_path, monkeypatch, relocatable=True)
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


def test_migration_never_strips_path_when_staging_fails(tmp_path, monkeypatch):
    """No venv sources → launchers can't stage → PATH must stay untouched."""
    home = tmp_path / "hermes"
    root = home / "hermes-agent"
    (root / "venv" / "Scripts").mkdir(parents=True)  # no launcher exes inside
    monkeypatch.setenv("HERMES_HOME", str(home))
    legacy_bin = str(root / "bin")
    state, read, write = _fake_registry([legacy_bin])

    ok = migrate_windows_bin_path(
        root, windows=True, read_user_path=read, write_user_path=write
    )

    assert not ok
    assert state["entries"] == [legacy_bin]  # working entry preserved
    assert state["writes"] == 0


def test_migration_skips_source_checkouts(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    root = tmp_path / "src" / "hermes-agent"
    scripts = root / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    for name in _WINDOWS_BIN_LAUNCHERS:
        (scripts / f"{name}.exe").write_bytes(b"MZ")
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
    import subprocess

    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        pytest.skip("not running from a git checkout")

    result = subprocess.run(
        ["git", "-C", str(repo_root), "check-ignore", "-q", "bin/hermes.exe"],
        capture_output=True,
    )
    assert result.returncode == 0, (
        "bin/hermes.exe is not gitignored — hermes update's autostash "
        "(--include-untracked) would sweep pre-migration launchers off disk"
    )
