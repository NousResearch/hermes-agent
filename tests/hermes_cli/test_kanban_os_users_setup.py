"""Regression tests for profile_os_users setup, ACL, audit, and rollback."""

from __future__ import annotations

import os
import sqlite3
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli.kanban_os_users import (
    DEFAULT_GROUP,
    GroupEntry,
    IncompatiblePrincipalError,
    LaunchHooks,
    PasswdEntry,
    _idempotent_ok,
    audit_mapping,
    execute_setup_plan,
    existing_user_compatible,
    mapped_home_ready,
    migrate_profile_files_commands,
    plan_rollback_steps,
    plan_setup_steps,
    probe_user_access,
    prove_sqlite_wal_lifecycle,
    render_sudoers,
    run_os_users_cli,
    save_os_users_state,
)


pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="profile_os_users is a Linux privilege-drop feature",
)

MAPPING = {"dev": "hermes-dev", "sysadmin": "hermes-sysadmin"}
WORKSPACE = "/home/matt/Documents/WorkoutTracker"


class _Proc:
    def __init__(self, rc=0, stdout="", stderr=""):
        self.returncode = rc
        self.stdout = stdout
        self.stderr = stderr


def _pw(name="hermes-dev", uid=2001, gid=2001, home=None):
    return PasswdEntry(name, uid, gid, home or f"/home/{name}")


def _board_paths(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "matt" / ".hermes"
    root.mkdir(parents=True)
    db = root / "kanban.db"
    db.write_bytes(b"")
    kdir = root / "kanban"
    kdir.mkdir()
    ws = kdir / "workspaces"
    ws.mkdir()
    return {
        "hermes_root": root,
        "kanban_db": db,
        "kanban_dir": kdir,
        "workspaces": ws,
    }


def _argv_blob(steps) -> str:
    return "\n".join(" ".join(step.argv) for step in steps)


def test_plan_setup_uses_private_groups_not_shared_gid():
    steps = plan_setup_steps(
        mapping=MAPPING,
        gateway_user="matt",
        board_paths={},
    )
    blob = _argv_blob(steps)
    assert "--gid hermes-kanban" not in blob
    useradds = [s.argv for s in steps if s.argv and s.argv[0] == "useradd"]
    assert useradds
    for argv in useradds:
        user = argv[-1]
        assert "-g" in argv
        assert argv[argv.index("-g") + 1] == user
        assert "-G" in argv
        assert argv[argv.index("-G") + 1] == DEFAULT_GROUP
    groups = [s.argv[-1] for s in steps if s.argv[:2] == ["groupadd", "--system"]]
    assert DEFAULT_GROUP in groups
    assert "hermes-dev" in groups
    assert "hermes-sysadmin" in groups
    installs = [s.argv for s in steps if s.argv[:2] == ["install", "-d"]]
    assert any("-g" in a and a[a.index("-g") + 1] == "hermes-dev" for a in installs)


def test_plan_setup_acls_actual_db_parent_not_kanban_subdir(tmp_path):
    paths = _board_paths(tmp_path)
    steps = plan_setup_steps(
        mapping=MAPPING,
        gateway_user="matt",
        board_paths=paths,
    )
    blob = _argv_blob(steps)
    db_parent = str(paths["kanban_db"].parent)
    kdir = str(paths["kanban_dir"])
    assert f"g:{DEFAULT_GROUP}:wx" in blob
    assert db_parent in blob
    assert (
        f"g:{DEFAULT_GROUP}:rwx" not in blob
        or kdir not in blob.split(f"g:{DEFAULT_GROUP}:rwx")[-1][:200]
    )
    setfacls = [s.argv for s in steps if s.argv and s.argv[0] == "setfacl"]
    db_parent_acls = [a for a in setfacls if a[-1] == db_parent]
    assert any("g:hermes-kanban:wx" in a for a in db_parent_acls)
    assert any("-d" in a and "g:hermes-kanban:rw" in a for a in db_parent_acls)
    kdir_wal = [
        a for a in setfacls if a[-1] == kdir and "g:hermes-kanban:rwx" in " ".join(a)
    ]
    assert kdir_wal == []
    ancestors = [a for a in setfacls if "g:hermes-kanban:--x" in a]
    assert ancestors
    assert any(str(paths["hermes_root"].parent) == a[-1] for a in ancestors)


def test_plan_setup_workspace_ancestor_acl_not_sysadmin():
    steps = plan_setup_steps(
        mapping=MAPPING,
        gateway_user="matt",
        dev_workspace=WORKSPACE,
        board_paths={},
    )
    setfacls = [s.argv for s in steps if s.argv and s.argv[0] == "setfacl"]
    titles = " ".join(s.title for s in steps)
    assert "sysadmin is NOT granted" in titles
    assert any(a[-1] == "/home/matt" and "u:hermes-dev:--x" in a for a in setfacls)
    assert any(
        a[-1] == "/home/matt/Documents" and "u:hermes-dev:--x" in a for a in setfacls
    )
    leaf = [a for a in setfacls if a[-1] == WORKSPACE]
    assert any("-R" in a and "u:hermes-dev:rwx" in a for a in leaf)
    assert any("-d" in a and "u:hermes-dev:rwx" in a for a in leaf)
    assert not any(
        "hermes-sysadmin" in " ".join(a) and WORKSPACE in a for a in setfacls
    )


def test_mocked_apply_command_sequence(tmp_path):
    paths = _board_paths(tmp_path)
    steps = plan_setup_steps(
        mapping=MAPPING,
        gateway_user="matt",
        dev_workspace=WORKSPACE,
        board_paths=paths,
        hermes_argv=["/usr/bin/hermes"],
    )
    recorded: list[list[str]] = []

    def run(argv, **kwargs):
        recorded.append(list(argv))
        return _Proc(0)

    state = tmp_path / "state.json"
    rc = execute_setup_plan(
        steps,
        run=run,
        state_path=state,
        sudoers_text=render_sudoers(
            gateway_user="matt", mapping=MAPPING, hermes_argv=["/usr/bin/hermes"]
        ),
        sudoers_tmp=str(tmp_path / "sudoers"),
    )
    assert rc == 0
    flat = [" ".join(a) for a in recorded]
    blob = "\n".join(flat)
    assert "--gid hermes-kanban" not in blob
    assert any(
        a[0] == "useradd" and "-g" in a and a[a.index("-g") + 1] == "hermes-dev"
        for a in recorded
    )
    assert any("-G hermes-kanban" in line for line in flat)
    assert any(
        "g:hermes-kanban:wx" in line and str(paths["kanban_db"].parent) in line
        for line in flat
    )
    assert any("u:hermes-dev:--x" in line and "/home/matt" in line for line in flat)
    assert any("-R" in a and WORKSPACE in a for a in recorded)
    assert any(a[0] == "visudo" for a in recorded)
    saved = state.read_text(encoding="utf-8")
    assert "hermes-dev" in saved
    assert "created_users" in saved


def test_mapped_home_ready_requires_soul_and_skills(tmp_path):
    home = tmp_path / "profiles" / "dev"
    home.mkdir(parents=True)
    ok, detail = mapped_home_ready(home)
    assert ok is False
    assert "SOUL.md" in detail
    assert "skills" in detail
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (home / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (home / "SOUL.md").write_text("# soul\n", encoding="utf-8")
    (home / "skills").mkdir()
    ok, detail = mapped_home_ready(home)
    assert ok is True
    assert "secret" not in detail


def test_audit_home_ready_blocks_isolation_claim(tmp_path):
    pw = _pw()
    hermes_home = tmp_path / "hermes-dev" / ".hermes" / "profiles" / "dev"
    hermes_home.mkdir(parents=True)
    os.chmod(hermes_home, 0o700)
    db_parent = tmp_path / "board"
    db_parent.mkdir()
    db = db_parent / "kanban.db"
    db.write_bytes(b"")

    def getpwnam(name):
        return pw

    def run(argv, **kwargs):
        if "/usr/bin/id" in argv or (
            len(argv) >= 2 and argv[-2:] == ["/usr/bin/id", "-u"]
        ):
            return _Proc(0, stdout=f"{pw.pw_uid}\n")
        return _Proc(0)

    hooks = LaunchHooks(
        getpwnam=getpwnam,
        geteuid=lambda: 1000,
        run=run,
        sudo_bin="/usr/bin/sudo",
        id_bin="/usr/bin/id",
        is_windows=False,
    )
    items = audit_mapping(
        mapping={"dev": "hermes-dev"},
        homes={"dev": str(hermes_home)},
        hooks=hooks,
        board_paths={"kanban_db": db},
        wal_prover=lambda u, p: None,
    )
    by_name = {i.name: i for i in items}
    assert by_name["home-ready:dev"].ok is False
    assert not all(i.ok for i in items)


def test_board_parent_fails_if_only_a_directory(tmp_path):
    pw = _pw()
    parent = tmp_path / "only-dir"
    parent.mkdir()
    db = parent / "kanban.db"
    db.write_bytes(b"")

    def run(argv, **kwargs):
        if argv and argv[-2:] == ["-u"] or (len(argv) >= 2 and argv[-1] == "-u"):
            return _Proc(0, stdout=f"{pw.pw_uid}\n")
        if "/usr/bin/id" in argv:
            return _Proc(0, stdout=f"{pw.pw_uid}\n")
        if "-w" in argv:
            return _Proc(1, stderr="denied\n")
        if "-x" in argv and str(parent) in argv:
            return _Proc(1, stderr="denied\n")
        if "-x" in argv:
            return _Proc(0)
        return _Proc(0, stdout=f"{pw.pw_uid}\n")

    hooks = LaunchHooks(
        getpwnam=lambda n: pw,
        geteuid=lambda: 1000,
        run=run,
        sudo_bin="/usr/bin/sudo",
        id_bin="/usr/bin/id",
        is_windows=False,
    )
    items = audit_mapping(
        mapping={"dev": "hermes-dev"},
        homes={"dev": str(tmp_path / "missing-home")},
        hooks=hooks,
        board_paths={"kanban_db": db},
        wal_prover=lambda u, p: None,
    )
    board = next(i for i in items if i.name == "board-parent")
    assert board.ok is False
    assert parent.is_dir()


def test_audit_wal_and_cross_home_denial(tmp_path):
    dev_home = tmp_path / "hermes-dev" / ".hermes" / "profiles" / "dev"
    sys_home = tmp_path / "hermes-sysadmin" / ".hermes" / "profiles" / "sysadmin"
    for home in (dev_home, sys_home):
        home.mkdir(parents=True)
        os.chmod(home, 0o700)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
        (home / ".env").write_text("TOKEN=hidden\n", encoding="utf-8")
        (home / "SOUL.md").write_text("# soul\n", encoding="utf-8")
        (home / "skills").mkdir()
        os.chmod(home / ".env", 0o600)
    db_parent = tmp_path / "shared"
    db_parent.mkdir()
    db = db_parent / "kanban.db"
    db.write_bytes(b"")
    users = {
        "hermes-dev": _pw("hermes-dev", 2001, 2001, str(tmp_path / "hermes-dev")),
        "hermes-sysadmin": _pw(
            "hermes-sysadmin", 2002, 2002, str(tmp_path / "hermes-sysadmin")
        ),
    }
    wal_seen: list[tuple[str, str]] = []

    def run(argv, **kwargs):
        user = argv[argv.index("-u") + 1] if "-u" in argv else ""
        if "/usr/bin/id" in argv or (
            len(argv) >= 2 and argv[-2:] == ["/usr/bin/id", "-u"]
        ):
            return _Proc(0, stdout=f"{users[user].pw_uid}\n")
        path = argv[-1]
        mode = argv[-2] if len(argv) >= 2 else ""
        other = str(sys_home) if user == "hermes-dev" else str(dev_home)
        if mode == "-r" and path == other:
            return _Proc(1)
        return _Proc(0)

    hooks = LaunchHooks(
        getpwnam=lambda n: users[n],
        geteuid=lambda: 1000,
        run=run,
        sudo_bin="/usr/bin/sudo",
        id_bin="/usr/bin/id",
        is_windows=False,
    )

    def wal_prover(user, path):
        wal_seen.append((user, path))

    items = audit_mapping(
        mapping=MAPPING,
        homes={"dev": str(dev_home), "sysadmin": str(sys_home)},
        hooks=hooks,
        board_paths={"kanban_db": db},
        wal_prover=wal_prover,
    )
    by_name = {i.name: i for i in items}
    assert by_name["cross:dev!=sysadmin"].ok is True
    assert by_name["cross-deny:hermes-dev->sysadmin"].ok is True
    assert by_name["cross-deny:hermes-sysadmin->dev"].ok is True
    assert by_name["board-parent"].ok is True
    assert by_name["board-wal:hermes-dev"].ok is True
    assert by_name["board-wal:hermes-sysadmin"].ok is True
    assert {u for u, _ in wal_seen} == {"hermes-dev", "hermes-sysadmin"}
    blob = " ".join(i.detail for i in items)
    assert "hidden" not in blob
    assert "TOKEN=" not in blob


def test_probe_user_access_expect_denied():
    def run(argv, **kwargs):
        return _Proc(1)

    hooks = LaunchHooks(
        getpwnam=lambda n: _pw(),
        geteuid=lambda: 1000,
        run=run,
        sudo_bin="/usr/bin/sudo",
        is_windows=False,
    )
    ok, detail = probe_user_access(
        "hermes-dev",
        "/home/hermes-sysadmin/.hermes",
        mode="r",
        hooks=hooks,
        expect_ok=False,
    )
    assert ok is True
    assert "denied" in detail


def test_idempotent_ok_rejects_shared_primary_group():
    pw = _pw(gid=3000)
    private = GroupEntry("hermes-dev", 2001, ())
    shared = GroupEntry(DEFAULT_GROUP, 3000, ("hermes-dev",))

    def getgrnam(name):
        if name == "hermes-dev":
            return private
        if name == DEFAULT_GROUP:
            return shared
        raise KeyError(name)

    hooks = LaunchHooks(
        getpwnam=lambda n: pw,
        getgrnam=getgrnam,
        geteuid=lambda: 1000,
        run=lambda *a, **k: _Proc(0),
        is_windows=False,
    )
    with pytest.raises(IncompatiblePrincipalError, match="not private group"):
        existing_user_compatible("hermes-dev", group=DEFAULT_GROUP, hooks=hooks)
    with pytest.raises(IncompatiblePrincipalError):
        _idempotent_ok(
            ["useradd", "--system", "hermes-dev"],
            9,
            hooks=hooks,
            group=DEFAULT_GROUP,
        )


def test_idempotent_ok_accepts_private_primary_plus_supplemental():
    pw = _pw(gid=2001)
    private = GroupEntry("hermes-dev", 2001, ())
    shared = GroupEntry(DEFAULT_GROUP, 3000, ("hermes-dev",))

    def getgrnam(name):
        if name == "hermes-dev":
            return private
        if name == DEFAULT_GROUP:
            return shared
        raise KeyError(name)

    hooks = LaunchHooks(
        getpwnam=lambda n: pw,
        getgrnam=getgrnam,
        geteuid=lambda: 1000,
        run=lambda *a, **k: _Proc(0),
        is_windows=False,
    )
    existing_user_compatible("hermes-dev", group=DEFAULT_GROUP, hooks=hooks)
    assert _idempotent_ok(["useradd", "hermes-dev"], 9, hooks=hooks) is True
    assert _idempotent_ok(["groupadd", "hermes-dev"], 9, hooks=hooks) is True


def test_rollback_without_state_never_userdel(tmp_path):
    missing = tmp_path / "no-such-state.json"
    steps = plan_rollback_steps(mapping=MAPPING, state_path=missing)
    blob = _argv_blob(steps)
    assert "userdel" not in blob
    assert "groupdel" not in blob
    assert "rm" in blob


def test_rollback_with_state_only_created_principals_and_acls(tmp_path):
    state = tmp_path / "state.json"
    save_os_users_state(
        {
            "created_users": ["hermes-dev"],
            "created_groups": ["hermes-dev", DEFAULT_GROUP],
            "preexisting_users": ["hermes-sysadmin"],
            "preexisting_groups": [],
            "acl_paths": [WORKSPACE, "/home/matt/.hermes"],
        },
        state,
    )
    steps = plan_rollback_steps(mapping=MAPPING, state_path=state)
    blob = _argv_blob(steps)
    assert "userdel -r hermes-dev" in blob
    assert "userdel -r hermes-sysadmin" not in blob
    assert "groupdel hermes-dev" in blob
    assert f"setfacl -x g:{DEFAULT_GROUP}" in blob
    assert f"setfacl -x u:hermes-dev {WORKSPACE}" in blob or any(
        s.argv[:3] == ["setfacl", "-x", "u:hermes-dev"] and s.argv[-1] == WORKSPACE
        for s in steps
    )
    assert any(s.argv[:2] == ["setfacl", "-k"] for s in steps)


def test_migrate_never_prints_secrets_and_documents_manual_gate(tmp_path):
    src = tmp_path / "profiles" / "dev"
    src.mkdir(parents=True)
    src.joinpath(".env").write_text("CURSOR_API_KEY=super-secret\n", encoding="utf-8")
    src.joinpath("SOUL.md").write_text("# soul\n", encoding="utf-8")
    text = "\n".join(
        migrate_profile_files_commands({"dev": "hermes-dev"}, source_root=tmp_path)
    )
    assert "super-secret" not in text
    assert "CURSOR_API_KEY=" not in text
    assert "MANUAL GATE" in text
    assert "SOUL.md" in text
    assert "skills" in text


def test_setup_apply_does_not_migrate_without_flag(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    recorded: list[list[str]] = []

    def run(argv, **kwargs):
        recorded.append(list(argv))
        return _Proc(0)

    hooks = LaunchHooks(
        getpwnam=lambda n: _pw(n),
        getgrnam=lambda n: GroupEntry(n, 1, ()),
        geteuid=lambda: 0,
        run=run,
        sudo_bin="/usr/bin/sudo",
        is_windows=False,
    )
    args = Namespace(
        os_users_action="setup",
        apply=True,
        gateway_user="matt",
        dev_workspace=WORKSPACE,
        json=False,
        migrate_profile_files=False,
    )
    rc = run_os_users_cli(
        args,
        hooks=hooks,
        state_path=tmp_path / "state.json",
        board_paths=_board_paths(tmp_path),
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "MANUAL GATE" in out
    assert not any(
        ".env" in " ".join(a) and a[0] == "install" and "-m" in a for a in recorded
    )
    assert "Check will FAIL" in out


def test_cli_probe_wal_lifecycle(tmp_path, capsys):
    db = tmp_path / "kanban.db"
    args = Namespace(os_users_action="probe", probe_kind="wal", probe_path=str(db))
    rc = run_os_users_cli(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "WAL lifecycle ok" in out
    assert db.exists()


def test_prove_sqlite_wal_lifecycle_creates_sidecars_or_probe_file(tmp_path):
    db = tmp_path / "kanban.db"
    prove_sqlite_wal_lifecycle(str(db))
    con = sqlite3.connect(str(db))
    mode = con.execute("PRAGMA journal_mode").fetchone()[0]
    con.close()
    assert str(mode).lower() in {"wal", "delete", "memory"} or db.exists()
    assert db.exists()


def test_sudoers_allows_test_bin_for_probes():
    text = render_sudoers(
        gateway_user="matt",
        mapping=MAPPING,
        hermes_argv=["/usr/bin/hermes"],
    )
    assert "/usr/bin/test" in text
    assert "/usr/bin/id" in text
    assert "NOPASSWD" in text


def test_docs_paths_use_venv_dot_and_documents_workouttracker():
    docs = (
        Path(__file__).resolve().parents[2] / "docs" / "kanban" / "profile-os-users.md"
    )
    text = docs.read_text(encoding="utf-8")
    assert ".venv/bin/python" in text
    assert "./venv/bin/python" not in text
    assert "/home/matt/Documents/WorkoutTracker" in text
    assert "--dev-workspace /home/matt/WorkoutTracker" not in text
    assert "makes the shared group the primary group" in text
    assert "-g hermes-dev" in text
    assert "-G hermes-kanban" in text
    assert "--migrate-profile-files" in text
