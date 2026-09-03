"""Rollout helpers: dedicated board ACLs, sqlite backup, bounded copy, SHA gates."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest

from hermes_cli.kanban_os_users import (
    DEFAULT_GROUP,
    LaunchHooks,
    PasswdEntry,
    audit_mapping,
    migrate_profile_files_commands,
    plan_migrate_steps,
    plan_setup_steps,
)
from hermes_cli.kanban_os_users_rollout import (
    apply_command_hint,
    copy_tree_reject_symlinks,
    feature_source_root,
    format_rollout_and_rollback,
    hermes_argv_covers_feature,
    reject_write_acl_on_hermes_root,
    shared_board_acl_layout,
    sqlite_backup_copy,
    summarize_tree,
)


pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="profile_os_users is a Linux privilege-drop feature",
)

MAPPING = {"dev": "hermes-dev", "sysadmin": "hermes-sysadmin"}


class _Step:
    def __init__(self, argv):
        self.argv = argv


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


def test_layout_retargets_write_off_hermes_root(tmp_path):
    paths = _board_paths(tmp_path)
    layout = shared_board_acl_layout(paths)
    assert layout["writable_dir"] == paths["kanban_dir"]
    assert layout["target_db"] == paths["kanban_dir"] / "kanban.db"
    assert layout["needs_migration"] is True
    assert layout["writable_dir"] != paths["hermes_root"]


def test_plan_setup_never_write_acl_on_hermes_root(tmp_path):
    paths = _board_paths(tmp_path)
    steps = plan_setup_steps(mapping=MAPPING, gateway_user="matt", board_paths=paths)
    hermes_root = str(paths["hermes_root"])
    kdir = str(paths["kanban_dir"])
    setfacls = [s.argv for s in steps if s.argv and s.argv[0] == "setfacl"]
    root_acls = [a for a in setfacls if a[-1] == hermes_root]
    assert root_acls
    assert all("g:hermes-kanban:--x" in a for a in root_acls)
    assert not any(":wx" in " ".join(a) or ":rw" in " ".join(a) for a in root_acls)
    kdir_acls = [a for a in setfacls if a[-1] == kdir]
    assert any("g:hermes-kanban:wx" in a for a in kdir_acls)
    assert any("-d" in a and "g:hermes-kanban:rw" in a for a in kdir_acls)
    reject_write_acl_on_hermes_root(steps, paths["hermes_root"])


def test_reject_write_acl_on_hermes_root_raises(tmp_path):
    root = tmp_path / ".hermes"
    root.mkdir()
    with pytest.raises(ValueError, match="refusing write ACL"):
        reject_write_acl_on_hermes_root(
            [_Step(["setfacl", "-m", "g:hermes-kanban:wx", str(root)])],
            root,
        )


def test_sqlite_backup_not_raw_copy(tmp_path):
    src = tmp_path / "live.db"
    dst = tmp_path / "dedicated" / "kanban.db"
    con = sqlite3.connect(str(src))
    con.execute("CREATE TABLE t (id INTEGER, note TEXT)")
    con.execute("INSERT INTO t VALUES (1, 'secret-should-not-print')")
    con.commit()
    con.close()
    sqlite_backup_copy(src, dst)
    out = sqlite3.connect(str(dst))
    row = out.execute("SELECT id FROM t").fetchone()
    out.close()
    assert row == (1,)
    assert dst.is_file()


def test_copy_tree_rejects_symlinks_and_preserves_files(tmp_path):
    src = tmp_path / "skills"
    dst = tmp_path / "out"
    (src / "nested").mkdir(parents=True)
    (src / "nested" / "ok.md").write_text("# skill\n", encoding="utf-8")
    (src / ".env").write_text("TOKEN=super-secret\n", encoding="utf-8")
    os.symlink(src / "nested" / "ok.md", src / "link.md")
    stats = copy_tree_reject_symlinks(src, dst, owner="matt", group="matt")
    assert stats["rejected_symlinks"] >= 1
    assert (dst / "nested" / "ok.md").is_file()
    assert not (dst / "link.md").exists()
    assert (dst / ".env").read_text(encoding="utf-8") == "TOKEN=super-secret\n"
    assert oct(os.stat(dst / ".env").st_mode & 0o777) == "0o600"


def test_migrate_dry_run_summarizes_instead_of_per_file_flood(tmp_path):
    src = tmp_path / "profiles" / "dev" / "skills"
    src.mkdir(parents=True)
    for i in range(200):
        (src / f"skill-{i:03d}.md").write_text("# x\n", encoding="utf-8")
    (tmp_path / "profiles" / "dev" / ".env").write_text(
        "CURSOR_API_KEY=super-secret\n", encoding="utf-8"
    )
    text = "\n".join(
        migrate_profile_files_commands({"dev": "hermes-dev"}, source_root=tmp_path)
    )
    assert "super-secret" not in text
    assert "CURSOR_API_KEY=" not in text
    assert "files=200" in text
    assert text.count("copy-tree") == 1
    assert len(text) < 8000
    steps = plan_migrate_steps({"dev": "hermes-dev"}, source_root=tmp_path)
    assert len(steps) <= 8
    summary = summarize_tree(src)
    assert summary["files"] == 200


def test_installed_hermes_argv_does_not_cover_feature():
    assert hermes_argv_covers_feature(["/usr/bin/hermes"]) is False
    root = feature_source_root()
    assert hermes_argv_covers_feature([sys.executable, "-m", "hermes_cli.main"]) or (
        hermes_argv_covers_feature([
            str(root / ".venv" / "bin" / "python"),
            "-m",
            "hermes_cli.main",
        ])
        or hermes_argv_covers_feature([str(root / "cli.py")])
    )
    hint = apply_command_hint([sys.executable, "-m", "hermes_cli.main"])
    assert hint.startswith("sudo ")
    assert "sudo hermes " not in hint


def test_host_gates_fail_closed_on_old_runtime_and_ssh(tmp_path):
    pw = _pw()
    hermes_home = tmp_path / "hermes-dev" / ".hermes" / "profiles" / "dev"
    hermes_home.mkdir(parents=True)
    os.chmod(hermes_home, 0o700)
    (hermes_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (hermes_home / ".env").write_text("TOKEN=hidden\n", encoding="utf-8")
    (hermes_home / "SOUL.md").write_text("# soul\n", encoding="utf-8")
    (hermes_home / "skills").mkdir()
    os.chmod(hermes_home / ".env", 0o600)
    db_parent = tmp_path / "kanban"
    db_parent.mkdir()
    db = db_parent / "kanban.db"
    db.write_bytes(b"")

    def run(argv, **kwargs):
        if "/usr/bin/id" in argv or (
            len(argv) >= 2 and argv[-2:] == ["/usr/bin/id", "-u"]
        ):
            return _Proc(0, stdout=f"{pw.pw_uid}\n")
        path = argv[-1] if argv else ""
        if path == "/home/matt/.ssh" or "id_" in str(path):
            return _Proc(1)
        if argv and "gh" in argv[-3:]:
            return _Proc(0, stdout='{"login": "should-not-appear"}\n')
        if argv and "ls-remote" in argv:
            return _Proc(0, stdout="abc HEAD\n")
        return _Proc(0)

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
        homes={"dev": str(hermes_home)},
        hooks=hooks,
        board_paths={"kanban_db": db, "hermes_root": tmp_path / "not-parent"},
        wal_prover=lambda u, p: None,
        host_gates=True,
        hermes_argv=["/usr/bin/hermes"],
    )
    by_name = {i.name: i for i in items}
    assert by_name["runtime-sha"].ok is False
    assert by_name["runtime-sha"].isolation is False
    assert by_name["github-api"].ok is True
    assert "should-not-appear" not in by_name["github-api"].detail
    assert "hidden" not in " ".join(i.detail for i in items)
    assert by_name["deny-matt-ssh"].ok is True


def test_docs_and_rollout_text_cover_gates():
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
    assert "HERMES_KANBAN_DB" in text
    assert "/home/matt/.hermes/kanban/kanban.db" in text
    assert "never write" in text.lower() or "never put a write ACL" in text.lower()
    assert "sudo hermes" in text and "Do not use it" in text
    assert "/opt/hermes/kanban-os-users" in text
    assert "gh api user" in text
    blob = format_rollout_and_rollback(
        hermes_argv=[sys.executable, "-m", "hermes_cli.main"]
    )
    assert "Ordered rollout" in blob
    assert "Rollback" in blob
    assert "sqlite backup" in blob.lower() or "migrate-db" in blob
