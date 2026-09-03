"""Security tests for kanban.profile_os_users mapped worker launch."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli.kanban_os_users import (
    LaunchHooks,
    MappedLaunchError,
    PasswdEntry,
    apply_mapped_worker_launch,
    audit_mapping,
    build_mapped_env,
    build_sudo_argv,
    is_sudo_wrapped,
    lookup_mapped_os_user,
    migrate_profile_files_commands,
    parse_profile_os_homes,
    parse_profile_os_users,
    preflight_mapped_user,
    render_sudoers,
    run_os_users_cli,
    validate_os_username,
)


pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="profile_os_users is a Linux privilege-drop feature",
)


class _Proc:
    def __init__(self, rc=0, stdout="", stderr=""):
        self.returncode = rc
        self.stdout = stdout
        self.stderr = stderr


def _pw(name="hermes-dev", uid=2001, gid=2001, home=None):
    return PasswdEntry(name, uid, gid, home or f"/home/{name}")


def _hooks(pw, *, euid=1000, run_rc=0, run_stdout=None, run_err=None):
    def getpwnam(name):
        if name != pw.pw_name:
            raise MappedLaunchError(f"Mapped OS user {name!r} does not exist.")
        return pw

    def run(argv, **kwargs):
        uid_out = run_stdout if run_stdout is not None else f"{pw.pw_uid}\n"
        if run_err is not None:
            return _Proc(run_rc, stdout=uid_out, stderr=run_err)
        return _Proc(run_rc, stdout=uid_out)

    return LaunchHooks(
        getpwnam=getpwnam,
        geteuid=lambda: euid,
        run=run,
        sudo_bin="/usr/bin/sudo",
        id_bin="/usr/bin/id",
        is_windows=False,
    )


def test_parse_empty_mapping_is_trusted_local():
    assert parse_profile_os_users(None) == {}
    assert parse_profile_os_users({}) == {}
    assert lookup_mapped_os_user("dev", {}) is None


def test_parse_valid_mapping():
    parsed = parse_profile_os_users({
        "dev": "hermes-dev",
        "sysadmin": "hermes-sysadmin",
    })
    assert parsed == {"dev": "hermes-dev", "sysadmin": "hermes-sysadmin"}
    assert lookup_mapped_os_user("Dev", parsed) == "hermes-dev"


@pytest.mark.parametrize(
    "raw",
    [
        "root",
        "toor",
        "ROOT",
        "hermes-dev;id",
        "hermes-dev$(id)",
        "a b",
        "../etc",
        "matt\nroot",
    ],
)
def test_validate_username_rejects_root_and_injection(raw):
    with pytest.raises(ValueError):
        validate_os_username(raw)


def test_parse_rejects_root_mapping():
    with pytest.raises(ValueError, match="root"):
        parse_profile_os_users({"dev": "root"})


def test_parse_rejects_invalid_profile_id():
    with pytest.raises(ValueError):
        parse_profile_os_users({"Dev Profile": "hermes-dev"})


def test_parse_homes_requires_absolute():
    with pytest.raises(ValueError, match="absolute"):
        parse_profile_os_homes({"dev": "relative/path"})
    parsed = parse_profile_os_homes({"dev": "/var/lib/hermes-dev"})
    assert parsed["dev"] == "/var/lib/hermes-dev"


def test_build_sudo_argv_is_list_no_shell():
    inner = [
        "/opt/hermes bin/hermes",
        "-p",
        "dev",
        "chat",
        "-q",
        "work kanban task t_1",
    ]
    argv = build_sudo_argv("hermes-dev", inner)
    assert argv[0] == "/usr/bin/sudo"
    assert argv[1:7] == ["-n", "-H", "-E", "-u", "hermes-dev", "--"]
    assert argv[7:] == inner
    assert " ".join(argv).count(";") == 0 or ";" not in "".join(argv[:7])
    assert is_sudo_wrapped(argv)
    assert not is_sudo_wrapped(inner)


def test_sudo_argv_preserves_spaces_and_metacharacters_as_single_tokens():
    nasty = "/tmp/work space; rm -rf / and $(reboot)"
    argv = build_sudo_argv("hermes-dev", ["hermes", "--cwd", nasty])
    assert nasty in argv
    assert argv[argv.index("--") + 1 :] == ["hermes", "--cwd", nasty]
    joined_for_shell = "sudo -n -u hermes-dev hermes --cwd " + nasty
    assert argv != joined_for_shell.split()


def test_unmapped_apply_is_identity():
    argv = ["hermes", "-p", "elias", "chat", "-q", "work"]
    env = {"HERMES_HOME": "/tmp/h", "SSH_AUTH_SOCK": "/run/ssh"}
    out_argv, out_env = apply_mapped_worker_launch(
        profile="elias",
        argv=argv,
        env=env,
        mapping={},
        homes={},
        preflight=False,
    )
    assert out_argv == argv
    assert out_env == env
    assert not is_sudo_wrapped(out_argv)


def test_mapped_apply_wraps_and_rewrites_env():
    pw = _pw()
    argv = ["hermes", "-p", "dev", "chat", "-q", "work kanban task t_x"]
    env = {
        "HERMES_HOME": "/home/matt/.hermes/profiles/dev",
        "SSH_AUTH_SOCK": "/run/user/1000/ssh",
        "CURSOR_API_KEY": "secret-cursor-key",
        "OPENAI_API_KEY": "sk-test",
        "HERMES_KANBAN_TASK": "t_x",
    }
    out_argv, out_env = apply_mapped_worker_launch(
        profile="dev",
        argv=argv,
        env=env,
        mapping={"dev": "hermes-dev"},
        homes={},
        hooks=_hooks(pw),
        preflight=False,
        workspace="",
        board_db=None,
    )
    assert is_sudo_wrapped(out_argv)
    assert out_argv[:7] == ["/usr/bin/sudo", "-n", "-H", "-E", "-u", "hermes-dev", "--"]
    assert out_argv[7:] == argv
    assert out_env["HOME"] == pw.pw_dir
    assert out_env["USER"] == "hermes-dev"
    assert out_env["HERMES_HOME"] == "/home/hermes-dev/.hermes/profiles/dev"
    assert "SSH_AUTH_SOCK" not in out_env
    assert "CURSOR_API_KEY" not in out_env
    assert "OPENAI_API_KEY" not in out_env
    assert out_env["HERMES_KANBAN_TASK"] == "t_x"


def test_mapped_apply_never_returns_unwrapped_on_sudo_denial():
    pw = _pw()
    hooks = _hooks(pw, run_rc=1, run_err="sudo: a password is required\n")
    with pytest.raises(MappedLaunchError, match="sudo"):
        apply_mapped_worker_launch(
            profile="dev",
            argv=["hermes", "-p", "dev"],
            env={},
            mapping={"dev": "hermes-dev"},
            homes={},
            hooks=hooks,
            preflight=True,
        )


def test_preflight_rejects_missing_user():
    hooks = LaunchHooks(
        getpwnam=lambda n: (_ for _ in ()).throw(MappedLaunchError("missing")),
        geteuid=lambda: 1000,
        run=lambda *a, **k: _Proc(0, "1\n"),
        is_windows=False,
        sudo_bin="/usr/bin/sudo",
    )
    with pytest.raises(MappedLaunchError, match="missing"):
        preflight_mapped_user("hermes-dev", hooks=hooks, require_paths=False)


def test_preflight_rejects_same_uid_and_does_not_call_it_isolation():
    pw = _pw(uid=1000)
    with pytest.raises(MappedLaunchError, match="not isolation"):
        preflight_mapped_user(
            "hermes-dev", hooks=_hooks(pw, euid=1000), require_paths=False
        )


def test_preflight_rejects_wrong_reported_uid():
    pw = _pw(uid=2001)
    hooks = _hooks(pw, run_stdout="0\n")
    with pytest.raises(MappedLaunchError, match="expected 2001"):
        preflight_mapped_user("hermes-dev", hooks=hooks, require_paths=False)


def test_preflight_rejects_windows():
    pw = _pw()
    hooks = _hooks(pw)
    hooks.is_windows = True
    with pytest.raises(MappedLaunchError, match="Linux-only"):
        preflight_mapped_user("hermes-dev", hooks=hooks, require_paths=False)


def test_build_mapped_env_drops_gateway_secrets():
    env = build_mapped_env(
        {
            "SSH_AUTH_SOCK": "/tmp/ssh",
            "SUDO_USER": "matt",
            "CURSOR_API_KEY": "nope",
            "HERMES_KANBAN_DB": "/shared/kanban.db",
            "PATH": "/usr/bin",
        },
        username="hermes-dev",
        home="/home/hermes-dev",
        hermes_home="/home/hermes-dev/.hermes/profiles/dev",
    )
    assert "SSH_AUTH_SOCK" not in env
    assert "SUDO_USER" not in env
    assert "CURSOR_API_KEY" not in env
    assert env["HERMES_KANBAN_DB"] == "/shared/kanban.db"
    assert env["HOME"] == "/home/hermes-dev"


def test_retries_remain_mapped():
    pw = _pw()
    hooks = _hooks(pw)
    mapping = {"dev": "hermes-dev"}
    first, _ = apply_mapped_worker_launch(
        profile="dev",
        argv=["hermes", "-p", "dev"],
        env={},
        mapping=mapping,
        homes={},
        hooks=hooks,
        preflight=False,
    )
    second, _ = apply_mapped_worker_launch(
        profile="dev",
        argv=["hermes", "-p", "dev"],
        env={},
        mapping=mapping,
        homes={},
        hooks=hooks,
        preflight=False,
    )
    assert is_sudo_wrapped(first) and is_sudo_wrapped(second)
    assert first[5] == second[5] == "hermes-dev"


def test_audit_empty_mapping_is_not_isolation():
    items = audit_mapping(mapping={}, homes={})
    assert items[0].ok is True
    assert items[0].isolation is False
    assert "trusted-local-user" in items[0].detail


def test_sudoers_and_migrate_never_print_secret_values(tmp_path):
    src = tmp_path / "profiles" / "dev"
    src.mkdir(parents=True)
    src.joinpath(".env").write_text("CURSOR_API_KEY=super-secret\n", encoding="utf-8")
    text = "\n".join(
        migrate_profile_files_commands({"dev": "hermes-dev"}, source_root=tmp_path)
    )
    sudoers = render_sudoers(
        gateway_user="matt",
        mapping={"dev": "hermes-dev", "sysadmin": "hermes-sysadmin"},
        hermes_argv=["/usr/bin/hermes"],
    )
    blob = text + "\n" + sudoers
    assert "super-secret" not in blob
    assert "CURSOR_API_KEY=" not in blob
    assert "BEGIN OPENSSH" not in blob
    assert "install" in text
    assert "NOPASSWD" in sudoers
    assert "hermes-dev" in sudoers
    assert "hermes-sysadmin" in sudoers


def test_cli_setup_dry_run_does_not_require_root(capsys):
    args = Namespace(
        os_users_action="setup",
        apply=False,
        gateway_user="matt",
        dev_workspace="/home/matt/Documents/WorkoutTracker",
        json=False,
        migrate_profile_files=False,
    )
    rc = run_os_users_cli(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "Dry-run only" in out
    assert "/home/matt/Documents/WorkoutTracker" in out
    assert "sysadmin not granted" in out
    assert "world-readable is not isolation" in out
    assert "Do not use `sudo hermes`" in out
    assert "kanban os-users setup --apply" in out
    assert "MANUAL GATE" in out
    assert "--gid hermes-kanban" not in out
    assert "-g hermes-dev" in out
    assert "-G hermes-kanban" in out


def test_cli_setup_apply_without_root_fails_closed(capsys, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    args = Namespace(
        os_users_action="setup",
        apply=True,
        gateway_user="matt",
        dev_workspace=None,
        json=False,
        migrate_profile_files=False,
    )
    rc = run_os_users_cli(args)
    err = capsys.readouterr().err
    assert rc == 1
    assert "euid 0" in err
    assert "Not prompting" in err


def _make_task(kb, *, assignee: str):
    return kb.Task(
        id="t_os_users",
        title="spawn",
        body=None,
        assignee=assignee,
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
    )


def test_default_spawn_unmapped_stays_on_gateway_argv(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pid = kb._default_spawn(_make_task(kb, assignee="elias"), str(workspace))
    assert pid == 4242
    assert captured["cmd"][0] == "hermes"
    assert captured["cmd"][1:3] == ["-p", "elias"]
    assert not is_sudo_wrapped(captured["cmd"])


def test_default_spawn_mapped_missing_user_never_popen(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "dev"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text(
        "kanban:\n  profile_os_users:\n    dev: hermes-dev\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return type("P", (), {"pid": 1})()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with pytest.raises((MappedLaunchError, RuntimeError, ValueError, KeyError)):
        kb._default_spawn(_make_task(kb, assignee="dev"), str(workspace))
    assert "cmd" not in captured


def test_default_spawn_mapped_reports_target_user(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "dev"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text(
        "kanban:\n  profile_os_users:\n    dev: hermes-dev\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_cli import kanban_db as kb
    import hermes_cli.kanban_os_users as osu

    pw = _pw(home=str(tmp_path / "hermes-dev"))
    mapped_home = tmp_path / "hermes-dev" / ".hermes" / "profiles" / "dev"
    mapped_home.mkdir(parents=True)
    os.chmod(mapped_home, 0o700)
    os.chmod(mapped_home.parent, 0o700)
    os.chmod(mapped_home.parent.parent, 0o700)

    monkeypatch.setattr(osu, "_default_getpwnam", lambda name: pw)
    monkeypatch.setattr(osu, "_preflight_paths", lambda *a, **k: None)
    monkeypatch.setattr(
        osu,
        "_default_run",
        lambda argv, **k: _Proc(0, stdout=f"{pw.pw_uid}\n"),
    )
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 9001

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "work space; meta"
    workspace.mkdir()
    pid = kb._default_spawn(_make_task(kb, assignee="dev"), str(workspace))
    assert pid == 9001
    assert is_sudo_wrapped(captured["cmd"])
    assert captured["cmd"][5] == "hermes-dev"
    assert (
        captured["env"]["HOME"].endswith("hermes-dev")
        or "hermes-dev" in captured["env"]["HOME"]
    )
    assert (
        "SSH_AUTH_SOCK" not in captured["env"]
        or captured["env"].get("SSH_AUTH_SOCK") is None
    )


def test_default_spawn_restart_safe_wraps_outside_sudo(monkeypatch, tmp_path):
    """systemd-run must wrap sudo so the scope stays on the gateway UID.

    Rebase conflict in `_default_spawn` put both wraps at the same site.
    Inverting the order would put systemd-run inside sudoers (which only
    allows hermes) and drop the scope onto the mapped UID.
    """
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "dev"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text(
        "kanban:\n  profile_os_users:\n    dev: hermes-dev\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_cli import kanban_db as kb
    import hermes_cli.kanban_os_users as osu

    pw = _pw(home=str(tmp_path / "hermes-dev"))
    mapped_home = tmp_path / "hermes-dev" / ".hermes" / "profiles" / "dev"
    mapped_home.mkdir(parents=True)
    os.chmod(mapped_home, 0o700)
    os.chmod(mapped_home.parent, 0o700)
    os.chmod(mapped_home.parent.parent, 0o700)

    monkeypatch.setattr(osu, "_default_getpwnam", lambda name: pw)
    monkeypatch.setattr(osu, "_preflight_paths", lambda *a, **k: None)
    monkeypatch.setattr(
        osu,
        "_default_run",
        lambda argv, **k: _Proc(0, stdout=f"{pw.pw_uid}\n"),
    )
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    seen = {}

    def fake_restart_safe(task, command):
        seen["inner"] = list(command)
        return ["systemd-run", "--scope", "--", *command]

    monkeypatch.setattr(kb, "_restart_safe_worker_argv", fake_restart_safe)
    captured = {}

    class FakeProc:
        pid = 9002

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    pid = kb._default_spawn(_make_task(kb, assignee="dev"), str(workspace))
    assert pid == 9002
    assert is_sudo_wrapped(seen["inner"])
    assert seen["inner"][5] == "hermes-dev"
    assert captured["cmd"][:3] == ["systemd-run", "--scope", "--"]
    assert is_sudo_wrapped(captured["cmd"][3:])
    assert captured["cmd"][0] != "sudo"
    assert captured["cmd"][8] == "hermes-dev"


def test_cross_profile_homes_are_distinct_and_private(tmp_path):
    dev_home = tmp_path / "hermes-dev" / ".hermes" / "profiles" / "dev"
    sys_home = tmp_path / "hermes-sysadmin" / ".hermes" / "profiles" / "sysadmin"
    gw_ssh = tmp_path / "matt" / ".ssh" / "id_ed25519"
    for p in (dev_home, sys_home, gw_ssh.parent):
        p.mkdir(parents=True)
    dev_secret = dev_home / ".env"
    sys_secret = sys_home / ".env"
    dev_secret.write_text("DEV_TOKEN=aaa\n", encoding="utf-8")
    sys_secret.write_text("SYS_TOKEN=bbb\n", encoding="utf-8")
    gw_ssh.write_text("ssh-secret\n", encoding="utf-8")
    for p in (dev_home, sys_home, dev_secret, sys_secret, gw_ssh):
        os.chmod(p, 0o600 if p.is_file() else 0o700)
    for p in (dev_home, sys_home, gw_ssh):
        mode = stat.S_IMODE(p.stat().st_mode)
        assert mode & 0o007 == 0
    assert os.path.realpath(dev_home) != os.path.realpath(sys_home)
    # Shared board sibling WAL files live next to the db, not inside a profile home.
    board_dir = tmp_path / "shared-kanban"
    board_dir.mkdir()
    db = board_dir / "kanban.db"
    db.write_bytes(b"")
    wal = board_dir / "kanban.db-wal"
    wal.write_bytes(b"wal")
    os.chmod(board_dir, 0o770)
    assert db.exists() and wal.exists()
    assert not str(dev_home).startswith(str(sys_home))


def test_shared_board_lifecycle_paths_are_group_not_world(tmp_path):
    board_dir = tmp_path / "kanban"
    board_dir.mkdir()
    os.chmod(board_dir, 0o2770)
    mode = stat.S_IMODE(board_dir.stat().st_mode)
    assert mode & 0o007 == 0
    assert mode & 0o070 == 0o070
