"""Tests for `hermes skills nacos` subcommands."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from hermes_cli.skills_nacos_cmd import (
    _bump_version,
    _dir_hash,
    _zip_skill_dir,
    cmd_doctor,
    cmd_list,
    cmd_pull,
    cmd_push,
    cmd_sync,
)
from tools.nacos_cli_client import NacosSkillEntry


def _mk_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, width=200, force_terminal=False), buf


# ------------------------------------------------------------------ doctor

def test_doctor_installed_with_server(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / ".nacos-cli.conf").write_text("x")
    client = MagicMock()
    client.check_installed.return_value = True
    client.version.return_value = "3.2.1"
    console, buf = _mk_console()
    rc = cmd_doctor(client=client, server_addr="http://nacos:8848", console=console)
    out = buf.getvalue()
    assert rc == 0
    assert "installed" in out
    assert "3.2.1" in out
    assert "http://nacos:8848" in out


def test_doctor_missing_cli(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    client = MagicMock()
    client.check_installed.return_value = False
    console, buf = _mk_console()
    rc = cmd_doctor(client=client, server_addr=None, console=console)
    assert rc == 1
    out = buf.getvalue()
    assert "not installed" in out
    assert "@nacos-group/cli" in out


# ------------------------------------------------------------------ list

def test_list_prints_table():
    client = MagicMock()
    client.list_skills.return_value = [
        NacosSkillEntry(
            "code-review", "public", "hermes-skills", "1.0.0",
            "Review code", "alice", "t", "sha256:a",
        ),
    ]
    console, buf = _mk_console()
    rc = cmd_list(client=client, namespace="public", group="hermes-skills",
                  query=None, console=console)
    assert rc == 0
    out = buf.getvalue()
    assert "code-review" in out
    assert "1.0.0" in out


def test_list_empty():
    client = MagicMock()
    client.list_skills.return_value = []
    console, buf = _mk_console()
    rc = cmd_list(client=client, namespace="x", group="y", console=console)
    assert rc == 0
    assert "no skills" in buf.getvalue().lower()


def test_list_cli_error():
    from tools.nacos_cli_client import NacosCliError
    client = MagicMock()
    client.list_skills.side_effect = NacosCliError("boom")
    console, buf = _mk_console()
    rc = cmd_list(client=client, console=console)
    assert rc == 1


# ------------------------------------------------------------------ push helpers

def test_bump_version_patch(tmp_path):
    md = tmp_path / "SKILL.md"
    md.write_text("---\nname: x\nversion: 1.2.3\n---\n")
    assert _bump_version(md, "patch") == "1.2.4"
    assert "version: 1.2.4" in md.read_text()


def test_bump_version_minor(tmp_path):
    md = tmp_path / "SKILL.md"
    md.write_text("---\nname: x\nversion: 1.2.3\n---\n")
    assert _bump_version(md, "minor") == "1.3.0"


def test_bump_version_major(tmp_path):
    md = tmp_path / "SKILL.md"
    md.write_text("---\nname: x\nversion: 1.2.3\n---\n")
    assert _bump_version(md, "major") == "2.0.0"


def test_bump_version_no_semver_raises(tmp_path):
    md = tmp_path / "SKILL.md"
    md.write_text("---\nname: x\n---\n")
    with pytest.raises(ValueError):
        _bump_version(md, "patch")


def test_zip_skill_dir_excludes_git_and_pycache(tmp_path):
    src = tmp_path / "skill"
    (src / ".git").mkdir(parents=True)
    (src / ".git" / "HEAD").write_text("ref")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "x.pyc").write_text("bin")
    (src / "SKILL.md").write_text("ok")
    zp = tmp_path / "out.zip"
    _zip_skill_dir(src, zp)
    with zipfile.ZipFile(zp) as zf:
        names = zf.namelist()
    assert "SKILL.md" in names
    assert not any(".git" in n for n in names)
    assert not any("__pycache__" in n for n in names)


# ------------------------------------------------------------------ push command

def test_push_uploads_zip(tmp_path, monkeypatch):
    hermes = tmp_path / "hermes"
    hermes.mkdir()
    monkeypatch.setattr(
        "hermes_cli.skills_nacos_cmd.get_hermes_home", lambda: hermes
    )
    skill_dir = hermes / "skills" / "my-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: my-skill\nversion: 1.0.0\n---\n")

    client = MagicMock()
    client.upload_skill.return_value = {"uploaded": True, "version": "1.0.0"}
    args = SimpleNamespace(name="my-skill", namespace="team-a",
                            group="hermes-skills", bump=None)
    console, buf = _mk_console()
    rc = cmd_push(args, client_factory=lambda: client, console=console)
    assert rc == 0
    assert client.upload_skill.called
    assert "pushed" in buf.getvalue()


def test_push_bumps_version_before_upload(tmp_path, monkeypatch):
    hermes = tmp_path / "hermes"
    hermes.mkdir()
    monkeypatch.setattr(
        "hermes_cli.skills_nacos_cmd.get_hermes_home", lambda: hermes
    )
    skill_dir = hermes / "skills" / "my-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: my-skill\nversion: 1.2.3\n---\n")
    client = MagicMock()
    client.upload_skill.return_value = {"version": "1.3.0"}
    args = SimpleNamespace(name="my-skill", namespace=None, group=None, bump="minor")
    console, _ = _mk_console()
    cmd_push(args, client_factory=lambda: client, console=console)
    assert "version: 1.3.0" in (skill_dir / "SKILL.md").read_text()


def test_push_missing_skill(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.skills_nacos_cmd.get_hermes_home", lambda: tmp_path
    )
    args = SimpleNamespace(name="does-not-exist", namespace=None, group=None, bump=None)
    console, _ = _mk_console()
    rc = cmd_push(args, client_factory=lambda: MagicMock(), console=console)
    assert rc == 1


# ------------------------------------------------------------------ sync

def test_sync_success():
    client = MagicMock()
    client.sync_namespace.return_value = {"synced": ["a", "b"], "skipped": ["c"]}
    args = SimpleNamespace(namespace="public", group="hermes-skills")
    console, buf = _mk_console()
    rc = cmd_sync(args, client_factory=lambda: client, console=console)
    assert rc == 0
    out = buf.getvalue()
    assert "synced" in out
    assert "a, b" in out
    assert "c" in out


def test_sync_error():
    from tools.nacos_cli_client import NacosCliError
    client = MagicMock()
    client.sync_namespace.side_effect = NacosCliError("boom")
    args = SimpleNamespace(namespace="p", group="g")
    console, _ = _mk_console()
    rc = cmd_sync(args, client_factory=lambda: client, console=console)
    assert rc == 1


# ------------------------------------------------------------------ pull (conflict matrix)

def _lock_entry(content_hash: str = "sha256:orig"):
    return {
        "source": "nacos",
        "identifier": "nacos://public/hermes-skills/code-review",
        "trust_level": "community",
        "content_hash": content_hash,
    }


def test_pull_rejects_local_modifications_without_force(tmp_path, monkeypatch):
    # Arrange: a fake installed skill whose on-disk hash differs from the lock
    hermes = tmp_path / "hermes"
    skills_dir = hermes / "skills"
    target = skills_dir / "code-review"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("modified")

    fake_lock = MagicMock()
    fake_lock.get_installed.return_value = _lock_entry("sha256:original")
    monkeypatch.setattr("hermes_cli.skills_nacos_cmd._dir_hash", lambda p: "sha256:different")

    import tools.skills_hub as sh
    monkeypatch.setattr(sh, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(sh, "HubLockFile", lambda: fake_lock)

    args = SimpleNamespace(name="code-review", namespace=None, group=None,
                            version=None, update=False, force=False)
    installer = MagicMock()
    console, buf = _mk_console()
    rc = cmd_pull(args, client_factory=lambda: MagicMock(),
                  console=console, installer=installer)
    assert rc == 2
    installer.assert_not_called()
    assert "local modifications" in buf.getvalue()


def test_pull_already_installed_without_update_skips(tmp_path, monkeypatch):
    hermes = tmp_path / "hermes"
    skills_dir = hermes / "skills"
    target = skills_dir / "code-review"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("ok")

    fake_lock = MagicMock()
    fake_lock.get_installed.return_value = _lock_entry("sha256:same")
    monkeypatch.setattr("hermes_cli.skills_nacos_cmd._dir_hash", lambda p: "sha256:same")
    import tools.skills_hub as sh
    monkeypatch.setattr(sh, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(sh, "HubLockFile", lambda: fake_lock)

    args = SimpleNamespace(name="code-review", namespace=None, group=None,
                            version=None, update=False, force=False)
    installer = MagicMock()
    console, buf = _mk_console()
    rc = cmd_pull(args, client_factory=lambda: MagicMock(),
                  console=console, installer=installer)
    assert rc == 0
    installer.assert_not_called()
    assert "already installed" in buf.getvalue()


def test_pull_delegates_to_installer_on_happy_path(tmp_path, monkeypatch):
    hermes = tmp_path / "hermes"
    skills_dir = hermes / "skills"
    # target does not exist → first install path
    fake_lock = MagicMock()
    fake_lock.get_installed.return_value = None
    import tools.skills_hub as sh
    monkeypatch.setattr(sh, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(sh, "HubLockFile", lambda: fake_lock)

    args = SimpleNamespace(name="code-review", namespace=None, group=None,
                            version="1.0.0", update=False, force=False)
    installer = MagicMock()
    console, _ = _mk_console()
    rc = cmd_pull(args, client_factory=lambda: MagicMock(),
                  console=console, installer=installer)
    assert rc == 0
    installer.assert_called_once()
    ident_arg = installer.call_args.args[0]
    assert ident_arg == "nacos://public/hermes-skills/code-review@1.0.0"


def test_pull_missing_checksum_requires_update_or_force(tmp_path, monkeypatch):
    """Lock entry without content_hash should not silently report 'already installed'."""
    hermes = tmp_path / "hermes"
    skills_dir = hermes / "skills"
    target = skills_dir / "code-review"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("ok")

    fake_lock = MagicMock()
    # content_hash field is missing entirely
    fake_lock.get_installed.return_value = {
        "source": "nacos",
        "identifier": "nacos://public/hermes-skills/code-review",
        "trust_level": "community",
    }
    import tools.skills_hub as sh
    monkeypatch.setattr(sh, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(sh, "HubLockFile", lambda: fake_lock)

    args = SimpleNamespace(name="code-review", namespace=None, group=None,
                            version=None, update=False, force=False)
    installer = MagicMock()
    console, buf = _mk_console()
    rc = cmd_pull(args, client_factory=lambda: MagicMock(),
                  console=console, installer=installer)
    assert rc == 0
    installer.assert_not_called()
    # Message should call out the missing checksum specifically
    assert "no checksum" in buf.getvalue().lower()


def test_pull_missing_checksum_with_update_proceeds(tmp_path, monkeypatch):
    """When --update is set, missing-checksum entries should still reinstall."""
    hermes = tmp_path / "hermes"
    skills_dir = hermes / "skills"
    target = skills_dir / "code-review"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("ok")

    fake_lock = MagicMock()
    fake_lock.get_installed.return_value = {
        "source": "nacos",
        "identifier": "nacos://public/hermes-skills/code-review",
        "trust_level": "community",
    }
    import tools.skills_hub as sh
    monkeypatch.setattr(sh, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(sh, "HubLockFile", lambda: fake_lock)

    args = SimpleNamespace(name="code-review", namespace=None, group=None,
                            version=None, update=True, force=False)
    installer = MagicMock()
    console, _ = _mk_console()
    rc = cmd_pull(args, client_factory=lambda: MagicMock(),
                  console=console, installer=installer)
    assert rc == 0
    installer.assert_called_once()


def test_pull_update_flag_forces_installer(tmp_path, monkeypatch):
    hermes = tmp_path / "hermes"
    skills_dir = hermes / "skills"
    target = skills_dir / "code-review"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("ok")

    fake_lock = MagicMock()
    fake_lock.get_installed.return_value = _lock_entry("sha256:ok")
    monkeypatch.setattr("hermes_cli.skills_nacos_cmd._dir_hash", lambda p: "sha256:ok")
    import tools.skills_hub as sh
    monkeypatch.setattr(sh, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(sh, "HubLockFile", lambda: fake_lock)

    args = SimpleNamespace(name="code-review", namespace=None, group=None,
                            version=None, update=True, force=False)
    installer = MagicMock()
    console, _ = _mk_console()
    rc = cmd_pull(args, client_factory=lambda: MagicMock(),
                  console=console, installer=installer)
    assert rc == 0
    assert installer.call_args.kwargs.get("force") is True
    assert installer.call_args.kwargs.get("skip_confirm") is True


# ------------------------------------------------------------------ _dir_hash

def test_dir_hash_deterministic(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / "a.txt").write_text("a")
    (d / "b.txt").write_text("b")
    h1 = _dir_hash(d)
    # Reordering writes shouldn't change the hash because we sort
    h2 = _dir_hash(d)
    assert h1 == h2
    assert h1.startswith("sha256:")
