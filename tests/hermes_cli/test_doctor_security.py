"""Permission-audit coverage for ``hermes doctor``."""

import os
import stat

import hermes_cli.doctor as doctor
from hermes_cli import doctor_security


def _mode(path):
    return stat.S_IMODE(path.lstat().st_mode)


def _run(monkeypatch, home, should_fix=False):
    monkeypatch.setattr(doctor, "HERMES_HOME", home)
    return doctor_security._check_state_permissions(should_fix)


def test_state_permission_check_accepts_private_state_tree(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(mode=0o700)
    (home / "sessions").mkdir(mode=0o700)
    (home / "logs").mkdir(mode=0o700)
    for name in (".env", "config.yaml", "state.db"):
        path = home / name
        path.write_text("private", encoding="utf-8")
        path.chmod(0o600)

    finding = _run(monkeypatch, home)

    assert finding.issues == []
    assert finding.fixed == 0


def test_state_permission_check_reports_insecure_file_and_directory(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(mode=0o755)
    (home / "sessions").mkdir()
    (home / "sessions").chmod(0o777)
    config = home / "config.yaml"
    config.write_text("model: test", encoding="utf-8")
    config.chmod(0o664)

    finding = _run(monkeypatch, home)

    assert len(finding.issues) == 3
    assert any("config.yaml" in issue and "0664" in issue for issue in finding.issues)
    assert any("sessions" in issue and "0777" in issue for issue in finding.issues)


def test_state_permission_check_reports_symlink_without_following_it(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(mode=0o700)
    outside = tmp_path / "outside.env"
    outside.write_text("secret", encoding="utf-8")
    outside.chmod(0o666)
    (home / ".env").symlink_to(outside)

    finding = _run(monkeypatch, home, should_fix=True)

    assert any(".env is a symlink" in issue for issue in finding.issues)
    assert _mode(outside) == 0o666
    assert finding.fixed == 0


def test_state_permission_check_does_not_descend_into_a_symlinked_home(monkeypatch, tmp_path):
    outside_home = tmp_path / "outside-home"
    outside_home.mkdir(mode=0o700)
    outside_config = outside_home / "config.yaml"
    outside_config.write_text("model: test", encoding="utf-8")
    outside_config.chmod(0o666)
    home = tmp_path / ".hermes"
    home.symlink_to(outside_home, target_is_directory=True)

    finding = _run(monkeypatch, home, should_fix=True)

    assert len(finding.issues) == 1
    assert "is a symlink; refusing to inspect or modify its target" in finding.issues[0]
    assert _mode(outside_config) == 0o666


def test_state_permission_fix_tightens_read_only_and_insecure_paths(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(mode=0o755)
    (home / "logs").mkdir(mode=0o755)
    config = home / "config.yaml"
    config.write_text("model: test", encoding="utf-8")
    config.chmod(0o444)

    finding = _run(monkeypatch, home, should_fix=True)

    assert finding.issues == []
    assert finding.fixed == 3
    assert _mode(home) == 0o700
    assert _mode(home / "logs") == 0o700
    assert _mode(config) == 0o600


def test_state_permission_fix_keeps_finding_when_nonfollowing_chmod_fails(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(mode=0o700)
    config = home / "config.yaml"
    config.write_text("model: test", encoding="utf-8")
    config.chmod(0o644)

    monkeypatch.setattr(doctor_security.os, "fchmod", lambda *_args: (_ for _ in ()).throw(OSError("denied")))
    finding = _run(monkeypatch, home, should_fix=True)

    assert finding.fixed == 0
    assert any("could not tighten" in issue for issue in finding.issues)
    assert _mode(config) == 0o644
