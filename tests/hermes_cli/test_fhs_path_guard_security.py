"""Regression tests for root FHS PATH guard privilege-boundary safety."""

from pathlib import Path as RealPath

from hermes_cli import main as hermes_main


def test_fhs_path_guard_uses_passwd_root_home_not_inherited_home(monkeypatch, tmp_path):
    attacker_home = tmp_path / "attacker-home"
    root_home = tmp_path / "root-home"
    attacker_home.mkdir(mode=0o700)
    root_home.mkdir(mode=0o700)
    attacker_bashrc = attacker_home / ".bashrc"
    root_bashrc = root_home / ".bashrc"
    attacker_bashrc.write_text("echo attacker-controlled\n")
    root_bashrc.write_text("# root shell config\n")
    root_bashrc.chmod(0o600)
    fake_fhs_link = tmp_path / "usr-local-bin-hermes"
    fake_fhs_link.write_text("#!/bin/sh\n")

    monkeypatch.setattr(hermes_main.sys, "platform", "linux")
    monkeypatch.setattr(hermes_main.os, "geteuid", lambda: 0)
    monkeypatch.setenv("HOME", str(attacker_home))
    monkeypatch.setattr(hermes_main, "_root_home_dir", lambda: root_home)
    monkeypatch.setattr(
        hermes_main,
        "_is_safe_root_owned_path",
        lambda path, *, require_file: path in {root_home, root_bashrc},
    )
    monkeypatch.setattr(
        hermes_main,
        "Path",
        (
            lambda value: fake_fhs_link
            if value == "/usr/local/bin/hermes"
            else RealPath(value)
        ),
    )

    def fail_if_probe_is_reintroduced(*_args, **_kwargs):
        raise AssertionError("root PATH guard must not launch a shell probe")

    monkeypatch.setattr(hermes_main.subprocess, "run", fail_if_probe_is_reintroduced)

    hermes_main._ensure_fhs_path_guard()

    assert "Hermes Agent" in root_bashrc.read_text()
    assert attacker_bashrc.read_text() == "echo attacker-controlled\n"


def test_install_sh_root_fhs_guard_does_not_run_interactive_bash_probe():
    install_sh = RealPath("scripts/install.sh").read_text()

    assert "bash -i -c 'command -v hermes'" not in install_sh
    assert 'env -i HOME="$HOME"' not in install_sh
    assert 'root_home="$(get_root_home_dir)"' in install_sh
    assert 'for SHELL_CONFIG in "$root_home/.bashrc" "$root_home/.bash_profile"' in install_sh
