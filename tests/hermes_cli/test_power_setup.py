from __future__ import annotations

import platform
import stat
import subprocess
from pathlib import Path

from hermes_cli import power_setup


def test_non_macos_does_not_attempt_install(monkeypatch, capsys):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert power_setup.run_power_setup() == 2
    assert "only available on macOS" in capsys.readouterr().out


def test_existing_hermes_rule_does_not_prompt(monkeypatch, capsys):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        power_setup.os,
        "stat",
        lambda *_args, **_kwargs: type(
            "Metadata", (), {"st_mode": stat.S_IFREG | 0o440, "st_uid": 0}
        )(),
    )
    monkeypatch.setattr(power_setup, "_verify_rule", lambda: True)
    monkeypatch.setattr(
        power_setup,
        "_run_native_install",
        lambda _source: (_ for _ in ()).throw(AssertionError),
    )

    assert power_setup.run_power_setup() == 0
    assert "already installed" in capsys.readouterr().out


def test_verify_rule_requires_both_exact_commands(monkeypatch):
    result = subprocess.CompletedProcess(
        ["sudo"],
        0,
        stdout="NOPASSWD: /usr/bin/pmset -a disablesleep 1\n/usr/bin/pmset -a disablesleep 0\n",
        stderr="",
    )
    monkeypatch.setattr(power_setup.subprocess, "run", lambda *args, **kwargs: result)
    assert power_setup._verify_rule() is True


def test_native_install_uses_os_auth_and_visudo(tmp_path, monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(power_setup.subprocess, "run", fake_run)
    source = Path(tmp_path) / "hermes-power-protect"
    result = power_setup._run_native_install(source)

    assert result.returncode == 0
    assert calls[0][0][:2] == ["/usr/bin/osascript", "-e"]
    script = calls[0][0][2]
    assert "/usr/sbin/visudo -c -f" in script
    assert "/usr/bin/install -o root -g wheel -m 0440" in script
    assert "/etc/sudoers.d/hermes-power-protect" in script
