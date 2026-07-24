from __future__ import annotations

import subprocess
from pathlib import Path

from hermes_cli import linux_desktop_integration as integration


def _completed(command: list[str], *, stdout: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")


def test_registers_local_build_as_user_protocol_handler(tmp_path, monkeypatch):
    executable = tmp_path / "checkout with spaces" / "linux-unpacked" / "Hermes"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    icon = tmp_path / "icon.png"
    icon.write_bytes(b"png")
    data_home = tmp_path / "xdg-data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.setattr(integration.sys, "platform", "linux")
    monkeypatch.setattr(
        integration.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )
    calls: list[list[str]] = []

    def run(command: list[str]):
        calls.append(command)
        if command[1:3] == ["query", "default"]:
            query_count = sum(call[1:3] == ["query", "default"] for call in calls)
            return _completed(
                command, stdout="hermes.desktop\n" if query_count > 1 else ""
            )
        return _completed(command)

    monkeypatch.setattr(integration, "_run", run)

    assert integration.register_linux_deep_link_protocol(executable, icon=icon) is True

    desktop_file = data_home / "applications" / "hermes.desktop"
    content = desktop_file.read_text(encoding="utf-8")
    assert f'Exec="{executable.resolve()}" %U' in content
    assert f"Icon={icon.resolve()}" in content
    assert "MimeType=x-scheme-handler/hermes;" in content
    assert calls == [
        ["/usr/bin/update-desktop-database", str(data_home / "applications")],
        ["/usr/bin/xdg-mime", "query", "default", "x-scheme-handler/hermes"],
        ["/usr/bin/xdg-mime", "default", "hermes.desktop", "x-scheme-handler/hermes"],
        ["/usr/bin/xdg-mime", "query", "default", "x-scheme-handler/hermes"],
    ]


def test_exec_path_escapes_desktop_entry_field_codes(tmp_path, monkeypatch):
    executable = tmp_path / 'cash$-quote"-percent%-tick`' / "Hermes"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    monkeypatch.setattr(integration.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(
        integration.shutil,
        "which",
        lambda name: "/usr/bin/xdg-mime" if name == "xdg-mime" else None,
    )
    monkeypatch.setattr(
        integration,
        "_run",
        lambda command: _completed(
            command, stdout="hermes.desktop\n" if "query" in command else ""
        ),
    )

    assert integration.register_linux_deep_link_protocol(executable) is True
    content = (tmp_path / "data" / "applications" / "hermes.desktop").read_text(
        encoding="utf-8"
    )
    exec_line = next(line for line in content.splitlines() if line.startswith("Exec="))
    assert "\\$" in exec_line
    assert '\\"' in exec_line
    assert "%%" in exec_line
    assert "\\`" in exec_line


def test_failed_xdg_registration_keeps_entry_and_prints_actionable_warning(
    tmp_path, monkeypatch, capsys
):
    executable = tmp_path / "Hermes"
    executable.write_text("", encoding="utf-8")
    data_home = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.setattr(integration.sys, "platform", "linux")
    monkeypatch.setattr(
        integration.shutil,
        "which",
        lambda name: "/usr/bin/xdg-mime" if name == "xdg-mime" else None,
    )
    monkeypatch.setattr(
        integration,
        "_run",
        lambda command: _completed(command, returncode=1),
    )

    assert integration.register_linux_deep_link_protocol(executable) is False
    assert (data_home / "applications" / "hermes.desktop").is_file()
    output = capsys.readouterr().out
    assert "Could not register hermes://" in output
    assert "xdg-mime default hermes.desktop x-scheme-handler/hermes" in output


def test_non_linux_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(integration.sys, "platform", "darwin")
    monkeypatch.setattr(
        integration, "_applications_dir", lambda: tmp_path / "applications"
    )

    assert integration.register_linux_deep_link_protocol(tmp_path / "missing") is True
    assert not (tmp_path / "applications").exists()
