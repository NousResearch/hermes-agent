"""``hermes desktop --install-apparmor-profile`` outside Linux (#130177 review)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import main_desktop_apparmor

pytestmark = pytest.mark.platforms("not linux")


def test_install_apparmor_profile_refuses_before_any_warning_or_sudo(tmp_path, monkeypatch, capsys):
    """No security warning and no raw "apparmor_parser not found": one Linux-only line."""
    calls: list[list[str]] = []
    monkeypatch.setattr(main_desktop_apparmor, "_run", lambda command, stdin_text=None: calls.append(command))

    assert main_desktop_apparmor.install_profile(Path(tmp_path / "Hermes")) is False
    assert calls == []
    assert capsys.readouterr().out.strip() == (
        "✗ --install-apparmor-profile is Linux only: AppArmor profiles do not exist on this OS.")
