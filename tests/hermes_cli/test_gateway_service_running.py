"""A macOS LaunchAgent counts as running only while launchd supervises a process for it."""

import pytest

from hermes_cli import gateway


@pytest.mark.platforms("macos")
@pytest.mark.parametrize(("supervised_pid", "expected"), [(None, False), (4242, True)])
def test_launchd_service_running_requires_a_supervised_process(monkeypatch, tmp_path, supervised_pid, expected):
    plist = tmp_path / "ai.hermes.gateway.plist"
    plist.write_text("<plist/>")
    monkeypatch.setattr(gateway, "get_launchd_plist_path", lambda: plist)
    monkeypatch.setattr(gateway, "_launchctl_supervised_pid", lambda label: supervised_pid)

    assert gateway._is_service_running() is expected
