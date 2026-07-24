"""Regression tests for launchd-owned gateway status detection."""

from pathlib import Path
from types import SimpleNamespace

import hermes_cli.gateway as gateway_cli
from gateway import status


def test_launchd_status_falls_back_to_authoritative_print(tmp_path, monkeypatch, capsys):
    calls = []
    plist = tmp_path / "ai.hermes.gateway.plist"
    plist.write_text("plist", encoding="utf-8")

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[:2] == ["launchctl", "list"]:
            return SimpleNamespace(returncode=113, stdout="", stderr="not found")
        if command[:2] == ["launchctl", "print"]:
            return SimpleNamespace(
                returncode=0,
                stdout="state = running\n\tpid = 7957\nlast exit code = (never exited)\n",
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/502")
    monkeypatch.setattr(
        gateway_cli,
        "get_launchd_plist_path",
        lambda: plist,
    )
    monkeypatch.setattr(gateway_cli, "launchd_plist_is_current", lambda: True)
    monkeypatch.setattr(gateway_cli, "_launchd_unsupported_marker_exists", lambda: False)
    monkeypatch.setattr(status, "get_running_pid", lambda cleanup_stale=False: 7957)

    assert gateway_cli._probe_launchd_service_running() is True
    gateway_cli.launchd_status()

    output = capsys.readouterr().out
    assert calls == [
        ["launchctl", "list", "ai.hermes.gateway"],
        ["launchctl", "print", "gui/502/ai.hermes.gateway"],
        ["launchctl", "list", "ai.hermes.gateway"],
        ["launchctl", "print", "gui/502/ai.hermes.gateway"],
    ]
    assert "Gateway is supervised by launchd (PID 7957)" in output
    assert "Gateway service is not loaded" not in output
    assert "detached gateway process" not in output
