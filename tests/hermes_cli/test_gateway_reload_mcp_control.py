from types import SimpleNamespace
from pathlib import Path

import pytest

import hermes_cli.gateway as gateway


def test_gateway_reload_mcp_requests_local_reload(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(gateway, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(gateway, "host_multiplexer_serving", lambda: None)
    monkeypatch.setattr(
        "gateway.control_socket.reload_gateway_mcp",
        lambda home: {"reloading": True, "pid": 123},
    )

    gateway._cmd_reload_mcp(SimpleNamespace())

    assert capsys.readouterr().out == "mcp reload started on gateway pid 123\n"


@pytest.mark.parametrize("host_profile", ["default", "host"])
@pytest.mark.parametrize("standalone", [False, True])
def test_gateway_reload_mcp_uses_live_host_unless_standalone(
    monkeypatch, tmp_path, capsys, host_profile, standalone,
):
    root = tmp_path / ".hermes"
    active_profile = root / "profiles" / "worker"
    active_profile.mkdir(parents=True)
    (active_profile / "config.yaml").write_text(
        f"gateway:\n  standalone: {str(standalone).lower()}\n", encoding="utf-8",
    )
    host_home = root if host_profile == "default" else root / "profiles" / host_profile
    seen = []
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(active_profile))
    monkeypatch.setattr(gateway, "get_hermes_home", lambda: active_profile)
    monkeypatch.setattr(
        gateway, "host_multiplexer_serving", lambda: SimpleNamespace(home=host_home),
    )
    monkeypatch.setattr(
        "gateway.control_socket.reload_gateway_mcp",
        lambda home: seen.append(home) or {"reloading": True},
    )

    gateway._cmd_reload_mcp(SimpleNamespace())

    assert seen == [active_profile if standalone else host_home]
    assert capsys.readouterr().out == "mcp reload started\n"
