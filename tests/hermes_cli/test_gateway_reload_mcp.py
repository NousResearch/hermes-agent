from types import SimpleNamespace

import hermes_cli.gateway as gateway


def test_gateway_reload_mcp_requests_local_reload(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(gateway, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        "gateway.control_socket.reload_gateway_mcp",
        lambda home: {"reloading": True, "pid": 123},
    )

    gateway._cmd_reload_mcp(SimpleNamespace())

    assert capsys.readouterr().out == "mcp reload started on gateway pid 123\n"


def test_gateway_reload_mcp_uses_default_root_for_live_multiplexer(monkeypatch, tmp_path, capsys):
    root = tmp_path / "hermes"
    active_profile = root / "profiles" / "worker"
    seen = []
    monkeypatch.setattr(gateway, "get_hermes_home", lambda: active_profile)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.setattr(
        "hermes_cli.gateway_multiplex_mode.default_gateway_multiplexes",
        lambda default_home=None: default_home == root,
    )
    monkeypatch.setattr(
        "gateway.control_socket.reload_gateway_mcp",
        lambda home: seen.append(home) or {"reloading": True},
    )

    gateway._cmd_reload_mcp(SimpleNamespace())

    assert seen == [root]
    assert capsys.readouterr().out == "mcp reload started\n"
