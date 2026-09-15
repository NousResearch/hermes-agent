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
