"""The dashboard auth gate refuses a gated bind with no provider as EX_CONFIG (#142281)."""

import pytest

from hermes_cli import dashboard_auth, web_server


def test_a_gated_bind_without_a_provider_exits_ex_config(monkeypatch, capsys):
    monkeypatch.setattr(web_server, "_dashboard_public_hosts", lambda: frozenset())
    monkeypatch.setattr(web_server, "should_require_dashboard_auth", lambda host, hosts: True)
    monkeypatch.setattr(dashboard_auth, "list_providers", list)
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.delenv("HERMES_DASHBOARD_SESSION_TOKEN", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        web_server._configure_auth_gate("127.0.0.1", False, None, None)

    assert excinfo.value.code == 78
    assert "Refusing to bind dashboard to 127.0.0.1" in capsys.readouterr().err
