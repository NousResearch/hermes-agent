import asyncio
from unittest.mock import patch


def test_on_slack_app_init_fires_before_socket_mode(slack_adapter_factory):
    """Hook fires after native handler registration and before Socket Mode."""
    captured = {}

    def hook(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        assert kwargs["adapter"]._handler is None
        return []

    adapter = slack_adapter_factory(profile_name="manager")

    with patch("hermes_cli.plugins.invoke_hook", side_effect=hook):
        result = asyncio.run(adapter.connect())

    assert result is True
    assert captured["name"] == "on_slack_app_init"
    assert captured["app"] is adapter._app
    assert captured["adapter"] is adapter
    assert captured["profile"] == "manager"
    assert captured["web_clients"] is adapter._team_clients
    assert captured["bot_user_id"] == adapter._bot_user_id
    assert adapter._handler is not None


def test_on_slack_app_init_failure_aborts_startup(slack_adapter_factory):
    """A hook invocation failure prevents Socket Mode startup."""

    adapter = slack_adapter_factory(profile_name="contents")

    with patch("hermes_cli.plugins.invoke_hook", side_effect=RuntimeError("plugin boom")):
        result = asyncio.run(adapter.connect())

    assert result is False
    assert adapter._handler is None
