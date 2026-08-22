from types import SimpleNamespace
from unittest.mock import patch

from gateway.slash_commands import _apply_status_extensions


def test_status_extensions_append_without_changing_canonical_dispatch_or_auth():
    event = SimpleNamespace(text="/status", source=SimpleNamespace(platform="telegram"))
    with patch("hermes_cli.plugins.invoke_hook") as invoke:
        invoke.return_value = ["Base status\n回報模式：brief"]
        result = _apply_status_extensions("Base status", event=event, gateway=object())

    assert result == "Base status\n回報模式：brief"
    assert event.text == "/status"
    invoke.assert_called_once()
    kwargs = invoke.call_args.kwargs
    assert kwargs["status"] == "Base status"
    assert kwargs["event"] is event


def test_status_extensions_ignore_non_string_or_empty_plugin_results():
    event = SimpleNamespace(text="/status", source=SimpleNamespace(platform="telegram"))
    with patch("hermes_cli.plugins.invoke_hook", return_value=[None, {}, "", "extended"]):
        assert _apply_status_extensions("base", event=event, gateway=object()) == "extended"
