"""Regression tests for native signal-cli setup guidance."""

import shutil

import pytest

from hermes_cli import gateway
from hermes_cli.config import OPTIONAL_ENV_VARS
from hermes_cli.web_routers.messaging import _messaging_env_info
from hermes_cli.web_server_messaging import _build_catalog_entry
from plugins.platforms.signal.setup import interactive_setup


def test_signal_setup_recommends_only_native_http_daemon(monkeypatch, capsys):
    monkeypatch.setattr(gateway, "get_env_value", lambda _key: "")
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    def cancel_setup(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", cancel_setup)

    interactive_setup()

    output = capsys.readouterr().out
    assert "native signal-cli HTTP daemon" in output
    assert "signal-cli --account +YOURNUMBER daemon --http" in output
    assert "bbernhard/signal-cli-rest-api" not in output


@pytest.mark.parametrize("use_fallback", [False, True], ids=["plugin", "fallback"])
def test_signal_dashboard_metadata_describes_native_http_daemon(monkeypatch, use_fallback):
    if use_fallback:
        monkeypatch.delitem(OPTIONAL_ENV_VARS, "SIGNAL_HTTP_URL")
        monkeypatch.delitem(OPTIONAL_ENV_VARS, "SIGNAL_ACCOUNT")

    signal = _build_catalog_entry("signal")
    http_url = _messaging_env_info("SIGNAL_HTTP_URL")
    account = _messaging_env_info("SIGNAL_ACCOUNT")

    assert signal["description"] == "Connect through the native signal-cli HTTP daemon."
    assert signal["docs_url"] == (
        "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/signal"
    )
    assert http_url["description"] == (
        "Native signal-cli HTTP daemon URL, e.g. http://127.0.0.1:8080"
    )
    assert http_url["prompt"] == "signal-cli HTTP URL"
    assert http_url["url"] == signal["docs_url"]
    assert "bridge" not in " ".join(
        (
            signal["description"],
            http_url["description"],
            http_url["prompt"],
            account["description"],
        )
    ).lower()
