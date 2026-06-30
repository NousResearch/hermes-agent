"""Regression tests for native signal-cli setup guidance."""

import shutil

from hermes_cli import gateway, web_server


def test_signal_setup_recommends_only_native_http_daemon(monkeypatch, capsys):
    monkeypatch.setattr(gateway, "get_env_value", lambda _key: "")
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    def cancel_setup(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", cancel_setup)

    gateway._setup_signal()

    output = capsys.readouterr().out
    assert "native signal-cli HTTP daemon" in output
    assert "signal-cli --account +YOURNUMBER daemon --http" in output
    assert "bbernhard/signal-cli-rest-api" not in output


def test_signal_dashboard_metadata_describes_native_http_daemon():
    signal = web_server._build_catalog_entry("signal")
    http_url = web_server._messaging_env_info("SIGNAL_HTTP_URL")
    account = web_server._messaging_env_info("SIGNAL_ACCOUNT")

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
