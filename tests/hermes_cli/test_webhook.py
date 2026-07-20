"""Regression tests for loopback-safe webhook CLI guidance."""

from unittest.mock import patch

import hermes_cli.webhook as webhook


def test_base_url_defaults_to_loopback():
    with patch.object(webhook, "_get_webhook_config", return_value={"extra": {}}):
        assert webhook._get_webhook_base_url() == "http://127.0.0.1:8644"


def test_setup_hint_scaffolds_loopback_bind():
    with patch.object(webhook, "display_hermes_home", return_value="/tmp/hermes"):
        hint = webhook._setup_hint()

    assert 'host: "127.0.0.1"' in hint
    assert 'host: "0.0.0.0"' not in hint
