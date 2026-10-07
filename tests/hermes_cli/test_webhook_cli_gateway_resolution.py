"""`hermes webhook` must see the webhook platform exactly as the gateway resolves it.

``hermes gateway setup`` and the dashboard enable webhooks by writing WEBHOOK_ENABLED /
WEBHOOK_PORT to .env; the gateway honours them, so the CLI gate and the URLs it prints must too.
"""

import os
from pathlib import Path

import pytest

import hermes_cli.webhook as wh
from gateway.config import Platform, load_gateway_config


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


@pytest.mark.parametrize("yaml", ["", "platforms:\n  webhook:\n    enabled: false\n"],
                         ids=["env-enabled", "yaml-explicit-disable"])
def test_cli_gate_and_url_follow_the_gateway_resolution(monkeypatch, yaml):
    if yaml:
        (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(yaml, encoding="utf-8")
    monkeypatch.setenv("WEBHOOK_ENABLED", "true")
    monkeypatch.setenv("WEBHOOK_PORT", "9123")

    gw = load_gateway_config().platforms.get(Platform.WEBHOOK)
    gateway_enabled = bool(gw and gw.enabled)

    assert wh._is_webhook_enabled() is gateway_enabled
    if gateway_enabled:
        assert wh._get_webhook_base_url().endswith(f":{gw.extra['port']}")
