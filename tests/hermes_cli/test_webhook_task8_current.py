"""Current-train regression contracts for webhook Task 8."""
from __future__ import annotations

import inspect
import threading
import time
from types import SimpleNamespace

from hermes_cli import webhook as webhook_cli
from hermes_cli import webhook_secrets


def _args(**overrides):
    values = {
        "name": "alerts",
        "secret": "",
        "events": "",
        "description": "",
        "prompt": "",
        "skills": "",
        "deliver": "log",
        "deliver_only": False,
        "script": "",
        "deliver_chat_id": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_update_of_legacy_no_secret_route_mints_reference(monkeypatch):
    saved = {}
    stored = []
    monkeypatch.setattr(webhook_cli, "_load_subscriptions", lambda: {"alerts": {"prompt": "old"}})
    monkeypatch.setattr(webhook_cli, "_save_subscriptions", lambda value: saved.update(value))
    monkeypatch.setattr(webhook_cli, "_store_route_secret", lambda name, value: stored.append((name, value)) or "WEBHOOK_ROUTE_ALERTS")
    monkeypatch.setattr(webhook_cli, "_get_webhook_base_url", lambda: "http://localhost:8644")

    webhook_cli._cmd_subscribe(_args())

    route = saved["alerts"]
    assert route["secret_ref"] == "WEBHOOK_ROUTE_ALERTS"
    assert "secret" not in route
    assert stored and stored[0][0] == "alerts"
    assert stored[0][1]


def test_campaign_watermark_is_not_shipped():
    assert "WEBHOOK_REVOLUTION_TASK8_MIGRATION_COMMAND_V1" not in inspect.getsource(webhook_cli)


def test_secret_writers_are_serialized(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    active = 0
    maximum = 0
    guard = threading.Lock()

    def fake_save(_key, _value):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.05)
        with guard:
            active -= 1

    monkeypatch.setattr("hermes_cli.config.save_env_value", fake_save)
    threads = [
        threading.Thread(
            target=webhook_secrets.store_webhook_secret,
            args=(f"WEBHOOK_ROUTE_{index}", f"secret-{index}"),
        )
        for index in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert maximum == 1
