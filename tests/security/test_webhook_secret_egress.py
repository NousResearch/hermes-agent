"""Secret egress regression tests for webhook route surfaces."""
import json

from hermes_cli.webhook import _load_subscriptions, _save_subscriptions


def test_route_json_and_loaded_route_never_contain_secret_after_migration(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    route_path = tmp_path / "webhook_subscriptions.json"
    route_path.write_text(json.dumps({"alerts": {"secret_ref": "WEBHOOK_ROUTE_ALERTS", "prompt": "ok"}}))
    loaded = json.dumps(_load_subscriptions())
    assert "sentinel-webhook-secret" not in loaded
    assert '"secret":' not in loaded
    assert '"secret_value":' not in loaded


def test_save_route_reference_does_not_write_plaintext(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _save_subscriptions({"alerts": {"secret_ref": "WEBHOOK_ROUTE_ALERTS", "prompt": "ok"}})
    text = (tmp_path / "webhook_subscriptions.json").read_text()
    assert "sentinel-webhook-secret" not in text
    assert "WEBHOOK_ROUTE_ALERTS" in text
