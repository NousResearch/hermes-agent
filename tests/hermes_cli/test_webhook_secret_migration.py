"""Webhook secret migration contract tests."""
import json

import pytest

from hermes_cli.migrations.webhook_secret_refs import (
    WebhookSecretMigrationError,
    migrate_webhook_routes,
)


def test_failure_before_secure_persistence_leaves_source_untouched(tmp_path):
    source = tmp_path / "routes.json"
    original = json.dumps({"alerts": {"secret": "sentinel-route-secret", "prompt": "ok"}}, indent=2)
    source.write_text(original, encoding="utf-8")

    def fail_store(_ref, _value):
        raise OSError("backend unavailable")

    with pytest.raises(WebhookSecretMigrationError, match="source left untouched"):
        migrate_webhook_routes(source, store=fail_store)
    assert source.read_text(encoding="utf-8") == original


def test_success_verifies_then_switches_and_scrubs_backup(tmp_path):
    source = tmp_path / "routes.json"
    backup = tmp_path / "routes.json.bak"
    payload = {"alerts": {"secret": "sentinel-route-secret", "prompt": "ok"}}
    source.write_text(json.dumps(payload), encoding="utf-8")
    backup.write_text(json.dumps(payload), encoding="utf-8")
    stored = {}

    def store(ref, value):
        stored[ref] = value

    result = migrate_webhook_routes(
        source,
        store=store,
        resolve=lambda ref: stored.get(ref),
        backup_paths=(backup,),
    )

    migrated = json.loads(source.read_text(encoding="utf-8"))
    scrubbed = json.loads(backup.read_text(encoding="utf-8"))
    assert migrated["alerts"]["secret_ref"] == "WEBHOOK_ROUTE_ALERTS"
    assert "secret" not in migrated["alerts"]
    assert "sentinel-route-secret" not in source.read_text(encoding="utf-8")
    assert "secret" not in scrubbed["alerts"]
    assert "sentinel-route-secret" not in backup.read_text(encoding="utf-8")
    assert result["receipts"][0]["stored"] is True
    assert result["receipts"][0]["verified"] is True
    assert result["rollback"]["source_preserved_on_pre_switch_failure"] is True
