"""End-to-end secret egress gates for Webhook Revolution Task 8."""

import json

from hermes_cli.migrations.webhook_secret_refs import migrate_webhook_routes


SENTINEL = "WR_SENTINEL_WEBHOOK_SECRET_7f39d8"


def test_real_sentinel_is_removed_from_routes_backups_and_receipts(tmp_path):
    source = tmp_path / "webhook_subscriptions.json"
    backup = tmp_path / "webhook_subscriptions.json.bak"
    value = {"alerts": {"secret": SENTINEL, "prompt": "ok"}}
    source.write_text(json.dumps(value), encoding="utf-8")
    backup.write_text(json.dumps(value), encoding="utf-8")
    secret_backend = {}

    result = migrate_webhook_routes(
        source,
        store=lambda ref, secret: secret_backend.__setitem__(ref, secret),
        resolve=secret_backend.get,
        backup_paths=(backup,),
    )

    assert secret_backend["WEBHOOK_ROUTE_ALERTS"] == SENTINEL
    assert SENTINEL not in source.read_text(encoding="utf-8")
    assert SENTINEL not in backup.read_text(encoding="utf-8")
    assert SENTINEL not in json.dumps(result)
    assert json.loads(source.read_text())["alerts"]["secret_ref"] == "WEBHOOK_ROUTE_ALERTS"


def test_pre_switch_failure_preserves_exact_plaintext_source_for_retry(tmp_path):
    source = tmp_path / "webhook_subscriptions.json"
    original = json.dumps({"alerts": {"secret": SENTINEL}}, indent=2)
    source.write_text(original, encoding="utf-8")

    try:
        migrate_webhook_routes(
            source,
            store=lambda _ref, _value: (_ for _ in ()).throw(OSError("offline")),
        )
    except Exception:
        pass

    assert source.read_text(encoding="utf-8") == original
