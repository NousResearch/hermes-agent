"""Tests for the cron alert novelty ledger.

The ledger is intentionally hard state, not prompt-only guidance: repeated
cron alert items must be suppressed before delivery unless an item is new or
materially changed.
"""

from __future__ import annotations

import json

from cron.alert_novelty import AlertNoveltyLedger, apply_alert_novelty_gate
from cron.scheduler import SILENT_MARKER


def test_new_items_pass_and_are_persisted(tmp_path):
    ledger = AlertNoveltyLedger(tmp_path / "ledger.json")
    text = "Habs trade update: https://example.com/news?id=1&utm_source=x"

    decision = ledger.evaluate("job-1", text)

    assert decision.should_deliver is True
    assert decision.reason == "new"
    assert len(decision.items) == 1
    assert decision.items[0].key == "url:https://example.com/news?id=1"

    ledger.commit(decision)
    raw = json.loads((tmp_path / "ledger.json").read_text())
    record = raw["jobs"]["job-1"]["items"]["url:https://example.com/news?id=1"]
    assert record["first_seen"]
    assert record["last_reported"]
    assert record["content_hash"] == decision.items[0].content_hash


def test_unchanged_repeat_is_silent_after_first_report(tmp_path):
    ledger = AlertNoveltyLedger(tmp_path / "ledger.json")
    text = "Habs trade update: https://example.com/news?id=1&utm_source=x"

    first = ledger.evaluate("job-1", text)
    ledger.commit(first)

    repeat = ledger.evaluate("job-1", text)

    assert repeat.should_deliver is False
    assert repeat.reason == "unchanged"
    assert repeat.final_response == SILENT_MARKER


def test_tracking_params_do_not_create_new_url_keys(tmp_path):
    ledger = AlertNoveltyLedger(tmp_path / "ledger.json")
    ledger.commit(ledger.evaluate("job-1", "https://example.com/news?id=1&utm_source=x"))

    repeat = ledger.evaluate("job-1", "https://example.com/news?utm_medium=social&id=1")

    assert repeat.should_deliver is False
    assert repeat.items[0].key == "url:https://example.com/news?id=1"


def test_material_update_same_key_is_delivered_and_hash_updated(tmp_path):
    ledger = AlertNoveltyLedger(tmp_path / "ledger.json")
    first = "Habs trade update: https://example.com/news?id=1\nOld details"
    updated = "Habs trade update: https://example.com/news?id=1\nMaterial new details"

    ledger.commit(ledger.evaluate("job-1", first))
    decision = ledger.evaluate("job-1", updated)

    assert decision.should_deliver is True
    assert decision.reason == "material_update"
    assert decision.items[0].is_material_update is True

    ledger.commit(decision)
    repeat = ledger.evaluate("job-1", updated)
    assert repeat.should_deliver is False


def test_mixed_new_and_repeated_items_delivers_only_reportable_items(tmp_path):
    ledger = AlertNoveltyLedger(tmp_path / "ledger.json")
    old_item = "Old Habs item: https://example.com/news/old"
    new_item = "New Habs item: https://example.com/news/new"
    ledger.commit(ledger.evaluate("job-1", old_item))

    decision = ledger.evaluate("job-1", f"{old_item}\n{new_item}")

    assert decision.should_deliver is True
    assert decision.reason == "new"
    assert old_item not in decision.final_response
    assert new_item in decision.final_response


def test_mixed_material_update_and_repeated_items_delivers_only_update(tmp_path):
    ledger = AlertNoveltyLedger(tmp_path / "ledger.json")
    repeated = "Repeated item: https://example.com/news/repeat"
    old_version = "Tracked item: https://example.com/news/track\nOld details"
    updated = "Tracked item: https://example.com/news/track\nUpdated details"
    ledger.commit(ledger.evaluate("job-1", repeated))
    ledger.commit(ledger.evaluate("job-1", old_version))

    decision = ledger.evaluate("job-1", f"{repeated}\n{updated}")

    assert decision.should_deliver is True
    assert decision.reason == "material_update"
    assert repeated not in decision.final_response
    assert updated in decision.final_response


def test_apply_gate_silences_repeated_successful_alert_and_preserves_failures(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    job = {"id": "job-1", "novelty_ledger": True}
    text = "Alert: https://example.com/item/1"

    first = apply_alert_novelty_gate(job, True, text, ledger_path=ledger_path)
    assert first == text

    second = apply_alert_novelty_gate(job, True, text, ledger_path=ledger_path)
    assert second == SILENT_MARKER

    failed = apply_alert_novelty_gate(job, False, text, ledger_path=ledger_path)
    assert failed == text


def test_gate_is_opt_in_to_avoid_suppressing_reminders(tmp_path):
    job = {"id": "job-1"}
    text = "Stand up and stretch"

    assert apply_alert_novelty_gate(job, True, text, ledger_path=tmp_path / "ledger.json") == text
    assert apply_alert_novelty_gate(job, True, text, ledger_path=tmp_path / "ledger.json") == text
