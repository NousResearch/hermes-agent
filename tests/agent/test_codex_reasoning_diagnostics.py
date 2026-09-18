"""Issuer mismatch diagnostics must not expose stored or supplied metadata."""
import logging

import pytest

from agent import codex_responses_adapter as adapter


@pytest.mark.parametrize("foreign_model", [False, True])
def test_reasoning_replay_preserves_routing_without_logging_values(monkeypatch, caplog, foreign_model):
    monkeypatch.setattr(adapter, "_CROSS_ISSUER_WARN_EMITTED", False)
    current_model = "active-model-private-marker"
    stored_model = "stored-model-private-marker" if foreign_model else current_model
    encrypted = "opaque-test-ciphertext"
    message = {"codex_reasoning_items": [{
        "id": "reasoning-item", "type": "reasoning", "encrypted_content": encrypted,
        "_issuer_model": stored_model,
    }]}
    seen = set()
    with caplog.at_level(logging.WARNING, logger=adapter.logger.name):
        result = adapter._replay_reasoning_items(
            message, seen_item_ids=seen, current_issuer_kind=None,
            current_issuer_model=current_model, native_compaction_eligible=False,
        )
    if foreign_model:
        assert result == [] and seen == set()
        assert "Dropping reasoning item" in caplog.text
    else:
        assert result == [{"type": "reasoning", "encrypted_content": encrypted}]
        assert seen == {"reasoning-item"} and not caplog.records
    for value in (current_model, stored_model, encrypted):
        assert value not in caplog.text
