"""Issuer mismatch diagnostics must not expose stored or supplied metadata."""
import logging

import pytest

from agent import codex_responses_adapter as adapter


@pytest.mark.parametrize("foreign_model", [False, True])
@pytest.mark.parametrize("foreign_issuer", [False, True])
def test_reasoning_replay_preserves_routing_without_logging_values(
    monkeypatch, caplog, foreign_model, foreign_issuer,
):
    monkeypatch.setattr(adapter, "_CROSS_ISSUER_WARN_EMITTED", False)
    current_model = "active-model-private-marker"
    stored_model = "stored-model-private-marker" if foreign_model else current_model
    current_issuer = "active-issuer-private-marker"
    stored_issuer = "stored-issuer-private-marker" if foreign_issuer else current_issuer
    encrypted = "opaque-test-ciphertext"
    message = {"codex_reasoning_items": [{
        "id": "reasoning-item", "type": "reasoning", "encrypted_content": encrypted,
        "_issuer_model": stored_model,
        "_issuer_kind": stored_issuer,
    }]}
    seen = set()
    with caplog.at_level(logging.WARNING, logger=adapter.logger.name):
        result = adapter._replay_reasoning_items(
            message, seen_item_ids=seen, current_issuer_kind=current_issuer,
            current_issuer_model=current_model, native_compaction_eligible=False,
        )
    if foreign_model or foreign_issuer:
        assert result == [] and seen == set()
        assert "Dropping reasoning item" in caplog.text
    else:
        assert result == [{"type": "reasoning", "encrypted_content": encrypted}]
        assert seen == {"reasoning-item"} and not caplog.records
    for value in (current_model, stored_model, current_issuer, stored_issuer, encrypted):
        assert value not in caplog.text
