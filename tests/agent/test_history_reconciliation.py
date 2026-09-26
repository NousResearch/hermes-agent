"""Keep model-only payloads only while their addressed durable row agrees."""
import pytest

from agent.history_reconciliation import reconcile_model_history


def test_active_row_keeps_image_and_pending_tail():
    image = {"role": "tool", "tool_call_id": "t1", "_row_id": 4,
             "content": [{"type": "text", "text": "view"}, {"type": "image_url", "image_url": {"url": "fixture"}}]}
    row = {"role": "tool", "tool_call_id": "t1", "_row_id": 4, "content": "view\n[screenshot]"}
    pending = {"role": "user", "content": "unsaved input"}
    current = [image, pending]
    result = reconcile_model_history([row], current)
    assert result == current
    assert result[0] is image and result[1] is pending


@pytest.mark.parametrize("change", ["append", "rewrite", "compact"])
def test_row_identity_and_content_both_fence_enrichment(change):
    old = {"role": "assistant", "_row_id": 1, "content": "old"}
    if change == "append":
        fresh = [dict(old), {"role": "user", "_row_id": 2, "content": "external"}]
    elif change == "rewrite":
        fresh = [{**old, "content": "changed in place"}]
    else:
        fresh = [{**old, "_row_id": 3, "content": "compressed"}]
    result = reconcile_model_history(fresh, [old])
    assert result == fresh
    assert (result[0] is old) == (change == "append")


def test_api_projection_and_reasoning_are_compared():
    memory = {"role": "assistant", "_row_id": 2, "content": " actual ", "reasoning_content": "a"}
    row = {"role": "assistant", "_row_id": 2, "content": "actual", "api_content": " actual ", "reasoning_content": "a"}
    assert reconcile_model_history([row], [memory])[0] is memory
    row["reasoning_content"] = "changed"
    assert reconcile_model_history([row], [memory])[0] is row


def test_unbound_durable_markers_do_not_prove_row_identity():
    arbitrary = {"role": "user", "content": "caller-owned", "_db_persisted": True}
    with pytest.raises(ValueError, match="unaddressed"):
        reconcile_model_history([{"role": "user", "content": "DB", "_row_id": 1}], [arbitrary])
