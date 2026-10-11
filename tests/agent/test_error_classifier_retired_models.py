"""A retired versioned SKU is a missing MODEL, not credit depletion.

Split out of ``tests/agent/test_error_classifier.py``: that file is past its line ratchet and may
only shrink, and new tests belong in a ``test_<stem>_<topic>`` sibling.

The Nous body names the model itself leaving the free tier and points at a sibling variant, so a
different model is the fix and the failure must reach the fallback chain. Distinct from
``test_404_free_tier_model_block_is_billing`` in the parent module, whose wording is the
account-level free-TIER wall.
"""

from agent.error_classifier import FailoverReason, classify_api_error
from tests.agent.test_error_classifier import MockAPIError


def test_404_retired_free_sku_is_model_not_found():
    """A retired versioned SKU is a missing MODEL, not credit depletion."""
    e = MockAPIError(
        "Not Found",
        status_code=404,
        body={
            "status": 404,
            "message": (
                "This model is no longer free. To continue using the paid variant, "
                "switch to 'meituan/longcat-2.0'."
            ),
        },
    )
    result = classify_api_error(e, provider="nous", model="meituan/longcat-2.0:free")
    assert result.reason == FailoverReason.model_not_found
    assert result.should_fallback is True
    assert result.retryable is False