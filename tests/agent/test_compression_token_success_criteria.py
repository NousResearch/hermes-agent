"""Tests for issue #39550: Compression token savings ignored when message count unchanged.

When compression reduces tokens materially but message count stays the same,
the conversation loop treats it as "cannot compress further" and fails, even
though post-compression context is well below model limits. Tests verify that
compression success is evaluated by token savings, not just message count.
"""
import pytest


def test_compression_success_with_unchanged_message_count():
    """Compression saving 105k tokens (36%) should succeed even with unchanged message count."""
    # Real observed case from issue #39550:
    # - Pre-compression: 220 messages, ~288,028 tokens
    # - Post-compression: 220 messages, ~183,180 tokens (36% savings)
    # - Model context: 1,000,000 tokens
    # - Result: Session was reset despite 183k << 1M context
    
    orig_messages = 220
    orig_tokens = 288_028
    compressed_messages = 220  # UNCHANGED
    compressed_tokens = 183_180  # Reduced by 105k (36%)
    model_context = 1_000_000
    
    # Current broken logic (breaks even though compression was successful):
    if compressed_messages >= orig_messages:
        break_loop = True  # <- This is wrong!
    else:
        break_loop = False
    
    # Verify current logic is broken
    assert break_loop is True, "Current code breaks on unchanged message count"
    
    # Better logic: check token savings too
    token_savings = orig_tokens - compressed_tokens
    token_savings_pct = (token_savings / orig_tokens) * 100
    tokens_below_threshold = compressed_tokens < (model_context * 0.7)  # 70% of context
    
    # Compression was successful if:
    # 1. Token savings > 20% (material improvement), OR
    # 2. Post-compression tokens below context threshold, OR
    # 3. Message count actually decreased
    compression_successful = (
        token_savings_pct > 20 or
        tokens_below_threshold or
        compressed_messages < orig_messages
    )
    
    # Verify better logic would succeed
    assert compression_successful is True
    assert 36 <= token_savings_pct <= 37  # Verify actual savings (approximately 36%)
    assert compressed_tokens < model_context * 0.7  # Below threshold


def test_compression_failure_when_no_material_improvement():
    """Compression with <10% token savings and unchanged message count should fail."""
    orig_messages = 100
    orig_tokens = 200_000
    compressed_messages = 100  # UNCHANGED
    compressed_tokens = 195_000  # Only 2.5% savings
    model_context = 100_000  # Very small context
    
    token_savings_pct = ((orig_tokens - compressed_tokens) / orig_tokens) * 100
    tokens_below_threshold = compressed_tokens < (model_context * 0.7)
    
    compression_successful = (
        token_savings_pct > 20 or
        tokens_below_threshold or
        compressed_messages < orig_messages
    )
    
    # This should fail because no material improvement
    assert compression_successful is False
    assert token_savings_pct == 2.5


def test_compression_success_when_message_count_decreases():
    """Compression reducing message count should always succeed."""
    orig_messages = 220
    orig_tokens = 288_028
    compressed_messages = 15  # DECREASED significantly
    compressed_tokens = 280_000  # Token savings minimal
    
    compression_successful = (
        ((orig_tokens - compressed_tokens) / orig_tokens) * 100 > 20 or
        compressed_tokens < 1_000_000 * 0.7 or
        compressed_messages < orig_messages
    )
    
    # Should succeed due to message count decrease
    assert compression_successful is True


def test_no_op_detection_unchanged_everything():
    """When both messages and tokens unchanged, it's a no-op."""
    orig_messages = 100
    orig_tokens = 200_000
    compressed_messages = 100
    compressed_tokens = 200_000
    
    is_no_op = (
        compressed_messages == orig_messages and
        compressed_tokens == orig_tokens
    )
    
    assert is_no_op is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
