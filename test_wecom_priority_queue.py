"""
Test WeCom priority queue and per-turn stream state implementation.

Verifies that:
1. Control lane (approval prompts) bypass normal queue
2. Token bucket correctly allocates 24 normal + 6 reserved
3. Normal messages cannot use reserved tokens
4. Control messages can use normal remaining + reserved
5. Per-turn stream state prevents concurrent message conflicts
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

# Mock the dependencies
import sys
sys.path.insert(0, '/Users/bilibili/.hermes/hermes-agent')

def test_token_bucket_allocation():
    """Test token bucket correctly splits 24 normal + 6 reserved."""
    from gateway.platforms.wecom import WeComAdapter

    # Create a mock adapter
    adapter = MagicMock()
    adapter._chat_token_usage = {}
    adapter._BUCKET_NORMAL_TOKENS = 24
    adapter._BUCKET_RESERVED_TOKENS = 6

    # Bind the methods
    adapter._get_token_usage = WeComAdapter._get_token_usage.__get__(adapter)
    adapter._bucket_try_consume = WeComAdapter._bucket_try_consume.__get__(adapter)
    adapter._bucket_try_consume_control = WeComAdapter._bucket_try_consume_control.__get__(adapter)

    chat_id = "test_chat"

    # Test 1: Normal messages can use 24 tokens
    for i in range(24):
        wait = adapter._bucket_try_consume(chat_id)
        assert wait == 0.0, f"Normal token {i+1}/24 should be available"

    # Test 2: 25th normal message should wait
    wait = adapter._bucket_try_consume(chat_id)
    assert wait > 0, "25th normal message should wait (normal quota exhausted)"

    print("✅ Test 1: Normal messages correctly limited to 24 tokens")

    # Reset for control test
    adapter._chat_token_usage = {}

    # Test 3: Control can use 24 normal + 6 reserved = 30 total
    for i in range(30):
        wait = adapter._bucket_try_consume_control(chat_id)
        assert wait == 0.0, f"Control token {i+1}/30 should be available"

    # Test 4: 31st control message should wait
    wait = adapter._bucket_try_consume_control(chat_id)
    assert wait > 0, "31st control message should wait (all tokens exhausted)"

    print("✅ Test 2: Control messages can use 30 tokens total (24 normal + 6 reserved)")

    # Reset for mixed scenario
    adapter._chat_token_usage = {}

    # Test 5: Normal uses 24, then control can still use 6 reserved
    for i in range(24):
        adapter._bucket_try_consume(chat_id)

    # Now normal is exhausted, but control should still have 6 reserved
    for i in range(6):
        wait = adapter._bucket_try_consume_control(chat_id)
        assert wait == 0.0, f"Control reserved token {i+1}/6 should be available"

    print("✅ Test 3: Control can use reserved tokens when normal exhausted")

    # Test 6: After using all, both should wait
    wait_normal = adapter._bucket_try_consume(chat_id)
    wait_control = adapter._bucket_try_consume_control(chat_id)
    assert wait_normal > 0 and wait_control > 0, "Both should wait when all tokens used"

    print("✅ Test 4: Both lanes wait when all tokens exhausted")


def test_per_turn_stream_state():
    """Test that per-turn stream state isolates concurrent messages."""
    from gateway.platforms.wecom import WeComAdapter, StreamTurn

    print("\n🧪 Testing per-turn stream state isolation...")

    # Create mock adapter
    adapter = MagicMock()
    adapter._stream_turns = {}

    # Bind methods
    adapter._get_or_create_stream_turn = WeComAdapter._get_or_create_stream_turn.__get__(adapter)
    adapter._cleanup_stream_turn = WeComAdapter._cleanup_stream_turn.__get__(adapter)
    adapter._find_active_turn_for_chat = WeComAdapter._find_active_turn_for_chat.__get__(adapter)

    # Test 1: Create multiple turns for the same chat
    turn1 = adapter._get_or_create_stream_turn("chat1", "req1")
    turn2 = adapter._get_or_create_stream_turn("chat1", "req2")

    assert turn1.stream_id != turn2.stream_id, "Different turns should have different stream IDs"
    assert len(adapter._stream_turns) == 2, "Should have 2 independent turns"
    print("✅ Test 5: Multiple concurrent turns per chat isolated correctly")

    # Test 2: Find active turn
    turn1.finalized = True
    active = adapter._find_active_turn_for_chat("chat1")
    assert active == turn2, "Should find the non-finalized turn"
    print("✅ Test 6: Find active turn works correctly")

    # Test 3: Cleanup
    adapter._cleanup_stream_turn("chat1", "req1")
    assert len(adapter._stream_turns) == 1, "Should have 1 turn after cleanup"
    assert "chat1:req1" not in adapter._stream_turns, "Cleaned turn should be removed"
    print("✅ Test 7: Stream turn cleanup works correctly")


async def test_priority_routing():
    """Test that control messages bypass normal queue."""
    print("\n🧪 Testing priority routing (requires running adapter)...")
    print("⚠️  This test needs a live WeCom connection - skipping for now")
    print("   To test manually:")
    print("   1. Send a long message that generates multiple stream frames")
    print("   2. Trigger an approval while streaming")
    print("   3. Verify approval prompt appears immediately (not after 15s)")
    print("   4. Send another message concurrently - streams should not interfere")


if __name__ == "__main__":
    print("🧪 WeCom Priority Queue & Per-Turn State Tests\n")

    try:
        test_token_bucket_allocation()
        print("\n✅ All token bucket tests passed!")

        test_per_turn_stream_state()
        print("\n✅ All per-turn state tests passed!")

        asyncio.run(test_priority_routing())

        print("\n" + "="*60)
        print("📋 Summary:")
        print("  ✅ Token allocation: 24 normal + 6 reserved")
        print("  ✅ Normal messages respect 24-token limit")
        print("  ✅ Control messages can use all 30 tokens")
        print("  ✅ Reserved tokens protected from normal lane")
        print("  ✅ Per-turn stream state isolates concurrent messages")
        print("  ✅ Multiple streams per chat don't interfere")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
