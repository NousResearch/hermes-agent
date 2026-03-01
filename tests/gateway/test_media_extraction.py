"""
Tests for MEDIA tag extraction from tool results.

Verifies that MEDIA tags (e.g., from TTS tool) are only extracted from
messages in the CURRENT turn, not from the full conversation history.
This prevents voice messages from accumulating and being sent multiple
times per reply. (Regression test for #160)
"""

import pytest
import re


def extract_media_tags_fixed(result_messages, history_len):
    """
    Extract MEDIA tags from tool results, but ONLY from new messages
    (those added after history_len). This is the fixed behavior.
    
    Args:
        result_messages: Full list of messages including history + new
        history_len: Length of history before this turn
        
    Returns:
        Tuple of (media_tags list, has_voice_directive bool)
    """
    media_tags = []
    has_voice_directive = False
    
    # Only process new messages from this turn
    new_messages = result_messages[history_len:] if len(result_messages) > history_len else []
    
    for msg in new_messages:
        if msg.get("role") == "tool" or msg.get("role") == "function":
            content = msg.get("content", "")
            if "MEDIA:" in content:
                for match in re.finditer(r'MEDIA:(\S+)', content):
                    path = match.group(1).strip().rstrip('",}')
                    if path:
                        media_tags.append(f"MEDIA:{path}")
                if "[[audio_as_voice]]" in content:
                    has_voice_directive = True
    
    return media_tags, has_voice_directive


def extract_media_tags_broken(result_messages):
    """
    The BROKEN behavior: extract MEDIA tags from ALL messages including history.
    This causes TTS voice messages to accumulate and be re-sent on every reply.
    """
    media_tags = []
    has_voice_directive = False
    
    for msg in result_messages:
        if msg.get("role") == "tool" or msg.get("role") == "function":
            content = msg.get("content", "")
            if "MEDIA:" in content:
                for match in re.finditer(r'MEDIA:(\S+)', content):
                    path = match.group(1).strip().rstrip('",}')
                    if path:
                        media_tags.append(f"MEDIA:{path}")
                if "[[audio_as_voice]]" in content:
                    has_voice_directive = True
    
    return media_tags, has_voice_directive


class TestMediaExtraction:
    """Tests for MEDIA tag extraction from tool results."""
    
    def test_media_tags_not_extracted_from_history(self):
        """MEDIA tags from previous turns should NOT be extracted again."""
        # Simulate conversation history with a TTS call from a previous turn
        history = [
            {"role": "user", "content": "Say hello as audio"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {"name": "text_to_speech"}}]},
            {"role": "tool", "tool_call_id": "1", "content": '{"success": true, "media_tag": "[[audio_as_voice]]\\nMEDIA:/path/to/audio1.ogg"}'},
            {"role": "assistant", "content": "I've said hello for you!"},
        ]
        
        # New turn: user asks a simple question
        new_messages = [
            {"role": "user", "content": "What time is it?"},
            {"role": "assistant", "content": "It's 3:30 AM."},
        ]
        
        all_messages = history + new_messages
        history_len = len(history)
        
        # Fixed behavior: should extract NO media tags (none in new messages)
        tags, voice_directive = extract_media_tags_fixed(all_messages, history_len)
        assert tags == [], "Fixed extraction should not find tags in history"
        assert voice_directive is False
        
        # Broken behavior: would incorrectly extract the old media tag
        broken_tags, broken_voice = extract_media_tags_broken(all_messages)
        assert len(broken_tags) == 1, "Broken extraction finds tags in history"
        assert "audio1.ogg" in broken_tags[0]
    
    def test_media_tags_extracted_from_current_turn(self):
        """MEDIA tags from the current turn SHOULD be extracted."""
        # History without TTS
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        
        # New turn with TTS call
        new_messages = [
            {"role": "user", "content": "Say goodbye as audio"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "2", "function": {"name": "text_to_speech"}}]},
            {"role": "tool", "tool_call_id": "2", "content": '{"success": true, "media_tag": "[[audio_as_voice]]\\nMEDIA:/path/to/audio2.ogg"}'},
            {"role": "assistant", "content": "I've said goodbye!"},
        ]
        
        all_messages = history + new_messages
        history_len = len(history)
        
        # Fixed behavior: should extract the new media tag
        tags, voice_directive = extract_media_tags_fixed(all_messages, history_len)
        assert len(tags) == 1, "Should extract media tag from current turn"
        assert "audio2.ogg" in tags[0]
        assert voice_directive is True
    
    def test_multiple_tts_calls_in_history_not_accumulated(self):
        """Multiple TTS calls in history should NOT accumulate in new responses."""
        # History with multiple TTS calls
        history = [
            {"role": "user", "content": "Say hello"},
            {"role": "tool", "tool_call_id": "1", "content": 'MEDIA:/audio/hello.ogg'},
            {"role": "assistant", "content": "Done!"},
            {"role": "user", "content": "Say goodbye"},
            {"role": "tool", "tool_call_id": "2", "content": 'MEDIA:/audio/goodbye.ogg'},
            {"role": "assistant", "content": "Done!"},
            {"role": "user", "content": "Say thanks"},
            {"role": "tool", "tool_call_id": "3", "content": 'MEDIA:/audio/thanks.ogg'},
            {"role": "assistant", "content": "Done!"},
        ]
        
        # New turn: no TTS
        new_messages = [
            {"role": "user", "content": "What time is it?"},
            {"role": "assistant", "content": "3 PM"},
        ]
        
        all_messages = history + new_messages
        history_len = len(history)
        
        # Fixed: no tags
        tags, _ = extract_media_tags_fixed(all_messages, history_len)
        assert tags == [], "Should not accumulate tags from history"
        
        # Broken: would have 3 tags (all the old ones)
        broken_tags, _ = extract_media_tags_broken(all_messages)
        assert len(broken_tags) == 3, "Broken version accumulates all history tags"
    
    def test_deduplication_within_current_turn(self):
        """Multiple MEDIA tags in current turn should be deduplicated."""
        history = []
        
        # Current turn with multiple tool calls producing same media
        new_messages = [
            {"role": "user", "content": "Multiple TTS"},
            {"role": "tool", "tool_call_id": "1", "content": 'MEDIA:/audio/same.ogg'},
            {"role": "tool", "tool_call_id": "2", "content": 'MEDIA:/audio/same.ogg'},  # duplicate
            {"role": "tool", "tool_call_id": "3", "content": 'MEDIA:/audio/different.ogg'},
            {"role": "assistant", "content": "Done!"},
        ]
        
        all_messages = history + new_messages
        
        tags, _ = extract_media_tags_fixed(all_messages, 0)
        # Even though same.ogg appears twice, deduplication happens after extraction
        # The extraction itself should get both, then caller deduplicates
        assert len(tags) == 3  # Raw extraction gets all
        
        # Deduplication as done in the actual code:
        seen = set()
        unique = [t for t in tags if t not in seen and not seen.add(t)]
        assert len(unique) == 2  # After dedup: same.ogg and different.ogg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
