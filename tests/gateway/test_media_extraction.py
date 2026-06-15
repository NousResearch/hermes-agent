"""
Tests for MEDIA tag extraction from tool results.

Verifies that MEDIA tags (e.g., from TTS tool) are only extracted from
messages in the CURRENT turn, not from the full conversation history.
This prevents voice messages from accumulating and being sent multiple
times per reply. (Regression test for #160)

Also covers #34608: a stale MEDIA: path emitted by an execute_code /
make_image tool several turns earlier must not leak onto a later
text-only reply, even when the path-based dedup set fails to capture it.
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


def extract_media_tags_production(result_messages, history_len, history_media_paths):
    """Mirror of the production scan in gateway/run.py after the #34608 fix.

    Primary guard: scope the scan to the current turn via ``history_len``
    slicing (matching how ``agent_history`` is passed as
    ``conversation_history`` into ``run_conversation``). Secondary guard:
    path-based dedup against ``history_media_paths`` (the #160 compression-safe
    fallback, also used when compression shrinks the list below history_len).
    """
    media_tags = []
    has_voice_directive = False

    if len(result_messages) >= history_len and history_len:
        scan_msgs = result_messages[history_len:]
    else:
        scan_msgs = result_messages

    for msg in scan_msgs:
        if msg.get("role") == "tool" or msg.get("role") == "function":
            content = msg.get("content", "")
            if "MEDIA:" in content:
                for match in re.finditer(r'MEDIA:(\S+)', content):
                    path = match.group(1).strip().rstrip('",}')
                    if path and path not in history_media_paths:
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

    def test_gateway_auto_append_ignores_media_examples_in_skill_docs(self):
        """Skill/documentation examples must not be appended as real attachments."""
        from gateway.run import _collect_auto_append_media_tags

        messages = [
            {"role": "user", "content": "How should I format gateway media?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_skill", "function": {"name": "skill_view"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_skill",
                "content": """
Recommended pattern:
```text
MEDIA:/absolute/path/to/image.png
```
Second message:
```text
caption
```
""",
            },
            {"role": "assistant", "content": "Use a standalone media message."},
        ]

        tags, voice = _collect_auto_append_media_tags(messages, history_offset=0)
        assert tags == []
        assert voice is False

    def test_gateway_auto_append_keeps_real_tts_media_tag(self):
        """TTS tool media tags are still auto-appended when the model omits them."""
        from gateway.run import _collect_auto_append_media_tags

        messages = [
            {"role": "user", "content": "Say this as audio"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_tts", "function": {"name": "text_to_speech"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_tts",
                "content": '{"success": true, "media_tag": "[[audio_as_voice]]\\nMEDIA:/tmp/voice.ogg"}',
            },
            {"role": "assistant", "content": "Done."},
        ]

        tags, voice = _collect_auto_append_media_tags(messages, history_offset=0)
        assert tags == ["MEDIA:/tmp/voice.ogg"]
        assert voice is True

    def test_gateway_auto_append_image_generate_json_path(self):
        """image_generate returns a local path in JSON (no MEDIA: tag); it is
        auto-appended so delivery doesn't depend on the model restating it."""
        from gateway.run import _collect_auto_append_media_tags

        messages = [
            {"role": "user", "content": "Make me a cat"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_img", "function": {"name": "image_generate"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_img",
                "content": '{"success": true, "image": "/tmp/gen/cat.png", "agent_visible_image": "/tmp/gen/cat.png"}',
            },
            {"role": "assistant", "content": "Here's your cat."},
        ]

        tags, voice = _collect_auto_append_media_tags(messages, history_offset=0)
        assert tags == ["MEDIA:/tmp/gen/cat.png"]
        assert voice is False

    def test_gateway_auto_append_image_generate_prefers_host_path(self):
        """When host and sandbox paths differ, the host-deliverable path wins."""
        from gateway.run import _collect_auto_append_media_tags

        messages = [
            {"role": "user", "content": "Make me a dog"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_img", "function": {"name": "image_generate"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_img",
                "content": '{"success": true, "host_image": "/host/dog.jpg", "image": "/host/dog.jpg", "agent_visible_image": "/sandbox/dog.jpg"}',
            },
        ]

        tags, _ = _collect_auto_append_media_tags(messages, history_offset=0)
        assert tags == ["MEDIA:/host/dog.jpg"]

    def test_gateway_auto_append_image_generate_failure_and_url_ignored(self):
        """Failed generations and remote URLs are not auto-delivered."""
        from gateway.run import _collect_auto_append_media_tags

        def _img_msgs(content):
            return [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c", "function": {"name": "image_generate"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c", "content": content},
            ]

        # Failed generation
        tags, _ = _collect_auto_append_media_tags(
            _img_msgs('{"success": false, "image": null, "error": "boom"}'),
            history_offset=0,
        )
        assert tags == []

        # Remote URL is not a local file path
        tags, _ = _collect_auto_append_media_tags(
            _img_msgs('{"success": true, "image": "https://fal.media/x/cat.png"}'),
            history_offset=0,
        )
        assert tags == []

    def test_gateway_auto_append_image_generate_dedupes_history(self):
        """A generated image path already in history is not re-sent."""
        from gateway.run import _collect_auto_append_media_tags

        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c", "function": {"name": "image_generate"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c",
                "content": '{"success": true, "image": "/tmp/gen/cat.png"}',
            },
        ]

        tags, _ = _collect_auto_append_media_tags(
            messages, history_offset=0, history_media_paths={"/tmp/gen/cat.png"}
        )
        assert tags == []

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


class TestStaleToolMediaLeak:
    """Regression tests for #34608.

    A MEDIA: path emitted by an execute_code / make_image tool several turns
    earlier remains in the full conversation message list. A later text-only
    reply (zero MEDIA directives) must NOT attach that stale image.

    The production code previously relied solely on path-based dedup against
    paths reconstructed from the replayable transcript. When that
    reconstruction does not byte-match the in-memory tool content (timestamp
    stripping, observed-context withholding, compression rewrites), the stale
    path is absent from the dedup set and leaks. Turn-scoped slicing closes
    this class of bug deterministically.
    """

    def test_stale_execute_code_media_not_attached_to_text_only_reply(self):
        """The exact #34608 scenario: make_image cover from an earlier turn."""
        # Prior turn generated an image via execute_code stdout.
        history = [
            {"role": "user", "content": "Make a cover image"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "1", "function": {"name": "execute_code"}}]},
            {"role": "tool", "tool_call_id": "1",
             "content": "Generating cover...\nMEDIA:/tmp/seosmi_cover.png\nDone."},
            {"role": "assistant", "content": "Here is your cover."},
        ]
        # Current turn: plain text status update, zero MEDIA directives.
        new_messages = [
            {"role": "user", "content": "What skill version am I on?"},
            {"role": "assistant", "content": "You're on v0.15.1."},
        ]
        all_messages = history + new_messages
        history_len = len(history)

        # Simulate the dedup set FAILING to capture the stale path (the real
        # #34608 condition: replayable-history reconstruction diverged from
        # the in-memory tool content, so the path is not in the set).
        history_media_paths = set()

        tags, voice = extract_media_tags_production(
            all_messages, history_len, history_media_paths
        )
        assert tags == [], (
            "Stale tool MEDIA from a prior turn must not leak onto a "
            f"later text-only reply, got {tags}"
        )
        assert voice is False

        # The pre-fix production behaviour (scan everything, dedup only) would
        # have leaked the stale path when the dedup set missed it.
        broken_tags, _ = extract_media_tags_broken(all_messages)
        assert any("seosmi_cover.png" in t for t in broken_tags), (
            "Sanity: the unscoped scan does surface the stale path"
        )

    def test_current_turn_media_still_attached_when_dedup_set_empty(self):
        """Turn-scoping must not suppress genuinely new media."""
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        new_messages = [
            {"role": "user", "content": "Make me a cover image"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "9", "function": {"name": "execute_code"}}]},
            {"role": "tool", "tool_call_id": "9",
             "content": "MEDIA:/tmp/fresh_cover.png"},
            {"role": "assistant", "content": "Here it is."},
        ]
        all_messages = history + new_messages
        tags, _ = extract_media_tags_production(
            all_messages, len(history), set()
        )
        assert len(tags) == 1 and "fresh_cover.png" in tags[0]

    def test_compression_shrink_falls_back_to_path_dedup(self):
        """When the list is shorter than history_len (mid-run compression),
        fall back to scanning everything with path-based dedup so the #160
        compression-safe guarantee is preserved."""
        # Post-compression list is shorter than the original history length.
        compressed_messages = [
            {"role": "user", "content": "summary so far..."},
            {"role": "tool", "tool_call_id": "7",
             "content": "MEDIA:/tmp/old_from_history.png"},
            {"role": "assistant", "content": "ok"},
        ]
        original_history_len = 12  # larger than the compressed list
        # The old path IS captured in the dedup set here (history scan ran
        # before compression), so it must still be excluded.
        history_media_paths = {"/tmp/old_from_history.png"}
        tags, _ = extract_media_tags_production(
            compressed_messages, original_history_len, history_media_paths
        )
        assert tags == [], (
            "On the compression fallback path, path-dedup must still exclude "
            f"known-old media, got {tags}"
        )


    def test_compression_fallback_dedupes_image_generate_json_paths(self):
        """After context compression, image_generate JSON paths from earlier
        turns must NOT be re-appended on the fallback scan path.

        Regression test for #46627: the _history_media_paths collection code
        only scanned for MEDIA: text patterns but missed image_generate JSON
        payloads, so the fallback path (taken when compression shrinks the
        message list below history_offset) re-appended the same images on
        every subsequent message.
        """
        from gateway.run import _collect_auto_append_media_tags

        # Simulate a compressed message list that includes an old
        # image_generate result AND a new text-only turn. The list is
        # shorter than history_offset, triggering the fallback path.
        compressed_messages = [
            # Old image_generate tool call + result (from previous turn)
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "old_img", "function": {"name": "image_generate"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "old_img",
                "content": '{"success": true, "image": "/tmp/gen/cat.png"}',
            },
            # New turn: text only, no media
            {"role": "user", "content": "What time is it?"},
            {"role": "assistant", "content": "It is 3 PM."},
        ]
        # history_offset > len(compressed_messages) triggers fallback
        original_history_len = 20

        # With the fix, _history_media_paths should contain the JSON path.
        # Simulate that by passing it as the dedup set.
        history_media_paths_with_fix = {"/tmp/gen/cat.png"}
        tags, _ = _collect_auto_append_media_tags(
            compressed_messages,
            history_offset=original_history_len,
            history_media_paths=history_media_paths_with_fix,
        )
        assert tags == [], (
            "image_generate path already in history_media_paths must not be "
            f"re-appended on the compression fallback path, got {tags}"
        )

        # Without the fix, _history_media_paths would be empty (JSON paths
        # not collected), and the fallback would re-append the image.
        history_media_paths_broken = set()
        broken_tags, _ = _collect_auto_append_media_tags(
            compressed_messages,
            history_offset=original_history_len,
            history_media_paths=history_media_paths_broken,
        )
        assert any("cat.png" in t for t in broken_tags), (
            "Sanity: without JSON path in dedup set, fallback re-appends image"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
