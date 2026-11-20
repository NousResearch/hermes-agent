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

import json
from unittest.mock import MagicMock

import pytest


class TestMediaExtraction:
    """Tests for MEDIA tag extraction from tool results."""

    def test_repairs_explicit_computer_use_media_path_from_json_result(self):
        from gateway.media_repair import (
            repair_explicit_computer_use_media_paths as _repair_explicit_computer_use_media_paths,
        )

        capture_name = "computer_use_0123456789abcdef0123456789abcdef.png"
        canonical = rf"C:\Users\Alice\AppData\Local\hermes\cache\images\{capture_name}"
        response = (
            "Here is the screenshot.\n"
            f"MEDIA:/Users/Alice/AppData/Local/hermes/cache/images/{capture_name}"
        )
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "capture", "function": {"name": "computer_use"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "capture",
                "content": json.dumps({"screenshot_path": canonical}),
            },
        ]

        repaired = _repair_explicit_computer_use_media_paths(response, messages)

        assert repaired == f"Here is the screenshot.\nMEDIA:{canonical}"

    def test_repairs_explicit_path_from_multimodal_text_summary(self):
        from gateway.media_repair import (
            repair_explicit_computer_use_media_paths as _repair_explicit_computer_use_media_paths,
        )

        capture_name = "computer_use_fedcba9876543210fedcba9876543210.jpg"
        canonical = rf"D:\Hermes Data\cache\images\{capture_name}"
        response = f'MEDIA:"/Users/Alice/Hermes Data/cache/images/{capture_name}"'
        messages = [
            {
                "role": "tool",
                "name": "computer_use",
                "tool_call_id": "capture",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "capture mode=screen 1920x1080\n"
                            f"  (shareable screenshot saved to {canonical})"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,AAAA"},
                    },
                ],
            }
        ]

        repaired = _repair_explicit_computer_use_media_paths(response, messages)

        assert repaired == f'MEDIA:"{canonical}"'

    def test_does_not_auto_attach_computer_use_capture(self):
        from gateway.media_repair import (
            repair_explicit_computer_use_media_paths as _repair_explicit_computer_use_media_paths,
        )

        capture_name = "computer_use_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.png"
        canonical = rf"C:\Users\Alice\AppData\Local\hermes\cache\images\{capture_name}"
        messages = [
            {
                "role": "tool",
                "name": "computer_use",
                "content": json.dumps({"screenshot_path": canonical}),
            }
        ]

        assert (
            _repair_explicit_computer_use_media_paths("Done.", messages)
            == "Done."
        )

    def test_does_not_rewrite_unmatched_or_previous_turn_capture(self):
        from gateway.media_repair import (
            repair_explicit_computer_use_media_paths as _repair_explicit_computer_use_media_paths,
        )

        old_name = "computer_use_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.png"
        current_name = "computer_use_cccccccccccccccccccccccccccccccc.png"
        old_canonical = rf"C:\cache\images\{old_name}"
        current_canonical = rf"C:\cache\images\{current_name}"
        history = [
            {
                "role": "tool",
                "name": "computer_use",
                "content": json.dumps({"screenshot_path": old_canonical}),
            },
        ]
        current_turn = [
            {"role": "user", "content": "Send the current screenshot."},
            {
                "role": "tool",
                "name": "computer_use",
                "content": json.dumps({"screenshot_path": current_canonical}),
            },
        ]
        response = f"MEDIA:/Users/Alice/cache/images/{old_name}"

        repaired = _repair_explicit_computer_use_media_paths(
            response,
            history + current_turn,
            history_offset=len(history),
        )

        assert repaired == response

    def test_malformed_json_result_fails_closed(self):
        """Truncated JSON must not repair to a doubled-backslash artifact.

        JSON escaping doubles backslashes; regex-scanning the raw string
        would yield ``C:\\\\Users\\\\...`` — a path that exists nowhere. When
        json.loads fails, the helper must yield nothing rather than rewrite
        the response to a corrupted path.
        """
        import json as _json

        from gateway.media_repair import (
            repair_explicit_computer_use_media_paths as _repair_explicit_computer_use_media_paths,
        )

        capture_name = "computer_use_dddddddddddddddddddddddddddddddd.png"
        canonical = rf"C:\Users\Alice\AppData\Local\hermes\cache\images\{capture_name}"
        payload = _json.dumps(
            {
                "summary": f"capture\n  (shareable screenshot saved to {canonical})",
                "screenshot_path": canonical,
            }
        )
        truncated = payload[:-5]  # starts with '{' but no longer parses
        response = f"MEDIA:/Users/Alice/AppData/Local/hermes/cache/images/{capture_name}"
        messages = [
            {"role": "tool", "name": "computer_use", "content": truncated},
        ]

        assert (
            _repair_explicit_computer_use_media_paths(response, messages)
            == response
        )

    def test_compression_fallback_slices_from_last_user_message(self):
        """When compression shrinks messages below history_offset, the repair
        recovers the current turn from the last user message — and fails
        closed (no rewrite) when no user message remains."""
        import json as _json

        from gateway.media_repair import (
            repair_explicit_computer_use_media_paths as _repair_explicit_computer_use_media_paths,
        )

        capture_name = "computer_use_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.png"
        canonical = rf"C:\cache\images\{capture_name}"
        response = f"MEDIA:/cache/images/{capture_name}"
        current_turn = [
            {"role": "user", "content": "Send the screenshot."},
            {
                "role": "tool",
                "name": "computer_use",
                "content": _json.dumps({"screenshot_path": canonical}),
            },
        ]

        # history_offset larger than the (compressed) message list forces the
        # fallback branch; the last-user slice still finds this turn's result.
        repaired = _repair_explicit_computer_use_media_paths(
            response, current_turn, history_offset=10
        )
        assert repaired == f"MEDIA:{canonical}"

        # No user message at all -> fail closed, nothing rewritten.
        no_user = [current_turn[1]]
        assert (
            _repair_explicit_computer_use_media_paths(
                response, no_user, history_offset=10
            )
            == response
        )

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


    def test_collect_history_media_paths_includes_image_generate_json(self):
        """Regression for #46627: the history media-path collector must pick up
        image_generate JSON-payload paths (no MEDIA: tag), not just MEDIA:
        text tags. Otherwise, after a compression boundary the auto-append
        fallback rescans full history, finds the generated path absent from
        the dedup set, and re-emits the same MEDIA tag every turn.
        """
        from gateway.run import _collect_history_media_paths

        history = [
            {"role": "user", "content": "make a cat"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "c", "function": {"name": "image_generate"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "c",
                "content": '{"success": true, "image": "/tmp/gen/cat.png"}',
            },
            # A separate MEDIA: text tag from another tool, to confirm both shapes.
            {
                "role": "tool",
                "tool_call_id": "d",
                "content": "Saved MEDIA:/tmp/voice/note.ogg done",
            },
        ]
        paths = _collect_history_media_paths(history)
        assert "/tmp/gen/cat.png" in paths  # JSON-payload path (the bug)
        assert "/tmp/voice/note.ogg" in paths  # MEDIA: text path (already worked)

    def test_non_streaming_dedup_excludes_current_turn_tool_output(self):
        from gateway.platforms.base import BasePlatformAdapter

        old_path = "/tmp/gen/old.png"
        current_path = "/tmp/gen/current.png"
        transcript = [
            {"role": "user", "content": "make the old image"},
            {"role": "assistant", "content": f"MEDIA:{old_path}"},
            {"role": "user", "content": "make a new image"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "current", "function": {"name": "image_generate"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "current",
                "content": f'{{"success": true, "image": "{current_path}"}}',
            },
            {"role": "assistant", "content": f"MEDIA:{current_path}"},
        ]
        adapter = MagicMock()
        adapter._session_store.peek_session_id.return_value = "session-id"
        adapter._session_store.load_transcript.return_value = transcript

        paths = BasePlatformAdapter._history_media_paths_for_session(
            adapter, "session-key"
        )
        assert paths == {old_path}

    @pytest.mark.parametrize(
        "current_path",
        ["/tmp/tts/current.ogg", "/tmp/tts/already-delivered.ogg"],
    )
    def test_non_streaming_dedup_scopes_tts_paths_to_prior_turns(
        self, current_path
    ):
        from gateway.platforms.base import BasePlatformAdapter

        old_path = "/tmp/tts/already-delivered.ogg"
        transcript = [
            {"role": "user", "content": "say the old message"},
            {"role": "assistant", "content": f"MEDIA:{old_path}"},
            {"role": "user", "content": "say the current message"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "tts", "function": {"name": "text_to_speech"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tts",
                "content": f"[[audio_as_voice]]\\nMEDIA:{current_path}",
            },
            {
                "role": "assistant",
                "content": f"[[audio_as_voice]]\\nMEDIA:{current_path}",
            },
        ]
        adapter = MagicMock()
        adapter._session_store.peek_session_id.return_value = "session-id"
        adapter._session_store.load_transcript.return_value = transcript

        paths = BasePlatformAdapter._history_media_paths_for_session(
            adapter, "session-key"
        )
        assert paths == {old_path}

    def test_image_generate_not_reemitted_after_compression(self):
        """End-to-end of the #46627 fix: collect history paths, then the
        compression-fallback rescan (history_offset stale) must dedup the
        generated image against them — no re-emission."""
        from gateway.run import (
            _collect_auto_append_media_tags,
            _collect_history_media_paths,
        )

        history = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c", "function": {"name": "image_generate"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "c",
                "content": '{"success": true, "image": "/tmp/gen/dog.png"}',
            },
        ]
        history_paths = _collect_history_media_paths(history)

        # Simulate the post-compression fallback: history_offset is stale
        # (larger than the shrunken message list), so the collector rescans
        # the full list. With the dedup set populated, the already-delivered
        # image must NOT be re-emitted.
        tags, _ = _collect_auto_append_media_tags(
            history, history_offset=9999, history_media_paths=history_paths
        )
        assert tags == [], f"generated image re-emitted after compression: {tags}"


    
    
    




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
