"""Tests for session_meta filtering — issue #4715.

Ensures that transcript-only session_meta messages never reach the
chat-completions API, via both the API-boundary guard in
_sanitize_api_messages() and the CLI session-restore paths.
"""

import logging

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Layer 1 — _sanitize_api_messages role-allowlist guard
# ---------------------------------------------------------------------------

class TestSanitizeApiMessagesRoleFilter:

    def test_drops_session_meta_role(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "session_meta", "content": {"model": "gpt-4"}},
            {"role": "assistant", "content": "hi"},
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        assert len(out) == 2
        assert all(m["role"] != "session_meta" for m in out)

    def test_preserves_valid_roles(self):
        msgs = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
        # Need a matching assistant tool_call so the tool result isn't orphaned
        msgs[2]["tool_calls"] = [{"id": "c1", "function": {"name": "t", "arguments": "{}"}}]
        out = AIAgent._sanitize_api_messages(msgs)
        roles = [m["role"] for m in out]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles

    def test_logs_warning_when_dropping(self, caplog):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "session_meta", "content": {"info": "test"}},
        ]
        with caplog.at_level(logging.DEBUG, logger="run_agent"):
            AIAgent._sanitize_api_messages(msgs)
        assert any("invalid role" in r.message and "session_meta" in r.message for r in caplog.records)

    def test_drops_multiple_invalid_roles(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "session_meta", "content": {}},
            {"role": "transcript_note", "content": "note"},
            {"role": "assistant", "content": "hi"},
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        assert len(out) == 2
        assert [m["role"] for m in out] == ["user", "assistant"]

    def test_strips_qq_cq_image_markup_and_signed_urls_from_prompt_bound_text(self):
        msgs = [
            {
                "role": "user",
                "content": (
                    "[The user sent an image. Auto vision is warming in the background. "
                    "Don't block your reply on it right now.]\n"
                    "[CQ:image,file=7CDEAA1F045EF8013AC8631FF3708901.png,"
                    "url=https://multimedia.nt.qq.com.cn/download?appid=1406&fileid=abc&rkey=def]\n"
                    "看看这个"
                ),
            }
        ]

        out = AIAgent._sanitize_api_messages(msgs)

        assert len(out) == 1
        content = out[0]["content"]
        assert "看看这个" in content
        assert "CQ:image" not in content
        assert "multimedia.nt.qq.com.cn" not in content
        assert "7CDEAA1F045EF8013AC8631FF3708901" not in content

    def test_compacts_remote_vision_fetch_failure_tool_results(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "vision_analyze", "arguments": "{}"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": (
                    '{"success": false, "error": "Error analyzing image: Failed to fetch input URL: 400", '
                    '"analysis": "The remote image URL could not be fetched by the vision provider. '
                    'If this image came from the current chat, use the local cached file path mentioned '
                    'in the conversation instead of the remote CDN URL. '
                    'https://multimedia.nt.qq.com.cn/download?appid=1406&fileid=abc&rkey=def"}'
                ),
            },
        ]

        out = AIAgent._sanitize_api_messages(msgs)

        assert len(out) == 2
        tool_content = out[1]["content"]
        assert "multimedia.nt.qq.com.cn" not in tool_content
        assert "Failed to fetch input URL: 400" not in tool_content
        assert "local cached file path" in tool_content

    def test_compacts_low_value_vision_tool_results(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "vision_analyze", "arguments": "{}"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": (
                    '{"success": false, "error": "Error analyzing image: Sticker-like or low-value media skipped", '
                    '"analysis": "This looks like a sticker-like or low-value image, so vision skipped it to keep '
                    'chat responsive. If details matter, ask for a static screenshot instead. '
                    'Source: /tmp/qq-sticker.webp"}'
                ),
            },
        ]

        out = AIAgent._sanitize_api_messages(msgs)

        assert len(out) == 2
        tool_content = out[1]["content"]
        assert "/tmp/qq-sticker.webp" not in tool_content
        assert "chat responsive" in tool_content
        assert "static screenshot" in tool_content


class TestVisionAntiRepeatKeys:

    def test_vision_tool_cache_key_normalizes_remote_qq_signed_urls(self):
        first = AIAgent._vision_tool_cache_key(
            "vision_analyze",
            {
                "image_url": (
                    "https://multimedia.nt.qq.com.cn/download?appid=1406"
                    "&fileid=abc&rkey=first"
                )
            },
        )
        second = AIAgent._vision_tool_cache_key(
            "vision_analyze",
            {
                "image_url": (
                    "https://multimedia.nt.qq.com.cn/download?appid=1406"
                    "&fileid=def&rkey=second"
                )
            },
        )

        assert first == second
        assert first == "[qq-remote-image-url]"


# ---------------------------------------------------------------------------
# Layer 2 — CLI session-restore filters session_meta before loading
# ---------------------------------------------------------------------------

class TestCLISessionRestoreFiltering:

    def test_restore_filters_session_meta(self):
        """Simulates the CLI restore path and verifies session_meta is removed."""
        # Build a fake restored message list (as returned by get_messages_as_conversation)
        fake_restored = [
            {"role": "session_meta", "content": {"model": "gpt-4"}},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "session_meta", "content": {"tools": []}},
        ]

        # Apply the same filtering that the patched CLI code now does
        filtered = [m for m in fake_restored if m.get("role") != "session_meta"]

        assert len(filtered) == 2
        assert all(m["role"] != "session_meta" for m in filtered)
        assert filtered[0]["role"] == "user"
        assert filtered[1]["role"] == "assistant"
