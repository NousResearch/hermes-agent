"""Tests for GatewayStreamConsumer — media directive stripping in streaming."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def test_stream_send_metadata_carries_original_reply_anchor():
    consumer = GatewayStreamConsumer(
        adapter=MagicMock(),
        chat_id="123",
        initial_reply_to_id="456",
    )

    assert consumer._metadata_for_send(final=False) == {
        "reply_to_message_id": "456",
    }
    assert consumer._metadata_for_send(final=True) == {
        "reply_to_message_id": "456",
        "notify": True,
    }


# ── _clean_for_display unit tests ────────────────────────────────────────


class TestCleanForDisplay:
    """Verify secrets and internal markers are stripped from display text."""

    def test_no_media_passthrough(self):
        """Text without MEDIA: passes through unchanged."""
        text = "Here is your analysis of the image."
        assert GatewayStreamConsumer._clean_for_display(text) == text


    def test_media_tag_single_quoted_stripped(self):
        """A single-quote-wrapped tag matches the known-ext cleanup pattern
        and is removed (delivery attempts it too — consistent)."""
        result = GatewayStreamConsumer._clean_for_display(
            "Result: 'MEDIA:/path/file.png'"
        )
        assert "MEDIA:" not in result

    def test_media_tag_double_quoted_json_context_stays_visible(self):
        """A double-quoted tag preceded by a colon sits in a JSON value
        context (#34375): extract_media masks it and never delivers, so
        display keeps it visible too instead of silently hiding a tag that
        produced no attachment (display/delivery consistency)."""
        result = GatewayStreamConsumer._clean_for_display(
            'Result: "MEDIA:/path/file.png"'
        )
        assert '"MEDIA:/path/file.png"' in result


    def test_media_tag_in_backticks_bogus_path_stays_visible(self):
        """A backtick-wrapped tag with a non-existent path is an inline-code
        example: extract_media does NOT deliver it, so display must not
        silently strip it either (display/delivery consistency, #16434)."""
        result = GatewayStreamConsumer._clean_for_display(
            "Result: `MEDIA:/path/file.png`"
        )
        assert "`MEDIA:/path/file.png`" in result

    def test_audio_as_voice_stripped(self):
        """[[audio_as_voice]] directive is removed."""
        text = "[[audio_as_voice]]\nMEDIA:/tmp/voice.ogg"
        result = GatewayStreamConsumer._clean_for_display(text)
        assert "[[audio_as_voice]]" not in result
        assert "MEDIA:" not in result

    def test_multiple_media_tags(self):
        """Multiple MEDIA: tags are all removed."""
        text = "Here are two files:\nMEDIA:/tmp/a.png\nMEDIA:/tmp/b.jpg"
        result = GatewayStreamConsumer._clean_for_display(text)
        assert "MEDIA:" not in result
        assert "Here are two files:" in result

    def test_excessive_newlines_collapsed(self):
        """Blank lines left by removed tags are collapsed."""
        text = "Before\n\n\nMEDIA:/tmp/file.png\n\n\nAfter"
        result = GatewayStreamConsumer._clean_for_display(text)
        # Should not have 3+ consecutive newlines
        assert "\n\n\n" not in result

    def test_media_only_response(self):
        """Response that is entirely MEDIA: tags returns empty/whitespace."""
        text = "MEDIA:/tmp/image.png"
        result = GatewayStreamConsumer._clean_for_display(text)
        assert result.strip() == ""

    def test_media_mid_sentence(self):
        """MEDIA: tag embedded in prose is stripped cleanly."""
        text = "I generated this image MEDIA:/tmp/art.png for you."
        result = GatewayStreamConsumer._clean_for_display(text)
        assert "MEDIA:" not in result
        assert "generated" in result
        assert "for you." in result

    def test_preserves_non_media_colons(self):
        """Normal colons and text with 'MEDIA' as a word aren't stripped."""
        text = "The media: files are stored in /tmp. Use social MEDIA carefully."
        result = GatewayStreamConsumer._clean_for_display(text)
        # "MEDIA:" in upper case without a path won't match \S+ (space follows)
        # But "media:" is lowercase so won't match either
        assert result == text

    def test_secret_redaction_is_forced_when_global_preference_is_off(self, monkeypatch):
        """Chat egress remains safe when general-purpose redaction is disabled."""
        secret = "sk-abc123def456ghi789jkl012mno345pqr678stu"
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)

        result = GatewayStreamConsumer._clean_for_display(f"token: {secret}")

        assert secret not in result
        assert "***" in result

    def test_tool_trace_banner_stripped(self):
        """Internal tool-failure diagnostics do not become chat content."""
        text = "Done.\n⚠️ 🛠️ `search repos (agent)` failed"

        assert GatewayStreamConsumer._clean_for_display(text) == "Done."

    def test_unmarked_backticked_failure_prose_remains_visible(self):
        """Ordinary assistant failure prose is not an internal tool trace."""
        text = "Result:\n`database migration` failed"

        assert GatewayStreamConsumer._clean_for_display(text) == text


# ── Integration: _send_or_edit strips MEDIA: ─────────────────────────────


class TestFinalizeCapabilityGate:
    """Verify REQUIRES_EDIT_FINALIZE gates the redundant final edit.

    Platforms that don't need an explicit finalize signal (Telegram,
    Slack, Matrix, …) should skip the redundant final edit when the
    mid-stream edit already delivered the final content.  Platforms that
    *do* need it (DingTalk AI Cards) must always receive a finalize=True
    edit at the end of the stream.
    """

    @pytest.mark.asyncio
    async def test_identical_text_skip_respects_adapter_flag(self):
        """_send_or_edit short-circuits identical-text only when the
        adapter doesn't require an explicit finalize signal."""
        # Adapter without finalize requirement — should skip identical edit.
        plain = MagicMock()
        plain.REQUIRES_EDIT_FINALIZE = False
        plain.send = AsyncMock(return_value=SimpleNamespace(
            success=True, message_id="m1",
        ))
        plain.edit_message = AsyncMock()
        plain.MAX_MESSAGE_LENGTH = 4096
        c1 = GatewayStreamConsumer(plain, "chat_1")
        await c1._send_or_edit("hello")  # first send
        await c1._send_or_edit("hello", finalize=True)  # identical → skip
        plain.edit_message.assert_not_called()

        # Adapter that requires finalize — must still fire the edit.
        picky = MagicMock()
        picky.REQUIRES_EDIT_FINALIZE = True
        picky.send = AsyncMock(return_value=SimpleNamespace(
            success=True, message_id="m1",
        ))
        picky.edit_message = AsyncMock(return_value=SimpleNamespace(
            success=True, message_id="m1",
        ))
        picky.MAX_MESSAGE_LENGTH = 4096
        c2 = GatewayStreamConsumer(picky, "chat_1")
        await c2._send_or_edit("hello")
        await c2._send_or_edit("hello", finalize=True)
        # Finalize edit must go through even on identical content.
        picky.edit_message.assert_called_once()
        assert picky.edit_message.call_args[1]["finalize"] is True


class TestEditMessageFinalizeSignature:
    """Every concrete platform adapter must accept the ``finalize`` kwarg.

    stream_consumer._send_or_edit always passes ``finalize=`` to
    ``adapter.edit_message(...)`` (see gateway/stream_consumer.py).  An
    adapter that overrides edit_message without accepting finalize raises
    TypeError the first time streaming hits a segment break or final edit.
    Guard the contract with an explicit signature check so it cannot
    silently regress — existing tests use MagicMock which swallows any
    kwarg and cannot catch this.
    """

    @pytest.mark.parametrize(
        "module_path,class_name",
        [
            ("plugins.platforms.telegram.adapter", "TelegramAdapter"),
            ("plugins.platforms.discord.adapter", "DiscordAdapter"),
            ("plugins.platforms.slack.adapter", "SlackAdapter"),
            ("plugins.platforms.matrix.adapter", "MatrixAdapter"),
            ("plugins.platforms.mattermost.adapter", "MattermostAdapter"),
            ("plugins.platforms.feishu.adapter", "FeishuAdapter"),
            ("plugins.platforms.whatsapp.adapter", "WhatsAppAdapter"),
            ("plugins.platforms.dingtalk.adapter", "DingTalkAdapter"),
        ],
    )
    def test_edit_message_accepts_finalize(self, module_path, class_name):
        import inspect

        module = pytest.importorskip(module_path)
        cls = getattr(module, class_name)
        params = inspect.signature(cls.edit_message).parameters
        assert "finalize" in params, (
            f"{class_name}.edit_message must accept 'finalize' kwarg; "
            f"stream_consumer._send_or_edit passes it unconditionally"
        )


class TestSendOrEditMediaStripping:
    """Verify _send_or_edit strips MEDIA: before sending to the platform."""

    @pytest.mark.asyncio
    async def test_first_send_strips_media(self):
        """Initial send removes MEDIA: tags from visible text."""
        adapter = MagicMock()
        send_result = SimpleNamespace(success=True, message_id="msg_1")
        adapter.send = AsyncMock(return_value=send_result)
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(adapter, "chat_123")
        await consumer._send_or_edit("Here is your image\nMEDIA:/tmp/test.png")

        adapter.send.assert_called_once()
        sent_text = adapter.send.call_args[1]["content"]
        assert "MEDIA:" not in sent_text
        assert "Here is your image" in sent_text

    @pytest.mark.asyncio
    async def test_edit_strips_media(self):
        """Edit call removes MEDIA: tags from visible text."""
        adapter = MagicMock()
        send_result = SimpleNamespace(success=True, message_id="msg_1")
        edit_result = SimpleNamespace(success=True)
        adapter.send = AsyncMock(return_value=send_result)
        adapter.edit_message = AsyncMock(return_value=edit_result)
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(adapter, "chat_123")
        # First send
        await consumer._send_or_edit("Starting response...")
        # Edit with MEDIA: tag
        await consumer._send_or_edit("Here is the result\nMEDIA:/tmp/image.png")

        adapter.edit_message.assert_called_once()
        edited_text = adapter.edit_message.call_args[1]["content"]
        assert "MEDIA:" not in edited_text


    @pytest.mark.asyncio
    async def test_short_text_with_cursor_skips_new_message(self):
        """Short text + cursor should not create a standalone new message.

        During rapid tool-calling the model often emits 1-2 tokens before
        switching to tool calls.  Sending 'I ▉' as a new message risks
        leaving the cursor permanently visible if the follow-up edit is
        rate-limited.  The guard should skip the first send and let the
        text accumulate into the next segment.
        """
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(cursor=" ▉"),
        )
        # No message_id yet (first send) — short text + cursor should be skipped
        assert consumer._message_id is None
        result = await consumer._send_or_edit("I ▉")
        assert result is True
        adapter.send.assert_not_called()

        # 3 chars is still under the threshold
        result = await consumer._send_or_edit("Hi! ▉")
        assert result is True
        adapter.send.assert_not_called()


# ── Integration: full stream run ─────────────────────────────────────────


class TestStreamRunMediaStripping:
    """End-to-end: deltas with MEDIA: produce clean visible text."""

    @pytest.mark.asyncio
    async def test_stream_with_media_tag(self):
        """Full stream run strips MEDIA: from the final visible message."""
        adapter = MagicMock()
        send_result = SimpleNamespace(success=True, message_id="msg_1")
        edit_result = SimpleNamespace(success=True)
        adapter.send = AsyncMock(return_value=send_result)
        adapter.edit_message = AsyncMock(return_value=edit_result)
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Feed deltas
        consumer.on_delta("Here is your generated image\n")
        consumer.on_delta("MEDIA:/home/user/.hermes/cache/images/abc123.png")
        consumer.finish()

        await consumer.run()

        # Verify the final text sent/edited doesn't contain MEDIA:
        all_calls = []
        for call in adapter.send.call_args_list:
            all_calls.append(call[1].get("content", ""))
        for call in adapter.edit_message.call_args_list:
            all_calls.append(call[1].get("content", ""))

        for sent_text in all_calls:
            assert "MEDIA:" not in sent_text, f"MEDIA: leaked into display: {sent_text!r}"

        assert consumer.already_sent


class TestStreamRunSecretRedaction:
    """Redact the complete display buffer before either overflow splitter."""

    _SECRET = "sk-abc123def456ghi789jkl012mno345pqr678stu"
    _PREFIX = "x" * 497 + " "
    _VISIBLE_PREFIX = "x" * 490 + "\n"

    @staticmethod
    def _adapter():
        adapter = MagicMock()
        adapter.REQUIRES_EDIT_FINALIZE = False
        adapter.MAX_MESSAGE_LENGTH = 600
        adapter.send = AsyncMock(
            side_effect=lambda **_kwargs: SimpleNamespace(
                success=True,
                message_id=f"msg_{adapter.send.call_count}",
            )
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="preview")
        )
        return adapter

    @pytest.mark.asyncio
    async def test_no_message_overflow_redacts_before_truncate(self, monkeypatch):
        """The first-send splitter never receives a reconstructible token."""
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        truncated_inputs = []

        def _truncate(text, limit, len_fn=len):
            truncated_inputs.append(text)
            return [text[:limit], text[limit:]]

        adapter.truncate_message.side_effect = _truncate
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(cursor="", buffer_only=True),
        )
        consumer.on_delta(self._PREFIX + self._SECRET)
        consumer.finish()

        await consumer.run()

        assert len(truncated_inputs) == 1
        assert self._SECRET not in truncated_inputs[0]
        payload = "".join(call.kwargs["content"] for call in adapter.send.call_args_list)
        assert self._SECRET not in payload
        assert "sk-abc...8stu" in payload

    @pytest.mark.asyncio
    async def test_existing_message_overflow_redacts_before_slice(self, monkeypatch):
        """The edit-overflow splitter never seals reconstructible fragments."""
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(cursor="", buffer_only=True),
        )
        consumer._message_id = "preview"
        consumer._last_sent_text = "preview"
        consumer.on_delta(self._PREFIX + self._SECRET)
        consumer.finish()

        await consumer.run()

        payloads = [
            call.kwargs["content"] for call in adapter.edit_message.call_args_list
        ]
        payloads.extend(call.kwargs["content"] for call in adapter.send.call_args_list)
        payload = "".join(payloads)
        assert self._SECRET not in payload
        assert "sk-abc...8stu" in payload

    @pytest.mark.asyncio
    async def test_split_secret_candidate_is_held_before_overflow(self, monkeypatch):
        """An unrecognized prefix cannot be sealed before its secret suffix."""
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        adapter.truncate_message.side_effect = (
            lambda text, limit, len_fn=len: [text[:limit], text[limit:]]
        )
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        secret = "sk-abcdefghij-klmnopqrstuv"
        task = asyncio.create_task(consumer.run())

        consumer.on_delta(self._VISIBLE_PREFIX + secret[:12])
        for _ in range(100):
            if adapter.send.call_count:
                break
            await asyncio.sleep(0.001)
        assert adapter.send.call_count

        consumer.on_delta(secret[12:])
        consumer.finish()
        await task

        payloads = [
            call.kwargs["content"] for call in adapter.send.call_args_list
        ]
        payloads.extend(
            call.kwargs["content"] for call in adapter.edit_message.call_args_list
        )
        assert all(secret[:12] not in payload for payload in payloads)
        assert secret not in "".join(payloads)
        assert any("sk-abc...stuv" in payload for payload in payloads)

    @pytest.mark.asyncio
    async def test_diverged_secret_prefix_is_released_during_stream(self, monkeypatch):
        """Legitimate prefix-shaped prose is delayed only until divergence."""
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())

        consumer.on_delta(self._VISIBLE_PREFIX + "sk-abc")
        for _ in range(100):
            if adapter.send.call_count:
                break
            await asyncio.sleep(0.001)
        assert adapter.send.call_count

        consumer.on_delta(" is documentation")
        for _ in range(100):
            if any(
                call.kwargs["content"].endswith("sk-abc is documentation")
                for call in (
                    adapter.send.call_args_list + adapter.edit_message.call_args_list
                )
            ):
                break
            await asyncio.sleep(0.001)
        assert any(
            call.kwargs["content"].endswith("sk-abc is documentation")
            for call in (
                adapter.send.call_args_list + adapter.edit_message.call_args_list
            )
        )

        consumer.finish()
        await task

    @pytest.mark.asyncio
    async def test_incomplete_database_password_never_reaches_adapter(self, monkeypatch):
        """A DB password stays private until its @ terminator arrives."""
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())
        password = "opaqueCredentialValue123"

        consumer.on_delta(f"Database: postgresql://operator:{password}")
        await asyncio.sleep(0.02)

        interim_payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list + adapter.edit_message.call_args_list
        ]
        assert all(password not in payload for payload in interim_payloads)

        consumer.on_delta("@db.example/app")
        consumer.finish()
        await task

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list + adapter.edit_message.call_args_list
        ]
        assert all(password not in payload for payload in payloads)
        assert any("postgresql://operator:***@db.example/app" in payload for payload in payloads)

    @pytest.mark.asyncio
    async def test_unterminated_private_key_body_never_reaches_adapter(self, monkeypatch):
        """A private-key body is suppressed even before the END marker."""
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        key_body = "SYNTHETICINERTPRIVATEKEYBODY1234567890"
        consumer.on_delta(
            "Key follows:\n-----BEGIN PRIVATE KEY-----\n" + key_body + "\n"
        )
        consumer.finish()

        await consumer.run()

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list + adapter.edit_message.call_args_list
        ]
        assert all(key_body not in payload for payload in payloads)
        assert any("[REDACTED PRIVATE KEY]" in payload for payload in payloads)

    @pytest.mark.asyncio
    async def test_json_secret_opener_is_not_sealed_before_value(self, monkeypatch):
        """Overflow cannot separate a JSON secret key from its later value."""
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        adapter.truncate_message.side_effect = (
            lambda text, limit, len_fn=len: [text[:limit], text[limit:]]
        )
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())
        secret = "opaqueJsonCredentialValue123"

        consumer.on_delta(self._VISIBLE_PREFIX + '{"token": "')
        await asyncio.sleep(0.02)
        consumer.on_delta(secret + '"}')
        consumer.finish()
        await task

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list + adapter.edit_message.call_args_list
        ]
        assert all(secret not in payload for payload in payloads)
        assert '"token": "***"' in "".join(payloads), [
            (len(payload), payload[-80:]) for payload in payloads
        ]

    @pytest.mark.asyncio
    async def test_known_prefix_state_survives_masked_preview(self, monkeypatch):
        """Later token bytes cannot escape after the first masked preview."""
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())
        first = "github_pat_ABCDEFGHIJ"
        tail = "KLMNOPQRSTUVWXYZ0123456789"

        consumer.on_delta("Credential: " + first)
        await asyncio.sleep(0.02)
        consumer.on_delta(tail)
        consumer.finish()
        await task

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list + adapter.edit_message.call_args_list
        ]
        assert all(first not in payload for payload in payloads)
        assert all(tail not in payload for payload in payloads)
        assert all(first + tail not in payload for payload in payloads)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("boundary", ["segment", "commentary"])
    async def test_known_prefix_state_survives_logical_boundary(
        self, monkeypatch, boundary,
    ):
        """Segment/commentary events do not terminate a credential candidate."""
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())
        first = "github_pat_ABCDE"
        tail = "FGHIJKLMNOPQRSTUVWXYZ0123456789"

        consumer.on_delta("Credential: " + first)
        if boundary == "segment":
            consumer.on_segment_break()
        else:
            consumer.on_commentary("Still working.")
        await asyncio.sleep(0.02)
        consumer.on_delta(tail + " finished")
        consumer.finish()
        await task

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list + adapter.edit_message.call_args_list
        ]
        assert all(first not in payload for payload in payloads)
        assert all(tail not in payload for payload in payloads)
        # The first commentary word is indistinguishable from a continuation
        # of the attacker-controlled token prefix and may be masked with it.
        assert any("working." in payload for payload in payloads) is (
            boundary == "commentary"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("boundary", ["segment", "commentary"])
    @pytest.mark.parametrize(
        ("first", "second", "leak_marker"),
        [
            (
                "Credential: github_pa",
                "t_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 done",
                "t_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            ),
            (
                'Data: {"tok',
                'en": "opaqueJsonCredentialValue123"}',
                "opaqueJsonCredentialValue123",
            ),
            (
                "URL: postgr",
                "es://user:opaqueDbCredentialValue123@db.example",
                "opaqueDbCredentialValue123",
            ),
            (
                "Key:\n-----BEG",
                (
                    "IN PRIVATE KEY-----\n"
                    "SYNTHETICINERTPRIVATEKEYBODY1234567890\n"
                    "-----END PRIVATE KEY-----"
                ),
                "SYNTHETICINERTPRIVATEKEYBODY1234567890",
            ),
            (
                "OPENAI_API_",
                "KEY=opaqueEnvCredentialValue123 done",
                "opaqueEnvCredentialValue123",
            ),
            (
                "Authoriza",
                "tion: Bearer opaqueAuthCredentialValue123 done",
                "opaqueAuthCredentialValue123",
            ),
            (
                "x-api-",
                "key: opaqueHeaderCredentialValue123 done",
                "opaqueHeaderCredentialValue123",
            ),
            (
                "mode=x&tok",
                "en=opaqueFormCredentialValue123",
                "opaqueFormCredentialValue123",
            ),
        ],
        ids=[
            "known-prefix",
            "json",
            "db",
            "pem",
            "env",
            "authorization",
            "api-key-header",
            "form-body",
        ],
    )
    async def test_partial_secret_opener_survives_logical_boundary(
        self, monkeypatch, boundary, first, second, leak_marker,
    ):
        """Partial grammar openers remain raw across segment/commentary events."""
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())

        consumer.on_delta(first)
        if boundary == "segment":
            consumer.on_segment_break()
        else:
            consumer.on_commentary("Still working.")
        await asyncio.sleep(0.02)
        early_payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list + adapter.edit_message.call_args_list
        ]
        assert all(first not in payload for payload in early_payloads)

        consumer.on_delta(second)
        consumer.finish()
        await task

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list + adapter.edit_message.call_args_list
        ]
        assert all(leak_marker not in payload for payload in payloads)
        assert any("Still working." in payload for payload in payloads) is (
            boundary == "commentary"
        )

    @pytest.mark.asyncio
    async def test_one_character_pat_segments_never_reconstruct(self, monkeypatch):
        """Every prefix character may arrive in its own logical segment."""
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())
        secret = "github_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        for char in secret:
            consumer.on_delta(char)
            consumer.on_segment_break()
        consumer.on_delta(" done")
        consumer.finish()
        await task

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list + adapter.edit_message.call_args_list
        ]
        joined = "".join(payloads)
        assert secret not in joined
        assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in joined

    @pytest.mark.asyncio
    async def test_empty_form_key_split_is_retained(self, monkeypatch):
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())

        consumer.on_delta("mode=x&")
        consumer.on_segment_break()
        await asyncio.sleep(0.02)
        consumer.on_delta("token=opaqueFormCredentialValue123")
        consumer.finish()
        await task

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list + adapter.edit_message.call_args_list
        ]
        assert all("opaqueFormCredentialValue123" not in payload for payload in payloads)

    @pytest.mark.asyncio
    async def test_progressive_authorization_scheme_retains_raw_state(self, monkeypatch):
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())

        consumer.on_delta("Authorization: Bearer")
        await asyncio.sleep(0.02)
        consumer.on_delta(" opaqueAuthCredentialValue123 done")
        consumer.finish()
        await task

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list + adapter.edit_message.call_args_list
        ]
        assert all("opaqueAuthCredentialValue123" not in payload for payload in payloads)
        assert any("Authorization: Bearer ***" in payload for payload in payloads)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("first", "second", "leak"),
        [
            (
                "prefix12345678:" + "A" * 29,
                "A done",
                "A" * 29,
            ),
            (
                "prefixeyJ" + "A" * 9,
                "A done",
                "eyJ" + "A" * 9,
            ),
            (
                "prefix+123456",
                "7 done",
                "+123456",
            ),
        ],
        ids=["telegram", "jwt", "phone"],
    )
    async def test_embedded_prethreshold_candidate_is_never_previewed(
        self, monkeypatch, first, second, leak,
    ):
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())

        consumer.on_delta(first)
        await asyncio.sleep(0.02)
        early = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list + adapter.edit_message.call_args_list
        ]
        assert all(leak not in payload for payload in early)
        consumer.on_delta(second)
        consumer.finish()
        await task

    @pytest.mark.asyncio
    async def test_commentary_secret_state_spans_messages(self, monkeypatch):
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )

        consumer.on_commentary("g")
        consumer.on_commentary(
            "ithub_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        consumer.finish()
        await consumer.run()

        payloads = [
            call.kwargs["content"] for call in adapter.send.call_args_list
        ]
        assert all(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in payload for payload in payloads
        )

    @pytest.mark.asyncio
    async def test_commentary_prefix_divergence_preserves_ordinary_text(self):
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )

        consumer.on_commentary("g")
        consumer.on_commentary("ood")
        consumer.finish()
        await consumer.run()

        assert [
            call.kwargs["content"] for call in adapter.send.call_args_list
        ] == ["good"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("first", "second", "leak"),
        [
            ("12345678:", "A" * 30 + " done", "A" * 30),
            ("ey", "J" + "A" * 10 + " done", "eyJ" + "A" * 10),
            ("+", "1234567 done", "+1234567"),
            (
                "OPENAI_API_KEY   ",
                "=opaqueWhitespaceCredential123 done",
                "opaqueWhitespaceCredential123",
            ),
        ],
        ids=["telegram-empty-body", "jwt-before-j", "phone-plus", "env-whitespace"],
    )
    async def test_earliest_candidate_prefix_survives_boundary(
        self, monkeypatch, first, second, leak,
    ):
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())

        consumer.on_delta(first)
        consumer.on_segment_break()
        await asyncio.sleep(0.02)
        assert all(
            first not in call.kwargs["content"]
            for call in adapter.send.call_args_list
            + adapter.edit_message.call_args_list
        )
        consumer.on_delta(second)
        consumer.finish()
        await task

        assert all(
            leak not in call.kwargs["content"]
            for call in adapter.send.call_args_list
            + adapter.edit_message.call_args_list
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("first", "second", "leak"),
        [
            (
                "OPENAI_API_KEY=opaqueEnvCredentialPart",
                "Tail123 done",
                "Tail123",
            ),
            (
                "mode=x&token=opaqueFormCredentialPart",
                "Tail123&safe=y",
                "Tail123",
            ),
        ],
        ids=["env", "form"],
    )
    async def test_active_value_state_survives_mid_value_boundary(
        self, monkeypatch, first, second, leak,
    ):
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())

        consumer.on_delta(first)
        consumer.on_segment_break()
        await asyncio.sleep(0.02)
        consumer.on_delta(second)
        consumer.finish()
        await task

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list
            + adapter.edit_message.call_args_list
        ]
        assert all(leak not in payload for payload in payloads)

    @pytest.mark.asyncio
    async def test_completed_private_key_is_never_previewed_raw(self, monkeypatch):
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())
        body = "SYNTHETICINERTPRIVATEKEYBODY1234567890"
        pem = (
            "-----BEGIN PRIVATE KEY-----\n"
            f"{body}\n"
            "-----END PRIVATE KEY-----"
        )

        consumer.on_delta(pem)
        await asyncio.sleep(0.02)
        consumer.on_delta("\nDone.")
        consumer.finish()
        await task

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list
            + adapter.edit_message.call_args_list
        ]
        assert all(body not in payload for payload in payloads)
        assert any("[REDACTED PRIVATE KEY]" in payload for payload in payloads)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("first", "second", "leak"),
        [
            ('prefix"', 'token": "opaqueEmbeddedJsonCredential"', "opaqueEmbedded"),
            (
                "prefixpost",
                "gresql://user:opaqueEmbeddedDbCredential@host",
                "opaqueEmbedded",
            ),
            (
                "prefixe",
                "yJABCDEFGHIJK.payloadSegment",
                "ABCDEFGHIJK",
            ),
            (
                "token=opaqueFirstForm",
                "Credential&safe=value",
                "opaqueFirstForm",
            ),
            (
                'OPENAI_API_KEY=""',
                "opaqueEmptyQuotedCredential done",
                "opaqueEmptyQuotedCredential",
            ),
        ],
        ids=[
            "embedded-json",
            "embedded-db",
            "embedded-jwt",
            "first-sensitive-form-pair",
            "empty-quoted-env-concatenation",
        ],
    )
    async def test_canonical_embedded_and_first_pair_state_survives_boundary(
        self, monkeypatch, first, second, leak,
    ):
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())

        consumer.on_delta(first)
        consumer.on_segment_break()
        await asyncio.sleep(0.02)
        consumer.on_delta(second)
        consumer.finish()
        await task

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list
            + adapter.edit_message.call_args_list
        ]
        assert all(leak not in payload for payload in payloads)

    @pytest.mark.asyncio
    async def test_paced_embedded_json_never_replays_raw_at_preview(
        self, monkeypatch,
    ):
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())
        secret = "opaquePacedEmbeddedJsonCredential123"

        consumer.on_delta('prefix"')
        consumer.on_segment_break()
        await asyncio.sleep(0.05)
        consumer.on_delta(f'token": "{secret}"')
        await asyncio.sleep(0.1)

        preview_payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list
            + adapter.edit_message.call_args_list
        ]
        assert all(secret not in payload for payload in preview_payloads)

        consumer.finish()
        await task
        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list
            + adapter.edit_message.call_args_list
        ]
        assert all(secret not in payload for payload in payloads)
        assert any('"token": "***"' in payload for payload in payloads)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("answer", "commentary", "leak"),
        [
            ("g", "ithub_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            ('"token": "', "opaqueCrossChannelJsonCredential", "opaqueCrossChannel"),
            ("postgres://user:", "opaqueCrossChannelDbCredential@host", "opaqueCrossChannel"),
            (
                "-----BEGIN PRIVATE KEY-----\n",
                "SYNTHETICINERTPRIVATEKEYBODY123\n-----END PRIVATE KEY-----",
                "SYNTHETICINERTPRIVATEKEYBODY",
            ),
            ("OPENAI_API_KEY=", "opaqueCrossChannelEnvCredential", "opaqueCrossChannel"),
            ("token=", "opaqueCrossChannelFormCredential&safe=x", "opaqueCrossChannel"),
        ],
        ids=["known-prefix", "json", "db", "pem", "env", "form"],
    )
    async def test_answer_state_protects_following_commentary(
        self, monkeypatch, answer, commentary, leak,
    ):
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )

        consumer.on_delta(answer)
        consumer.on_commentary(commentary)
        consumer.finish()
        await consumer.run()

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list
            + adapter.edit_message.call_args_list
        ]
        assert all(leak not in payload for payload in payloads)

    @pytest.mark.asyncio
    async def test_cancellation_after_cross_channel_completion_never_leaks(
        self, monkeypatch,
    ):
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())

        consumer.on_delta("g")
        consumer.on_commentary(
            "ithub_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        await asyncio.sleep(0.02)
        task.cancel()
        await task

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list
            + adapter.edit_message.call_args_list
        ]
        assert all("ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in payload for payload in payloads)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("terminal", ["finish", "cancel"])
    async def test_multi_callback_commentary_keeps_one_cross_channel_state(
        self, monkeypatch, terminal,
    ):
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())
        first = "pat_"
        second = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!"

        consumer.on_delta("github_")
        consumer.on_commentary(first)
        await asyncio.sleep(0.05)
        consumer.on_commentary(second)
        await asyncio.sleep(0.05)
        if terminal == "finish":
            consumer.finish()
        else:
            task.cancel()
        await task

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list
            + adapter.edit_message.call_args_list
        ]
        assert all(first not in payload for payload in payloads)
        assert all(second not in payload for payload in payloads)
        assert any("github...6789" in payload for payload in payloads)

    @pytest.mark.asyncio
    async def test_finish_preserves_benign_cross_channel_commentary(
        self, monkeypatch,
    ):
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())

        consumer.on_delta("g")
        consumer.on_commentary("ithub_pa")
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        commentary_payloads = [
            call.kwargs["content"] for call in adapter.send.call_args_list
        ]
        assert "ithub_pa" in commentary_payloads

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("commentary", "final_tail", "leak"),
        [
            (
                "g",
                "ithub_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            ),
            (
                'Data: {"token": "opaqueCommentary',
                "FinalCredential123",
                "FinalCredential123",
            ),
            (
                "OPENAI_API_KEY=opaqueCommentary",
                "FinalCredential123",
                "FinalCredential123",
            ),
        ],
        ids=["known-prefix", "json", "env"],
    )
    async def test_commentary_state_continues_into_final_stream(
        self, monkeypatch, commentary, final_tail, leak,
    ):
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )

        consumer.on_commentary(commentary)
        consumer.on_delta(final_tail)
        consumer.finish()
        await consumer.run()

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list
            + adapter.edit_message.call_args_list
        ]
        assert all(leak not in payload for payload in payloads)

    @pytest.mark.asyncio
    async def test_cancellation_terminally_flushes_retained_state(self, monkeypatch):
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())
        secret = "github_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        consumer.on_delta(secret)
        await asyncio.sleep(0.02)
        task.cancel()
        await task

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list
            + adapter.edit_message.call_args_list
        ]
        assert payloads
        assert all(secret not in payload for payload in payloads)
        assert any("github...6789" in payload for payload in payloads)

    @pytest.mark.asyncio
    async def test_cancellation_preserves_benign_commentary_suffix(self):
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())

        consumer.on_commentary("g")
        await asyncio.sleep(0.02)
        task.cancel()
        await task

        assert [
            call.kwargs["content"] for call in adapter.send.call_args_list
        ] == ["g"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("first", "tail", "leak_marker"),
        [
            ("12345678:" + "A" * 29, "A done", "A" * 29),
            ("eyJ" + "A" * 9, "A done", "eyJ" + "A" * 9),
            ("+123456", "7 done", "+123456"),
        ],
        ids=["telegram", "jwt", "phone"],
    )
    async def test_prethreshold_secret_candidate_is_never_previewed(
        self, monkeypatch, first, tail, leak_marker,
    ):
        """A recognizable opener is retained before the static minimum."""
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())

        consumer.on_delta(first)
        await asyncio.sleep(0.02)
        early_payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list + adapter.edit_message.call_args_list
        ]
        assert all(leak_marker not in payload for payload in early_payloads)

        consumer.on_delta(tail)
        consumer.finish()
        await task
        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list + adapter.edit_message.call_args_list
        ]
        assert all(leak_marker not in payload for payload in payloads)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text",
        [
            "12345678:short prose",
            "eyJshort prose",
            "+123 nope",
        ],
        ids=["telegram", "jwt", "phone"],
    )
    async def test_prethreshold_candidate_releases_on_divergence(self, text):
        """Ordinary prose is preserved after the terminal ambiguity boundary."""
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())

        split = text.find("short") if "short" in text else text.find(" nope")
        consumer.on_delta(text[:split])
        await asyncio.sleep(0.02)
        consumer.on_delta(text[split:])
        consumer.finish()
        await task

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list + adapter.edit_message.call_args_list
        ]
        assert text in payloads[-1]

    @pytest.mark.asyncio
    async def test_unterminated_commentary_is_terminally_sanitized(self, monkeypatch):
        """Completed commentary cannot bypass structured-secret termination."""
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        body = "opaqueCommentaryCredentialValue123"

        consumer.on_commentary('Data: {"token": "' + body)
        consumer.finish()
        await consumer.run()

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list
        ]
        assert all(body not in payload for payload in payloads)
        assert any('"token": "***"' in payload for payload in payloads)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("initial", "tail", "terminator", "leak_marker", "safe_marker"),
        [
            (
                "OPENAI_API_KEY=opaqueCredentialValue123",
                "UNREDACTEDSUFFIX9876543210",
                " done",
                "UNREDACTEDSUFFIX",
                "OPENAI_API_KEY=***",
            ),
            (
                "Authorization: Bearer opaqueCredentialValue123",
                "UNREDACTEDSUFFIX9876543210",
                " done",
                "UNREDACTEDSUFFIX",
                "Authorization: Bearer ***",
            ),
            (
                "x-api-key: opaqueCredentialValue123",
                "UNREDACTEDSUFFIX9876543210",
                " done",
                "UNREDACTEDSUFFIX",
                "x-api-key: ***",
            ),
            (
                "12345678:" + "A" * 30,
                "UNREDACTEDSUFFIX9876543210",
                " done",
                "UNREDACTEDSUFFIX",
                "12345678:***",
            ),
            (
                "eyJ" + "A" * 12,
                "UNREDACTEDSUFFIX9876543210",
                " done",
                "UNREDACTEDSUFFIX",
                "eyJAAA...3210",
            ),
            (
                "+1234567",
                "89012345",
                " done",
                "8901",
                "+123****2345",
            ),
            (
                "mode=x&token=opaqueCredentialValue123",
                "UNREDACTEDSUFFIX9876543210",
                "",
                "UNREDACTEDSUFFIX",
                "mode=x&token=***",
            ),
        ],
        ids=[
            "env",
            "authorization",
            "api-key-header",
            "telegram",
            "jwt",
            "phone",
            "form-body",
        ],
    )
    async def test_nonprefix_secret_state_survives_later_suffix(
        self,
        monkeypatch,
        initial,
        tail,
        terminator,
        leak_marker,
        safe_marker,
    ):
        """A grammar matched on one tick retains raw state through its delimiter."""
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())

        consumer.on_delta(initial)
        await asyncio.sleep(0.02)
        consumer.on_delta(tail + terminator)
        consumer.finish()
        await task

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list + adapter.edit_message.call_args_list
        ]
        assert all(leak_marker not in payload for payload in payloads)
        assert all(initial + tail not in payload for payload in payloads)
        assert any(safe_marker in payload for payload in payloads), payloads

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text",
        [
            "eyJshort prose",
            "12345678:short prose",
            "+123 nope",
        ],
        ids=["jwt", "telegram", "phone"],
    )
    async def test_nonprefix_candidate_below_minimum_is_released(self, text):
        """Incomplete token-shaped prose remains visible when it diverges."""
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        consumer.on_delta(text)
        consumer.finish()

        await consumer.run()

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list + adapter.edit_message.call_args_list
        ]
        assert text in "".join(payloads)

    @pytest.mark.asyncio
    async def test_jwt_state_survives_partial_dot_segment(self, monkeypatch):
        """A short in-progress JWT payload segment cannot break raw state."""
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())
        header = "eyJ" + "A" * 12

        consumer.on_delta(header)
        await asyncio.sleep(0.02)
        consumer.on_delta(".ab")
        await asyncio.sleep(0.02)
        consumer.on_delta("cd.EFGH done")
        consumer.finish()
        await task

        payloads = [
            call.kwargs["content"]
            for call in adapter.send.call_args_list + adapter.edit_message.call_args_list
        ]
        assert all(header not in payload for payload in payloads)
        assert all(".abcd.EFGH" not in payload for payload in payloads)
        assert any("eyJAAA...EFGH done" in payload for payload in payloads)


class TestStreamRunToolTraceFiltering:
    """Do not seal internal tool-trace fragments during progressive overflow."""

    _VISIBLE_PREFIX = "x" * 490 + "\n"

    @staticmethod
    def _adapter():
        adapter = MagicMock()
        adapter.REQUIRES_EDIT_FINALIZE = False
        adapter.MAX_MESSAGE_LENGTH = 600
        adapter.send = AsyncMock(
            side_effect=lambda **_kwargs: SimpleNamespace(
                success=True,
                message_id=f"msg_{adapter.send.call_count}",
            )
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="preview")
        )
        adapter.truncate_message.side_effect = (
            lambda text, limit, len_fn=len: [text[:limit], text[limit:]]
        )
        return adapter

    @pytest.mark.asyncio
    async def test_split_tool_trace_is_held_before_overflow(self):
        """No incomplete banner fragment becomes a permanent platform write."""
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())

        consumer.on_delta(self._VISIBLE_PREFIX + "⚠️ 🛠️ `search repos")
        for _ in range(100):
            if adapter.send.call_count:
                break
            await asyncio.sleep(0.001)
        assert adapter.send.call_count

        consumer.on_delta(" (agent)` failed")
        consumer.finish()
        await task

        payloads = [
            call.kwargs["content"] for call in adapter.send.call_args_list
        ]
        payloads.extend(
            call.kwargs["content"] for call in adapter.edit_message.call_args_list
        )
        assert all("⚠️" not in payload for payload in payloads)
        assert all("🛠️" not in payload for payload in payloads)
        assert all("search repos" not in payload for payload in payloads)
        assert any(self._VISIBLE_PREFIX.rstrip() in payload for payload in payloads)

    @pytest.mark.asyncio
    async def test_incomplete_marked_prose_is_preserved_at_stream_end(self):
        """An unterminated marker-shaped line remains legitimate visible text."""
        adapter = self._adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        )
        text = self._VISIBLE_PREFIX + "⚠️ 🛠️ `deployment failed unexpectedly"
        consumer.on_delta(text)
        consumer.finish()

        await consumer.run()

        payload = "".join(
            call.kwargs["content"] for call in adapter.send.call_args_list
        )
        assert text == payload


class TestBeforeFinalizeHook:
    """Verify the optional pre-finalize hook fires at the right time."""

    @pytest.mark.asyncio
    async def test_hook_runs_before_finalize_edit(self):
        """Adapters that require finalize should pause typing before the edit."""
        events = []
        adapter = MagicMock()
        adapter.REQUIRES_EDIT_FINALIZE = True
        adapter.send = AsyncMock(
            side_effect=lambda **_kw: (
                events.append("send"),
                SimpleNamespace(success=True, message_id="msg_1"),
            )[1]
        )
        adapter.edit_message = AsyncMock(
            side_effect=lambda **_kw: (
                events.append("edit"),
                SimpleNamespace(success=True, message_id="msg_1"),
            )[1]
        )
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5),
            on_before_finalize=lambda: events.append("pause"),
        )
        consumer.on_delta("Hello")
        consumer.finish()

        await consumer.run()

        assert events == ["send", "pause", "edit"]


# ── Segment break (tool boundary) tests ──────────────────────────────────


class TestSegmentBreakOnToolBoundary:
    """Verify that on_delta(None) finalizes the current message and starts a
    new one so the final response appears below tool-progress messages."""


    @pytest.mark.asyncio
    async def test_segment_break_removes_cursor(self):
        """The finalized segment message should not have a cursor."""
        adapter = MagicMock()
        send_result = SimpleNamespace(success=True, message_id="msg_1")
        edit_result = SimpleNamespace(success=True)
        adapter.send = AsyncMock(return_value=send_result)
        adapter.edit_message = AsyncMock(return_value=edit_result)
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=" ▉")
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        consumer.on_delta("Thinking...")
        consumer.on_delta(None)
        consumer.on_delta("Done.")
        consumer.finish()

        await consumer.run()

        # The first segment should have been finalized without cursor.
        # Check all edit_message calls + the initial send for the first segment.
        # The last state of msg_1 should NOT have the cursor.
        all_texts = []
        for call in adapter.send.call_args_list:
            all_texts.append(call[1].get("content", ""))
        for call in adapter.edit_message.call_args_list:
            all_texts.append(call[1].get("content", ""))

        # Find the text(s) that contain "Thinking" — the finalized version
        # should not have the cursor.
        thinking_texts = [t for t in all_texts if "Thinking" in t]
        assert thinking_texts, "Expected at least one message with 'Thinking'"
        # The LAST occurrence is the finalized version
        assert "▉" not in thinking_texts[-1], (
            f"Cursor found in finalized segment: {thinking_texts[-1]!r}"
        )


    @pytest.mark.asyncio
    async def test_segment_break_clears_failed_edit_fallback_state(self):
        """A tool boundary after edit failure must flush the undelivered tail
        without duplicating the prefix the user already saw (#8124)."""
        adapter = MagicMock()
        send_results = [
            SimpleNamespace(success=True, message_id="msg_1"),
            SimpleNamespace(success=True, message_id="msg_2"),
            SimpleNamespace(success=True, message_id="msg_3"),
        ]
        adapter.send = AsyncMock(side_effect=send_results)
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=False, error="flood_control:6"))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=" ▉")
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        consumer.on_delta("Hello")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.08)
        consumer.on_delta(" world")
        await asyncio.sleep(0.08)
        consumer.on_delta(None)
        consumer.on_delta("Next segment")
        consumer.finish()
        await task

        sent_texts = [call[1]["content"] for call in adapter.send.call_args_list]
        # The undelivered "world" tail must reach the user, and the next
        # segment must not duplicate "Hello" that was already visible.
        assert sent_texts == ["Hello ▉", "world", "Next segment"]

    @pytest.mark.asyncio
    async def test_segment_break_after_mid_stream_edit_failure_preserves_tail(self):
        """Regression for #8124: when an earlier edit succeeded but later edits
        fail (persistent flood control) and a tool boundary arrives before the
        fallback threshold is reached, the pre-boundary tail must still be
        delivered — not silently dropped by the segment reset."""
        adapter = MagicMock()
        # msg_1 for the initial partial, msg_2 for the flushed tail,
        # msg_3 for the post-boundary segment.
        send_results = [
            SimpleNamespace(success=True, message_id="msg_1"),
            SimpleNamespace(success=True, message_id="msg_2"),
            SimpleNamespace(success=True, message_id="msg_3"),
        ]
        adapter.send = AsyncMock(side_effect=send_results)

        # First two edits succeed, everything after fails with flood control
        # — simulating Telegram's "edit once then get rate-limited" pattern.
        edit_results = [
            SimpleNamespace(success=True),   # "Hello world ▉"  — succeeds
            SimpleNamespace(success=False, error="flood_control:6.0"),  # "Hello world more ▉" — flood triggered
            SimpleNamespace(success=False, error="flood_control:6.0"),  # finalize edit at segment break
            SimpleNamespace(success=False, error="flood_control:6.0"),  # cursor-strip attempt
        ]
        adapter.edit_message = AsyncMock(side_effect=edit_results + [edit_results[-1]] * 10)
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=" ▉")
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        consumer.on_delta("Hello")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.08)
        consumer.on_delta(" world")
        await asyncio.sleep(0.08)
        consumer.on_delta(" more")
        await asyncio.sleep(0.08)
        consumer.on_delta(None)  # tool boundary
        consumer.on_delta("Here is the tool result.")
        consumer.finish()
        await task

        sent_texts = [call[1]["content"] for call in adapter.send.call_args_list]
        # "more" must have been delivered, not dropped.
        all_text = " ".join(sent_texts)
        assert "more" in all_text, (
            f"Pre-boundary tail 'more' was silently dropped: sends={sent_texts}"
        )
        # Post-boundary text must also reach the user.
        assert "Here is the tool result." in all_text


    @pytest.mark.asyncio
    async def test_fallback_final_sends_full_text_at_tool_boundary(self):
        """After a tool call, the streamed prefix is stale (from the pre-tool
        segment).  _send_fallback_final must still send the post-tool response
        even when continuation_text calculates as empty (#10807)."""
        adapter = MagicMock()
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_1"),
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True),
        )
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Simulate a pre-tool streamed segment that becomes the visible prefix
        pre_tool_text = "I'll run that code now."
        consumer.on_delta(pre_tool_text)
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)

        # After the tool call, the model returns a SHORT final response that
        # does NOT start with the pre-tool prefix.  The continuation calculator
        # would return empty (no prefix match → full text returned, but if the
        # streaming edit already showed pre_tool_text, the prefix-based logic
        # wrongly matches).  Simulate this by setting _last_sent_text to the
        # pre-tool content, then finishing with different post-tool content.
        consumer._last_sent_text = pre_tool_text
        post_tool_response = "⏰ Script timed out after 30s and was killed."
        consumer.finish()
        await task

        # The fallback should send the post-tool response via
        # _send_fallback_final.
        await consumer._send_fallback_final(post_tool_response)

        # Verify the final text was sent (not silently dropped)
        sent = False
        for call in adapter.send.call_args_list:
            content = call[1].get("content", call[0][0] if call[0] else "")
            if "timed out" in str(content):
                sent = True
                break
        assert sent, (
            "Post-tool timeout response was silently dropped by "
            "_send_fallback_final — the #10807 fix should prevent this"
        )

    @pytest.mark.asyncio
    async def test_fallback_final_deletes_partial_after_full_resend(self):
        """After fallback re-sends the COMPLETE response, the frozen partial
        must be deleted so the user sees only the complete response (#16668).
        Full resend happens when the visible prefix doesn't match the final
        text (e.g. post-segment-break content, #10807)."""
        adapter = MagicMock()
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_new"),
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True),
        )
        adapter.delete_message = AsyncMock(return_value=None)
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # The stale partial shows pre-tool text that is NOT a prefix of the
        # final response — fallback re-sends the complete final text.
        consumer._message_id = "msg_partial"
        consumer._last_sent_text = "Let me check that for you…"

        await consumer._send_fallback_final("Working on it. Done!")

        adapter.delete_message.assert_awaited_once_with("chat_123", "msg_partial")
        assert consumer._final_response_sent is True

    @pytest.mark.asyncio
    async def test_fallback_final_keeps_partial_after_tail_only_send(self):
        """When the fallback sends only the missing TAIL (visible prefix
        matches the final text), the partial message IS the head of the
        answer — deleting it would leave the user with only the last part
        of the response (the 'model sent only the second half' bug)."""
        adapter = MagicMock()
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_new"),
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True),
        )
        adapter.delete_message = AsyncMock(return_value=None)
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Visible partial is a true prefix of the final response — the
        # fallback dedup sends only the tail.
        consumer._message_id = "msg_partial"
        consumer._last_sent_text = "Working on i"

        await consumer._send_fallback_final("Working on it. Done!")

        # Tail was sent...
        sent_contents = [
            c.kwargs.get("content", "") for c in adapter.send.call_args_list
        ]
        assert any("Done!" in s and "Working on i" not in s for s in sent_contents)
        # ...and the head-bearing partial was NOT deleted.
        adapter.delete_message.assert_not_awaited()
        assert consumer._final_response_sent is True


class TestFinalResponseDeliveryGuard:
    """Regression coverage for #10748 — _final_response_sent must reflect
    actual delivery of the *current* chunked send, not the cumulative
    `_already_sent` flag (which earlier tool-progress edits or fallback-mode
    promotion can taint)."""

    @pytest.mark.asyncio
    async def test_split_overflow_failed_send_does_not_mark_final_sent(self):
        """Split-overflow path: if every chunk send fails on done frame,
        _final_response_sent must stay False so the gateway falls back."""
        adapter = MagicMock()
        # Every send fails — _send_new_chunk returns the passed-in reply_to.
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=False, error="network down"),
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True),
        )
        adapter.MAX_MESSAGE_LENGTH = 100
        adapter.truncate_message = MagicMock(
            side_effect=lambda text, limit: [text[:limit], text[limit:]],
        )

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Simulate prior tool-progress edits that set _already_sent
        consumer._already_sent = True

        # Long text > MAX_MESSAGE_LENGTH, no existing message id (fresh send path)
        long_text = "x" * 200
        consumer.on_delta(long_text)
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        assert consumer._final_response_sent is False, (
            "_already_sent leaked into _final_response_sent — gateway will "
            "wrongly suppress its fallback delivery (#10748)"
        )


class TestFinalContentDeliveredGuard:
    """Regression coverage for #25010 — _final_content_delivered must only be
    set when the final response is actually confirmed delivered to the user,
    not when a mid-stream edit happened to show partial content.  Prematurely
    setting this flag causes the gateway to suppress the normal final send,
    leaving the user with an incomplete partial message."""

    @pytest.mark.asyncio
    async def test_mid_stream_edit_success_does_not_mark_content_delivered(self):
        """When the mid-stream edit with finalize=True succeeds but the
        subsequent finalize edit fails, _final_content_delivered must stay
        False so the gateway does not suppress its fallback send (#25010).

        Simulates TelegramAdapter which sets REQUIRES_EDIT_FINALIZE=True,
        requiring a second finalize edit even when content is unchanged."""
        adapter = MagicMock()
        adapter.REQUIRES_EDIT_FINALIZE = True  # Telegram adapter behavior
        # First send (initial streaming message) succeeds.
        # Mid-stream edit succeeds.
        # Final finalize edit fails, and the consumer's own fallback send also
        # fails, so no path has confirmed the complete final response reached
        # the user.
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=False))
        adapter.send = AsyncMock(side_effect=[
            SimpleNamespace(success=True, message_id="msg_1"),
            SimpleNamespace(success=False, error="network down"),
        ])
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Simulate streaming: send initial text, then more text, then done
        consumer.on_delta("Part one of the response...\n")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        # Keep the second delta buffered until finish so the complete answer is
        # not already visible before the final edit attempt fails.
        consumer.cfg.buffer_threshold = 10_000
        consumer._current_edit_interval = 10.0

        consumer.on_delta("Part two, the complete final answer.\n")
        await asyncio.sleep(0.05)

        consumer.finish()
        await task

        # The key assertion: _final_content_delivered must NOT be True,
        # because the final edit failed and the complete response was never
        # confirmed delivered.
        assert consumer._final_content_delivered is False, (
            "_final_content_delivered was prematurely set to True — gateway "
            "will wrongly suppress its fallback send, leaving the user with "
            "an incomplete partial message (#25010)"
        )
        # The gateway must still be allowed to send the complete response
        assert consumer._final_response_sent is False, (
            "_final_response_sent must also be False when the final edit failed"
        )


    @pytest.mark.asyncio
    async def test_fallback_partial_send_does_not_mark_final_sent(self):
        """When fallback final send delivers only some chunks before failing,
        _final_response_sent must stay False so the gateway can still attempt
        a complete final send (#25010)."""
        call_count = 0

        async def fake_send(*, chat_id, content, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return SimpleNamespace(success=True, message_id="msg_1")
            # Third chunk (fallback continuation) FAILS
            return SimpleNamespace(success=False, error="flood_control:13.0")

        adapter = MagicMock()
        adapter.send = AsyncMock(side_effect=fake_send)
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=False, error="flood_control:13.0"),
        )
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Trigger enough delta to enter fallback mode
        consumer.on_delta("Initial streaming text...\n")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)

        # Send a very long text that will trigger overflow/fallback
        long_text = ("x" * 3000 + "\n") + ("y" * 3000 + "\n") + "Final answer.\n"
        consumer.on_delta(long_text)
        await asyncio.sleep(0.1)

        consumer.finish()
        await task

        assert consumer._final_response_sent is False, (
            "Partial fallback send must not set _final_response_sent — gateway "
            "must still be able to deliver the complete response (#25010)"
        )


class TestInitialOverflowRollingEdit:
    @pytest.mark.asyncio
    async def test_initial_overflow_keeps_last_chunk_as_edit_target(self):
        """When the first visible flush already overflows, only sealed head
        chunks should be posted as fixed messages.  The trailing chunk must
        remain the active edit target so later streamed deltas update that
        second message instead of overwriting or posting a new one."""
        adapter = MagicMock()
        msg_ids = iter(["msg_1", "msg_2"])
        adapter.send = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                success=True,
                message_id=next(msg_ids),
            )
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_2"),
        )
        adapter.MAX_MESSAGE_LENGTH = 700

        config = StreamConsumerConfig(
            edit_interval=0.01,
            buffer_threshold=5,
            cursor=" ▉",
        )
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        head = "x " * 325
        tail = "B" * 25
        consumer.on_delta(head)
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.08)
        consumer.on_delta(tail)
        await asyncio.sleep(0.08)
        consumer.finish()
        await task

        assert adapter.send.call_count == 2
        assert adapter.edit_message.call_count >= 1
        edited_texts = [call.kwargs["content"] for call in adapter.edit_message.call_args_list]
        assert any("x " * 10 in text and tail in text for text in edited_texts), (
            "the second overflow chunk should be edited with its existing tail "
            "plus later deltas, not overwritten by only the later delta"
        )
        assert consumer.final_response_sent is True

    @pytest.mark.asyncio
    async def test_initial_overflow_uses_adapter_fence_aware_split(self):
        """Initial rolling sends must preserve the adapter's fence contract."""
        adapter = TestUtf16OverflowDetection()._make_telegram_like_adapter()
        from gateway.platforms.base import utf16_len

        msg_ids = iter(["msg_1", "msg_2", "msg_3"])
        adapter.send = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                success=True,
                message_id=next(msg_ids),
            )
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_3"),
        )
        raw_limit = 700
        setattr(adapter, "MAX_MESSAGE_LENGTH", raw_limit)
        splitter = MagicMock(side_effect=adapter.truncate_message)
        adapter.truncate_message = splitter

        config = StreamConsumerConfig(
            edit_interval=0.01,
            buffer_threshold=5,
            cursor=" ▉",
        )
        consumer = GatewayStreamConsumer(adapter, "chat_fenced", config)
        fenced = "```python\n" + ("print('x')\n" * 100) + "```"
        safe_limit = raw_limit - utf16_len(config.cursor) - 100
        expected_chunks = adapter.truncate_message(
            fenced, safe_limit, len_fn=adapter.message_len_fn,
        )
        splitter.reset_mock()

        consumer.on_delta(fenced)
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.08)
        consumer.on_delta("\nTail after the fenced stream.")
        await asyncio.sleep(0.08)
        consumer.finish()
        await task

        sent_texts = [call.kwargs["content"] for call in adapter.send.call_args_list]
        edited_texts = [call.kwargs["content"] for call in adapter.edit_message.call_args_list]
        assert splitter.call_count >= 1
        assert all(text.count("```") % 2 == 0 for text in sent_texts + edited_texts)
        assert len(sent_texts) == len(expected_chunks)
        assert sent_texts[:-1] == expected_chunks[:-1]
        assert sent_texts[-1].startswith(expected_chunks[-1])
        assert any("Tail after the fenced stream." in text for text in edited_texts)
        assert all(utf16_len(text) <= safe_limit for text in sent_texts)


class TestEditOverflowSplitAndDeliver:
    """When edit_message split-and-delivers an oversized payload across the
    original message + N continuations (Telegram >4096 UTF-16), the consumer
    must update _message_id to the latest continuation, reset _last_sent_text,
    and fire on_new_message so subsequent tool-progress bubbles linearize
    below the new visible message."""

    @pytest.mark.asyncio
    async def test_consumer_advances_message_id_on_split_and_deliver(self):
        adapter = MagicMock()
        # Simulate edit_message split-and-deliver: success=True with the
        # final continuation's id and a populated continuation_message_ids
        # tuple (the new SendResult contract).
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(
            success=True,
            message_id="msg_continuation_2",
            continuation_message_ids=("msg_continuation_1", "msg_continuation_2"),
        ))
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_initial"),
        )
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(
            edit_interval=0.01, buffer_threshold=5, cursor="",
        )
        consumer = GatewayStreamConsumer(adapter, "chat_999", config)

        # Track on_new_message firings.
        new_msg_count = [0]
        consumer._on_new_message = lambda: new_msg_count.__setitem__(0, new_msg_count[0] + 1)

        # Seed the consumer as if a first send succeeded already.
        consumer._message_id = "msg_initial"
        consumer._last_sent_text = "old"
        consumer._already_sent = True

        # Drive an edit that the adapter "split and delivers".
        ok = await consumer._send_or_edit("new full text after overflow")

        assert ok is True
        # Consumer advanced to the latest continuation id.
        assert consumer._message_id == "msg_continuation_2"
        # Skip-if-same cache reset so the next edit doesn't false-positive.
        assert consumer._last_sent_text == ""
        # on_new_message fired so the tool-progress bubble breaks below
        # the new continuation (per the openclaw #32535 lesson).
        assert new_msg_count[0] == 1


class TestInterimCommentaryMessages:
    @pytest.mark.asyncio
    async def test_commentary_message_stays_separate_from_final_stream(self):
        adapter = MagicMock()
        adapter.send = AsyncMock(side_effect=[
            SimpleNamespace(success=True, message_id="msg_1"),
            SimpleNamespace(success=True, message_id="msg_2"),
        ])
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5),
        )

        consumer.on_commentary("I'll inspect the repository first.")
        consumer.on_delta("Done.")
        consumer.finish()

        await consumer.run()

        sent_texts = [call[1]["content"] for call in adapter.send.call_args_list]
        assert sent_texts == ["I'll inspect the repository first.", "Done."]
        assert consumer.final_response_sent is True


class TestCancelledConsumerSetsFlags:
    """Cancellation must set final_response_sent when already_sent is True.

    The 5-second stream_task timeout in gateway/run.py can cancel the
    consumer while it's still processing.  If final_response_sent stays
    False, the gateway falls through to the normal send path and the
    user sees a duplicate message.
    """

    @pytest.mark.asyncio
    async def test_cancelled_with_already_sent_marks_final_response_sent(self):
        """Cancelling after content was sent should set final_response_sent."""
        adapter = MagicMock()
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_1")
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True)
        )
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5),
        )

        # Stream some text — the consumer sends it and sets already_sent
        consumer.on_delta("Hello world")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.08)

        assert consumer.already_sent is True

        # Cancel the task (simulates the 5-second timeout in gateway)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # The fix: final_response_sent should be True even though _DONE
        # was never processed, preventing a duplicate message.
        assert consumer.final_response_sent is True


# ── Think-block filtering unit tests ─────────────────────────────────────


def _make_consumer() -> GatewayStreamConsumer:
    """Create a bare consumer for unit-testing the filter (no adapter needed)."""
    adapter = MagicMock()
    return GatewayStreamConsumer(adapter, "chat_test")


class TestFilterAndAccumulate:
    """Unit tests for _filter_and_accumulate think-block suppression."""

    def test_plain_text_passes_through(self):
        c = _make_consumer()
        c._filter_and_accumulate("Hello world")
        assert c._accumulated == "Hello world"

    def test_complete_think_block_stripped(self):
        c = _make_consumer()
        c._filter_and_accumulate("<think>internal reasoning</think>Answer here")
        assert c._accumulated == "Answer here"


    def test_opening_tag_split_across_deltas(self):
        c = _make_consumer()
        c._filter_and_accumulate("<thi")
        # Partial tag held back
        assert c._accumulated == ""
        c._filter_and_accumulate("nk>hidden</think>shown")
        assert c._accumulated == "shown"


    def test_multiple_think_blocks_with_text_between(self):
        """Think tag after non-whitespace is NOT a boundary (prose safety)."""
        c = _make_consumer()
        c._filter_and_accumulate(
            "<think>block1</think>A<think>block2</think>B"
        )
        # Second <think> follows 'A' (not a block boundary) — treated as prose
        assert "A" in c._accumulated
        assert "B" in c._accumulated


    @pytest.mark.parametrize(
        "tag",
        ["THINK", "Think", "ThInK", "THOUGHT", "REASONING", "Thinking"],
    )
    def test_reasoning_tags_are_case_insensitive(self, tag):
        c = _make_consumer()
        c._filter_and_accumulate(f"<{tag}>hidden reasoning</{tag}>Visible answer")
        assert c._accumulated == "Visible answer"
        assert "hidden reasoning" not in c._accumulated

    def test_prose_mention_not_stripped(self):
        """<think> mentioned mid-line in prose should NOT trigger filtering."""
        c = _make_consumer()
        c._filter_and_accumulate("The <think> tag is used for reasoning")
        assert "<think>" in c._accumulated
        assert "used for reasoning" in c._accumulated


    def test_think_with_only_whitespace_before(self):
        """<think> preceded by only whitespace on its line is a boundary."""
        c = _make_consumer()
        c._filter_and_accumulate("  <think>hidden</think>visible")
        # Leading whitespace before the tag is emitted, then block is stripped
        assert c._accumulated == "  visible"

    def test_flush_think_buffer_on_non_tag(self):
        """Partial tag that turns out not to be a tag is flushed."""
        c = _make_consumer()
        c._filter_and_accumulate("<thi")
        assert c._accumulated == ""
        # Flush explicitly (simulates stream end)
        c._flush_think_buffer()
        assert c._accumulated == "<thi"


class TestFilterAndAccumulateIntegration:
    """Integration: verify think blocks don't leak through the full run() path."""

    @pytest.mark.asyncio
    async def test_think_block_not_sent_to_platform(self):
        """Think blocks should be filtered before platform edit."""
        adapter = MagicMock()
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_1")
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True)
        )
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(
            adapter,
            "chat_test",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5),
        )

        # Simulate streaming: think block then visible text
        consumer.on_delta("<think>deep reasoning here</think>")
        consumer.on_delta("The answer is 42.")
        consumer.finish()

        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.15)

        # The final text sent to the platform should NOT contain <think>
        all_calls = list(adapter.send.call_args_list) + list(
            adapter.edit_message.call_args_list
        )
        for call in all_calls:
            args, kwargs = call
            content = kwargs.get("content") or (args[0] if args else "")
            assert "<think>" not in content, f"Think tag leaked: {content}"
            assert "deep reasoning" not in content

        try:
            task.cancel()
            await task
        except asyncio.CancelledError:
            pass


# ── buffer_only mode tests ─────────────────────────────────────────────


class TestBufferOnlyMode:
    """Verify buffer_only mode suppresses intermediate edits and only
    flushes on structural boundaries (done, segment break, commentary)."""


    @pytest.mark.asyncio
    async def test_flushes_on_segment_break(self):
        """A segment break (tool call boundary) flushes accumulated text."""
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.send = AsyncMock(side_effect=[
            SimpleNamespace(success=True, message_id="msg1"),
            SimpleNamespace(success=True, message_id="msg2"),
        ])
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))

        cfg = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor="", buffer_only=True)
        consumer = GatewayStreamConsumer(adapter, "!room:server", config=cfg)

        consumer.on_delta("Before tool call")
        consumer.on_delta(None)
        consumer.on_delta("After tool call")
        consumer.finish()

        await consumer.run()

        assert adapter.send.call_count == 2
        assert "Before tool call" in adapter.send.call_args_list[0][1]["content"]
        assert "After tool call" in adapter.send.call_args_list[1][1]["content"]
        adapter.edit_message.assert_not_called()


# ── Cursor stripping on fallback (#7183) ────────────────────────────────────


class TestCursorStrippingOnFallback:
    """Regression: cursor must be stripped when fallback continuation is empty (#7183).

    When _send_fallback_final is called with nothing new to deliver (the visible
    partial already matches final_text), the last edit may still show the cursor
    character because fallback mode was entered after a failed edit.  Before the
    fix this would leave the message permanently frozen with a visible ▉.
    """

    @pytest.mark.asyncio
    async def test_cursor_stripped_when_continuation_empty(self):
        """_send_fallback_final must attempt a final edit to strip the cursor."""
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg-1")
        )

        consumer = GatewayStreamConsumer(
            adapter, "chat-1",
            config=StreamConsumerConfig(cursor=" ▉"),
        )
        consumer._message_id = "msg-1"
        consumer._last_sent_text = "Hello world ▉"
        consumer._fallback_final_send = False

        await consumer._send_fallback_final("Hello world")

        adapter.edit_message.assert_called_once()
        call_args = adapter.edit_message.call_args
        assert call_args.kwargs["content"] == "Hello world"
        assert consumer._already_sent is True
        # _last_sent_text should reflect the cleaned text after a successful strip
        assert consumer._last_sent_text == "Hello world"


# ── on_new_message callback (tool-progress linearization) ─────────────


class TestOnNewMessageCallback:
    """The on_new_message callback fires whenever a fresh content bubble
    lands on the platform. Gateway uses this to close off the current
    tool-progress bubble so the next tool.started opens a new bubble
    below the content — preserving chronological order in the chat.

    Before this callback existed (post PR #7885), content messages got
    their own bubbles after segment breaks, but the tool-progress task
    kept editing the ORIGINAL progress bubble above all new content.
    Result: tool lines appeared stacked in the upper bubble while
    content messages lined up below, making the timeline look scrambled.
    """


    @pytest.mark.asyncio
    async def test_callback_fires_once_per_segment(self):
        """A new first-send fires the callback again after segment break."""
        adapter = MagicMock()
        msg_counter = iter(["msg_1", "msg_2", "msg_3"])
        adapter.send = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(success=True, message_id=next(msg_counter))
        )
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        events = []
        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1)
        consumer = GatewayStreamConsumer(
            adapter, "chat", config,
            on_new_message=lambda: events.append("reset"),
        )

        consumer.on_delta("A")
        consumer.on_delta(None)
        consumer.on_delta("B")
        consumer.on_delta(None)
        consumer.on_delta("C")
        consumer.finish()
        await consumer.run()

        # Three content bubbles ⇒ three reset notifications
        assert events == ["reset", "reset", "reset"]


    @pytest.mark.asyncio
    async def test_no_callback_when_none(self):
        """Consumer works correctly when on_new_message is None (default)."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1)
        consumer = GatewayStreamConsumer(adapter, "chat", config)  # no callback

        consumer.on_delta("Hello")
        consumer.finish()
        await consumer.run()

        assert consumer.already_sent is True


class TestUtf16OverflowDetection:
    """Regression coverage for #11170 — Telegram counts message length in
    UTF-16 code units, not Python codepoints. A response with supplementary
    characters (emoji, CJK in some ranges) can have len()=3000 codepoints
    but utf16_len()=5000+ units, blowing past Telegram's 4096 limit."""

    def _make_telegram_like_adapter(self):
        """Construct a minimal BasePlatformAdapter subclass that overrides
        message_len_fn like Telegram does."""
        from gateway.platforms.base import utf16_len, BasePlatformAdapter

        TelegramLikeAdapter = type(
            "TelegramLikeAdapter",
            (BasePlatformAdapter,),
            {
                "MAX_MESSAGE_LENGTH": 4096,
                "message_len_fn": property(lambda self: utf16_len),
            },
        )
        # Defeat ABCMeta abstract-instantiation guard by clearing the cached
        # abstract methods set after class creation.
        TelegramLikeAdapter.__abstractmethods__ = frozenset()
        adapter = TelegramLikeAdapter.__new__(TelegramLikeAdapter)
        adapter._typing_paused = set()
        adapter._fatal_error_message = None
        return adapter

    @pytest.mark.asyncio
    async def test_emoji_text_exceeding_utf16_limit_triggers_overflow_split(self):
        """A response that is under 4096 codepoints but over 4096 UTF-16
        units must trigger the overflow-split path."""
        from gateway.platforms.base import utf16_len

        adapter = self._make_telegram_like_adapter()
        # Mock the send/edit methods we actually call
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_1"),
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True),
        )
        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # 🚀 is 1 codepoint = 2 UTF-16 units. 2200 of them = 2200 codepoints,
        # 4400 UTF-16 units. Under the codepoint-equivalent limit (would not
        # trigger split with len()) but over Telegram's UTF-16 4096 limit.
        emoji_text = "🚀" * 2200
        assert len(emoji_text) < adapter.MAX_MESSAGE_LENGTH, (
            "Test setup invariant: codepoint count under limit"
        )
        assert utf16_len(emoji_text) > adapter.MAX_MESSAGE_LENGTH, (
            "Test setup invariant: UTF-16 count over limit"
        )

        consumer.on_delta(emoji_text)
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        # The fix: stream consumer detects UTF-16 overflow using the adapter's
        # length function.  Without that, len() would return 2200 (under the
        # limit) and Hermes would attempt a single over-limit Telegram send.
        sent_texts = [call.kwargs["content"] for call in adapter.send.call_args_list]
        assert len(sent_texts) == 2, (
            "UTF-16 overflow not detected — emoji text bypassed split path"
        )
        max_units = 4096
        assert all(utf16_len(text) <= max_units for text in sent_texts), (
            f"split chunks still exceed Telegram UTF-16 limit: "
            f"{[utf16_len(text) for text in sent_texts]}"
        )


class TestFreshFinalRespectsAdapterDecline:
    """Regression: when an adapter explicitly declines fresh-final via
    ``prefers_fresh_final_streaming = False``, the time-based
    ``_should_send_fresh_final()`` must NOT override that decision.
    (#47048 — Telegram rich-message overlap with legacy MarkdownV2 preview)
    """

    @pytest.mark.asyncio
    async def test_adapter_decline_fresh_final_overrides_time_threshold(self):
        """Adapter with prefers_fresh_final_streaming=False must NOT take
        the fresh-final path even when fresh_final_after_seconds is large."""
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="rich_msg"),
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="edit_msg"),
        )
        adapter.delete_message = AsyncMock(return_value=True)
        # Adapter explicitly declines fresh-final (like Telegram)
        adapter.prefers_fresh_final_streaming = MagicMock(return_value=False)

        config = StreamConsumerConfig(
            edit_interval=0.01,
            buffer_threshold=5,
            fresh_final_after_seconds=1.0,  # time threshold would trigger
            cursor=" ▉",
        )
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Simulate: first message sent during streaming
        consumer.on_delta("Hello world")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        # First message should have been sent
        assert consumer._message_id is not None
        # Simulate time passing (beyond threshold)
        consumer._message_created_ts -= 10.0

        # Finalize
        consumer.on_delta("Hello world final")
        consumer.finish()
        await task

        # The adapter declined fresh-final, so send() should NOT have been
        # called for the final message — only edit_message(finalize=True).
        adapter.send.assert_called_once()  # Only the initial send
        adapter.edit_message.assert_called()  # Finalize edit
        # Verify edit was called with finalize=True
        edit_calls = [
            c for c in adapter.edit_message.call_args_list
            if c.kwargs.get("finalize") or (len(c.args) > 3 and c.args[3])
        ]
        assert len(edit_calls) >= 1, (
            "Expected finalize=True edit call, got none"
        )


# ── run_still_current staleness guard ────────────────────────────────────

class TestRunStillCurrentGuard:
    """Verify that the stream consumer abandons delivery when the session is
    reset (e.g. /new or /stop), preventing stale deltas from reaching the user."""


    @pytest.mark.asyncio
    async def test_abandons_stream_after_one_edit_when_session_reset(self):
        """If staleness flips after the first edit, the consumer stops
        on the next loop iteration and does not send the final response."""
        adapter = MagicMock()
        send_result = SimpleNamespace(success=True, message_id="msg_1")
        adapter.send = AsyncMock(return_value=send_result)
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        call_count = [0]

        def is_current():
            call_count[0] += 1
            return call_count[0] == 1

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=3)
        consumer = GatewayStreamConsumer(
            adapter, "chat_123", config,
            run_still_current=is_current,
        )

        consumer.on_delta("First segment")
        consumer.on_delta(None)  # segment break → resets message_id
        consumer.on_delta("Second segment text that will be stale")
        # No finish() — staleness should prevent second segment from sending

        await consumer.run()

        # First segment was sent, second was abandoned
        assert adapter.send.call_count == 1
        assert "First segment" in adapter.send.call_args_list[0][1]["content"]
        assert consumer._final_response_sent is False

    @pytest.mark.asyncio
    async def test_normal_delivery_when_session_stays_current(self):
        """When _run_still_current always returns True, the consumer
        behaves normally and delivers the full response."""
        adapter = MagicMock()
        send_result = SimpleNamespace(success=True, message_id="msg_1")
        adapter.send = AsyncMock(return_value=send_result)
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(
            adapter, "chat_123", config,
            run_still_current=lambda: True,
        )

        consumer.on_delta("Hello, world!")
        consumer.finish()

        await consumer.run()

        assert adapter.send.call_count >= 1
        assert consumer._final_response_sent is True


# ── _strip_orphan_close_tags regression tests ──────────────────────────
# Regression guard for the /think tag leak: when the stream consumer is
# NOT inside a think block, stray close tags like </think> must be
# stripped before text is accumulated — otherwise they leak to Telegram.
# (Reported by Tony on 2026-06-09.)


class TestStripOrphanCloseTags:
    """Verify orphan close tags are stripped from text the stream consumer
    would accumulate while NOT inside a think block."""

    @pytest.mark.parametrize(
        "tag",
        [
            "</think>",
            "</thinking>",
            "</thought>",
            "</reasoning>",
            "</REASONING_SCRATCHPAD>",
            "</THINKING>",
        ],
    )
    def test_all_close_tag_variants_stripped(self, tag):
        text = f"before{tag}after"
        result = GatewayStreamConsumer._strip_orphan_close_tags(text)
        assert tag not in result
        assert "before" in result and "after" in result

    def test_no_close_tag_passthrough(self):
        text = "Just normal text with no tags."
        assert GatewayStreamConsumer._strip_orphan_close_tags(text) == text

    def test_empty_string(self):
        assert GatewayStreamConsumer._strip_orphan_close_tags("") == ""


class TestHasDeliveredTextAfterSegmentBreak:
    """has_delivered_text must find a delivered segment after a segment break,
    but must not claim text from a failed delivery. (#65919 review)"""

    def test_finds_delivered_segment_after_segment_break(self):
        """A successfully delivered segment must still be found by
        has_delivered_text after _reset_segment_state runs."""
        c = _make_consumer()
        # Simulate a successfully delivered segment
        c._last_sent_text = "Here is the first segment"
        c._reset_segment_state()
        # After the reset, has_delivered_text must still find it
        assert c.has_delivered_text("Here is the first segment") is True


# ── Flush barrier (clarify-ordering) tests ───────────────────────────────


class TestFlushPendingSync:
    """flush_pending_sync() is the ordering barrier that guarantees buffered
    prose lands on the platform BEFORE a blocking interactive prompt (clarify
    poll) is sent. Regression coverage for the bug where the poll raced ahead
    of its own explanation, rendering the question above the prose.
    """

    @pytest.mark.asyncio
    async def test_flush_delivers_buffered_commentary_before_returning(self):
        """A commentary message queued before the flush barrier must be sent
        to the adapter before flush_pending_sync() returns True."""
        adapter = MagicMock()
        sent_order = []

        async def _send(*args, **kwargs):
            sent_order.append(("send", kwargs.get("content", "")))
            return SimpleNamespace(success=True, message_id="msg_1")

        async def _edit(*args, **kwargs):
            sent_order.append(("edit", kwargs.get("content", "")))
            return SimpleNamespace(success=True)

        adapter.send = AsyncMock(side_effect=_send)
        adapter.edit_message = AsyncMock(side_effect=_edit)
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Mirror the live config that exhibits the bug: streaming off but
        # interim assistant messages on, so prose arrives as commentary.
        consumer.on_commentary("Now I get the full picture. Here's the situation.")

        # Run the drain loop concurrently while we block on the flush barrier
        # from a worker thread (mirrors the agent thread calling the clarify
        # callback while the consumer task drains on the event loop).
        task = asyncio.create_task(consumer.run())
        flushed = await asyncio.to_thread(consumer.flush_pending_sync, 3.0)

        assert flushed is True, "flush_pending_sync should complete within timeout"
        # The commentary must already be on screen by the time flush returns.
        assert any(
            "full picture" in content for _kind, content in sent_order
        ), f"Commentary not delivered before flush returned: {sent_order!r}"

        consumer.finish()
        await task


    @pytest.mark.asyncio
    async def test_flush_completes_on_oversized_buffered_prose(self):
        """Regression: oversized prose takes the overflow-split `continue`
        path in run(), which previously skipped the flush-event set, stalling
        the caller for the full timeout. The flush must still complete promptly
        and the prose must be delivered before it returns."""
        adapter = MagicMock()
        sent = []

        async def _send(*args, **kwargs):
            sent.append(kwargs.get("content", ""))
            return SimpleNamespace(success=True, message_id=f"m{len(sent)}")

        adapter.send = AsyncMock(side_effect=_send)
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096
        # Real splitter so the overflow branch (message_id is None +
        # len > safe_limit) is actually taken.
        adapter.truncate_message = (
            lambda text, limit, len_fn=len: [
                text[i:i + limit] for i in range(0, len(text), limit)
            ]
        )

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Commentary far larger than the platform limit → overflow split path.
        big = "X" * 9000
        consumer.on_commentary(big)

        task = asyncio.create_task(consumer.run())
        # Tight timeout: if the continue-path skips the set, this returns False.
        flushed = await asyncio.to_thread(consumer.flush_pending_sync, 2.0)

        assert flushed is True, (
            "flush stalled — overflow `continue` path did not signal the barrier"
        )
        assert sent, "oversized prose was not delivered before flush returned"
        assert sum(len(c) for c in sent) >= 9000

        consumer.finish()
        await task
