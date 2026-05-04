"""Tests for audio routing on the API server's /v1/chat/completions endpoint.

Mirrors the image-routing test pattern: mock decide_audio_input_mode to
control routing, mock transcribe_audio for transcription, and verify that
the user_message passed to _run_agent is correctly enriched or passed through.
"""

import base64
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _AUDIO_PART_TYPES,
    _content_has_visible_payload,
    _extract_audio_from_content,
    _normalize_multimodal_content,
)


# ---------------------------------------------------------------------------
# Helpers — mirror test_api_server_multimodal.py
# ---------------------------------------------------------------------------


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    from gateway.platforms.api_server import cors_middleware, security_headers_middleware

    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


def _b64_audio() -> str:
    """Return a tiny valid base64 payload (not actual audio, but valid shape)."""
    return base64.b64encode(b"\x00\x01\x02\x03").decode("ascii")


def _audio_content_parts() -> list:
    """Return multimodal content with one input_audio part + text."""
    return [
        {"type": "text", "text": "Transcribe this"},
        {
            "type": "input_audio",
            "input_audio": {"data": _b64_audio(), "format": "wav"},
        },
    ]


# ---------------------------------------------------------------------------
# Unit tests for _extract_audio_from_content
# ---------------------------------------------------------------------------


class TestExtractAudioFromContent:
    def test_extracts_base64_audio(self):
        content = _audio_content_parts()
        items = _extract_audio_from_content(content)
        assert len(items) == 1
        assert items[0]["source"] == "base64"
        assert items[0]["data"] == _b64_audio()
        assert items[0]["format"] == "wav"

    def test_empty_list_when_no_audio(self):
        content = [{"type": "text", "text": "hello"}]
        assert _extract_audio_from_content(content) == []

    def test_coerces_invalid_format_to_wav(self):
        content = [
            {
                "type": "input_audio",
                "input_audio": {"data": _b64_audio(), "format": "flac"},
            },
        ]
        items = _extract_audio_from_content(content)
        assert items[0]["format"] == "wav"

    def test_skips_empty_data(self):
        content = [
            {
                "type": "input_audio",
                "input_audio": {"data": "", "format": "wav"},
            },
        ]
        assert _extract_audio_from_content(content) == []


# ---------------------------------------------------------------------------
# Unit tests for _normalize_multimodal_content — input_audio
# ---------------------------------------------------------------------------


class TestNormalizeInputAudio:
    def test_input_audio_validated_and_passed_through(self):
        b64 = _b64_audio()
        content = [
            {"type": "text", "text": "hi"},
            {
                "type": "input_audio",
                "input_audio": {"data": b64, "format": "wav"},
            },
        ]
        out = _normalize_multimodal_content(content)
        assert isinstance(out, list)
        assert out[1] == {
            "type": "input_audio",
            "input_audio": {"data": b64, "format": "wav"},
        }

    def test_input_audio_missing_object_rejected(self):
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content([{"type": "input_audio"}])
        assert "input_audio parts must include" in str(exc.value)

    def test_input_audio_empty_data_rejected(self):
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content([
                {
                    "type": "input_audio",
                    "input_audio": {"data": "", "format": "wav"},
                },
            ])
        assert "non-empty base64" in str(exc.value)

    def test_input_audio_bad_format_rejected(self):
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content([
                {
                    "type": "input_audio",
                    "input_audio": {"data": _b64_audio(), "format": "xyz"},
                },
            ])
        assert "unsupported audio format" in str(exc.value)


# ---------------------------------------------------------------------------
# Unit tests for _content_has_visible_payload — input_audio
# ---------------------------------------------------------------------------


class TestContentHasVisiblePayloadAudio:
    def test_audio_only_is_visible(self):
        content = [
            {
                "type": "input_audio",
                "input_audio": {"data": _b64_audio(), "format": "wav"},
            },
        ]
        assert _content_has_visible_payload(content) is True

    def test_audio_with_text_is_visible(self):
        assert _content_has_visible_payload(_audio_content_parts()) is True


# ---------------------------------------------------------------------------
# HTTP integration — audio routing through _handle_chat_completions
# ---------------------------------------------------------------------------


class TestChatCompletionsAudioRouting:
    @pytest.mark.asyncio
    async def test_text_mode_transcribes_audio(self, monkeypatch):
        """When audio_mode == "text", audio is transcribed and user_message is a string."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        # Mock the routing decision → text
        monkeypatch.setattr(
            "agent.audio_routing.decide_audio_input_mode",
            lambda *a, **kw: "text",
        )
        # Mock transcription → success
        monkeypatch.setattr(
            "tools.transcription_tools.transcribe_audio",
            lambda path, model=None: {
                "success": True,
                "transcript": "hello world",
                "provider": "groq",
            },
        )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "ok", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )

                mock_run.side_effect = _stub

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "user", "content": _audio_content_parts()},
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            user_msg = mock_run.captured["user_message"]
            # Should be a plain string after transcription
            assert isinstance(user_msg, str)
            assert "hello world" in user_msg
            assert "Transcribe this" in user_msg

    @pytest.mark.asyncio
    async def test_native_mode_passes_audio_through(self, monkeypatch):
        """When audio_mode == "native", input_audio parts pass through unchanged."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        monkeypatch.setattr(
            "agent.audio_routing.decide_audio_input_mode",
            lambda *a, **kw: "native",
        )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "ok", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )

                mock_run.side_effect = _stub

                audio_parts = _audio_content_parts()
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": audio_parts}],
                    },
                )

            assert resp.status == 200, await resp.text()
            user_msg = mock_run.captured["user_message"]
            # Should still be a list with the input_audio part intact
            assert isinstance(user_msg, list)
            types = [p["type"] for p in user_msg if isinstance(p, dict)]
            assert "input_audio" in types

    @pytest.mark.asyncio
    async def test_text_only_skips_routing(self, monkeypatch):
        """Text-only messages bypass audio routing entirely."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        # The routing module should never be called
        called = []
        monkeypatch.setattr(
            "agent.audio_routing.decide_audio_input_mode",
            lambda *a, **kw: called.append(1) or "text",
        )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "ok", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )

                mock_run.side_effect = _stub

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "just text"}],
                    },
                )

            assert resp.status == 200, await resp.text()
            assert called == [], "routing should not have been called for text-only"

    @pytest.mark.asyncio
    async def test_transcription_failure_graceful_fallback(self, monkeypatch):
        """When transcribe_audio fails, a fallback note is prepended."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        monkeypatch.setattr(
            "agent.audio_routing.decide_audio_input_mode",
            lambda *a, **kw: "text",
        )
        monkeypatch.setattr(
            "tools.transcription_tools.transcribe_audio",
            lambda path, model=None: {
                "success": False,
                "transcript": "",
                "error": "STT is disabled",
            },
        )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "ok", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )

                mock_run.side_effect = _stub

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "user", "content": _audio_content_parts()},
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            user_msg = mock_run.captured["user_message"]
            assert isinstance(user_msg, str)
            assert "couldn't transcribe" in user_msg.lower()

    @pytest.mark.asyncio
    async def test_transcription_backend_timeout_fallback(self, monkeypatch):
        """Catastrophic exception in transcribe_audio produces graceful fallback, not 500."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        monkeypatch.setattr(
            "agent.audio_routing.decide_audio_input_mode",
            lambda *a, **kw: "text",
        )
        monkeypatch.setattr(
            "tools.transcription_tools.transcribe_audio",
            lambda path, model=None: (_ for _ in ()).throw(
                TimeoutError("simulated provider timeout")
            ),
        )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "ok", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )

                mock_run.side_effect = _stub

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "user", "content": _audio_content_parts()},
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            user_msg = mock_run.captured["user_message"]
            assert isinstance(user_msg, str)
            assert "something went wrong" in user_msg.lower()

    @pytest.mark.asyncio
    async def test_routing_failure_passes_through(self, monkeypatch):
        """When the routing decision itself fails, audio passes through unchanged."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        def _crashing_router(*a, **kw):
            raise RuntimeError("config unavailable")

        monkeypatch.setattr(
            "agent.audio_routing.decide_audio_input_mode",
            _crashing_router,
        )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "ok", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )

                mock_run.side_effect = _stub

                audio_parts = _audio_content_parts()
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": audio_parts}],
                    },
                )

            assert resp.status == 200, await resp.text()
            user_msg = mock_run.captured["user_message"]
            assert isinstance(user_msg, list)
            types = [p["type"] for p in user_msg if isinstance(p, dict)]
            assert "input_audio" in types

    @pytest.mark.asyncio
    async def test_audio_with_text_preserves_original_text(self, monkeypatch):
        """When audio is transcribed, the original text part is preserved."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        monkeypatch.setattr(
            "agent.audio_routing.decide_audio_input_mode",
            lambda *a, **kw: "text",
        )
        monkeypatch.setattr(
            "tools.transcription_tools.transcribe_audio",
            lambda path, model=None: {
                "success": True,
                "transcript": "the transcript",
                "provider": "local",
            },
        )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "ok", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )

                mock_run.side_effect = _stub

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "what does this say?"},
                                    {
                                        "type": "input_audio",
                                        "input_audio": {
                                            "data": _b64_audio(),
                                            "format": "wav",
                                        },
                                    },
                                ],
                            },
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            user_msg = mock_run.captured["user_message"]
            assert isinstance(user_msg, str)
            assert "the transcript" in user_msg
            assert "what does this say?" in user_msg
