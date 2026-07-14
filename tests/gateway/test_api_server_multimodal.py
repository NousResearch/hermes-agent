"""End-to-end tests for inline multimodal inputs on API server endpoints.

Covers the multimodal normalization path added to the API server.  Unlike the
adapter-level tests that patch ``_run_agent``, these tests patch
``AIAgent.run_conversation`` instead so the adapter's full request-handling
path (including the ``run_agent`` prologue that used to crash on list content)
executes against a real aiohttp app.
"""

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _content_has_visible_payload,
    _normalize_multimodal_content,
    cors_middleware,
    security_headers_middleware,
)


# ---------------------------------------------------------------------------
# Pure-function tests for _normalize_multimodal_content
# ---------------------------------------------------------------------------


def _audio_part(data: bytes = b"hello", audio_format: str = "wav") -> dict:
    return {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(data).decode("ascii"),
            "format": audio_format,
        },
    }


class TestNormalizeMultimodalContent:
    def test_string_passthrough(self):
        assert _normalize_multimodal_content("hello") == "hello"

    def test_none_returns_empty_string(self):
        assert _normalize_multimodal_content(None) == ""

    def test_text_only_list_collapses_to_string(self):
        content = [{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]
        assert _normalize_multimodal_content(content) == "hi\nthere"

    def test_responses_input_text_canonicalized(self):
        content = [{"type": "input_text", "text": "hello"}]
        assert _normalize_multimodal_content(content) == "hello"

    def test_image_url_preserved_with_text(self):
        content = [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}},
        ]
        out = _normalize_multimodal_content(content)
        assert isinstance(out, list)
        assert out == [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}},
        ]

    def test_input_image_converted_to_canonical_shape(self):
        content = [
            {"type": "input_text", "text": "hi"},
            {"type": "input_image", "image_url": "https://example.com/cat.png"},
        ]
        out = _normalize_multimodal_content(content)
        assert out == [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ]

    def test_data_image_url_accepted(self):
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        out = _normalize_multimodal_content(content)
        assert out == [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]

    def test_non_image_data_url_rejected(self):
        content = [{"type": "image_url", "image_url": {"url": "data:text/plain;base64,SGVsbG8="}}]
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content(content)
        assert str(exc.value).startswith("unsupported_content_type:")

    def test_file_part_rejected(self):
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content([{"type": "file", "file": {"file_id": "f_1"}}])
        assert str(exc.value).startswith("unsupported_content_type:")

    def test_input_file_part_rejected(self):
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content([{"type": "input_file", "file_id": "f_1"}])
        assert str(exc.value).startswith("unsupported_content_type:")

    def test_missing_url_rejected(self):
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content([{"type": "image_url", "image_url": {}}])
        assert str(exc.value).startswith("invalid_image_url:")

    def test_bad_scheme_rejected(self):
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content([{"type": "image_url", "image_url": {"url": "ftp://example.com/x.png"}}])
        assert str(exc.value).startswith("invalid_image_url:")

    def test_unknown_part_type_rejected(self):
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content([{"type": "audio", "audio": {}}])
        assert str(exc.value).startswith("unsupported_content_type:")

    def test_input_audio_rejected_by_default(self):
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content([_audio_part()])
        assert str(exc.value).startswith("unsupported_content_type:")

    def test_input_audio_allowed_and_canonicalized(self):
        out = _normalize_multimodal_content(
            [_audio_part(data=b"voice", audio_format=".WAV")],
            allow_audio=True,
        )
        assert out == [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64.b64encode(b"voice").decode("ascii"),
                    "format": "wav",
                },
            }
        ]

    @pytest.mark.parametrize(
        "part",
        [
            {"type": "input_audio"},
            {"type": "input_audio", "input_audio": "not-an-object"},
            {"type": "input_audio", "input_audio": {"format": "wav"}},
            {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}},
            {"type": "input_audio", "input_audio": {"data": "not base64!", "format": "wav"}},
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64.b64encode(b"voice").decode("ascii"),
                    "format": "../wav",
                },
            },
        ],
    )
    def test_malformed_input_audio_rejected_when_allowed(self, part):
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content([part], allow_audio=True)
        assert str(exc.value).startswith("invalid_content_part:")

    def test_unsupported_input_audio_format_rejected_when_allowed(self):
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content([_audio_part(audio_format="exe")], allow_audio=True)
        assert str(exc.value).startswith("unsupported_content_type:")


class TestContentHasVisiblePayload:
    def test_non_empty_string(self):
        assert _content_has_visible_payload("hello")

    def test_whitespace_only_string(self):
        assert not _content_has_visible_payload("   ")

    def test_list_with_image_only(self):
        assert _content_has_visible_payload([{"type": "image_url", "image_url": {"url": "x"}}])

    def test_list_with_audio_only(self):
        assert _content_has_visible_payload([_audio_part()])

    def test_list_with_only_empty_text(self):
        assert not _content_has_visible_payload([{"type": "text", "text": ""}])


# ---------------------------------------------------------------------------
# HTTP integration — real aiohttp client hitting the adapter handlers
# ---------------------------------------------------------------------------


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)
    return app


def _capture_run_agent(mock_run: MagicMock, final_response: str = "ok") -> None:
    async def _stub(**kwargs):
        mock_run.captured = kwargs
        return (
            {"final_response": final_response, "messages": [], "api_calls": 1},
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

    mock_run.side_effect = _stub


@pytest.fixture
def adapter():
    return _make_adapter()


class TestChatCompletionsMultimodalHTTP:
    @pytest.mark.asyncio
    async def test_inline_image_preserved_to_run_agent(self, adapter):
        """Multimodal user content reaches _run_agent as a list of parts."""
        image_payload = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_run_agent",
                new=MagicMock(),
            ) as mock_run:
                _capture_run_agent(mock_run, final_response="A cat.")

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": image_payload}],
                    },
                )

            assert resp.status == 200, await resp.text()
            assert mock_run.captured["user_message"] == image_payload

    @pytest.mark.asyncio
    async def test_text_only_array_collapses_to_string(self, adapter):
        """Text-only array becomes a plain string so logging stays unchanged."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                _capture_run_agent(mock_run)

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            assert mock_run.captured["user_message"] == "hello"

    @pytest.mark.asyncio
    async def test_audio_only_transcribed_before_run_agent(self, adapter):
        """Audio-only turns become visible transcript text instead of empty requests."""
        temp_path = {}

        def _fake_transcribe(path):
            temp_path["path"] = path
            assert Path(path).exists()
            return {"success": True, "transcript": "hello from audio"}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with (
                patch("tools.transcription_tools.transcribe_audio", side_effect=_fake_transcribe),
                patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run,
            ):
                _capture_run_agent(mock_run)

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": [_audio_part()]}],
                    },
                )

            assert resp.status == 200, await resp.text()
            assert "Transcript: \"hello from audio\"" in mock_run.captured["user_message"]
            assert temp_path["path"]
            assert not Path(temp_path["path"]).exists()

    @pytest.mark.asyncio
    async def test_mixed_text_image_audio_preserves_image_and_adds_transcript(self, adapter):
        image_part = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/cat.png", "detail": "high"},
        }
        content = [
            {"type": "text", "text": "Please inspect this."},
            image_part,
            _audio_part(),
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with (
                patch(
                    "tools.transcription_tools.transcribe_audio",
                    return_value={"success": True, "transcript": "also check the tail"},
                ),
                patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run,
            ):
                _capture_run_agent(mock_run)

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": content}]},
                )

            assert resp.status == 200, await resp.text()
            assert mock_run.captured["user_message"] == [
                {"type": "text", "text": "Please inspect this."},
                image_part,
                {"type": "text", "text": '[The user sent a voice message. Transcript: "also check the tail"]'},
            ]

    @pytest.mark.parametrize(
        ("transcribe_result", "expected_text"),
        [
            ({"success": False, "error": "no speech detected"}, "transcription failed: no speech detected"),
            ({"success": True, "transcript": ""}, "transcribed to empty text"),
        ],
    )
    @pytest.mark.asyncio
    async def test_audio_transcription_non_text_outcomes_reach_run_agent_as_note(
        self, adapter, transcribe_result, expected_text
    ):
        temp_path = {}

        def _fake_transcribe(path):
            temp_path["path"] = path
            assert Path(path).exists()
            return transcribe_result

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with (
                patch("tools.transcription_tools.transcribe_audio", side_effect=_fake_transcribe),
                patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run,
            ):
                _capture_run_agent(mock_run)

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": [_audio_part()]}],
                    },
                )

            assert resp.status == 200, await resp.text()
            assert expected_text in mock_run.captured["user_message"]
            assert temp_path["path"]
            assert not Path(temp_path["path"]).exists()

    @pytest.mark.asyncio
    async def test_invalid_audio_base64_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "hermes-agent",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_audio", "input_audio": {"data": "nope!", "format": "wav"}},
                            ],
                        },
                    ],
                },
            )
            assert resp.status == 400
            body = await resp.json()
        assert body["error"]["code"] == "invalid_content_part"
        assert body["error"]["param"] == "messages[0].content"

    @pytest.mark.parametrize(
        ("messages", "expected_param"),
        [
            ([{"role": "assistant", "content": [_audio_part()]}], "messages[0].content"),
            (
                [
                    {"role": "user", "content": [_audio_part()]},
                    {"role": "user", "content": "next turn"},
                ],
                "messages[0].content",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_audio_outside_final_user_message_returns_400(self, adapter, messages, expected_param):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "hermes-agent",
                    "messages": messages,
                },
            )
            assert resp.status == 400
            body = await resp.json()
        assert body["error"]["code"] == "unsupported_content_type"
        assert body["error"]["param"] == expected_param

    @pytest.mark.asyncio
    async def test_file_part_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "hermes-agent",
                    "messages": [
                        {"role": "user", "content": [{"type": "file", "file": {"file_id": "f_1"}}]},
                    ],
                },
            )
            assert resp.status == 400
            body = await resp.json()
        assert body["error"]["code"] == "unsupported_content_type"
        assert body["error"]["param"] == "messages[0].content"

    @pytest.mark.asyncio
    async def test_non_image_data_url_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "hermes-agent",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "data:text/plain;base64,SGVsbG8="},
                                },
                            ],
                        },
                    ],
                },
            )
            assert resp.status == 400
            body = await resp.json()
        assert body["error"]["code"] == "unsupported_content_type"


class TestResponsesMultimodalHTTP:
    @pytest.mark.asyncio
    async def test_input_image_canonicalized_and_forwarded(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                _capture_run_agent(mock_run)

                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": "Describe."},
                                    {
                                        "type": "input_image",
                                        "image_url": "https://example.com/cat.png",
                                    },
                                ],
                            }
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            expected = [
                {"type": "text", "text": "Describe."},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ]
            assert mock_run.captured["user_message"] == expected

    @pytest.mark.asyncio
    async def test_input_file_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={
                    "model": "hermes-agent",
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_file", "file_id": "f_1"}],
                        }
                    ],
                },
            )
            assert resp.status == 400
            body = await resp.json()
        assert body["error"]["code"] == "unsupported_content_type"

    @pytest.mark.asyncio
    async def test_input_audio_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={
                    "model": "hermes-agent",
                    "input": [
                        {
                            "role": "user",
                            "content": [_audio_part()],
                        }
                    ],
                },
            )
            assert resp.status == 400
            body = await resp.json()
        assert body["error"]["code"] == "unsupported_content_type"
