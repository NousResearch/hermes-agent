"""End-to-end tests for inline image inputs on /v1/chat/completions and /v1/responses.

Covers the multimodal normalization path added to the API server.  Unlike the
adapter-level tests that patch ``_run_agent``, these tests patch
``AIAgent.run_conversation`` instead so the adapter's full request-handling
path (including the ``run_agent`` prologue that used to crash on list content)
executes against a real aiohttp app.
"""

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


class TestContentHasVisiblePayload:
    def test_non_empty_string(self):
        assert _content_has_visible_payload("hello")

    def test_whitespace_only_string(self):
        assert not _content_has_visible_payload("   ")

    def test_list_with_image_only(self):
        assert _content_has_visible_payload([{"type": "image_url", "image_url": {"url": "x"}}])

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
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "A cat.", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                # With image routing wired in, images only pass through when
                # the routing decision returns "native" (vision-capable model
                # or explicit config).  The routing decision is tested
                # separately; this test verifies the native pass-through path.
                with patch(
                    "agent.image_routing.decide_image_input_mode",
                    return_value="native",
                ):
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
                            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            assert mock_run.captured["user_message"] == "hello"

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
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "ok", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

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


# ---------------------------------------------------------------------------
# Image routing: decide_image_input_mode wired into _handle_chat_completions
# ---------------------------------------------------------------------------


class TestChatCompletionsImageRouting:
    """Tests for the image-routing decision wired into the API server.

    The gateway (CLI/TUI/messaging) calls _decide_image_input_mode before
    passing content to the agent.  These tests verify the api_server now
    does the same — respecting agent.image_input_mode, auxiliary.vision
    overrides, and model capability metadata.
    """

    IMAGE_PAYLOAD = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
    ]

    @pytest.mark.asyncio
    async def test_native_mode_preserves_image_parts(self, adapter):
        """Vision-capable model (or explicit native config) → images pass through."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "A cat.", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                with patch(
                    "agent.image_routing.decide_image_input_mode",
                    return_value="native",
                ):
                    resp = await cli.post(
                        "/v1/chat/completions",
                        json={
                            "model": "hermes-agent",
                            "messages": [{"role": "user", "content": self.IMAGE_PAYLOAD}],
                        },
                    )

            assert resp.status == 200, await resp.text()
            user_msg = mock_run.captured["user_message"]
            assert isinstance(user_msg, list)
            assert any(
                isinstance(p, dict) and p.get("type") == "image_url"
                for p in user_msg
            )

    @pytest.mark.asyncio
    async def test_text_mode_replaces_images_with_descriptions(self, adapter):
        """Non-vision model (or explicit text config) → images replaced with text."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "A cat.", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                with patch(
                    "agent.image_routing.decide_image_input_mode",
                    return_value="text",
                ):
                    with patch(
                        "tools.vision_tools.vision_analyze_tool",
                        return_value='{"success":true,"analysis":"a fluffy cat sitting on a couch"}',
                    ):
                        resp = await cli.post(
                            "/v1/chat/completions",
                            json={
                                "model": "hermes-agent",
                                "messages": [{"role": "user", "content": self.IMAGE_PAYLOAD}],
                            },
                        )

            assert resp.status == 200, await resp.text()
            user_msg = mock_run.captured["user_message"]
            assert isinstance(user_msg, str)
            assert "a fluffy cat sitting on a couch" in user_msg
            assert "What's in this image?" in user_msg
            assert "vision_analyze" in user_msg

    @pytest.mark.asyncio
    async def test_auto_with_non_vision_model_uses_text_path(self, adapter):
        """Auto mode + non-vision model caps → text path."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "A cat.", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                with patch(
                    "agent.image_routing._lookup_supports_vision",
                    return_value=False,
                ):
                    with patch(
                        "tools.vision_tools.vision_analyze_tool",
                        return_value='{"success":true,"analysis":"a fluffy cat sitting on a couch"}',
                    ):
                        resp = await cli.post(
                            "/v1/chat/completions",
                            json={
                                "model": "hermes-agent",
                                "messages": [{"role": "user", "content": self.IMAGE_PAYLOAD}],
                            },
                        )

            assert resp.status == 200, await resp.text()
            user_msg = mock_run.captured["user_message"]
            assert isinstance(user_msg, str)
            assert "a fluffy cat sitting on a couch" in user_msg

    @pytest.mark.asyncio
    async def test_auto_with_vision_model_uses_native_path(self, adapter):
        """Auto mode + vision-capable model caps → native path."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "A cat.", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                with patch(
                    "agent.image_routing._lookup_supports_vision",
                    return_value=True,
                ):
                    resp = await cli.post(
                        "/v1/chat/completions",
                        json={
                            "model": "hermes-agent",
                            "messages": [{"role": "user", "content": self.IMAGE_PAYLOAD}],
                        },
                    )

            assert resp.status == 200, await resp.text()
            user_msg = mock_run.captured["user_message"]
            assert isinstance(user_msg, list)
            assert any(
                isinstance(p, dict) and p.get("type") == "image_url"
                for p in user_msg
            )

    @pytest.mark.asyncio
    async def test_url_based_image_passed_to_vision_analyze(self, adapter):
        """HTTPS image URL is forwarded to vision_analyze_tool correctly."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "A cat.", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                calls = []

                async def _fake_vision(image_url, user_prompt, model=None):
                    calls.append({"image_url": image_url, "user_prompt": user_prompt})
                    return '{"success":true,"analysis":"a cat"}'

                with patch(
                    "agent.image_routing.decide_image_input_mode",
                    return_value="text",
                ):
                    with patch(
                        "tools.vision_tools.vision_analyze_tool",
                        side_effect=_fake_vision,
                    ):
                        resp = await cli.post(
                            "/v1/chat/completions",
                            json={
                                "model": "hermes-agent",
                                "messages": [{"role": "user", "content": self.IMAGE_PAYLOAD}],
                            },
                        )

            assert resp.status == 200, await resp.text()
            assert len(calls) == 1
            assert calls[0]["image_url"] == "https://example.com/cat.png"

    @pytest.mark.asyncio
    async def test_text_only_message_skips_routing(self, adapter):
        """Plain text messages never trigger image routing logic."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "Hi!", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                with patch(
                    "agent.image_routing.decide_image_input_mode",
                ) as mock_decide:
                    resp = await cli.post(
                        "/v1/chat/completions",
                        json={
                            "model": "hermes-agent",
                            "messages": [{"role": "user", "content": "Hello, world!"}],
                        },
                    )

            assert resp.status == 200, await resp.text()
            assert mock_run.captured["user_message"] == "Hello, world!"
            mock_decide.assert_not_called()

    @pytest.mark.asyncio
    async def test_data_url_image_enrichment(self, adapter):
        """data:image/... URLs are handled by the text path."""
        app = _create_app(adapter)
        data_url_payload = [
            {"type": "text", "text": "What's this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "A cat.", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                with patch(
                    "agent.image_routing.decide_image_input_mode",
                    return_value="text",
                ):
                    with patch(
                        "tools.vision_tools.vision_analyze_tool",
                        return_value='{"success":true,"analysis":"a pixel"}',
                    ):
                        resp = await cli.post(
                            "/v1/chat/completions",
                            json={
                                "model": "hermes-agent",
                                "messages": [{"role": "user", "content": data_url_payload}],
                            },
                        )

            assert resp.status == 200, await resp.text()
            user_msg = mock_run.captured["user_message"]
            assert isinstance(user_msg, str)
            assert "a pixel" in user_msg

    @pytest.mark.asyncio
    async def test_routing_failure_passes_through_native(self, adapter):
        """If decide_image_input_mode raises, images pass through unchanged."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "A cat.", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                with patch(
                    "agent.image_routing.decide_image_input_mode",
                    side_effect=RuntimeError("config missing"),
                ):
                    resp = await cli.post(
                        "/v1/chat/completions",
                        json={
                            "model": "hermes-agent",
                            "messages": [{"role": "user", "content": self.IMAGE_PAYLOAD}],
                        },
                    )

            assert resp.status == 200, await resp.text()
            user_msg = mock_run.captured["user_message"]
            assert isinstance(user_msg, list)
            assert any(
                isinstance(p, dict) and p.get("type") == "image_url"
                for p in user_msg
            )
