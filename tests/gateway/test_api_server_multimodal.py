"""End-to-end tests for inline multimodal inputs on /v1/chat/completions and /v1/responses.

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
    body_limit_middleware,
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

    def test_file_part_normalized_when_allowed(self):
        out = _normalize_multimodal_content(
            [{"type": "file", "file": {"filename": "notes.txt", "file_data": _b64(b"hello")}}],
            allow_files=True,
        )
        assert out == [
            {"type": "file", "file": {"filename": "notes.txt", "file_data": _b64(b"hello")}},
        ]

    def test_input_file_part_normalized_when_allowed(self):
        out = _normalize_multimodal_content(
            [{"type": "input_file", "filename": "report.pdf", "file_data": _b64(b"%PDF-1.4")}],
            allow_files=True,
        )
        assert out == [
            {"type": "file", "file": {"filename": "report.pdf", "file_data": _b64(b"%PDF-1.4")}},
        ]

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

    def test_list_with_file_only(self):
        assert _content_has_visible_payload([{"type": "file", "file": {"filename": "x.txt"}}])

    def test_list_with_only_empty_text(self):
        assert not _content_has_visible_payload([{"type": "text", "text": ""}])


# ---------------------------------------------------------------------------
# HTTP integration — real aiohttp client hitting the adapter handlers
# ---------------------------------------------------------------------------


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    import gateway.platforms.api_server as api_server_mod

    mws = [mw for mw in (cors_middleware, body_limit_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws, client_max_size=api_server_mod._aiohttp_client_max_size())
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)
    return app


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@pytest.fixture
def hermes_cache_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    document_cache = hermes_home / "cache" / "documents"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", document_cache)
    return hermes_home


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
    async def test_inline_text_file_cached_and_injected(self, adapter, hermes_cache_home):
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
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Read this."},
                                    {
                                        "type": "file",
                                        "file": {
                                            "filename": "notes.txt",
                                            "file_data": _b64(b"hello from the document"),
                                        },
                                    },
                                ],
                            },
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            user_message = mock_run.captured["user_message"]
            assert isinstance(user_message, str)
            assert "Read this." in user_message
            assert "The user sent a text document: 'notes.txt'" in user_message
            assert "Its content has been included below" in user_message
            assert "/root/.hermes/cache/documents/" in user_message
            assert str(hermes_cache_home) not in user_message
            assert "[Content of notes.txt]:\nhello from the document" in user_message

            cached_files = list((hermes_cache_home / "cache" / "documents").iterdir())
            assert len(cached_files) == 1
            assert cached_files[0].name.startswith("doc_")
            assert cached_files[0].name.endswith("_notes.txt")

    @pytest.mark.asyncio
    async def test_inline_pdf_file_uses_gateway_style_note(self, adapter, hermes_cache_home):
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
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_file",
                                        "filename": "report.pdf",
                                        "file_data": _b64(b"%PDF-1.4\n%fake"),
                                    },
                                ],
                            },
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            user_message = mock_run.captured["user_message"]
            assert isinstance(user_message, str)
            assert "The user sent a document: 'report.pdf'" in user_message
            assert "application/pdf" in user_message
            assert "/root/.hermes/cache/documents/" in user_message
            assert "[Content of report.pdf]" not in user_message

    @pytest.mark.asyncio
    async def test_inline_file_keeps_image_parts(self, adapter, hermes_cache_home):
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
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                                    {
                                        "type": "file",
                                        "file": {"filename": "notes.md", "file_data": _b64(b"# note")},
                                    },
                                ],
                            },
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            user_message = mock_run.captured["user_message"]
            assert isinstance(user_message, list)
            assert user_message[0]["type"] == "image_url"
            assert any(
                part.get("type") == "text" and "The user sent a text document: 'notes.md'" in part.get("text", "")
                for part in user_message
            )

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
    async def test_historical_file_part_returns_400(self, adapter):
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
                                    "type": "file",
                                    "file": {"filename": "old.txt", "file_data": _b64(b"old")},
                                },
                            ],
                        },
                        {"role": "assistant", "content": "ok"},
                        {"role": "user", "content": "new turn"},
                    ],
                },
            )
            assert resp.status == 400
            body = await resp.json()
        assert body["error"]["code"] == "unsupported_content_type"
        assert body["error"]["param"] == "messages[0].content"

    @pytest.mark.asyncio
    async def test_assistant_file_part_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "hermes-agent",
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "file",
                                    "file": {"filename": "assistant.txt", "file_data": _b64(b"no")},
                                },
                            ],
                        },
                    ],
                },
            )
            assert resp.status == 400
            body = await resp.json()
        assert body["error"]["code"] == "unsupported_content_type"
        assert body["error"]["param"] == "messages[1].content"

    @pytest.mark.asyncio
    async def test_invalid_base64_file_returns_400(self, adapter, hermes_cache_home):
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
                                {"type": "file", "file": {"filename": "notes.txt", "file_data": "not base64"}},
                            ],
                        },
                    ],
                },
            )
            assert resp.status == 400
            body = await resp.json()
        assert body["error"]["code"] == "invalid_content_part"
        assert body["error"]["param"] == "messages[0].content"
        assert not (hermes_cache_home / "cache" / "documents").exists()

    @pytest.mark.asyncio
    async def test_data_url_file_data_prefix_is_accepted(self, adapter, hermes_cache_home):
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
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "file",
                                        "file": {
                                            "filename": "notes.txt",
                                            "file_data": f"data:text/plain;base64,{_b64(b'prefixed')}",
                                        },
                                    },
                                ],
                            },
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            assert "[Content of notes.txt]:\nprefixed" in mock_run.captured["user_message"]

    @pytest.mark.asyncio
    async def test_oversized_decoded_file_returns_400(self, adapter, hermes_cache_home):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.platforms.api_server.MAX_INLINE_FILE_BYTES", 4):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "file", "file": {"filename": "notes.txt", "file_data": _b64(b"12345")}},
                                ],
                            },
                        ],
                    },
                )
            assert resp.status == 400
            body = await resp.json()
        assert body["error"]["code"] == "unsupported_content_type"
        assert body["error"]["param"] == "messages[0].content"
        assert not (hermes_cache_home / "cache" / "documents").exists()

    @pytest.mark.asyncio
    async def test_unsupported_file_extension_returns_400(self, adapter, hermes_cache_home):
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
                                {"type": "file", "file": {"filename": "script.exe", "file_data": _b64(b"MZ")}},
                            ],
                        },
                    ],
                },
            )
            assert resp.status == 400
            body = await resp.json()
        assert body["error"]["code"] == "unsupported_content_type"
        assert body["error"]["param"] == "messages[0].content"
        assert not (hermes_cache_home / "cache" / "documents").exists()

    @pytest.mark.asyncio
    async def test_file_part_missing_filename_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "hermes-agent",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "file", "file": {"file_data": _b64(b"hello")}}],
                        },
                    ],
                },
            )
            assert resp.status == 400
            body = await resp.json()
        assert body["error"]["code"] == "invalid_content_part"

    @pytest.mark.asyncio
    async def test_content_length_above_limit_returns_413(self, adapter):
        with patch("gateway.platforms.api_server.MAX_REQUEST_BYTES", 10):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    data=b"x" * 11,
                    headers={"Content-Type": "application/json"},
                )
                assert resp.status == 413
                body = await resp.json()
        assert body["error"]["code"] == "body_too_large"

    @pytest.mark.asyncio
    async def test_file_request_above_default_body_limit_succeeds_when_limit_raised(
        self,
        adapter,
        hermes_cache_home,
    ):
        raw_file = b"x" * (1_100_000)
        with patch("gateway.platforms.api_server.MAX_REQUEST_BYTES", 2_000_000):
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
                                        {
                                            "type": "file",
                                            "file": {"filename": "large.txt", "file_data": _b64(raw_file)},
                                        },
                                    ],
                                },
                            ],
                        },
                    )

            assert resp.status == 200, await resp.text()
            assert "The user sent a document: 'large.txt'" in mock_run.captured["user_message"]
            assert "[Content of large.txt]" not in mock_run.captured["user_message"]

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

    @pytest.mark.asyncio
    async def test_inline_input_file_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={
                    "model": "hermes-agent",
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_file",
                                    "filename": "notes.txt",
                                    "file_data": _b64(b"hello"),
                                },
                            ],
                        }
                    ],
                },
            )
            assert resp.status == 400
            body = await resp.json()
        assert body["error"]["code"] == "unsupported_content_type"
