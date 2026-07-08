"""End-to-end tests for inline image inputs on /v1/chat/completions and /v1/responses.

Covers the multimodal normalization path added to the API server.  Unlike the
adapter-level tests that patch ``_run_agent``, these tests patch
``AIAgent.run_conversation`` instead so the adapter's full request-handling
path (including the ``run_agent`` prologue that used to crash on list content)
executes against a real aiohttp app.
"""

import base64
import importlib
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

# Minimal valid 1x1 PNG (magic bytes only matter for cache_image_from_bytes's sniff).
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
_PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(_PNG_BYTES).decode("ascii")


def _reload_api_server(monkeypatch, hermes_home: Path):
    """Reload api_server (and its cache-dir dependencies) under an isolated HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import hermes_constants
    importlib.reload(hermes_constants)
    import gateway.platforms.base as base_mod
    importlib.reload(base_mod)
    import gateway.platforms.api_server as api_server_mod
    importlib.reload(api_server_mod)
    return api_server_mod


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
        # "AAAA" decodes to non-image bytes, so gateway persistence silently
        # no-ops (see TestGatewayDataUrlPersistence for the persisted case) —
        # the original image_url part must still be forwarded either way.
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
# Gateway-layer persistence of data: URL images (aligns HTTP API ingestion
# with what platform adapters like Weixin already do: save inbound media to
# a stable local cache path before the agent runs).
# ---------------------------------------------------------------------------


class TestGatewayDataUrlPersistence:
    def test_data_url_image_persisted_and_note_injected(self, tmp_path, monkeypatch):
        api_server_mod = _reload_api_server(monkeypatch, tmp_path / "hermes")
        content = [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": _PNG_DATA_URL}},
        ]
        out = api_server_mod._normalize_multimodal_content(content)

        assert isinstance(out, list)
        # Original image_url part is preserved unchanged for native-vision providers.
        image_parts = [p for p in out if p.get("type") == "image_url"]
        assert image_parts == [{"type": "image_url", "image_url": {"url": _PNG_DATA_URL}}]

        # A sibling text note with the stable local path was appended.
        text_parts = [p for p in out if p.get("type") == "text"]
        assert len(text_parts) == 2
        note = text_parts[-1]["text"]
        assert "saved at:" in note
        cache_dir = tmp_path / "hermes" / "cache" / "images"
        assert str(cache_dir) in note

        # The file actually exists with the decoded bytes.
        cached_files = list(cache_dir.glob("*"))
        assert len(cached_files) == 1
        assert cached_files[0].read_bytes() == _PNG_BYTES

    def test_http_url_not_persisted(self, tmp_path, monkeypatch):
        api_server_mod = _reload_api_server(monkeypatch, tmp_path / "hermes")
        content = [{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}]
        out = api_server_mod._normalize_multimodal_content(content)

        assert out == [{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}]
        cache_dir = tmp_path / "hermes" / "cache" / "images"
        assert not cache_dir.exists()

    def test_invalid_base64_data_url_not_persisted_but_still_forwarded(self, tmp_path, monkeypatch):
        api_server_mod = _reload_api_server(monkeypatch, tmp_path / "hermes")
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        out = api_server_mod._normalize_multimodal_content(content)

        assert out == [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        cache_dir = tmp_path / "hermes" / "cache" / "images"
        assert not cache_dir.exists()

    def test_idempotent_when_note_already_present(self, tmp_path, monkeypatch):
        """Replayed conversation history must not stack duplicate 'saved at' notes."""
        api_server_mod = _reload_api_server(monkeypatch, tmp_path / "hermes")
        content = [{"type": "image_url", "image_url": {"url": _PNG_DATA_URL}}]

        first = api_server_mod._normalize_multimodal_content(content)
        first_note = next(p["text"] for p in first if p.get("type") == "text")

        # Re-normalize the *original* request content again (simulating a
        # fresh request with the same inline image) — the cached path is
        # content-addressed, so the note text is identical and must not be
        # duplicated within a single normalization pass either.
        second = api_server_mod._normalize_multimodal_content(content)
        second_note = next(p["text"] for p in second if p.get("type") == "text")
        assert second_note == first_note
        assert len([p for p in second if p.get("type") == "text"]) == 1

        # Simulate replaying a message that ALREADY contains the note
        # (e.g. stored conversation_history from a prior turn) — normalizing
        # it again must not append a second copy.
        replayed = [
            {"type": "text", "text": first_note},
            {"type": "image_url", "image_url": {"url": _PNG_DATA_URL}},
        ]
        third = api_server_mod._normalize_multimodal_content(replayed)
        assert len([p for p in third if p.get("type") == "text"]) == 1

    def test_responses_input_image_persisted(self, tmp_path, monkeypatch):
        """The same helper covers the Responses API's input_image shape."""
        api_server_mod = _reload_api_server(monkeypatch, tmp_path / "hermes")
        content = [{"type": "input_image", "image_url": _PNG_DATA_URL}]
        out = api_server_mod._normalize_multimodal_content(content)

        image_parts = [p for p in out if p.get("type") == "image_url"]
        assert image_parts == [{"type": "image_url", "image_url": {"url": _PNG_DATA_URL}}]
        text_parts = [p for p in out if p.get("type") == "text"]
        assert len(text_parts) == 1
        assert "saved at:" in text_parts[0]["text"]


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
