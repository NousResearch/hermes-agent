"""Regression tests for OpenAI-API gateway vision routing (issue #27232).

Open WebUI users (and any other OpenAI-compatible client) upload photos
as ``image_url`` parts on ``/v1/chat/completions``. Before the fix the
adapter forwarded those parts verbatim to the agent, the main provider
rejected them (DeepSeek, Mistral, etc. don't accept ``image_url``), and
``run_agent`` fell back to text-only mode after stripping the images —
the user got "no photo seen" responses even with ``auxiliary.vision``
configured.

This module covers the new pipeline that mirrors the TUI gateway:
``_decide_api_server_image_mode`` + ``_describe_image_via_vision`` +
``apply_vision_routing`` are wired into ``_run_agent`` so every endpoint
(``/v1/chat/completions``, ``/v1/responses``, ``/v1/runs``) benefits.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _decide_api_server_image_mode,
    _describe_image_via_vision,
    _extract_text_and_image_parts,
    _image_url_from_part,
    _materialize_image_for_vision,
    _route_content_for_text_mode,
    apply_vision_routing,
    cors_middleware,
    security_headers_middleware,
)


# 1×1 transparent PNG — minimum valid bytes for data-URL materialisation tests.
_PNG_1x1_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
_PNG_DATA_URL = f"data:image/png;base64,{_PNG_1x1_BASE64}"


# ---------------------------------------------------------------------------
# _extract_text_and_image_parts — pure helper, splits multimodal content
# ---------------------------------------------------------------------------


class TestExtractTextAndImageParts:
    def test_string_passthrough_keeps_text_and_no_images(self):
        assert _extract_text_and_image_parts("hello world") == ("hello world", [])

    def test_text_only_list_joins_with_newlines(self):
        text, images = _extract_text_and_image_parts(
            [{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]
        )
        assert text == "hi\nthere"
        assert images == []

    def test_mixed_list_separates_text_and_image_parts(self):
        text, images = _extract_text_and_image_parts(
            [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                {"type": "input_image", "image_url": "https://x/z.png"},
            ]
        )
        assert text == "what is this?"
        assert [p["type"] for p in images] == ["image_url", "input_image"]

    def test_responses_input_text_part_counted_as_text(self):
        text, images = _extract_text_and_image_parts(
            [{"type": "input_text", "text": "Describe."}]
        )
        assert text == "Describe."
        assert images == []

    def test_unknown_part_types_ignored(self):
        text, images = _extract_text_and_image_parts(
            [{"type": "weird", "value": 1}, {"type": "text", "text": "ok"}]
        )
        assert text == "ok"
        assert images == []

    def test_non_list_non_string_returns_empty(self):
        text, images = _extract_text_and_image_parts({"role": "user"})
        assert text == ""
        assert images == []


# ---------------------------------------------------------------------------
# _image_url_from_part — pulls a URL out of either Chat or Responses shape
# ---------------------------------------------------------------------------


class TestImageUrlFromPart:
    def test_chat_completions_shape_returns_inner_url(self):
        assert _image_url_from_part({"image_url": {"url": "https://x/y.png"}}) == (
            "https://x/y.png"
        )

    def test_responses_shape_returns_top_level_string(self):
        assert _image_url_from_part({"image_url": "https://x/y.png"}) == (
            "https://x/y.png"
        )

    def test_missing_url_returns_none(self):
        assert _image_url_from_part({"image_url": {}}) is None
        assert _image_url_from_part({"image_url": None}) is None
        assert _image_url_from_part({}) is None

    def test_whitespace_stripped_and_empty_treated_as_missing(self):
        assert _image_url_from_part({"image_url": {"url": "  https://x  "}}) == "https://x"
        assert _image_url_from_part({"image_url": {"url": "   "}}) is None


# ---------------------------------------------------------------------------
# _materialize_image_for_vision — data URLs → temp files, http(s) → passthrough
# ---------------------------------------------------------------------------


class TestMaterializeImageForVision:
    def test_http_url_passes_through_without_temp_file(self):
        path_or_url, is_temp = _materialize_image_for_vision("https://x/y.png")
        assert path_or_url == "https://x/y.png"
        assert is_temp is False

    def test_https_url_passes_through_without_temp_file(self):
        path_or_url, is_temp = _materialize_image_for_vision("https://cdn.example/y.png")
        assert path_or_url == "https://cdn.example/y.png"
        assert is_temp is False

    def test_data_image_url_decoded_to_temp_png(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        path, is_temp = _materialize_image_for_vision(_PNG_DATA_URL)
        assert is_temp is True
        assert path is not None
        p = Path(path)
        try:
            assert p.exists()
            assert p.suffix == ".png"
            assert p.read_bytes() == base64.b64decode(_PNG_1x1_BASE64)
            assert "api_server_vision" in str(p.parent)
        finally:
            p.unlink(missing_ok=True)

    def test_unsupported_scheme_returns_none(self):
        assert _materialize_image_for_vision("ftp://x/y.png") == (None, False)

    def test_non_image_data_url_returns_none(self):
        assert _materialize_image_for_vision(
            "data:text/plain;base64,SGVsbG8="
        ) == (None, False)

    def test_data_url_without_comma_returns_none(self):
        assert _materialize_image_for_vision("data:image/png;base64") == (None, False)

    def test_malformed_base64_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # ``base64.b64decode(validate=False)`` accepts garbage by ignoring it,
        # so the file actually does get written. The contract here is: as long
        # as we don't raise, we return ``(path, True)`` and let vision_analyze
        # surface the real error downstream.
        path, is_temp = _materialize_image_for_vision("data:image/png;base64,!!!")
        if path:
            Path(path).unlink(missing_ok=True)
        assert is_temp is bool(path)

    def test_empty_url_returns_none(self):
        assert _materialize_image_for_vision("") == (None, False)


# ---------------------------------------------------------------------------
# _describe_image_via_vision — async wrapper around vision_analyze_tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDescribeImageViaVision:
    async def test_returns_analysis_text_on_success(self):
        with patch(
            "tools.vision_tools.vision_analyze_tool",
            new=AsyncMock(
                return_value=json.dumps({"success": True, "analysis": "A red cat sits on a mat."})
            ),
        ) as fake:
            out = await _describe_image_via_vision("https://x/y.png")
        assert out == "A red cat sits on a mat."
        fake.assert_awaited_once()
        # vision_analyze_tool was called with the URL untouched (no temp file
        # for http(s) inputs).
        kwargs = fake.await_args.kwargs
        assert kwargs["image_url"] == "https://x/y.png"

    async def test_returns_none_on_unsupported_scheme(self):
        with patch(
            "tools.vision_tools.vision_analyze_tool",
            new=AsyncMock(),
        ) as fake:
            out = await _describe_image_via_vision("ftp://x/y.png")
        assert out is None
        fake.assert_not_called()

    async def test_returns_none_when_tool_reports_failure(self):
        with patch(
            "tools.vision_tools.vision_analyze_tool",
            new=AsyncMock(return_value=json.dumps({"success": False, "analysis": "boom"})),
        ):
            out = await _describe_image_via_vision("https://x/y.png")
        assert out is None

    async def test_returns_none_on_tool_exception(self):
        with patch(
            "tools.vision_tools.vision_analyze_tool",
            new=AsyncMock(side_effect=RuntimeError("vision provider down")),
        ):
            out = await _describe_image_via_vision("https://x/y.png")
        assert out is None

    async def test_returns_none_on_invalid_json(self):
        with patch(
            "tools.vision_tools.vision_analyze_tool",
            new=AsyncMock(return_value="not-json"),
        ):
            out = await _describe_image_via_vision("https://x/y.png")
        assert out is None

    async def test_data_url_writes_temp_and_cleans_up(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        seen_paths: list[str] = []

        async def _fake_tool(image_url, user_prompt, **_kwargs):
            seen_paths.append(image_url)
            assert os.path.exists(image_url), "temp file must exist while tool runs"
            return json.dumps({"success": True, "analysis": "tiny test image"})

        with patch("tools.vision_tools.vision_analyze_tool", new=_fake_tool):
            out = await _describe_image_via_vision(_PNG_DATA_URL)

        assert out == "tiny test image"
        assert seen_paths and not seen_paths[0].startswith("data:")
        # Temp file must be removed after analysis.
        assert not os.path.exists(seen_paths[0])


# ---------------------------------------------------------------------------
# _route_content_for_text_mode — replaces image parts with descriptions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRouteContentForTextMode:
    async def test_string_content_returned_unchanged(self):
        assert await _route_content_for_text_mode("hello") == "hello"

    async def test_list_without_images_returned_unchanged(self):
        content = [{"type": "text", "text": "hello"}]
        out = await _route_content_for_text_mode(content)
        assert out == content

    async def test_image_replaced_with_description_block_before_text(self):
        content = [
            {"type": "text", "text": "What is in this photo?"},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ]
        with patch(
            "gateway.platforms.api_server._describe_image_via_vision",
            new=AsyncMock(return_value="A red cat."),
        ):
            out = await _route_content_for_text_mode(content)
        assert isinstance(out, str)
        assert "A red cat." in out
        assert "[The user sent an image~" in out
        assert out.endswith("What is in this photo?")

    async def test_pure_image_turn_gets_caption_placeholder(self):
        content = [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]
        with patch(
            "gateway.platforms.api_server._describe_image_via_vision",
            new=AsyncMock(return_value="A red cat."),
        ):
            out = await _route_content_for_text_mode(content)
        assert isinstance(out, str)
        assert "A red cat." in out

    async def test_failed_description_keeps_user_aware_image_was_sent(self):
        content = [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ]
        with patch(
            "gateway.platforms.api_server._describe_image_via_vision",
            new=AsyncMock(return_value=None),
        ):
            out = await _route_content_for_text_mode(content)
        assert isinstance(out, str)
        assert "couldn't quite see it" in out
        assert "look" in out  # user text preserved

    async def test_multiple_images_each_get_their_own_block(self):
        content = [
            {"type": "text", "text": "compare"},
            {"type": "image_url", "image_url": {"url": "https://x/a.png"}},
            {"type": "input_image", "image_url": "https://x/b.png"},
        ]
        with patch(
            "gateway.platforms.api_server._describe_image_via_vision",
            new=AsyncMock(side_effect=["first description", "second description"]),
        ):
            out = await _route_content_for_text_mode(content)
        assert out.count("[The user sent an image~") == 2
        assert "first description" in out
        assert "second description" in out
        assert out.endswith("compare")

    async def test_image_part_without_url_skipped(self):
        content = [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {}},
        ]
        with patch(
            "gateway.platforms.api_server._describe_image_via_vision",
            new=AsyncMock(return_value="should not run"),
        ) as fake:
            out = await _route_content_for_text_mode(content)
        # No image parts had a URL → nothing to enrich → original content returned.
        assert out == content
        fake.assert_not_called()


# ---------------------------------------------------------------------------
# apply_vision_routing — gate that consults _decide_api_server_image_mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestApplyVisionRouting:
    async def test_native_mode_keeps_image_parts_intact(self):
        content = [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ]
        with patch(
            "gateway.platforms.api_server._decide_api_server_image_mode",
            return_value="native",
        ), patch(
            "gateway.platforms.api_server._describe_image_via_vision",
            new=AsyncMock(return_value="should not run"),
        ) as fake_describe:
            out = await apply_vision_routing(content)
        assert out == content
        fake_describe.assert_not_called()

    async def test_text_mode_replaces_images_with_descriptions(self):
        content = [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ]
        with patch(
            "gateway.platforms.api_server._decide_api_server_image_mode",
            return_value="text",
        ), patch(
            "gateway.platforms.api_server._describe_image_via_vision",
            new=AsyncMock(return_value="A red cat."),
        ):
            out = await apply_vision_routing(content)
        assert isinstance(out, str)
        assert "A red cat." in out
        assert out.endswith("what is this?")

    async def test_no_images_skips_mode_decision_entirely(self):
        content = "hello"
        with patch(
            "gateway.platforms.api_server._decide_api_server_image_mode",
            side_effect=AssertionError("should not be consulted for text-only content"),
        ):
            out = await apply_vision_routing(content)
        assert out == "hello"

    async def test_pipeline_exception_falls_back_to_passthrough(self):
        content = [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ]
        with patch(
            "gateway.platforms.api_server._decide_api_server_image_mode",
            side_effect=RuntimeError("config broken"),
        ):
            out = await apply_vision_routing(content)
        assert out == content


# ---------------------------------------------------------------------------
# _decide_api_server_image_mode — wraps agent.image_routing.decide_image_input_mode
# ---------------------------------------------------------------------------


class TestDecideApiServerImageMode:
    def test_returns_native_when_decision_helper_raises(self):
        with patch("agent.image_routing.decide_image_input_mode", side_effect=RuntimeError):
            assert _decide_api_server_image_mode() == "native"

    def test_passes_provider_model_and_config_through(self):
        with patch(
            "agent.image_routing.decide_image_input_mode",
            return_value="text",
        ) as fake_decide, patch(
            "agent.auxiliary_client._read_main_provider", return_value="deepseek"
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="deepseek-chat"
        ), patch(
            "hermes_cli.config.load_config",
            return_value={"auxiliary": {"vision": {"provider": "openrouter"}}},
        ):
            mode = _decide_api_server_image_mode()
        assert mode == "text"
        args, _ = fake_decide.call_args
        assert args[0] == "deepseek"
        assert args[1] == "deepseek-chat"
        assert args[2] == {"auxiliary": {"vision": {"provider": "openrouter"}}}


# ---------------------------------------------------------------------------
# HTTP integration — full request path with vision routing enabled
# ---------------------------------------------------------------------------


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


class TestChatCompletionsVisionRoutingHTTP:
    @pytest.mark.asyncio
    async def test_text_mode_replaces_image_with_description_before_agent(self, adapter):
        """End-to-end: Open WebUI upload flow with text-mode routing — the
        agent receives a flat string with the vision description, not the
        raw ``image_url`` part that the main provider would reject."""
        fake_agent = MagicMock()
        fake_agent.run_conversation.return_value = {
            "final_response": "A red cat.", "messages": [], "api_calls": 1
        }
        fake_agent.session_prompt_tokens = 0
        fake_agent.session_completion_tokens = 0
        fake_agent.session_total_tokens = 0
        fake_agent.session_id = "sid"

        app = _create_app(adapter)
        with patch(
            "gateway.platforms.api_server._decide_api_server_image_mode",
            return_value="text",
        ), patch(
            "gateway.platforms.api_server._describe_image_via_vision",
            new=AsyncMock(return_value="A red cat curled on a mat."),
        ), patch.object(
            adapter, "_create_agent", return_value=fake_agent,
        ):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "What's in this photo?"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": "https://example.com/cat.png"},
                                    },
                                ],
                            }
                        ],
                    },
                )
                assert resp.status == 200, await resp.text()

        forwarded_user_message = fake_agent.run_conversation.call_args.kwargs["user_message"]
        assert isinstance(forwarded_user_message, str), (
            "text-mode routing must hand the agent a plain string, not the "
            "raw multimodal list (which the main provider would reject)"
        )
        assert "A red cat curled on a mat." in forwarded_user_message
        assert forwarded_user_message.endswith("What's in this photo?")

    @pytest.mark.asyncio
    async def test_native_mode_forwards_image_parts_untouched(self, adapter):
        """When the main model supports vision (native mode), the image
        parts must reach the agent verbatim so providers like GPT-4o can
        consume them directly."""
        captured: dict = {}

        async def _stub(**kwargs):
            captured.update(kwargs)
            return (
                {"final_response": "ok", "messages": [], "api_calls": 1},
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

        app = _create_app(adapter)
        with patch(
            "gateway.platforms.api_server._decide_api_server_image_mode",
            return_value="native",
        ), patch(
            "gateway.platforms.api_server._describe_image_via_vision",
            new=AsyncMock(return_value="should not run"),
        ) as fake_describe, patch.object(
            adapter, "_run_agent", new=MagicMock(side_effect=_stub),
        ):
            async with TestClient(TestServer(app)) as cli:
                payload = [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                ]
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": payload}],
                    },
                )
                assert resp.status == 200, await resp.text()

        fake_describe.assert_not_called()
        assert captured["user_message"] == payload

    @pytest.mark.asyncio
    async def test_data_image_url_is_materialised_and_described(
        self, adapter, monkeypatch, tmp_path
    ):
        """The real Open WebUI photo-upload format is a data URL — it must
        be decoded to a temp file and passed to ``vision_analyze``."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        seen_paths: list[str] = []

        async def _fake_vision_tool(image_url, user_prompt, **_kwargs):
            seen_paths.append(image_url)
            return json.dumps({"success": True, "analysis": "1×1 PNG"})

        fake_agent = MagicMock()
        fake_agent.run_conversation.return_value = {
            "final_response": "ok", "messages": [], "api_calls": 1
        }
        fake_agent.session_prompt_tokens = 0
        fake_agent.session_completion_tokens = 0
        fake_agent.session_total_tokens = 0
        fake_agent.session_id = "sid"

        app = _create_app(adapter)
        with patch(
            "gateway.platforms.api_server._decide_api_server_image_mode",
            return_value="text",
        ), patch(
            "tools.vision_tools.vision_analyze_tool", new=_fake_vision_tool,
        ), patch.object(
            adapter, "_create_agent", return_value=fake_agent,
        ):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "what is this?"},
                                    {"type": "image_url", "image_url": {"url": _PNG_DATA_URL}},
                                ],
                            }
                        ],
                    },
                )
                assert resp.status == 200, await resp.text()

        assert seen_paths, "vision tool must be invoked for data:image URLs"
        assert not seen_paths[0].startswith("data:"), seen_paths
        forwarded = fake_agent.run_conversation.call_args.kwargs["user_message"]
        assert isinstance(forwarded, str)
        assert "1×1 PNG" in forwarded


class TestResponsesVisionRoutingHTTP:
    @pytest.mark.asyncio
    async def test_input_image_routed_through_vision_pipeline(self, adapter):
        """``/v1/responses`` shares ``_run_agent`` so it gets the same
        routing — verify the Responses-style ``input_image`` shape works."""
        fake_agent = MagicMock()
        fake_agent.run_conversation.return_value = {
            "final_response": "ok", "messages": [], "api_calls": 1
        }
        fake_agent.session_prompt_tokens = 0
        fake_agent.session_completion_tokens = 0
        fake_agent.session_total_tokens = 0
        fake_agent.session_id = "sid"

        app = _create_app(adapter)
        with patch(
            "gateway.platforms.api_server._decide_api_server_image_mode",
            return_value="text",
        ), patch(
            "gateway.platforms.api_server._describe_image_via_vision",
            new=AsyncMock(return_value="A cat on a mat."),
        ), patch.object(
            adapter, "_create_agent", return_value=fake_agent,
        ):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": "Describe."},
                                    {"type": "input_image", "image_url": "https://example.com/cat.png"},
                                ],
                            }
                        ],
                    },
                )
                assert resp.status == 200, await resp.text()

        forwarded = fake_agent.run_conversation.call_args.kwargs["user_message"]
        assert isinstance(forwarded, str)
        assert "A cat on a mat." in forwarded
        assert forwarded.endswith("Describe.")
