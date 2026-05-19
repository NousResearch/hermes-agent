"""Regression coverage for the OpenAI-compatible API server's Unicode
handling, reported as #28646 (Hermes-web-ui showed ``\\u65e0 kanban
\\u76ee\\u5f55`` in tool-call rendering instead of ``无 kanban 目录``).

Background
----------
The API server emits multiple JSON payloads to the wire:

* ``function_call.arguments`` — a JSON-encoded *string* embedded in
  larger SSE event payloads. OpenAI-compatible web UIs frequently
  display this string verbatim instead of parsing it again, so any
  ``\\uXXXX`` escape sequences inside reach the user as literal text.
* ``function_call_output.output[0].text`` — same shape for tool
  results.
* ``response.completed`` — re-serializes the trimmed arguments map.
* ``data:`` chunks of ``/v1/chat/completions`` SSE and ``event:`` /
  ``data:`` envelopes of ``/v1/responses`` SSE.

Python's ``json.dumps`` defaults to ``ensure_ascii=True``, which
encodes every non-ASCII codepoint as a ``\\uXXXX`` escape — the root
cause of the mojibake. These tests pin ``ensure_ascii=False`` on
every wire-facing serialization in ``gateway/platforms/api_server.py``.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

# Reuse the helper that constructs an APIServerAdapter with the same
# config the rest of the suite uses. Importing as a module-level helper
# instead of a fixture keeps each test self-contained and easy to read.
from tests.gateway.test_api_server import _create_app, _make_adapter  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402


# ---------------------------------------------------------------------------
# Sample CJK fragments
# ---------------------------------------------------------------------------
#
# ``无`` (U+65E0, "no / none") and ``目录`` (U+76EE U+5F55, "directory")
# are the exact characters from the issue's screenshot. We also throw
# in an accented Latin character and an emoji so the regression covers
# the whole non-ASCII surface, not just CJK.

CJK_COMMAND = 'echo "无 kanban 目录"'           # 无 + 目录 — issue #28646
LATIN_DIACRITIC = "café"                       # U+00E9
EMOJI = "🚀"                                    # U+1F680


# ---------------------------------------------------------------------------
# /v1/responses streaming path — function_call.arguments
# ---------------------------------------------------------------------------


class TestFunctionCallArgumentsUnicode:
    """``_emit_tool_started`` JSON-encodes the ``arguments`` dict into a
    string that web UIs render directly. ``ensure_ascii=False`` is the
    only fix that lets them render CJK natively."""

    @pytest.mark.asyncio
    async def test_arguments_string_contains_native_cjk_chars(self):
        """The exact mojibake from issue #28646 must not reappear.

        Pre-fix: the SSE body contained ``"echo \\\"\\u65e0 kanban
        \\u76ee\\u5f55\\\""`` — six ASCII characters per CJK codepoint.
        Post-fix: the body carries the native ``无`` and ``目录``
        bytes, which is what the issue's screenshot reporter expected.
        """
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                start_cb = kwargs.get("tool_start_callback")
                complete_cb = kwargs.get("tool_complete_callback")
                if start_cb:
                    start_cb("call_cjk", "terminal", {"command": CJK_COMMAND})
                if complete_cb:
                    complete_cb("call_cjk", "terminal", {"command": CJK_COMMAND}, "ok")
                return (
                    {"final_response": "done", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "ls", "stream": True},
                )
                assert resp.status == 200
                body = await resp.text()

        # Positive: the native CJK chars survive end-to-end.
        assert "无" in body
        assert "目录" in body
        # Negative: the escape-sequence form (the user-visible mojibake)
        # must not appear anywhere in the stream.
        assert "\\u65e0" not in body
        assert "\\u76ee" not in body
        assert "\\u5f55" not in body

    @pytest.mark.asyncio
    async def test_arguments_string_is_still_valid_inner_json(self):
        """The fix mustn't break round-tripping: clients that *do*
        ``JSON.parse`` the inner ``arguments`` string still need a
        valid JSON object back.
        """
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                start_cb = kwargs.get("tool_start_callback")
                complete_cb = kwargs.get("tool_complete_callback")
                if start_cb:
                    start_cb("call_cjk", "terminal", {"command": CJK_COMMAND})
                if complete_cb:
                    complete_cb("call_cjk", "terminal", {"command": CJK_COMMAND}, "ok")
                return (
                    {"final_response": "done", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "ls", "stream": True},
                )
                body = await resp.text()

        # Find an output_item.added event for function_call and re-parse
        # its inner arguments string the way a smart client would.
        parsed_inner_args = None
        for line in body.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                continue
            item = payload.get("item") or {}
            if item.get("type") == "function_call" and item.get("name") == "terminal":
                parsed_inner_args = json.loads(item["arguments"])
                break
        assert parsed_inner_args is not None, "no function_call event found in SSE body"
        assert parsed_inner_args == {"command": CJK_COMMAND}


# ---------------------------------------------------------------------------
# /v1/responses streaming path — function_call_output text
# ---------------------------------------------------------------------------


class TestFunctionCallOutputUnicode:
    @pytest.mark.asyncio
    async def test_tool_result_dict_preserves_cjk(self):
        """When the tool returns a *dict*, ``_emit_tool_completed``
        JSON-encodes it. CJK characters in the result map must reach
        the client as native Unicode."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                start_cb = kwargs.get("tool_start_callback")
                complete_cb = kwargs.get("tool_complete_callback")
                if start_cb:
                    start_cb("call_a", "kanban_show", {"id": "work"})
                if complete_cb:
                    # ``result`` is a dict here (not a string), so it goes
                    # through ``json.dumps`` inside the API server.
                    complete_cb(
                        "call_a", "kanban_show", {"id": "work"},
                        {"title": "目录", "note": "café 🚀"},
                    )
                return (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "show", "stream": True},
                )
                body = await resp.text()

        assert "目录" in body
        assert "café" in body
        assert "🚀" in body
        assert "\\u76ee" not in body
        assert "\\u00e9" not in body


# ---------------------------------------------------------------------------
# response.completed envelope — re-serialized trimmed arguments
# ---------------------------------------------------------------------------


class TestResponseCompletedTrimmingUnicode:
    @pytest.mark.asyncio
    async def test_completed_envelope_carries_native_cjk_arguments(self):
        """``response.completed`` re-runs ``json.dumps`` on trimmed
        tool-call arguments. Without ``ensure_ascii=False`` it would
        silently reintroduce the mojibake fixed for the streaming
        path."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                start_cb = kwargs.get("tool_start_callback")
                complete_cb = kwargs.get("tool_complete_callback")
                if start_cb:
                    start_cb("call_cjk", "terminal", {"command": CJK_COMMAND})
                if complete_cb:
                    complete_cb("call_cjk", "terminal", {"command": CJK_COMMAND}, "ok")
                return (
                    {"final_response": "done", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "ls", "stream": True},
                )
                body = await resp.text()

        completed_arguments = None
        for line in body.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                continue
            if payload.get("type") != "response.completed":
                continue
            for item in payload.get("response", {}).get("output", []):
                if item.get("type") == "function_call":
                    completed_arguments = item["arguments"]
                    break
            break

        assert completed_arguments is not None, "no terminal function_call in response.completed envelope"
        # The string sent over the wire is native UTF-8…
        assert "无" in completed_arguments
        assert "目录" in completed_arguments
        # …and still re-parses back into a structured dict.
        assert json.loads(completed_arguments) == {"command": CJK_COMMAND}


# ---------------------------------------------------------------------------
# /v1/chat/completions delta.content envelope
# ---------------------------------------------------------------------------


class TestChatCompletionsDeltaUnicode:
    @pytest.mark.asyncio
    async def test_streamed_chat_completion_delivers_native_cjk_in_delta(self):
        """OpenAI Chat Completions SSE wraps each token in
        ``choices[0].delta.content``. Smart clients ``JSON.parse`` the
        chunk and recover Unicode either way, but naive log inspectors
        and dumps care about the literal wire bytes — keep CJK native
        to avoid stack traces that surface as ``\\u65e0`` in support
        threads."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("你好,café 🚀")
                    cb(None)
                return (
                    {"final_response": "你好,café 🚀", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                body = await resp.text()

        assert "你好" in body
        assert "café" in body
        assert "🚀" in body
        assert "\\u4f60" not in body  # 你
        assert "\\u597d" not in body  # 好
        assert "\\u00e9" not in body  # é
