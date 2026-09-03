"""Tests for reasoning/thinking exposure across the API server (#48024).

Covers the sweeper's core requirement: /v1/chat/completions and /v1/responses
must source reasoning from the STRUCTURED ``reasoning_callback`` path (the one
``_fire_reasoning_delta`` drives), NOT from the ``reasoning.available`` progress
event (which conversation_loop derives from the assistant message content).

Both endpoints are gated on ``display.platforms.api_server.show_reasoning`` and
emit reasoning in the correct streaming shape for each surface:

- chat/completions: ``delta.reasoning_content`` chunks (stream) +
  ``message.reasoning_content`` (non-stream).
- responses: the spec reasoning event family (``response.output_item.added`` →
  ``reasoning_summary_part.added`` → ``reasoning_summary_text.delta`` → ``.done``
  → ``output_item.done``) plus ``reasoning`` output items in the envelope.
"""

import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tests.gateway.test_api_server import _create_app


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True))


def _enable_reasoning(adapter):
    """Force the show_reasoning gate on for a test."""
    adapter._reasoning_exposure_enabled = lambda: True


def _disable_reasoning(adapter):
    adapter._reasoning_exposure_enabled = lambda: False


# ---------------------------------------------------------------------------
# Gate helper
# ---------------------------------------------------------------------------


class TestReasoningExposureGate:
    def test_gate_reads_display_setting(self, adapter, monkeypatch):
        monkeypatch.setattr(
            "gateway.run._load_gateway_config", lambda: {"_fake": True}
        )
        monkeypatch.setattr(
            "gateway.display_config.resolve_display_setting",
            lambda cfg, platform, name, default: True,
        )
        assert adapter._reasoning_exposure_enabled() is True

    def test_gate_defaults_off(self, adapter, monkeypatch):
        monkeypatch.setattr(
            "gateway.run._load_gateway_config", lambda: {}
        )
        monkeypatch.setattr(
            "gateway.display_config.resolve_display_setting",
            lambda cfg, platform, name, default: default,
        )
        assert adapter._reasoning_exposure_enabled() is False

    def test_gate_fails_closed_on_config_error(self, adapter, monkeypatch):
        def _boom():
            raise RuntimeError("config parse failed")

        monkeypatch.setattr("gateway.run._load_gateway_config", _boom)
        # Must not raise — a broken config can't 500 every request.
        assert adapter._reasoning_exposure_enabled() is False


# ---------------------------------------------------------------------------
# chat/completions — structured reasoning
# ---------------------------------------------------------------------------


class TestChatCompletionsReasoning:
    @pytest.mark.asyncio
    async def test_stream_emits_reasoning_content_from_callback(self, adapter):
        """Structured reasoning_callback deltas → delta.reasoning_content."""
        _enable_reasoning(adapter)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                rc = kwargs.get("reasoning_callback")
                cb = kwargs.get("stream_delta_callback")
                # The gateway must wire a real reasoning_callback when the
                # gate is on — this is the structured path, not
                # reasoning.available.
                assert rc is not None
                rc("Let me think ")
                rc("about this.")
                if cb:
                    cb("Answer.")
                return (
                    {"final_response": "Answer.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            import unittest.mock as _m
            with _m.patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        reasoning_deltas = []
        content_deltas = []
        for line in body.splitlines():
            if not line.startswith("data: "):
                continue
            raw = line[len("data: "):]
            if raw.strip() == "[DONE]":
                continue
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if "reasoning_content" in delta:
                    reasoning_deltas.append(delta["reasoning_content"])
                if delta.get("content"):
                    content_deltas.append(delta["content"])

        assert "".join(reasoning_deltas) == "Let me think about this."
        assert "".join(content_deltas) == "Answer."

    @pytest.mark.asyncio
    async def test_stream_no_reasoning_when_gate_off(self, adapter):
        """Gate off → no reasoning_callback wired, byte-identical wire."""
        _disable_reasoning(adapter)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            captured = {}

            async def _mock_run_agent(**kwargs):
                captured["reasoning_callback"] = kwargs.get("reasoning_callback")
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Answer.")
                return (
                    {"final_response": "Answer.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            import unittest.mock as _m
            with _m.patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        assert captured["reasoning_callback"] is None
        assert "reasoning_content" not in body

    @pytest.mark.asyncio
    async def test_non_stream_reasoning_content_sibling(self, adapter):
        """Non-stream: reasoning_content is a sibling of content."""
        _enable_reasoning(adapter)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                rc = kwargs.get("reasoning_callback")
                assert rc is not None
                rc("Structured thinking.")
                return (
                    {"final_response": "Final.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            import unittest.mock as _m
            with _m.patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert resp.status == 200
                data = await resp.json()

        msg = data["choices"][0]["message"]
        assert msg["content"] == "Final."
        assert msg["reasoning_content"] == "Structured thinking."

    @pytest.mark.asyncio
    async def test_non_stream_no_reasoning_key_when_gate_off(self, adapter):
        _disable_reasoning(adapter)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            captured = {}

            async def _mock_run_agent(**kwargs):
                captured["reasoning_callback"] = kwargs.get("reasoning_callback")
                return (
                    {"final_response": "Final.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            import unittest.mock as _m
            with _m.patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert resp.status == 200
                data = await resp.json()

        assert captured["reasoning_callback"] is None
        assert "reasoning_content" not in data["choices"][0]["message"]


# ---------------------------------------------------------------------------
# responses — spec reasoning event family
# ---------------------------------------------------------------------------


class TestResponsesReasoning:
    @pytest.mark.asyncio
    async def test_stream_emits_reasoning_event_family(self, adapter):
        """Structured reasoning_callback → the spec reasoning event family."""
        _enable_reasoning(adapter)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                rc = kwargs.get("reasoning_callback")
                cb = kwargs.get("stream_delta_callback")
                assert rc is not None
                rc("Thinking hard ")
                rc("about it.")
                if cb:
                    cb("The answer.")
                return (
                    {"final_response": "The answer.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            import unittest.mock as _m
            with _m.patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Question?",
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        events = [
            line[len("event: "):]
            for line in body.splitlines()
            if line.startswith("event: ")
        ]
        # The reasoning family must appear, and in order relative to the item.
        assert "response.reasoning_summary_part.added" in events
        assert "response.reasoning_summary_text.delta" in events
        assert "response.reasoning_summary_text.done" in events

        # Collect the reasoning text from the summary deltas.
        deltas = []
        for line in body.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "response.reasoning_summary_text.delta":
                deltas.append(payload["delta"])
        assert "".join(deltas) == "Thinking hard about it."

        # A reasoning output_item.added must precede the summary part, and its
        # item type must be "reasoning".
        reasoning_item_added = False
        for line in body.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                continue
            if (
                payload.get("type") == "response.output_item.added"
                and payload.get("item", {}).get("type") == "reasoning"
            ):
                reasoning_item_added = True
        assert reasoning_item_added

    @pytest.mark.asyncio
    async def test_stream_reasoning_closes_before_text(self, adapter):
        """A reasoning burst closes (done) before answer text is emitted."""
        _enable_reasoning(adapter)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                rc = kwargs.get("reasoning_callback")
                cb = kwargs.get("stream_delta_callback")
                rc("Reason.")
                if cb:
                    cb("Text.")
                return (
                    {"final_response": "Text.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            import unittest.mock as _m
            with _m.patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Q?",
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        ordered_types = []
        for line in body.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                continue
            t = payload.get("type")
            if t in (
                "response.reasoning_summary_text.done",
                "response.output_text.delta",
            ):
                ordered_types.append(t)
        # The reasoning summary must be finalized before the first text delta.
        assert ordered_types[0] == "response.reasoning_summary_text.done"
        assert "response.output_text.delta" in ordered_types

    @pytest.mark.asyncio
    async def test_stream_no_reasoning_when_gate_off(self, adapter):
        _disable_reasoning(adapter)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            captured = {}

            async def _mock_run_agent(**kwargs):
                captured["reasoning_callback"] = kwargs.get("reasoning_callback")
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Answer.")
                return (
                    {"final_response": "Answer.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            import unittest.mock as _m
            with _m.patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Q?",
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        assert captured["reasoning_callback"] is None
        assert "reasoning_summary" not in body


# ---------------------------------------------------------------------------
# responses non-stream — _extract_output_items include_reasoning
# ---------------------------------------------------------------------------


class TestExtractOutputItemsReasoning:
    def test_reasoning_item_from_message_field(self):
        result = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "hi",
                    "reasoning_content": "my structured reasoning",
                },
            ],
            "final_response": "hi",
        }
        items = APIServerAdapter._extract_output_items(result, include_reasoning=True)
        reasoning = [it for it in items if it.get("type") == "reasoning"]
        assert len(reasoning) == 1
        assert reasoning[0]["summary"][0]["text"] == "my structured reasoning"
        assert reasoning[0]["status"] == "completed"

    def test_reasoning_falls_back_to_reasoning_key(self):
        result = {
            "messages": [
                {"role": "assistant", "content": "hi", "reasoning": "alt field"},
            ],
            "final_response": "hi",
        }
        items = APIServerAdapter._extract_output_items(result, include_reasoning=True)
        reasoning = [it for it in items if it.get("type") == "reasoning"]
        assert len(reasoning) == 1
        assert reasoning[0]["summary"][0]["text"] == "alt field"

    def test_no_reasoning_item_when_flag_off(self):
        result = {
            "messages": [
                {"role": "assistant", "content": "hi", "reasoning_content": "x"},
            ],
            "final_response": "hi",
        }
        items = APIServerAdapter._extract_output_items(result, include_reasoning=False)
        assert not any(it.get("type") == "reasoning" for it in items)

    def test_no_reasoning_item_when_field_blank(self):
        result = {
            "messages": [
                {"role": "assistant", "content": "hi", "reasoning_content": "   "},
            ],
            "final_response": "hi",
        }
        items = APIServerAdapter._extract_output_items(result, include_reasoning=True)
        assert not any(it.get("type") == "reasoning" for it in items)

    @pytest.mark.asyncio
    async def test_non_stream_response_includes_reasoning_output_item(self, adapter):
        _enable_reasoning(adapter)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            import unittest.mock as _m
            with _m.patch.object(adapter, "_run_agent", new_callable=_m.AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "answer",
                        "messages": [
                            {
                                "role": "assistant",
                                "content": "answer",
                                "reasoning_content": "the thinking",
                            },
                        ],
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Q?"},
                )
                assert resp.status == 200
                data = await resp.json()

        types = [it.get("type") for it in data["output"]]
        assert "reasoning" in types


# ---------------------------------------------------------------------------
# responses input hardening — echoed reasoning items are skipped
# ---------------------------------------------------------------------------


class TestResponsesInputHardening:
    @pytest.mark.asyncio
    async def test_echoed_reasoning_item_in_input_is_skipped(self, adapter):
        """A trailing reasoning item in ``input`` must not 400 the request."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            captured = {}

            import unittest.mock as _m

            async def _mock_run_agent(**kwargs):
                captured["user_message"] = kwargs.get("user_message")
                captured["history"] = kwargs.get("conversation_history")
                return (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )

            with _m.patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": [
                            {"role": "user", "content": "hi"},
                            {
                                "type": "reasoning",
                                "id": "rs_abc",
                                "summary": [{"type": "summary_text", "text": "prev"}],
                            },
                        ],
                    },
                )
                assert resp.status == 200

        # The reasoning item was dropped, so the real user turn is the input.
        assert captured["user_message"] == "hi"

    @pytest.mark.asyncio
    async def test_echoed_reasoning_item_in_history_is_skipped(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            import unittest.mock as _m

            async def _mock_run_agent(**kwargs):
                return (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )

            with _m.patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "next",
                        "conversation_history": [
                            {"role": "user", "content": "earlier"},
                            {
                                "type": "reasoning",
                                "id": "rs_xyz",
                                "summary": [{"type": "summary_text", "text": "t"}],
                            },
                        ],
                    },
                )
                # Without the skip, the reasoning entry (no role/content) 400s.
                assert resp.status == 200
