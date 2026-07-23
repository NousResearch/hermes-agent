"""Tests for /v1/responses history image pruning and the compression-duplication fix.

Covers:
- ``_prune_history_images`` (pure function) — keeps the newest N image parts,
  replaces older ones with a text placeholder, never mutates its input.
- ``APIServerAdapter._resolve_max_history_images`` — config default/override/
  garbage-fallback for ``gateway.api_server.max_history_images``.
- Integration: pruning applied on the chain-load and store paths of
  /v1/responses.
- ``_build_response_conversation_history`` honoring ``result["history_compressed"]``
  to avoid duplicating history when agent-side context compression rewrote the
  in-memory transcript mid-turn.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _prune_history_images,
    cors_middleware,
    security_headers_middleware,
)


def _img(name: str) -> dict:
    # A real-looking https URL: fixtures that flow through the HTTP endpoint
    # hit ``_normalize_multimodal_content``'s URL-scheme validation, while
    # fixtures injected straight into the response store don't care either
    # way — so use one shape everywhere.
    return {"type": "image_url", "image_url": {"url": f"https://example.com/{name}.png"}}


def _text(text: str) -> dict:
    return {"type": "text", "text": text}


# ---------------------------------------------------------------------------
# _prune_history_images
# ---------------------------------------------------------------------------


class TestPruneHistoryImages:
    def _sample_history(self):
        # 4 image parts total, oldest-to-newest: img1, img2, img3, img4.
        return [
            {"role": "user", "content": [_text("look"), _img("img1")]},
            {"role": "assistant", "content": "ok, describing"},
            {"role": "user", "content": [_img("img2"), _text("and this one")]},
            {"role": "user", "content": [_img("img3")]},
            {"role": "user", "content": [_img("img4"), _text("last one")]},
        ]

    def test_keeps_newest_n_replaces_older(self):
        history = self._sample_history()
        pruned = _prune_history_images(history, keep_last=3)

        # img1 (oldest) is replaced; img2/img3/img4 survive verbatim.
        assert pruned[0]["content"] == [_text("look"), {"type": "text", "text": "[older image omitted from context]"}]
        assert pruned[2]["content"] == [_img("img2"), _text("and this one")]
        assert pruned[3]["content"] == [_img("img3")]
        assert pruned[4]["content"] == [_img("img4"), _text("last one")]

    def test_text_and_string_content_preserved(self):
        history = self._sample_history()
        pruned = _prune_history_images(history, keep_last=3)

        # Plain-string content untouched, and unmodified messages pass
        # through by reference (not deep-copied).
        assert pruned[1]["content"] == "ok, describing"
        assert pruned[1] is history[1]

    def test_no_messages_deleted(self):
        history = self._sample_history()
        pruned = _prune_history_images(history, keep_last=0)
        assert len(pruned) == len(history)
        assert [m["role"] for m in pruned] == [m["role"] for m in history]

    def test_does_not_mutate_input_list_or_dicts(self):
        history = self._sample_history()
        original_first_content = list(history[0]["content"])
        _prune_history_images(history, keep_last=0)
        assert history[0]["content"] == original_first_content
        assert history[0]["content"][1] == _img("img1")

    def test_keep_last_negative_one_is_noop(self):
        history = self._sample_history()
        pruned = _prune_history_images(history, keep_last=-1)
        assert pruned is history

    def test_keep_last_zero_replaces_all_images(self):
        history = self._sample_history()
        pruned = _prune_history_images(history, keep_last=0)
        placeholder = {"type": "text", "text": "[older image omitted from context]"}
        assert pruned[0]["content"] == [_text("look"), placeholder]
        assert pruned[2]["content"] == [placeholder, _text("and this one")]
        assert pruned[3]["content"] == [placeholder]
        assert pruned[4]["content"] == [placeholder, _text("last one")]

    def test_history_with_no_images_returned_equivalent(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [_text("hi there")]},
        ]
        pruned = _prune_history_images(history, keep_last=3)
        assert pruned == history

    def test_keep_last_at_or_above_total_is_noop(self):
        history = self._sample_history()
        pruned = _prune_history_images(history, keep_last=4)
        assert pruned == history

    def test_anthropic_style_image_type_pruned(self):
        # Stored history can carry Anthropic-shape "image" blocks, not just
        # "image_url"/"input_image".
        history = [
            {"role": "user", "content": [{"type": "image", "source": {"type": "base64", "data": "AAAA"}}]},
            {"role": "user", "content": [_img("img2")]},
        ]
        pruned = _prune_history_images(history, keep_last=1)
        assert pruned[0]["content"] == [{"type": "text", "text": "[older image omitted from context]"}]
        assert pruned[1]["content"] == [_img("img2")]


# ---------------------------------------------------------------------------
# gateway.api_server.max_history_images config resolution
# ---------------------------------------------------------------------------


class TestMaxHistoryImagesConfig:
    def test_resolve_defaults_to_3_when_unset(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            assert APIServerAdapter._resolve_max_history_images() == 3

    def test_resolve_reads_config_value(self):
        cfg = {"gateway": {"api_server": {"max_history_images": 7}}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert APIServerAdapter._resolve_max_history_images() == 7

    def test_resolve_honors_disable_sentinel(self):
        cfg = {"gateway": {"api_server": {"max_history_images": -1}}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert APIServerAdapter._resolve_max_history_images() == -1

    def test_resolve_garbage_falls_back_to_default(self):
        cfg = {"gateway": {"api_server": {"max_history_images": "not-an-int"}}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert APIServerAdapter._resolve_max_history_images() == 3


# ---------------------------------------------------------------------------
# Integration: pruning applied on the /v1/responses chain-load/store paths
# ---------------------------------------------------------------------------


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/responses", adapter._handle_responses)
    return app


@pytest.fixture
def adapter():
    a = _make_adapter()
    a._max_history_images = 3
    return a


class TestHistoryPruningIntegration:
    @pytest.mark.asyncio
    async def test_loaded_chain_history_passed_to_agent_is_pruned(self, adapter):
        """Loading a previous_response_id chain prunes before it reaches _run_agent."""
        stored_history = [
            {"role": "user", "content": [_img("img1"), _text("first")]},
            {"role": "user", "content": [_img("img2")]},
            {"role": "user", "content": [_img("img3")]},
            {"role": "user", "content": [_img("img4")]},
        ]
        adapter._response_store.put(
            "resp_prev",
            {
                "response": {"id": "resp_prev", "status": "completed"},
                "conversation_history": stored_history,
                "session_id": "s1",
            },
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "one more",
                        "previous_response_id": "resp_prev",
                    },
                )

            assert resp.status == 200
            passed_history = mock_run.call_args.kwargs["conversation_history"]
            assert passed_history[0]["content"] == [
                {"type": "text", "text": "[older image omitted from context]"},
                _text("first"),
            ]
            assert passed_history[1]["content"] == [_img("img2")]
            assert passed_history[2]["content"] == [_img("img3")]
            assert passed_history[3]["content"] == [_img("img4")]
            # Original stored row untouched.
            assert adapter._response_store.get("resp_prev")["conversation_history"] == stored_history

    @pytest.mark.asyncio
    async def test_stored_history_after_call_is_pruned(self, adapter):
        """The conversation_history written to the response store is pruned."""
        agent_transcript = [
            {"role": "user", "content": [_img("img1")]},
            {"role": "assistant", "content": [_img("img2"), _text("described")]},
            {"role": "user", "content": [_img("img3")]},
            {"role": "assistant", "content": [_img("img4"), _text("described again")]},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "described again", "messages": agent_transcript, "api_calls": 1},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": [{"role": "user", "content": [_img("img1")]}]},
                )

            assert resp.status == 200
            data = await resp.json()
            stored = adapter._response_store.get(data["id"])
            image_parts = [
                part
                for msg in stored["conversation_history"]
                if isinstance(msg.get("content"), list)
                for part in msg["content"]
                if part.get("type") == "image_url"
            ]
            assert len(image_parts) == 3
            assert stored["conversation_history"][0]["content"] == [
                {"type": "text", "text": "[older image omitted from context]"}
            ]


# ---------------------------------------------------------------------------
# Compression-duplication fix — _build_response_conversation_history
# ---------------------------------------------------------------------------


class TestBuildResponseConversationHistoryCompressionFlag:
    def test_history_compressed_flag_returns_agent_transcript_verbatim(self):
        """When compression rewrote history mid-turn, the agent transcript is
        the canonical post-compression history — no prior-prefix duplication."""
        prior = [
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": "resp1"},
        ]
        # Compaction dropped the old turn entirely — no prefix relationship
        # to `prior` at all.
        agent_messages = [
            {"role": "system", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "turn2"},
            {"role": "assistant", "content": "resp2"},
        ]
        result = {"messages": agent_messages, "history_compressed": True}

        out = APIServerAdapter._build_response_conversation_history(
            prior, "turn2", result, "resp2"
        )
        assert out == agent_messages
        assert out[: len(prior)] != prior

    def test_no_flag_exact_prefix_uses_old_path(self):
        """Regression: without the flag, exact-prefix match still short-circuits."""
        prior = [{"role": "user", "content": "turn1"}]
        agent_messages = prior + [
            {"role": "user", "content": "turn2"},
            {"role": "assistant", "content": "resp2"},
        ]
        result = {"messages": agent_messages}

        out = APIServerAdapter._build_response_conversation_history(
            prior, "turn2", result, "resp2"
        )
        assert out == agent_messages

    def test_no_flag_suffix_shaped_result_uses_old_append_path(self):
        """Regression: without the flag, a turn-suffix-only messages list still
        gets appended behind prior + current_user (byte-identical to today)."""
        prior = [{"role": "user", "content": "turn1"}]
        agent_messages = [{"role": "assistant", "content": "resp2"}]
        result = {"messages": agent_messages}

        out = APIServerAdapter._build_response_conversation_history(
            prior, "turn2", result, "resp2"
        )
        assert out == [
            {"role": "user", "content": "turn1"},
            {"role": "user", "content": "turn2"},
            {"role": "assistant", "content": "resp2"},
        ]

    def test_flag_set_but_no_messages_falls_back_to_final_response_path(self):
        """history_compressed=True with no agent transcript still falls back
        to the final_response append path (flag alone isn't enough)."""
        prior = [{"role": "user", "content": "turn1"}]
        result = {"messages": [], "history_compressed": True}

        out = APIServerAdapter._build_response_conversation_history(
            prior, "turn2", result, "resp2"
        )
        assert out == [
            {"role": "user", "content": "turn1"},
            {"role": "user", "content": "turn2"},
            {"role": "assistant", "content": "resp2"},
        ]


class TestConversationHistoryAfterCompressionSetsFlag:
    def test_sets_history_compressed_this_turn(self):
        from agent.conversation_compression import conversation_history_after_compression

        agent = SimpleNamespace(_history_compressed_this_turn=False)
        conversation_history_after_compression(agent, [{"role": "user", "content": "x"}])
        assert agent._history_compressed_this_turn is True
