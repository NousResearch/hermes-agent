"""Client projections must not expose model-only compaction scaffolding."""

from __future__ import annotations

from aiohttp.test_utils import TestClient, TestServer
import pytest

from agent.compaction_display import project_compaction_message_for_display
from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
)
from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _is_compressed_summary_message,
)
from hermes_state import SessionDB


STANDALONE_SUMMARY = (
    f"{SUMMARY_PREFIX}\n\n"
    f"{HISTORICAL_TASK_HEADING}\nold work\n\n"
    f"{_SUMMARY_END_MARKER}"
)
MERGED_CARRIER = (
    f"{_MERGED_PRIOR_CONTEXT_HEADER}\n"
    "Refactor complete.\n\n"
    f"{_MERGED_SUMMARY_DELIMITER}\n\n"
    f"{STANDALONE_SUMMARY}"
)
REAL_USER = "test the browser controller again"
USER_CARRIER = f"{STANDALONE_SUMMARY}\n\n{REAL_USER}"


def _row(role: str, content, **extra) -> dict:
    row = {"id": 1, "session_id": "s1", "role": role, "content": content}
    row.update(extra)
    return row


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


def _messages_app(adapter: APIServerAdapter):
    from aiohttp import web

    app = web.Application()
    app.router.add_get(
        "/api/sessions/{session_id}/messages",
        adapter._handle_session_messages,
    )
    return app


class TestMessageProjection:
    def test_standalone_summary_is_hidden_without_scaffolding(self):
        projected = APIServerAdapter._message_response(
            _row(
                "user",
                STANDALONE_SUMMARY,
                tool_calls=[{"id": "stale"}],
                reasoning="internal compression reasoning",
                reasoning_content="internal compression reasoning",
                reasoning_details=[{"type": "reasoning.summary", "summary": "internal"}],
                codex_reasoning_items=[{"type": "reasoning", "id": "internal"}],
                codex_message_items=[{"type": "message", "id": "internal"}],
            )
        )

        assert projected["content"] == ""
        assert projected["display_kind"] == "hidden"
        assert "tool_calls" not in projected
        assert "finish_reason" not in projected
        assert "reasoning" not in projected
        assert "reasoning_content" not in projected
        assert "reasoning_details" not in projected
        assert "codex_reasoning_items" not in projected
        assert "codex_message_items" not in projected

    def test_merged_carrier_preserves_only_real_prior_content(self):
        projected = APIServerAdapter._message_response(
            _row(
                "assistant",
                MERGED_CARRIER,
                tool_calls=[{"id": "prior-call"}],
                finish_reason="tool_calls",
            )
        )

        assert projected["content"] == "Refactor complete."
        assert "tool_calls" not in projected
        assert "finish_reason" not in projected
        assert "PRIOR CONTEXT" not in projected["content"]
        assert "CONTEXT COMPACTION" not in projected["content"]

    def test_merged_content_array_preserves_blocks_before_summary(self):
        projected = APIServerAdapter._message_response(
            _row(
                "user",
                [
                    {
                        "type": "text",
                        "text": f"{_MERGED_PRIOR_CONTEXT_HEADER}\n{REAL_USER}",
                    },
                    {
                        "type": "text",
                        "text": f"{_MERGED_SUMMARY_DELIMITER}\n{STANDALONE_SUMMARY}",
                    },
                ],
            )
        )

        assert projected["content"] == [{"type": "text", "text": REAL_USER}]

    def test_real_message_that_mentions_marker_text_is_untouched(self):
        message = _row(
            "user",
            "please explain the string [CONTEXT COMPACTION] in this bug report",
            tool_calls=[{"id": "real"}],
            reasoning="real provider payload",
        )
        projected = project_compaction_message_for_display(message)

        assert projected == message
        assert projected is not message

    def test_unrelated_hidden_message_is_not_reclassified_as_compaction(self):
        message = _row("assistant", "ordinary hidden control row", display_kind="hidden")

        assert _is_compressed_summary_message(message) is False


class TestSummaryRecognizer:
    @pytest.mark.parametrize(
        "message",
        [
            _row("user", STANDALONE_SUMMARY),
            _row("assistant", MERGED_CARRIER),
            _row("assistant", "metadata-only", **{COMPRESSED_SUMMARY_METADATA_KEY: True}),
        ],
    )
    def test_recognizes_all_compaction_carrier_shapes(self, message):
        assert _is_compressed_summary_message(message) is True

    def test_ignores_real_message(self):
        assert _is_compressed_summary_message(_row("user", REAL_USER)) is False


class TestTurnTranscriptProjection:
    def test_run_completed_strips_scaffolding_but_keeps_real_carrier_content(self):
        result = {
            "messages": [
                {"role": "user", "content": REAL_USER},
                {"role": "assistant", "content": "checking the controller"},
                _row("user", STANDALONE_SUMMARY),
                _row("assistant", MERGED_CARRIER),
                {"role": "assistant", "content": "the controller is ready"},
            ],
            "final_response": "the controller is ready",
        }

        turn = APIServerAdapter._turn_transcript_messages(
            [{"role": "user", "content": REAL_USER}],
            REAL_USER,
            result,
        )

        assert [message.get("content") for message in turn] == [
            "checking the controller",
            "Refactor complete.",
            "the controller is ready",
        ]


class TestMessagesEndpointProjection:
    @pytest.mark.asyncio
    async def test_messages_endpoint_dedupes_projected_compaction_carrier(
        self,
        adapter,
        session_db,
    ):
        session_id = session_db.create_session(
            "projection-dedupe", "api_server"
        )
        session_db.append_message(
            session_id, "user", REAL_USER, timestamp=123.0
        )
        session_db.archive_and_compact(
            session_id,
            [{"role": "user", "content": USER_CARRIER, "timestamp": 123.0}],
        )
        carrier_id = session_db.get_messages(session_id)[0]["id"]
        later_id = session_db.append_message(
            session_id, "assistant", "later answer", timestamp=124.0
        )
        raw_page = session_db.get_display_message_page(
            session_id,
            limit=120,
            latest=False,
            include_compacted=True,
        )
        assert [message["content"] for message in raw_page["messages"]] == [
            REAL_USER,
            USER_CARRIER,
            "later answer",
        ]

        async with TestClient(TestServer(_messages_app(adapter))) as client:
            first_response = await client.get(
                f"/api/sessions/{session_id}/messages"
                "?include_compacted=true&limit=2&offset=0&order=oldest"
            )
            second_response = await client.get(
                f"/api/sessions/{session_id}/messages"
                "?include_compacted=true&limit=2&offset=2&order=oldest"
            )
            assert first_response.status == 200
            assert second_response.status == 200
            payload = await first_response.json()
            second_payload = await second_response.json()

        assert len(payload["data"]) == 1
        assert payload["data"][0]["id"] == carrier_id
        assert payload["data"][0]["content"] == REAL_USER
        assert [message["id"] for message in second_payload["data"]] == [
            later_id
        ]
        assert not {
            message["id"] for message in payload["data"]
        } & {message["id"] for message in second_payload["data"]}
        assert payload["pagination"] == {
            "limit": 2,
            "offset": 0,
            "order": "oldest",
            "returned": 2,
        }

    @pytest.mark.asyncio
    async def test_messages_endpoint_preserves_reasoning_distinct_projection(
        self,
        adapter,
        session_db,
    ):
        session_id = session_db.create_session(
            "projection-reasoning-distinct", "api_server"
        )
        session_db.append_message(
            session_id,
            "assistant",
            "Refactor complete.",
            reasoning="source reasoning",
            timestamp=123.0,
        )
        session_db.archive_and_compact(
            session_id,
            [{"role": "assistant", "content": MERGED_CARRIER, "timestamp": 123.0}],
        )

        async with TestClient(TestServer(_messages_app(adapter))) as client:
            response = await client.get(
                f"/api/sessions/{session_id}/messages?include_compacted=true"
            )
            assert response.status == 200
            payload = await response.json()

        assert [message["content"] for message in payload["data"]] == [
            "Refactor complete.",
            "Refactor complete.",
        ]
        assert [message.get("reasoning") for message in payload["data"]] == [
            "source reasoning",
            None,
        ]

    @pytest.mark.asyncio
    async def test_messages_endpoint_never_serves_compaction_scaffolding(
        self,
        adapter,
        session_db,
    ):
        session_id = session_db.create_session("projection-session", "api_server")
        session_db.replace_messages(
            session_id,
            [
                _row("user", STANDALONE_SUMMARY),
                _row("assistant", MERGED_CARRIER),
                _row("user", REAL_USER),
            ],
        )

        async with TestClient(TestServer(_messages_app(adapter))) as client:
            response = await client.get(f"/api/sessions/{session_id}/messages")
            assert response.status == 200
            payload = await response.json()

        messages = payload["data"]
        assert payload["session_id"] == session_id
        assert payload["lineage_root_id"] == session_id
        assert payload["resolved_tip_id"] == session_id
        assert payload["display_revision"] == session_db.get_display_revision(session_id)
        assert payload["unchanged"] is False
        assert len(messages) == 3
        assert messages[0]["content"] == ""
        assert messages[0]["display_kind"] == "hidden"
        assert messages[1]["content"] == "Refactor complete."
        assert messages[2]["content"] == REAL_USER
        rendered = " ".join(str(message.get("content") or "") for message in messages)
        assert "PRIOR CONTEXT" not in rendered
        assert "CONTEXT COMPACTION" not in rendered
