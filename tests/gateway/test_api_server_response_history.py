"""Stored Responses transcripts must not duplicate prior history.

Regression: `_response_messages_turn_start_index` compared the agent's returned rows against
the plain input history with exact-dict equality. The agent stamps transport metadata
(``timestamp``, ``_row_id``, ``_db_persisted``, ``reasoning``, ...) onto those rows, so the
comparison failed and `_build_response_conversation_history` re-prepended the whole transcript
every turn (roughly doubling it until compaction).
"""

from gateway.platforms.api_server import APIServerAdapter


def _enriched(role: str, content: str, **extra) -> dict:
    row = {"role": role, "content": content}
    row.update(extra)
    return row


class TestResponseConversationHistory:
    def test_enriched_full_transcript_is_stored_once(self):
        prior = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = {
            "messages": [
                _enriched("user", "hello", _row_id=1, timestamp=1.0, _db_persisted=True),
                _enriched("assistant", "hi", finish_reason="stop", reasoning=""),
                _enriched("user", "next", _row_id=3, timestamp=2.0, _db_persisted=True),
                _enriched("assistant", "ok", finish_reason="stop"),
            ],
            "final_response": "ok",
        }

        history = APIServerAdapter._build_response_conversation_history(prior, "next", result, "ok")

        assert history == result["messages"]
        assert [message["role"] for message in history] == ["user", "assistant", "user", "assistant"]

    def test_turn_only_result_still_prepends_prior(self):
        prior = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        result = {"messages": [_enriched("assistant", "ok", finish_reason="stop")], "final_response": "ok"}

        history = APIServerAdapter._build_response_conversation_history(prior, "next", result, "ok")

        assert history == prior + [
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "ok", "finish_reason": "stop"},
        ]
