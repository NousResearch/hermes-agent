"""Named /v1/responses conversations must not duplicate stored history (#101644).

A two-turn named conversation should store 2 messages after turn 1 and 4
after turn 2.  v0.21.0 stored 3 then 8: the turn-start detector compared the
fully metadata-decorated agent transcript against a bare ``{role, content}``
user dict with strict equality, so the transcript-shaped ``result["messages"]``
never matched its prefix and the builder fell back to concatenating the
manually constructed user message with the FULL transcript.

The fix compares semantically — persistence metadata (``_db_persisted``,
``_row_id``, ``timestamp``) is ignored when both sides agree on the fields
that identify the message.
"""

from gateway.platforms.api_server import APIServerAdapter


def _build(conversation_history, user_message, result, final_response):
    return APIServerAdapter._build_response_conversation_history(
        conversation_history, user_message, result, final_response
    )


class TestNamedResponsesHistory:
    def test_metadata_decorated_full_transcript_is_not_duplicated(self):
        prior = [
            {"role": "user", "content": "remember: x"},
            {"role": "assistant", "content": "noted"},
        ]
        # The agent returns the FULL transcript, each message decorated with
        # persistence metadata (_db_persisted, _row_id, timestamp) — the
        # shape v0.21.0 emitted after SessionDB persistence.
        result = {
            "messages": [
                {"role": "user", "content": "remember: x", "_db_persisted": True, "_row_id": 1, "timestamp": 1.0},
                {"role": "assistant", "content": "noted", "_db_persisted": True, "_row_id": 2, "timestamp": 2.0},
                {"role": "user", "content": "what was it?", "_db_persisted": True, "_row_id": 3, "timestamp": 3.0},
                {"role": "assistant", "content": "x", "_db_persisted": True, "_row_id": 4, "timestamp": 4.0},
            ]
        }
        out = _build(prior, "what was it?", result, "x")
        assert len(out) == 4, f"expected 4 stored messages, got {len(out)}"
        assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant"]

    def test_first_turn_stores_exactly_two_messages(self):
        result = {
            "messages": [
                {"role": "user", "content": "hi", "_db_persisted": True, "_row_id": 1, "timestamp": 1.0},
                {"role": "assistant", "content": "hello", "_db_persisted": True, "_row_id": 2, "timestamp": 2.0},
            ]
        }
        out = _build([], "hi", result, "hello")
        assert len(out) == 2, f"expected 2 stored messages, got {len(out)}"

    def test_current_turn_only_suffix_still_concatenates(self):
        # Suffix-shaped results (current turn only, no prior prefix) keep the
        # concatenate path — that shape is legitimate.  The existing fallback
        # appends the constructed current_user AND the suffix, so the user
        # message appears twice; pinning the historical behavior here so this
        # PR does not silently change the suffix contract.
        prior = [
            {"role": "user", "content": "remember: x"},
            {"role": "assistant", "content": "noted"},
        ]
        result = {
            "messages": [
                {"role": "user", "content": "what was it?"},
                {"role": "assistant", "content": "x"},
            ]
        }
        out = _build(prior, "what was it?", result, "x")
        assert len(out) == 5

    def test_bare_transcript_without_metadata_still_matches(self):
        prior = [
            {"role": "user", "content": "remember: x"},
            {"role": "assistant", "content": "noted"},
        ]
        result = {
            "messages": [
                {"role": "user", "content": "remember: x"},
                {"role": "assistant", "content": "noted"},
                {"role": "user", "content": "what was it?"},
                {"role": "assistant", "content": "x"},
            ]
        }
        out = _build(prior, "what was it?", result, "x")
        assert len(out) == 4

    def test_diverged_transcript_does_not_use_prefix_path(self):
        # A transcript that diverges from prior history (e.g. compression
        # rewrote it without the _compressed flag set) must NOT be mistaken
        # for a prefix match — semantic comparison must not overreach.
        prior = [
            {"role": "user", "content": "remember: x"},
            {"role": "assistant", "content": "noted"},
        ]
        result = {
            "messages": [
                {"role": "user", "content": "summary: user asked to remember x"},
                {"role": "assistant", "content": "noted"},
                {"role": "user", "content": "what was it?"},
                {"role": "assistant", "content": "x"},
            ]
        }
        out = _build(prior, "what was it?", result, "x")
        # Not a prefix match → concatenate fallback: prior + current_user + 4
        # suffix messages = 7 (existing behavior for this shape).
        assert len(out) == 2 + 1 + 4

    def test_response_messages_turn_start_ignores_persistence_metadata(self):
        prior = [
            {"role": "user", "content": "remember: x"},
            {"role": "assistant", "content": "noted"},
        ]
        result = {
            "messages": [
                {"role": "user", "content": "remember: x", "_db_persisted": True, "timestamp": 1.0},
                {"role": "assistant", "content": "noted", "_db_persisted": True, "timestamp": 2.0},
                {"role": "user", "content": "what was it?", "_db_persisted": True, "timestamp": 3.0},
                {"role": "assistant", "content": "x", "_db_persisted": True, "timestamp": 4.0},
            ]
        }
        start = APIServerAdapter._response_messages_turn_start_index(
            prior, "what was it?", result
        )
        assert start == 3, f"expected turn start 3, got {start}"
