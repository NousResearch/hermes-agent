"""Tests for hermes_state.py — SessionDB SQLite CRUD, FTS5 search, export."""

import time
import pytest
from pathlib import Path

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temp database file."""
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


# =========================================================================
# Session lifecycle
# =========================================================================

class TestSessionLifecycle:
    def test_create_and_get_session(self, db):
        sid = db.create_session(
            session_id="s1",
            source="cli",
            model="test-model",
        )
        assert sid == "s1"

        session = db.get_session("s1")
        assert session is not None
        assert session["source"] == "cli"
        assert session["model"] == "test-model"
        assert session["ended_at"] is None

    def test_get_nonexistent_session(self, db):
        assert db.get_session("nonexistent") is None

    def test_end_session(self, db):
        db.create_session(session_id="s1", source="cli")
        db.end_session("s1", end_reason="user_exit")

        session = db.get_session("s1")
        assert isinstance(session["ended_at"], float)
        assert session["end_reason"] == "user_exit"

    def test_end_session_preserves_original_end_reason(self, db):
        """The first end_reason wins — compression splits must not be
        overwritten when a later stale ``end_session()`` call lands on the
        same row (e.g. from a CLI session_id that desynced after compression
        and then tried to /resume another session).
        """
        db.create_session(session_id="s1", source="cli")
        db.end_session("s1", end_reason="compression")
        first_ended_at = db.get_session("s1")["ended_at"]

        # Simulate a stale CLI holding the old session_id and calling
        # end_session() again with a different reason.
        time.sleep(0.01)
        db.end_session("s1", end_reason="resumed_other")

        session = db.get_session("s1")
        assert session["end_reason"] == "compression"
        assert session["ended_at"] == first_ended_at

    def test_end_session_after_reopen_allows_re_end(self, db):
        """reopen_session() is the explicit escape hatch for re-ending a
        closed session. After reopen, end_session() takes effect again.
        """
        db.create_session(session_id="s1", source="cli")
        db.end_session("s1", end_reason="compression")
        db.reopen_session("s1")
        db.end_session("s1", end_reason="user_exit")

        session = db.get_session("s1")
        assert session["end_reason"] == "user_exit"

    def test_update_system_prompt(self, db):
        db.create_session(session_id="s1", source="cli")
        db.update_system_prompt("s1", "You are a helpful assistant.")

        session = db.get_session("s1")
        assert session["system_prompt"] == "You are a helpful assistant."

    def test_update_token_counts(self, db):
        db.create_session(session_id="s1", source="cli")
        db.update_token_counts("s1", input_tokens=200, output_tokens=100)
        db.update_token_counts("s1", input_tokens=100, output_tokens=50)

        session = db.get_session("s1")
        assert session["input_tokens"] == 300
        assert session["output_tokens"] == 150

    def test_update_token_counts_backfills_model_when_null(self, db):
        db.create_session(session_id="s1", source="telegram")
        db.update_token_counts("s1", input_tokens=10, output_tokens=5, model="openai/gpt-5.4")

        session = db.get_session("s1")
        assert session["model"] == "openai/gpt-5.4"

    def test_update_token_counts_preserves_existing_model(self, db):
        db.create_session(session_id="s1", source="cli", model="anthropic/claude-opus-4.6")
        db.update_token_counts("s1", input_tokens=10, output_tokens=5, model="openai/gpt-5.4")

        session = db.get_session("s1")
        assert session["model"] == "anthropic/claude-opus-4.6"

    def test_parent_session(self, db):
        db.create_session(session_id="parent", source="cli")
        db.create_session(session_id="child", source="cli", parent_session_id="parent")

        child = db.get_session("child")
        assert child["parent_session_id"] == "parent"


# =========================================================================
# Message storage
# =========================================================================

class TestMessageStorage:
    def test_append_and_get_messages(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Hello")
        db.append_message("s1", role="assistant", content="Hi there!")

        messages = db.get_messages("s1")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"

    def test_message_increments_session_count(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Hello")
        db.append_message("s1", role="assistant", content="Hi")

        session = db.get_session("s1")
        assert session["message_count"] == 2

    def test_tool_response_does_not_increment_tool_count(self, db):
        """Tool responses (role=tool) should not increment tool_call_count.

        Only assistant messages with tool_calls should count.
        """
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="tool", content="result", tool_name="web_search")

        session = db.get_session("s1")
        assert session["tool_call_count"] == 0

    def test_assistant_tool_calls_increment_by_count(self, db):
        """An assistant message with N tool_calls should increment by N."""
        db.create_session(session_id="s1", source="cli")
        tool_calls = [
            {"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}},
        ]
        db.append_message("s1", role="assistant", content="", tool_calls=tool_calls)

        session = db.get_session("s1")
        assert session["tool_call_count"] == 1

    def test_tool_call_count_matches_actual_calls(self, db):
        """tool_call_count should equal the number of tool calls made, not messages."""
        db.create_session(session_id="s1", source="cli")

        # Assistant makes 2 parallel tool calls in one message
        tool_calls = [
            {"id": "call_1", "function": {"name": "ha_call_service", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "ha_call_service", "arguments": "{}"}},
        ]
        db.append_message("s1", role="assistant", content="", tool_calls=tool_calls)

        # Two tool responses come back
        db.append_message("s1", role="tool", content="ok", tool_name="ha_call_service")
        db.append_message("s1", role="tool", content="ok", tool_name="ha_call_service")

        session = db.get_session("s1")
        # Should be 2 (the actual number of tool calls), not 3
        assert session["tool_call_count"] == 2, (
            f"Expected 2 tool calls but got {session['tool_call_count']}. "
            "tool responses are double-counted and multi-call messages are under-counted"
        )

    def test_tool_calls_serialization(self, db):
        db.create_session(session_id="s1", source="cli")
        tool_calls = [{"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}}]
        db.append_message("s1", role="assistant", tool_calls=tool_calls)

        messages = db.get_messages("s1")
        assert messages[0]["tool_calls"] == tool_calls

    def test_get_messages_as_conversation(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Hello")
        db.append_message("s1", role="assistant", content="Hi!")

        conv = db.get_messages_as_conversation("s1")
        assert len(conv) == 2
        assert conv[0] == {"role": "user", "content": "Hello"}
        assert conv[1] == {"role": "assistant", "content": "Hi!"}

    def test_finish_reason_stored(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="assistant", content="Done", finish_reason="stop")

        messages = db.get_messages("s1")
        assert messages[0]["finish_reason"] == "stop"

    def test_reasoning_persisted_and_restored(self, db):
        """Reasoning text is stored for assistant messages and restored by
        get_messages_as_conversation() so providers receive coherent multi-turn
        reasoning context."""
        db.create_session(session_id="s1", source="telegram")
        db.append_message("s1", role="user", content="create a cron job")
        db.append_message(
            "s1",
            role="assistant",
            content=None,
            tool_calls=[{"function": {"name": "cronjob", "arguments": "{}"}, "id": "c1", "type": "function"}],
            reasoning="I should call the cronjob tool to schedule this.",
        )
        db.append_message("s1", role="tool", content='{"job_id": "abc"}', tool_call_id="c1")

        conv = db.get_messages_as_conversation("s1")
        assert len(conv) == 3
        # reasoning must be present on the assistant message
        assistant = conv[1]
        assert assistant["role"] == "assistant"
        assert assistant.get("reasoning") == "I should call the cronjob tool to schedule this."
        # user and tool messages must NOT carry reasoning
        assert "reasoning" not in conv[0]
        assert "reasoning" not in conv[2]

    def test_reasoning_details_persisted_and_restored(self, db):
        """reasoning_details (structured array) is round-tripped through JSON
        serialization in the DB."""
        db.create_session(session_id="s1", source="telegram")
        details = [
            {"type": "reasoning.summary", "summary": "Thinking about tools"},
            {"type": "reasoning.encrypted_content", "encrypted_content": "abc123"},
        ]
        db.append_message(
            "s1",
            role="assistant",
            content="Hello",
            reasoning="Thinking about what to say",
            reasoning_details=details,
        )

        conv = db.get_messages_as_conversation("s1")
        assert len(conv) == 1
        msg = conv[0]
        assert msg["reasoning"] == "Thinking about what to say"
        assert msg["reasoning_details"] == details

    def test_reasoning_not_set_for_non_assistant(self, db):
        """reasoning is never leaked onto user or tool messages."""
        db.create_session(session_id="s1", source="telegram")
        db.append_message("s1", role="user", content="hi")
        db.append_message("s1", role="assistant", content="hello", reasoning=None)

        conv = db.get_messages_as_conversation("s1")
        assert "reasoning" not in conv[0]
        assert "reasoning" not in conv[1]

    def test_reasoning_empty_string_not_restored(self, db):
        """Empty string reasoning is treated as absent."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="assistant", content="hi", reasoning="")

        conv = db.get_messages_as_conversation("s1")
        assert "reasoning" not in conv[0]

    def test_codex_reasoning_items_persisted_and_restored(self, db):
        """codex_reasoning_items (encrypted blobs for Codex Responses API) are
        round-tripped through JSON serialization in the DB."""
        db.create_session(session_id="s1", source="cli")
        codex_items = [
            {"type": "reasoning", "id": "rs_abc", "encrypted_content": "enc_blob_123"},
            {"type": "reasoning", "id": "rs_def", "encrypted_content": "enc_blob_456"},
        ]
        db.append_message(
            "s1",
            role="assistant",
            content="Done",
            codex_reasoning_items=codex_items,
        )

        conv = db.get_messages_as_conversation("s1")
        assert len(conv) == 1
        assert conv[0]["codex_reasoning_items"] == codex_items
        assert conv[0]["codex_reasoning_items"][0]["encrypted_content"] == "enc_blob_123"


# =========================================================================
# FTS5 search
# =========================================================================

class TestFTS5Search:
    def test_search_finds_content(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="How do I deploy with Docker?")
        db.append_message("s1", role="assistant", content="Use docker compose up.")

        results = db.search_messages("docker")
        assert len(results) == 2
        # At least one result should mention docker
        snippets = [r.get("snippet", "") for r in results]
        assert any("docker" in s.lower() or "Docker" in s for s in snippets)

    def test_search_empty_query(self, db):
        assert db.search_messages("") == []
        assert db.search_messages("   ") == []

    def test_search_with_source_filter(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="CLI question about Python")

        db.create_session(session_id="s2", source="telegram")
        db.append_message("s2", role="user", content="Telegram question about Python")

        results = db.search_messages("Python", source_filter=["telegram"])
        # Should only find the telegram message
        sources = [r["source"] for r in results]
        assert all(s == "telegram" for s in sources)

    def test_search_default_sources_include_acp(self, db):
        db.create_session(session_id="s1", source="acp")
        db.append_message("s1", role="user", content="ACP question about Python")

        results = db.search_messages("Python")
        sources = [r["source"] for r in results]
        assert "acp" in sources

    def test_search_default_includes_all_platforms(self, db):
        """Default search (no source_filter) should find sessions from any platform."""
        for src in ("cli", "telegram", "signal", "homeassistant", "acp", "matrix"):
            sid = f"s-{src}"
            db.create_session(session_id=sid, source=src)
            db.append_message(sid, role="user", content=f"universal search test from {src}")

        results = db.search_messages("universal search test")
        found_sources = {r["source"] for r in results}
        assert found_sources == {"cli", "telegram", "signal", "homeassistant", "acp", "matrix"}

    def test_search_with_role_filter(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="What is FastAPI?")
        db.append_message("s1", role="assistant", content="FastAPI is a web framework.")

        results = db.search_messages("FastAPI", role_filter=["assistant"])
        roles = [r["role"] for r in results]
        assert all(r == "assistant" for r in roles)

    def test_search_returns_context(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Tell me about Kubernetes")
        db.append_message("s1", role="assistant", content="Kubernetes is an orchestrator.")

        results = db.search_messages("Kubernetes")
        assert len(results) == 2
        assert "context" in results[0]
        assert isinstance(results[0]["context"], list)
        assert len(results[0]["context"]) > 0

    def test_search_special_chars_do_not_crash(self, db):
        """FTS5 special characters in queries must not raise OperationalError."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="How do I use C++ templates?")

        # Each of these previously caused sqlite3.OperationalError
        dangerous_queries = [
            'C++',              # + is FTS5 column filter
            '"unterminated',    # unbalanced double-quote
            '(problem',         # unbalanced parenthesis
            'hello AND',        # dangling boolean operator
            '***',              # repeated wildcard
            '{test}',           # curly braces (column reference)
            'OR hello',         # leading boolean operator
            'a AND OR b',       # adjacent operators
        ]
        for query in dangerous_queries:
            # Must not raise — should return list (possibly empty)
            results = db.search_messages(query)
            assert isinstance(results, list), f"Query {query!r} did not return a list"

    def test_search_sanitized_query_still_finds_content(self, db):
        """Sanitization must not break normal keyword search."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Learning C++ templates today")

        # "C++" sanitized to "C" should still match "C++"
        results = db.search_messages("C++")
        # The word "C" appears in the content, so FTS5 should find it
        assert isinstance(results, list)

    def test_search_hyphenated_term_does_not_crash(self, db):
        """Hyphenated terms like 'chat-send' must not crash FTS5."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Run the chat-send command")

        results = db.search_messages("chat-send")
        assert isinstance(results, list)
        assert len(results) >= 1
        assert any("chat-send" in (r.get("snippet") or r.get("content", "")).lower()
                    for r in results)

    def test_search_dotted_term_does_not_crash(self, db):
        """Dotted terms like 'P2.2' or 'simulate.p2.test.ts' should not crash FTS5."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Working on P2.2 session_search edge cases")
        db.append_message("s1", role="assistant", content="See simulate.p2.test.ts for details")

        results = db.search_messages("P2.2")
        assert isinstance(results, list)
        assert len(results) >= 1

        results2 = db.search_messages("simulate.p2.test.ts")
        assert isinstance(results2, list)
        assert len(results2) >= 1

    def test_search_quoted_phrase_preserved(self, db):
        """User-provided quoted phrases should be preserved for exact matching."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="docker networking is complex")
        db.append_message("s1", role="assistant", content="networking docker tips")

        # Quoted phrase should match only the exact order
        results = db.search_messages('"docker networking"')
        assert isinstance(results, list)
        # Should find the user message (exact phrase) but may or may not find
        # the assistant message depending on FTS5 phrase matching
        assert len(results) >= 1

    def test_sanitize_fts5_query_strips_dangerous_chars(self):
        """Unit test for _sanitize_fts5_query static method."""
        from hermes_state import SessionDB
        s = SessionDB._sanitize_fts5_query
        assert s('hello world') == 'hello world'
        assert '+' not in s('C++')
        assert '"' not in s('"unterminated')
        assert '(' not in s('(problem')
        assert '{' not in s('{test}')
        # Dangling operators removed
        assert s('hello AND') == 'hello'
        assert s('OR world') == 'world'
        # Leading bare * removed
        assert s('***') == ''
        # Valid prefix kept
        assert s('deploy*') == 'deploy*'

    def test_sanitize_fts5_preserves_quoted_phrases(self):
        """Properly paired double-quoted phrases should be preserved."""
        from hermes_state import SessionDB
        s = SessionDB._sanitize_fts5_query
        # Simple quoted phrase
        assert s('"exact phrase"') == '"exact phrase"'
        # Quoted phrase alongside unquoted terms
        assert '"docker networking"' in s('"docker networking" setup')
        # Multiple quoted phrases
        result = s('"hello world" OR "foo bar"')
        assert '"hello world"' in result
        assert '"foo bar"' in result
        # Unmatched quote still stripped
        assert '"' not in s('"unterminated')

    def test_sanitize_fts5_quotes_hyphenated_terms(self):
        """Hyphenated terms should be wrapped in quotes for exact matching."""
        from hermes_state import SessionDB
        s = SessionDB._sanitize_fts5_query
        # Simple hyphenated term
        assert s('chat-send') == '"chat-send"'
        # Multiple hyphens
        assert s('docker-compose-up') == '"docker-compose-up"'
        # Hyphenated term with other words
        result = s('fix chat-send bug')
        assert '"chat-send"' in result
        assert 'fix' in result
        assert 'bug' in result
        # Multiple hyphenated terms with OR
        result = s('chat-send OR deploy-prod')
        assert '"chat-send"' in result
        assert '"deploy-prod"' in result
        # Already-quoted hyphenated term — no double quoting
        assert s('"chat-send"') == '"chat-send"'
        # Hyphenated inside a quoted phrase stays as-is
        assert s('"my chat-send thing"') == '"my chat-send thing"'

    def test_sanitize_fts5_quotes_dotted_terms(self):
        """Dotted terms should be wrapped in quotes to avoid FTS5 query parse edge cases."""
        from hermes_state import SessionDB
        s = SessionDB._sanitize_fts5_query

        assert s('P2.2') == '"P2.2"'
        assert s('simulate.p2') == '"simulate.p2"'
        assert s('simulate.p2.test.ts') == '"simulate.p2.test.ts"'

        # Already quoted — no double quoting
        assert s('"P2.2"') == '"P2.2"'

        # Works with boolean syntax
        result = s('P2.2 OR simulate.p2')
        assert '"P2.2"' in result
        assert '"simulate.p2"' in result

        # Mixed dots and hyphens — single pass avoids double-quoting
        assert s('my-app.config') == '"my-app.config"'
        assert s('my-app.config.ts') == '"my-app.config.ts"'


# =========================================================================
# CJK (Chinese/Japanese/Korean) LIKE fallback
# =========================================================================

class TestCJKSearchFallback:
    """Regression tests for CJK search (see #11511).

    SQLite FTS5's default tokenizer treats contiguous CJK runs as a single
    token ("和其他agent的聊天记录" → one token), so substring queries like
    "记忆断裂" return 0 rows despite the data being present. SessionDB falls
    back to LIKE substring matching whenever FTS5 returns no results and
    the query contains CJK characters.
    """

    def test_cjk_detection_covers_all_ranges(self):
        from hermes_state import SessionDB
        f = SessionDB._contains_cjk
        # Chinese (CJK Unified Ideographs)
        assert f("记忆断裂") is True
        # Japanese Hiragana + Katakana
        assert f("こんにちは") is True
        assert f("カタカナ") is True
        # Korean Hangul syllables (both early and late — guards against
        # the \ud7a0-\ud7af typo seen in one of the duplicate PRs)
        assert f("안녕하세요") is True
        assert f("기억") is True
        # Non-CJK
        assert f("hello world") is False
        assert f("日本語mixedwithenglish") is True
        assert f("") is False

    def test_chinese_multichar_query_returns_results(self, db):
        """The headline bug: multi-char Chinese query must not return []."""
        db.create_session(session_id="s1", source="cli")
        db.append_message(
            "s1", role="user",
            content="昨天和其他Agent的聊天记录，记忆断裂问题复现了",
        )
        results = db.search_messages("记忆断裂")
        assert len(results) == 1
        assert results[0]["session_id"] == "s1"

    def test_chinese_bigram_query(self, db):
        db.create_session(session_id="s1", source="telegram")
        db.append_message("s1", role="user", content="今天讨论A2A通信协议的实现")
        results = db.search_messages("通信")
        assert len(results) == 1

    def test_korean_query_returns_results(self, db):
        """Guards against Hangul range typos (\\uac00-\\ud7af, not \\ud7a0-)."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="안녕하세요 반갑습니다")
        results = db.search_messages("안녕")
        assert len(results) == 1

    def test_japanese_query_returns_results(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="こんにちは世界")
        assert len(db.search_messages("こんにちは")) == 1
        assert len(db.search_messages("世界")) == 1

    def test_cjk_fallback_preserves_source_filter(self, db):
        """Guards against the SQL-builder bug where filter clauses land
        after LIMIT/OFFSET (seen in one of the duplicate PRs)."""
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")
        db.append_message("s1", role="user", content="记忆断裂在CLI")
        db.append_message("s2", role="user", content="记忆断裂在Telegram")

        results = db.search_messages("记忆断裂", source_filter=["telegram"])
        assert len(results) == 1
        assert results[0]["source"] == "telegram"

    def test_cjk_fallback_preserves_exclude_sources(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="tool")
        db.append_message("s1", role="user", content="记忆断裂在CLI")
        db.append_message("s2", role="assistant", content="记忆断裂在tool")

        results = db.search_messages("记忆断裂", exclude_sources=["tool"])
        sources = {r["source"] for r in results}
        assert "tool" not in sources
        assert "cli" in sources

    def test_cjk_fallback_preserves_role_filter(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="用户说的记忆断裂")
        db.append_message("s1", role="assistant", content="助手说的记忆断裂")

        results = db.search_messages("记忆断裂", role_filter=["assistant"])
        assert len(results) == 1
        assert results[0]["role"] == "assistant"

    def test_cjk_snippet_is_centered_on_match(self, db):
        """Snippet should contain the search term, not just the first N chars."""
        db.create_session(session_id="s1", source="cli")
        long_prefix = "这是一段很长的前缀用来把匹配位置推到文档中间" * 3
        long_suffix = "这是一段很长的后缀内容填充剩余空间" * 3
        db.append_message(
            "s1", role="user",
            content=f"{long_prefix}记忆断裂{long_suffix}",
        )
        results = db.search_messages("记忆断裂")
        assert len(results) == 1
        # The centered substr() snippet must include the matched term.
        assert "记忆断裂" in results[0]["snippet"]

    def test_english_query_still_uses_fts5_fast_path(self, db):
        """English queries must not trigger the LIKE fallback (fast path regression)."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Deploy docker containers")
        results = db.search_messages("docker")
        assert len(results) == 1
        # No CJK in query → LIKE fallback must not run. We don't assert this
        # directly (no instrumentation), but the FTS5 path produces an
        # FTS5-style snippet with highlight markers when the term is short.
        # At minimum: english queries must still match.

    def test_cjk_query_with_no_matches_returns_empty(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="unrelated English content")
        results = db.search_messages("记忆断裂")
        assert results == []

    def test_mixed_cjk_english_query(self, db):
        """Mixed queries should still fall back to LIKE when FTS5 misses."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="讨论Agent通信协议")
        # "Agent通信" is CJK+English — FTS5 default tokenizer indexes the
        # whole CJK run with embedded "agent" as separate tokens; the LIKE
        # fallback handles the substring correctly.
        results = db.search_messages("Agent通信")
        assert len(results) == 1


# =========================================================================
# Session search and listing
# =========================================================================

class TestSearchSessions:
    def test_list_all_sessions(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")

        sessions = db.search_sessions()
        assert len(sessions) == 2

    def test_filter_by_source(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")

        sessions = db.search_sessions(source="cli")
        assert len(sessions) == 1
        assert sessions[0]["source"] == "cli"

    def test_pagination(self, db):
        for i in range(5):
            db.create_session(session_id=f"s{i}", source="cli")

        page1 = db.search_sessions(limit=2)
        page2 = db.search_sessions(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0]["id"] != page2[0]["id"]


# =========================================================================
# Counts
# =========================================================================

class TestCounts:
    def test_session_count(self, db):
        assert db.session_count() == 0
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")
        assert db.session_count() == 2

    def test_session_count_by_source(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")
        db.create_session(session_id="s3", source="cli")
        assert db.session_count(source="cli") == 2
        assert db.session_count(source="telegram") == 1

    def test_message_count_total(self, db):
        assert db.message_count() == 0
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Hello")
        db.append_message("s1", role="assistant", content="Hi")
        assert db.message_count() == 2

    def test_message_count_per_session(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="cli")
        db.append_message("s1", role="user", content="A")
        db.append_message("s2", role="user", content="B")
        db.append_message("s2", role="user", content="C")
        assert db.message_count(session_id="s1") == 1
        assert db.message_count(session_id="s2") == 2


# =========================================================================
# Delete and export
# =========================================================================

class TestDeleteAndExport:
    def test_delete_session(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Hello")

        assert db.delete_session("s1") is True
        assert db.get_session("s1") is None
        assert db.message_count(session_id="s1") == 0

    def test_delete_nonexistent(self, db):
        assert db.delete_session("nope") is False

    def test_resolve_session_id_exact(self, db):
        db.create_session(session_id="20260315_092437_c9a6ff", source="cli")
        assert db.resolve_session_id("20260315_092437_c9a6ff") == "20260315_092437_c9a6ff"

    def test_resolve_session_id_unique_prefix(self, db):
        db.create_session(session_id="20260315_092437_c9a6ff", source="cli")
        assert db.resolve_session_id("20260315_092437_c9a6") == "20260315_092437_c9a6ff"

    def test_resolve_session_id_ambiguous_prefix_returns_none(self, db):
        db.create_session(session_id="20260315_092437_c9a6aa", source="cli")
        db.create_session(session_id="20260315_092437_c9a6bb", source="cli")
        assert db.resolve_session_id("20260315_092437_c9a6") is None

    def test_resolve_session_id_escapes_like_wildcards(self, db):
        db.create_session(session_id="20260315_092437_c9a6ff", source="cli")
        db.create_session(session_id="20260315X092437_c9a6ff", source="cli")
        assert db.resolve_session_id("20260315_092437") == "20260315_092437_c9a6ff"

    def test_export_session(self, db):
        db.create_session(session_id="s1", source="cli", model="test")
        db.append_message("s1", role="user", content="Hello")
        db.append_message("s1", role="assistant", content="Hi")

        export = db.export_session("s1")
        assert isinstance(export, dict)
        assert export["source"] == "cli"
        assert len(export["messages"]) == 2

    def test_export_nonexistent(self, db):
        assert db.export_session("nope") is None

    def test_export_all(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")
        db.append_message("s1", role="user", content="A")

        exports = db.export_all()
        assert len(exports) == 2

    def test_export_all_with_source(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")

        exports = db.export_all(source="cli")
        assert len(exports) == 1
        assert exports[0]["source"] == "cli"


# =========================================================================
# Prune
# =========================================================================

class TestPruneSessions:
    def test_prune_old_ended_sessions(self, db):
        # Create and end an "old" session
        db.create_session(session_id="old", source="cli")
        db.end_session("old", end_reason="done")
        # Manually backdate started_at
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (time.time() - 100 * 86400, "old"),
        )
        db._conn.commit()

        # Create a recent session
        db.create_session(session_id="new", source="cli")

        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 1
        assert db.get_session("old") is None
        session = db.get_session("new")
        assert session is not None
        assert session["id"] == "new"

    def test_prune_skips_active_sessions(self, db):
        db.create_session(session_id="active", source="cli")
        # Backdate but don't end
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (time.time() - 200 * 86400, "active"),
        )
        db._conn.commit()

        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 0
        assert db.get_session("active") is not None

    def test_prune_with_source_filter(self, db):
        for sid, src in [("old_cli", "cli"), ("old_tg", "telegram")]:
            db.create_session(session_id=sid, source=src)
            db.end_session(sid, end_reason="done")
            db._conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (time.time() - 200 * 86400, sid),
            )
        db._conn.commit()

        pruned = db.prune_sessions(older_than_days=90, source="cli")
        assert pruned == 1
        assert db.get_session("old_cli") is None
        assert db.get_session("old_tg") is not None

    def test_prune_with_multilevel_chain(self, db):
        """Pruning old sessions orphans newer children instead of crashing on FK."""
        old_ts = time.time() - 200 * 86400
        recent_ts = time.time() - 10 * 86400

        # Chain: A (old) -> B (old) -> C (recent) -> D (recent)
        db.create_session(session_id="A", source="cli")
        db.end_session("A", end_reason="compressed")
        db.create_session(session_id="B", source="cli", parent_session_id="A")
        db.end_session("B", end_reason="compressed")
        db.create_session(session_id="C", source="cli", parent_session_id="B")
        db.end_session("C", end_reason="compressed")
        db.create_session(session_id="D", source="cli", parent_session_id="C")
        db.end_session("D", end_reason="done")

        # Backdate A and B to be old; C and D stay recent
        for sid, ts in [("A", old_ts), ("B", old_ts), ("C", recent_ts), ("D", recent_ts)]:
            db._conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?", (ts, sid)
            )
        db._conn.commit()

        # Should not raise IntegrityError
        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 2  # only A and B
        assert db.get_session("A") is None
        assert db.get_session("B") is None
        # C and D survive, C is orphaned (parent_session_id NULL)
        c = db.get_session("C")
        assert c is not None
        assert c["parent_session_id"] is None
        d = db.get_session("D")
        assert d is not None
        assert d["parent_session_id"] == "C"

    def test_prune_entire_old_chain(self, db):
        """All sessions in a chain are old — entire chain is pruned."""
        old_ts = time.time() - 200 * 86400

        db.create_session(session_id="X", source="cli")
        db.end_session("X", end_reason="compressed")
        db.create_session(session_id="Y", source="cli", parent_session_id="X")
        db.end_session("Y", end_reason="compressed")
        db.create_session(session_id="Z", source="cli", parent_session_id="Y")
        db.end_session("Z", end_reason="done")

        for sid in ("X", "Y", "Z"):
            db._conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?", (old_ts, sid)
            )
        db._conn.commit()

        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 3
        for sid in ("X", "Y", "Z"):
            assert db.get_session(sid) is None


class TestDeleteSessionOrphansChildren:
    def test_delete_orphans_children(self, db):
        """Deleting a parent session orphans its children."""
        db.create_session(session_id="parent", source="cli")
        db.create_session(session_id="child", source="cli", parent_session_id="parent")
        db.create_session(session_id="grandchild", source="cli", parent_session_id="child")

        # Should not raise IntegrityError
        result = db.delete_session("parent")
        assert result is True
        assert db.get_session("parent") is None
        # Child is orphaned, not deleted
        child = db.get_session("child")
        assert child is not None
        assert child["parent_session_id"] is None
        # Grandchild is untouched
        grandchild = db.get_session("grandchild")
        assert grandchild is not None
        assert grandchild["parent_session_id"] == "child"


# =========================================================================
# Schema and WAL mode
# =========================================================================

# =========================================================================
# Session title
# =========================================================================

class TestSessionTitle:
    def test_set_and_get_title(self, db):
        db.create_session(session_id="s1", source="cli")
        assert db.set_session_title("s1", "My Session") is True

        session = db.get_session("s1")
        assert session["title"] == "My Session"

    def test_set_title_nonexistent_session(self, db):
        assert db.set_session_title("nonexistent", "Title") is False

    def test_title_initially_none(self, db):
        db.create_session(session_id="s1", source="cli")
        session = db.get_session("s1")
        assert session["title"] is None

    def test_update_title(self, db):
        db.create_session(session_id="s1", source="cli")
        db.set_session_title("s1", "First Title")
        db.set_session_title("s1", "Updated Title")

        session = db.get_session("s1")
        assert session["title"] == "Updated Title"

    def test_title_in_search_sessions(self, db):
        db.create_session(session_id="s1", source="cli")
        db.set_session_title("s1", "Debugging Auth")
        db.create_session(session_id="s2", source="cli")

        sessions = db.search_sessions()
        titled = [s for s in sessions if s.get("title") == "Debugging Auth"]
        assert len(titled) == 1
        assert titled[0]["id"] == "s1"

    def test_title_in_export(self, db):
        db.create_session(session_id="s1", source="cli")
        db.set_session_title("s1", "Export Test")
        db.append_message("s1", role="user", content="Hello")

        export = db.export_session("s1")
        assert export["title"] == "Export Test"

    def test_title_with_special_characters(self, db):
        db.create_session(session_id="s1", source="cli")
        title = "PR #438 — fixing the 'auth' middleware"
        db.set_session_title("s1", title)

        session = db.get_session("s1")
        assert session["title"] == title

    def test_title_empty_string_normalized_to_none(self, db):
        """Empty strings are normalized to None (clearing the title)."""
        db.create_session(session_id="s1", source="cli")
        db.set_session_title("s1", "My Title")
        # Setting to empty string should clear the title (normalize to None)
        db.set_session_title("s1", "")

        session = db.get_session("s1")
        assert session["title"] is None

    def test_multiple_empty_titles_no_conflict(self, db):
        """Multiple sessions can have empty-string (normalized to NULL) titles."""
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="cli")
        db.set_session_title("s1", "")
        db.set_session_title("s2", "")
        # Both should be None, no uniqueness conflict
        assert db.get_session("s1")["title"] is None
        assert db.get_session("s2")["title"] is None

    def test_title_survives_end_session(self, db):
        db.create_session(session_id="s1", source="cli")
        db.set_session_title("s1", "Before End")
        db.end_session("s1", end_reason="user_exit")

        session = db.get_session("s1")
        assert session["title"] == "Before End"
        assert session["ended_at"] is not None


class TestSanitizeTitle:
    """Tests for SessionDB.sanitize_title() validation and cleaning."""

    def test_normal_title_unchanged(self):
        assert SessionDB.sanitize_title("My Project") == "My Project"

    def test_strips_whitespace(self):
        assert SessionDB.sanitize_title("  hello world  ") == "hello world"

    def test_collapses_internal_whitespace(self):
        assert SessionDB.sanitize_title("hello   world") == "hello world"

    def test_tabs_and_newlines_collapsed(self):
        assert SessionDB.sanitize_title("hello\t\nworld") == "hello world"

    def test_none_returns_none(self):
        assert SessionDB.sanitize_title(None) is None

    def test_empty_string_returns_none(self):
        assert SessionDB.sanitize_title("") is None

    def test_whitespace_only_returns_none(self):
        assert SessionDB.sanitize_title("   \t\n  ") is None

    def test_control_chars_stripped(self):
        # Null byte, bell, backspace, etc.
        assert SessionDB.sanitize_title("hello\x00world") == "helloworld"
        assert SessionDB.sanitize_title("\x07\x08test\x1b") == "test"

    def test_del_char_stripped(self):
        assert SessionDB.sanitize_title("hello\x7fworld") == "helloworld"

    def test_zero_width_chars_stripped(self):
        # Zero-width space (U+200B), zero-width joiner (U+200D)
        assert SessionDB.sanitize_title("hello\u200bworld") == "helloworld"
        assert SessionDB.sanitize_title("hello\u200dworld") == "helloworld"

    def test_rtl_override_stripped(self):
        # Right-to-left override (U+202E) — used in filename spoofing attacks
        assert SessionDB.sanitize_title("hello\u202eworld") == "helloworld"

    def test_bom_stripped(self):
        # Byte order mark (U+FEFF)
        assert SessionDB.sanitize_title("\ufeffhello") == "hello"

    def test_only_control_chars_returns_none(self):
        assert SessionDB.sanitize_title("\x00\x01\x02\u200b\ufeff") is None

    def test_max_length_allowed(self):
        title = "A" * 100
        assert SessionDB.sanitize_title(title) == title

    def test_exceeds_max_length_raises(self):
        title = "A" * 101
        with pytest.raises(ValueError, match="too long"):
            SessionDB.sanitize_title(title)

    def test_unicode_emoji_allowed(self):
        assert SessionDB.sanitize_title("🚀 My Project 🎉") == "🚀 My Project 🎉"

    def test_cjk_characters_allowed(self):
        assert SessionDB.sanitize_title("我的项目") == "我的项目"

    def test_accented_characters_allowed(self):
        assert SessionDB.sanitize_title("Résumé éditing") == "Résumé éditing"

    def test_special_punctuation_allowed(self):
        title = "PR #438 — fixing the 'auth' middleware"
        assert SessionDB.sanitize_title(title) == title

    def test_sanitize_applied_in_set_session_title(self, db):
        """set_session_title applies sanitize_title internally."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "  hello\x00  world  ")
        assert db.get_session("s1")["title"] == "hello world"

    def test_too_long_title_rejected_by_set(self, db):
        """set_session_title raises ValueError for overly long titles."""
        db.create_session("s1", "cli")
        with pytest.raises(ValueError, match="too long"):
            db.set_session_title("s1", "X" * 150)


class TestSchemaInit:
    def test_wal_mode(self, db):
        cursor = db._conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode == "wal"

    def test_foreign_keys_enabled(self, db):
        cursor = db._conn.execute("PRAGMA foreign_keys")
        assert cursor.fetchone()[0] == 1

    def test_tables_exist(self, db):
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "sessions" in tables
        assert "messages" in tables
        assert "schema_version" in tables

    def test_schema_version(self, db):
        cursor = db._conn.execute("SELECT version FROM schema_version")
        version = cursor.fetchone()[0]
        assert version == 6

    def test_title_column_exists(self, db):
        """Verify the title column was created in the sessions table."""
        cursor = db._conn.execute("PRAGMA table_info(sessions)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "title" in columns

    def test_migration_from_v2(self, tmp_path):
        """Simulate a v2 database and verify migration adds title column."""
        import sqlite3

        db_path = tmp_path / "migrate_test.db"
        conn = sqlite3.connect(str(db_path))
        # Create v2 schema (without title column)
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (2);

            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                user_id TEXT,
                model TEXT,
                model_config TEXT,
                system_prompt TEXT,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL,
                end_reason TEXT,
                message_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0
            );

            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_count INTEGER,
                finish_reason TEXT
            );
        """)
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            ("existing", "cli", 1000.0),
        )
        conn.commit()
        conn.close()

        # Open with SessionDB — should migrate to v6
        migrated_db = SessionDB(db_path=db_path)

        # Verify migration
        cursor = migrated_db._conn.execute("SELECT version FROM schema_version")
        assert cursor.fetchone()[0] == 6

        # Verify title column exists and is NULL for existing sessions
        session = migrated_db.get_session("existing")
        assert session is not None
        assert session["title"] is None

        # Verify we can set title on migrated session
        assert migrated_db.set_session_title("existing", "Migrated Title") is True
        session = migrated_db.get_session("existing")
        assert session["title"] == "Migrated Title"

        migrated_db.close()


class TestTitleUniqueness:
    """Tests for unique title enforcement and title-based lookups."""

    def test_duplicate_title_raises(self, db):
        """Setting a title already used by another session raises ValueError."""
        db.create_session("s1", "cli")
        db.create_session("s2", "cli")
        db.set_session_title("s1", "my project")
        with pytest.raises(ValueError, match="already in use"):
            db.set_session_title("s2", "my project")

    def test_same_session_can_keep_title(self, db):
        """A session can re-set its own title without error."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "my project")
        # Should not raise — it's the same session
        assert db.set_session_title("s1", "my project") is True

    def test_null_titles_not_unique(self, db):
        """Multiple sessions can have NULL titles (no constraint violation)."""
        db.create_session("s1", "cli")
        db.create_session("s2", "cli")
        # Both have NULL titles — no error
        assert db.get_session("s1")["title"] is None
        assert db.get_session("s2")["title"] is None

    def test_get_session_by_title(self, db):
        db.create_session("s1", "cli")
        db.set_session_title("s1", "refactoring auth")
        result = db.get_session_by_title("refactoring auth")
        assert result is not None
        assert result["id"] == "s1"

    def test_get_session_by_title_not_found(self, db):
        assert db.get_session_by_title("nonexistent") is None

    def test_get_session_title(self, db):
        db.create_session("s1", "cli")
        assert db.get_session_title("s1") is None
        db.set_session_title("s1", "my title")
        assert db.get_session_title("s1") == "my title"

    def test_get_session_title_nonexistent(self, db):
        assert db.get_session_title("nonexistent") is None


class TestTitleLineage:
    """Tests for title lineage resolution and auto-numbering."""

    def test_resolve_exact_title(self, db):
        db.create_session("s1", "cli")
        db.set_session_title("s1", "my project")
        assert db.resolve_session_by_title("my project") == "s1"

    def test_resolve_returns_latest_numbered(self, db):
        """When numbered variants exist, return the most recent one."""
        import time
        db.create_session("s1", "cli")
        db.set_session_title("s1", "my project")
        time.sleep(0.01)
        db.create_session("s2", "cli")
        db.set_session_title("s2", "my project #2")
        time.sleep(0.01)
        db.create_session("s3", "cli")
        db.set_session_title("s3", "my project #3")
        # Resolving "my project" should return s3 (latest numbered variant)
        assert db.resolve_session_by_title("my project") == "s3"

    def test_resolve_exact_numbered(self, db):
        """Resolving an exact numbered title returns that specific session."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "my project")
        db.create_session("s2", "cli")
        db.set_session_title("s2", "my project #2")
        # Resolving "my project #2" exactly should return s2
        assert db.resolve_session_by_title("my project #2") == "s2"

    def test_resolve_nonexistent_title(self, db):
        assert db.resolve_session_by_title("nonexistent") is None

    def test_next_title_no_existing(self, db):
        """With no existing sessions, base title is returned as-is."""
        assert db.get_next_title_in_lineage("my project") == "my project"

    def test_next_title_first_continuation(self, db):
        """First continuation after the original gets #2."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "my project")
        assert db.get_next_title_in_lineage("my project") == "my project #2"

    def test_next_title_increments(self, db):
        """Each continuation increments the number."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "my project")
        db.create_session("s2", "cli")
        db.set_session_title("s2", "my project #2")
        db.create_session("s3", "cli")
        db.set_session_title("s3", "my project #3")
        assert db.get_next_title_in_lineage("my project") == "my project #4"

    def test_next_title_strips_existing_number(self, db):
        """Passing a numbered title strips the number and finds the base."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "my project")
        db.create_session("s2", "cli")
        db.set_session_title("s2", "my project #2")
        # Even when called with "my project #2", it should return #3
        assert db.get_next_title_in_lineage("my project #2") == "my project #3"


class TestTitleSqlWildcards:
    """Titles containing SQL LIKE wildcards (%, _) must not cause false matches."""

    def test_resolve_title_with_underscore(self, db):
        """A title like 'test_project' should not match 'testXproject #2'."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "test_project")
        db.create_session("s2", "cli")
        db.set_session_title("s2", "testXproject #2")
        # Resolving "test_project" should return s1 (exact), not s2
        assert db.resolve_session_by_title("test_project") == "s1"

    def test_resolve_title_with_percent(self, db):
        """A title with '%' should not wildcard-match unrelated sessions."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "100% done")
        db.create_session("s2", "cli")
        db.set_session_title("s2", "100X done #2")
        # Should resolve to s1 (exact), not s2
        assert db.resolve_session_by_title("100% done") == "s1"

    def test_next_lineage_with_underscore(self, db):
        """get_next_title_in_lineage with underscores doesn't match wrong sessions."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "test_project")
        db.create_session("s2", "cli")
        db.set_session_title("s2", "testXproject #2")
        # Only "test_project" exists, so next should be "test_project #2"
        assert db.get_next_title_in_lineage("test_project") == "test_project #2"


class TestListSessionsRich:
    """Tests for enhanced session listing with preview and last_active."""

    def test_preview_from_first_user_message(self, db):
        db.create_session("s1", "cli")
        db.append_message("s1", "system", "You are a helpful assistant.")
        db.append_message("s1", "user", "Help me refactor the auth module please")
        db.append_message("s1", "assistant", "Sure, let me look at it.")
        sessions = db.list_sessions_rich()
        assert len(sessions) == 1
        assert "Help me refactor the auth module" in sessions[0]["preview"]

    def test_preview_truncated_at_60(self, db):
        db.create_session("s1", "cli")
        long_msg = "A" * 100
        db.append_message("s1", "user", long_msg)
        sessions = db.list_sessions_rich()
        assert len(sessions[0]["preview"]) == 63  # 60 chars + "..."
        assert sessions[0]["preview"].endswith("...")

    def test_preview_empty_when_no_user_messages(self, db):
        db.create_session("s1", "cli")
        db.append_message("s1", "system", "System prompt")
        sessions = db.list_sessions_rich()
        assert sessions[0]["preview"] == ""

    def test_last_active_from_latest_message(self, db):
        import time
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")
        time.sleep(0.01)
        db.append_message("s1", "assistant", "Hi there!")
        sessions = db.list_sessions_rich()
        # last_active should be close to now (the assistant message)
        assert sessions[0]["last_active"] > sessions[0]["started_at"]

    def test_last_active_fallback_to_started_at(self, db):
        db.create_session("s1", "cli")
        sessions = db.list_sessions_rich()
        # No messages, so last_active falls back to started_at
        assert sessions[0]["last_active"] == sessions[0]["started_at"]

    def test_rich_list_includes_title(self, db):
        db.create_session("s1", "cli")
        db.set_session_title("s1", "refactoring auth")
        sessions = db.list_sessions_rich()
        assert sessions[0]["title"] == "refactoring auth"

    def test_rich_list_source_filter(self, db):
        db.create_session("s1", "cli")
        db.create_session("s2", "telegram")
        sessions = db.list_sessions_rich(source="cli")
        assert len(sessions) == 1
        assert sessions[0]["id"] == "s1"

    def test_preview_newlines_collapsed(self, db):
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Line one\nLine two\nLine three")
        sessions = db.list_sessions_rich()
        assert "\n" not in sessions[0]["preview"]
        assert "Line one Line two" in sessions[0]["preview"]


class TestCompressionChainProjection:
    """Tests for lineage-aware list_sessions_rich — compressed conversations
    surface as their live continuation tip, not the dead parent root.
    """

    def _build_compression_chain(self, db, t0: float):
        """Helper: builds root -> delegate -> compression-child -> tip chain.

        Returns (root_id, delegate_id, mid_id, tip_id).
        """
        import time as _time
        # Root that gets compressed
        db.create_session("root1", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, "root1"))
        db.append_message("root1", "user", "help me refactor auth")

        # Delegate subagent spawned while root1 was live (before it ended)
        db.create_session("delegate1", "cli", parent_session_id="root1")
        db._conn.execute(
            "UPDATE sessions SET started_at=?, ended_at=? WHERE id=?",
            (t0 + 600, t0 + 650, "delegate1"),
        )
        db.append_message("delegate1", "user", "delegate task")

        # root1 compressed at t0+1800
        t_compress_root = t0 + 1800
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t_compress_root, "compression", "root1"),
        )

        # Continuation mid created 1s after parent ended
        db.create_session("mid1", "cli", parent_session_id="root1")
        db._conn.execute(
            "UPDATE sessions SET started_at=? WHERE id=?",
            (t_compress_root + 1, "mid1"),
        )
        db.append_message("mid1", "user", "continuing")

        # mid1 also compressed
        t_compress_mid = t_compress_root + 1800
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t_compress_mid, "compression", "mid1"),
        )

        # Tip — latest continuation
        db.create_session("tip1", "cli", parent_session_id="mid1")
        db._conn.execute(
            "UPDATE sessions SET started_at=? WHERE id=?",
            (t_compress_mid + 1, "tip1"),
        )
        db.append_message("tip1", "user", "latest message")

        db._conn.commit()
        return ("root1", "delegate1", "mid1", "tip1")

    def test_get_compression_tip_walks_full_chain(self, db):
        import time as _time
        self._build_compression_chain(db, _time.time() - 3600)
        assert db.get_compression_tip("root1") == "tip1"
        assert db.get_compression_tip("mid1") == "tip1"
        assert db.get_compression_tip("tip1") == "tip1"

    def test_get_compression_tip_returns_self_for_uncompressed(self, db):
        db.create_session("solo", "cli")
        assert db.get_compression_tip("solo") == "solo"

    def test_get_compression_tip_skips_delegate_children(self, db):
        """Delegate subagents have parent_session_id set but were created
        BEFORE the parent ended. They must not be followed as compression
        continuations — the started_at >= ended_at guard handles this.
        """
        import time as _time
        self._build_compression_chain(db, _time.time() - 3600)
        # delegate1 is a child of root1 but NOT a compression continuation.
        # root1's tip must be tip1 (via mid1), not delegate1.
        assert db.get_compression_tip("root1") == "tip1"

    def test_list_surfaces_tip_for_compressed_root(self, db):
        """The list must show the tip's id/message_count/preview in place of
        the root row, so users can see and resume the live conversation.
        """
        import time as _time
        self._build_compression_chain(db, _time.time() - 3600)
        # Add an uncompressed root for comparison.
        db.create_session("solo", "cli")
        db.append_message("solo", "user", "standalone")
        db._conn.commit()

        sessions = db.list_sessions_rich(source="cli", limit=20)
        ids = [s["id"] for s in sessions]
        # Only top-level conversations appear: tip1 (projected from root1) + solo.
        # Delegate children, mid1, and the dead root1 must NOT be in the list.
        assert "tip1" in ids
        assert "solo" in ids
        assert "root1" not in ids
        assert "mid1" not in ids
        assert "delegate1" not in ids

        tip_row = next(s for s in sessions if s["id"] == "tip1")
        # The row surfaces the tip's identity but preserves the root's start
        # timestamp for stable ordering and lineage tracking.
        assert tip_row["_lineage_root_id"] == "root1"
        assert tip_row["preview"].startswith("latest message")
        assert tip_row["ended_at"] is None  # tip is still live
        assert tip_row["end_reason"] is None

    def test_list_without_projection_returns_raw_root(self, db):
        """project_compression_tips=False returns the raw parent-NULL root
        rows — useful for admin/debug UIs.
        """
        import time as _time
        self._build_compression_chain(db, _time.time() - 3600)
        sessions = db.list_sessions_rich(
            source="cli", limit=20, project_compression_tips=False
        )
        ids = [s["id"] for s in sessions]
        assert "root1" in ids
        assert "tip1" not in ids

        root_row = next(s for s in sessions if s["id"] == "root1")
        assert root_row["end_reason"] == "compression"
        assert "_lineage_root_id" not in root_row

    def test_list_preserves_sort_by_started_at(self, db):
        """Chronological ordering uses the ROOT's started_at (conversation
        start), not the tip's. This keeps lineage entries stable in the list
        even as new compressions push the tip forward in time.
        """
        import time as _time
        t0 = _time.time() - 3600
        self._build_compression_chain(db, t0)

        # Create a newer standalone session that should sort above the lineage
        # if we used tip.started_at, but below if we correctly use root.started_at.
        t_between = t0 + 120  # between root1 and its compression
        db.create_session("newer", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t_between, "newer"))
        db.append_message("newer", "user", "newer session started after root1")
        db._conn.commit()

        sessions = db.list_sessions_rich(source="cli", limit=20)
        ids_in_order = [s["id"] for s in sessions]
        # 'newer' started AFTER root1 but BEFORE tip1's actual started_at.
        # Correct ordering (by root started_at): newer > tip1's lineage entry.
        assert ids_in_order.index("newer") < ids_in_order.index("tip1")

    def test_list_handles_broken_chain_gracefully(self, db):
        """A compression root with no child (e.g. DB corruption or a partial
        end_session call that didn't finish creating the child) must not
        crash the list — it should fall back to surfacing the root as-is.
        """
        import time as _time
        t0 = _time.time() - 100
        db.create_session("orphan", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, "orphan"))
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t0 + 10, "compression", "orphan"),
        )
        db._conn.commit()

        sessions = db.list_sessions_rich(source="cli", limit=10)
        ids = [s["id"] for s in sessions]
        assert "orphan" in ids
        row = next(s for s in sessions if s["id"] == "orphan")
        # No tip means no projection — row stays raw.
        assert "_lineage_root_id" not in row
        assert row["end_reason"] == "compression"


# =========================================================================
# Session source exclusion (--source flag for third-party isolation)
# =========================================================================

class TestExcludeSources:
    """Tests for exclude_sources on list_sessions_rich and search_messages."""

    def test_list_sessions_rich_excludes_tool_source(self, db):
        db.create_session("s1", "cli")
        db.create_session("s2", "tool")
        db.create_session("s3", "telegram")
        sessions = db.list_sessions_rich(exclude_sources=["tool"])
        ids = [s["id"] for s in sessions]
        assert "s1" in ids
        assert "s3" in ids
        assert "s2" not in ids

    def test_list_sessions_rich_no_exclusion_returns_all(self, db):
        db.create_session("s1", "cli")
        db.create_session("s2", "tool")
        sessions = db.list_sessions_rich()
        ids = [s["id"] for s in sessions]
        assert "s1" in ids
        assert "s2" in ids

    def test_list_sessions_rich_source_and_exclude_combined(self, db):
        """When source= is explicit, exclude_sources should not conflict."""
        db.create_session("s1", "cli")
        db.create_session("s2", "tool")
        db.create_session("s3", "telegram")
        # Explicit source filter: only tool sessions, no exclusion
        sessions = db.list_sessions_rich(source="tool")
        ids = [s["id"] for s in sessions]
        assert ids == ["s2"]

    def test_list_sessions_rich_exclude_multiple_sources(self, db):
        db.create_session("s1", "cli")
        db.create_session("s2", "tool")
        db.create_session("s3", "cron")
        db.create_session("s4", "telegram")
        sessions = db.list_sessions_rich(exclude_sources=["tool", "cron"])
        ids = [s["id"] for s in sessions]
        assert "s1" in ids
        assert "s4" in ids
        assert "s2" not in ids
        assert "s3" not in ids

    def test_search_messages_excludes_tool_source(self, db):
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Python deployment question")
        db.create_session("s2", "tool")
        db.append_message("s2", "user", "Python automated question")
        results = db.search_messages("Python", exclude_sources=["tool"])
        sources = [r["source"] for r in results]
        assert "cli" in sources
        assert "tool" not in sources

    def test_search_messages_no_exclusion_returns_all_sources(self, db):
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Rust deployment question")
        db.create_session("s2", "tool")
        db.append_message("s2", "user", "Rust automated question")
        results = db.search_messages("Rust")
        sources = [r["source"] for r in results]
        assert "cli" in sources
        assert "tool" in sources

    def test_search_messages_source_include_and_exclude(self, db):
        """source_filter (include) and exclude_sources can coexist."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Golang test")
        db.create_session("s2", "telegram")
        db.append_message("s2", "user", "Golang test")
        db.create_session("s3", "tool")
        db.append_message("s3", "user", "Golang test")
        # Include cli+tool, but exclude tool → should only return cli
        results = db.search_messages(
            "Golang", source_filter=["cli", "tool"], exclude_sources=["tool"]
        )
        sources = [r["source"] for r in results]
        assert sources == ["cli"]


class TestResolveSessionByNameOrId:
    """Tests for the main.py helper that resolves names or IDs."""

    def test_resolve_by_id(self, db):
        db.create_session("test-id-123", "cli")
        session = db.get_session("test-id-123")
        assert session is not None
        assert session["id"] == "test-id-123"

    def test_resolve_by_title_falls_back(self, db):
        db.create_session("s1", "cli")
        db.set_session_title("s1", "my project")
        result = db.resolve_session_by_title("my project")
        assert result == "s1"


# =========================================================================
# Concurrent write safety / lock contention fixes (#3139)
# =========================================================================

class TestConcurrentWriteSafety:
    def test_create_session_insert_or_ignore_is_idempotent(self, db):
        """create_session with the same ID twice must not raise (INSERT OR IGNORE)."""
        db.create_session(session_id="dup-1", source="cli", model="m")
        # Second call should be silent — no IntegrityError
        db.create_session(session_id="dup-1", source="gateway", model="m2")
        session = db.get_session("dup-1")
        # Row should exist (first write wins with OR IGNORE)
        assert session is not None
        assert session["source"] == "cli"

    def test_ensure_session_creates_missing_row(self, db):
        """ensure_session must create a minimal row when the session doesn't exist."""
        assert db.get_session("orphan-session") is None
        db.ensure_session("orphan-session", source="gateway", model="test-model")
        row = db.get_session("orphan-session")
        assert row is not None
        assert row["source"] == "gateway"
        assert row["model"] == "test-model"

    def test_ensure_session_is_idempotent(self, db):
        """ensure_session on an existing row must be a no-op (no overwrite)."""
        db.create_session(session_id="existing", source="cli", model="original-model")
        db.ensure_session("existing", source="gateway", model="overwrite-model")
        row = db.get_session("existing")
        # First write wins — ensure_session must not overwrite
        assert row["source"] == "cli"
        assert row["model"] == "original-model"

    def test_ensure_session_allows_append_message_after_failed_create(self, db):
        """Messages can be flushed even when create_session failed at startup.

        Simulates the #3139 scenario: create_session raises (lock), then
        ensure_session is called during flush, then append_message succeeds.
        """
        # Simulate failed create_session — row absent
        db.ensure_session("late-session", source="gateway", model="gpt-4")
        db.append_message(
            session_id="late-session",
            role="user",
            content="hello after lock",
        )
        msgs = db.get_messages("late-session")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hello after lock"

    def test_sqlite_timeout_is_at_least_30s(self, db):
        """Connection timeout should be >= 30s to survive CLI/gateway contention."""
        # Access the underlying connection timeout via sqlite3 introspection.
        # There is no public API, so we check the kwarg via the module default.
        import sqlite3
        import inspect
        from hermes_state import SessionDB as _SessionDB
        src = inspect.getsource(_SessionDB.__init__)
        assert "30" in src, (
            "SQLite timeout should be at least 30s to handle CLI/gateway lock contention"
        )
