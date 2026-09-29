"""test_run_agent_session_persist: split from the former tests/agent/test_run_agent.py monolith (#99982)."""
import threading
from unittest.mock import MagicMock


def test_persist_user_message_override_rewrites_text_turns(agent):
    messages = [{"role": "user", "content": "API-only synthetic prefix\nhello"}]
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "hello"

    agent._apply_persist_user_message_override(messages)

    assert messages == [{"role": "user", "content": "hello"}]

def test_flush_persist_override_replaces_api_local_multimodal_note(agent):
    """A note-added multimodal API payload stores the original clean content."""
    clean_content = [
        {"type": "text", "text": "Describe this screenshot"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    api_content = [
        {"type": "text", "text": "[MODEL SWITCH NOTE]\n\nDescribe this screenshot"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    agent._session_db = MagicMock()
    agent._session_db_created = True
    agent.session_id = "session-123"
    agent._last_flushed_db_idx = 0
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = clean_content
    agent._persist_user_message_timestamp = None

    agent._flush_messages_to_session_db([{"role": "user", "content": api_content}], [])

    batch = agent._session_db.append_messages_batch.call_args.kwargs["messages"]
    assert batch[0]["content"] == "Describe this screenshot\n[screenshot]"
    assert api_content[0]["text"] == "[MODEL SWITCH NOTE]\n\nDescribe this screenshot"

def test_direct_session_db_flushes_share_marker_claim(agent):
    """A direct flush cannot interleave its marker check with `_persist_session`."""
    class _BarrierDB:
        def __init__(self):
            self.rows = []
            self.entered = threading.Event()
            self.release = threading.Event()
            self.calls = 0
            self.token_flushes = 0
            self._lock = threading.Lock()

        def flush_token_counts(self):
            self.token_flushes += 1

        def append_message(self, **kwargs):
            with self._lock:
                self.calls += 1
                first = self.calls == 1
            if first:
                self.entered.set()
                assert self.release.wait(timeout=5)
            self.rows.append(kwargs["content"])

        def append_messages_batch(self, session_id, messages, **kwargs):
            with self._lock:
                self.calls += 1
                first = self.calls == 1
            if first:
                self.entered.set()
                assert self.release.wait(timeout=5)
            for m in messages:
                self.rows.append(m["content"])
            return list(range(1, len(messages) + 1))

    db = _BarrierDB()
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = "session-123"
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._persist_user_message_timestamp = None
    agent._persist_disabled = False
    agent._session_persist_lock = threading.RLock()


    message = {"role": "user", "content": "exactly once"}
    normal = threading.Thread(target=lambda: agent._persist_session([message], []))
    direct = threading.Thread(target=lambda: agent._flush_messages_to_session_db([message], []))
    normal.start()
    assert db.entered.wait(timeout=5)
    direct.start()
    # Direct flush is blocked by the agent-wide persistence lock until the
    # normal writer stamps the message's durable marker.
    assert db.calls == 1
    db.release.set()
    normal.join(timeout=5)
    direct.join(timeout=5)

    assert not normal.is_alive()
    assert not direct.is_alive()
    assert db.rows == ["exactly once"]
    assert db.token_flushes == 1

class TestSessionFilenameSafety:
    def test_safe_session_filename_component_contains_traversal(self):
        # The sanitizer is the chokepoint: every session-ID-derived artifact
        # path goes through it, so it must always yield a single, traversal-free
        # path segment while leaving legitimate IDs untouched.
        from agent.session_persistence import _safe_session_filename_component as f
        for raw in ("../../etc/passwd", "/abs/path", "..\\win\\trav", "a/b/c"):
            out = f(raw)
            assert "/" not in out and "\\" not in out and ".." not in out, out
        # Legit IDs pass through unchanged; distinct IDs never collide.
        assert f("api-abc123def456") == "api-abc123def456"
        assert f("../a") != f("../b")

class TestGetMessagesUpToLastAssistant:
    def test_empty_list(self, agent):
        assert agent._get_messages_up_to_last_assistant([]) == []

    def test_no_assistant_returns_copy(self, agent):
        msgs = [{"role": "user", "content": "hi"}]
        result = agent._get_messages_up_to_last_assistant(msgs)
        assert result == msgs
        assert result is not msgs  # should be a copy

class TestPersistUserMessageOverride:
    """Synthetic API-only user prefixes should never leak into transcripts."""

    def test_persist_session_rewrites_current_turn_user_message(self, agent):
        agent._session_db = MagicMock()
        agent.session_id = "session-123"
        agent._last_flushed_db_idx = 0
        agent._persist_user_message_idx = 0
        agent._persist_user_message_override = "Hello there"
        messages = [
            {
                "role": "user",
                "content": (
                    "[Voice input — respond concisely and conversationally, "
                    "2-3 sentences max. No code blocks or markdown.] Hello there"
                ),
            },
            {"role": "assistant", "content": "Hi!"},
        ]

        agent._persist_session(messages, [])

        # The original messages list must NOT be mutated — the persist
        # override is applied only to the DB row (resolved inside the flush
        # chokepoint), so the live list keeps the original content for the
        # API call (#48677).
        assert (
            messages[0]["content"]
            == "[Voice input — respond concisely and conversationally, "
            "2-3 sentences max. No code blocks or markdown.] Hello there"
        )
        # But the DB write must get the override.
        batch = agent._session_db.append_messages_batch.call_args_list[0].kwargs[
            "messages"
        ]
        assert batch[0]["content"] == "Hello there"
