"""Session persistence, transcript-flush and redaction tests for run_agent.AIAgent.

Split out of the former monolithic ``test_run_agent.py`` (8,699 lines) so the
per-file CI shard runner (``scripts/run_tests_parallel.py`` spawns one
subprocess per file and cannot split within a file) is not pinned by a single
file. Pure move: test bodies are byte-identical to the original. Shared
fixtures live in ``conftest.py``; shared mock builders in
``_run_agent_helpers.py``.
"""

import json
import threading
from unittest.mock import MagicMock

import pytest

import run_agent


def test_persist_user_message_override_rewrites_text_turns(agent):
    messages = [{"role": "user", "content": "API-only synthetic prefix\nhello"}]
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "hello"

    agent._apply_persist_user_message_override(messages)

    assert messages == [{"role": "user", "content": "hello"}]

def test_persist_user_message_override_preserves_multimodal_turns(agent):
    multimodal_content = [
        {"type": "text", "text": "What color is this?"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
        },
    ]
    messages = [{"role": "user", "content": multimodal_content}]
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "What color is this? [Image attachment]"

    agent._apply_persist_user_message_override(messages)

    assert messages == [{"role": "user", "content": multimodal_content}]

def test_persist_user_message_override_restores_clean_multimodal_note(agent):
    clean_content = [
        {"type": "text", "text": "Describe this screenshot"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    api_content = [
        {"type": "text", "text": "[MODEL SWITCH NOTE]\n\nDescribe this screenshot"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    messages = [{"role": "user", "content": api_content}]
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = clean_content

    agent._apply_persist_user_message_override(messages)

    assert messages == [{"role": "user", "content": clean_content}]

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

    db_write = agent._session_db.append_message.call_args.kwargs
    assert db_write["content"] == "Describe this screenshot\n[screenshot]"
    assert api_content[0]["text"] == "[MODEL SWITCH NOTE]\n\nDescribe this screenshot"

def test_direct_session_db_flushes_share_marker_claim(agent):
    """A direct flush cannot interleave its marker check with `_persist_session`."""
    class _BarrierDB:
        def __init__(self):
            self.rows = []
            self.entered = threading.Event()
            self.release = threading.Event()
            self.calls = 0
            self._lock = threading.Lock()

        def append_message(self, **kwargs):
            with self._lock:
                self.calls += 1
                first = self.calls == 1
            if first:
                self.entered.set()
                assert self.release.wait(timeout=5)
            self.rows.append(kwargs["content"])

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
    agent._session_json_enabled = False

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

class TestSessionJsonSnapshotOptIn:
    """Regression: per-session JSON snapshot writer is opt-in via config.

    state.db is canonical (PR #29182).  ``sessions.write_json_snapshots``
    defaults to False, so the agent must NOT write ``session_{sid}.json``
    files by default — that behavior caused multi-GB sessions directories
    on heavy users.  Users can opt back in for external tooling that reads
    the JSON files directly.
    """

    def test_session_json_disabled_by_default(self, agent):
        # Default config: writer is gated off.
        assert getattr(agent, "_session_json_enabled", False) is False, (
            "sessions.write_json_snapshots must default to False"
        )

    def test_save_session_log_noops_when_disabled(self, agent, tmp_path):
        # When disabled, calling the method must not write any file even
        # if logs_dir is writable and messages are non-empty.
        agent._session_json_enabled = False
        agent.logs_dir = tmp_path
        agent._session_messages = [{"role": "user", "content": "hello"}]
        agent._save_session_log()
        # No session_*.json must appear under logs_dir.
        assert list(tmp_path.glob("session_*.json")) == []

    def test_save_session_log_writes_when_enabled(self, agent, tmp_path):
        # Opt-in path: with the flag on and a session_id, the writer must
        # produce ``session_{sid}.json`` under logs_dir.
        agent._session_json_enabled = True
        agent.logs_dir = tmp_path
        messages = [{"role": "user", "content": "hello"}]
        agent._save_session_log(messages)
        expected = tmp_path / f"session_{agent.session_id}.json"
        assert expected.exists(), (
            "Opt-in writer must produce session_{sid}.json under logs_dir"
        )

    def test_logs_dir_retained_for_request_dumps(self, agent):
        # logs_dir is kept unconditionally because
        # agent_runtime_helpers.dump_api_request_debug still writes
        # request_dump_*.json there (debug breadcrumb path), independent of
        # the session JSON opt-in.
        assert hasattr(agent, "logs_dir")

    def test_traversal_session_id_cannot_escape_logs_dir(self, agent, tmp_path):
        # Security regression (#5958): a traversal-shaped session ID (which can
        # originate from the untrusted X-Hermes-Session-Id API header) must not
        # redirect the session snapshot outside the sessions directory.
        agent._session_json_enabled = True
        agent.logs_dir = tmp_path
        agent.session_id = "../../../../outside_dir/pwned"
        agent._save_session_log([{"role": "user", "content": "hello"}])

        # Exactly one snapshot, and it lives directly under logs_dir.
        written = list(tmp_path.glob("session_*.json"))
        assert len(written) == 1, "writer must produce a single contained snapshot"
        assert written[0].resolve().parent == tmp_path.resolve()
        # Nothing escaped to the traversal target.
        assert not (tmp_path.parent.parent / "outside_dir").exists()

    def test_safe_session_filename_component_contains_traversal(self):
        # The sanitizer is the chokepoint: every session-ID-derived artifact
        # path goes through it, so it must always yield a single, traversal-free
        # path segment while leaving legitimate IDs untouched.
        f = run_agent._safe_session_filename_component
        for raw in ("../../etc/passwd", "/abs/path", "..\\win\\trav", "a/b/c"):
            out = f(raw)
            assert "/" not in out and "\\" not in out and ".." not in out, out
        # Legit IDs pass through unchanged; distinct IDs never collide.
        assert f("api-abc123def456") == "api-abc123def456"
        assert f("../a") != f("../b")

class TestSaveSessionLogRedactsSecrets:
    """Regression: session_*.json must not contain plaintext credentials (#19798, #19845)."""

    @pytest.fixture(autouse=True)
    def _ensure_redaction_enabled(self, monkeypatch):
        """Force redaction on regardless of host HERMES_REDACT_SECRETS state.
        The hermetic conftest blanks the env var; the module-level
        ``_REDACT_ENABLED`` constant is captured at import time, so we
        flip it directly for the duration of these tests."""
        monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)

    def test_redacts_api_key_in_tool_content(self, agent, tmp_path):
        agent._session_json_enabled = True
        agent.logs_dir = tmp_path
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "tool",
                "content": "Response: Authorization: Bearer sk-proj-abc123def456ghi789jkl012mno",
            },
        ]
        agent._save_session_log(messages)

        snapshot = (tmp_path / f"session_{agent.session_id}.json").read_text(encoding="utf-8")
        assert "sk-proj-abc123def456ghi789jkl012mno" not in snapshot

    def test_redacts_api_key_in_user_message(self, agent, tmp_path):
        agent._session_json_enabled = True
        agent.logs_dir = tmp_path
        messages = [
            {"role": "user", "content": "My key is sk-ant-api03-abc123def456ghi789jkl012mno please use it"},
        ]
        agent._save_session_log(messages)

        snapshot = (tmp_path / f"session_{agent.session_id}.json").read_text(encoding="utf-8")
        assert "sk-ant-api03-abc123def456ghi789jkl012mno" not in snapshot

    def test_redacts_system_prompt_credentials(self, agent, tmp_path):
        agent._session_json_enabled = True
        agent.logs_dir = tmp_path
        agent._cached_system_prompt = "Use key sk-proj-realkey1234567890123456 for API calls"
        agent._save_session_log([{"role": "user", "content": "test"}])

        snapshot = (tmp_path / f"session_{agent.session_id}.json").read_text(encoding="utf-8")
        assert "sk-proj-realkey1234567890123456" not in snapshot

    def test_redacts_list_type_multimodal_content(self, agent, tmp_path):
        """OpenAI/Anthropic multimodal shape: content = list of {type, text|image_url} parts."""
        agent._session_json_enabled = True
        agent.logs_dir = tmp_path
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Key: gsk_abc123def456ghi789jkl012mno"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            },
        ]
        agent._save_session_log(messages)

        snapshot_text = (tmp_path / f"session_{agent.session_id}.json").read_text(encoding="utf-8")
        snapshot = json.loads(snapshot_text)
        parts = snapshot["messages"][0]["content"]
        assert "gsk_abc123def456ghi789jkl012mno" not in parts[0]["text"]
        # Image part preserved untouched
        assert parts[1]["image_url"]["url"].startswith("data:image")

class TestGetMessagesUpToLastAssistant:
    def test_empty_list(self, agent):
        assert agent._get_messages_up_to_last_assistant([]) == []

    def test_no_assistant_returns_copy(self, agent):
        msgs = [{"role": "user", "content": "hi"}]
        result = agent._get_messages_up_to_last_assistant(msgs)
        assert result == msgs
        assert result is not msgs  # should be a copy

    def test_single_assistant(self, agent):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = agent._get_messages_up_to_last_assistant(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_multiple_assistants_returns_up_to_last(self, agent):
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        result = agent._get_messages_up_to_last_assistant(msgs)
        assert len(result) == 3
        assert result[-1]["content"] == "q2"

    def test_assistant_then_tool_messages(self, agent):
        msgs = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "result", "tool_call_id": "1"},
        ]
        # Last assistant is at index 1, so result = msgs[:1]
        result = agent._get_messages_up_to_last_assistant(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

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
        first_db_write = agent._session_db.append_message.call_args_list[0].kwargs
        assert first_db_write["content"] == "Hello there"
