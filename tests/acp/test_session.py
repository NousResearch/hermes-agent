"""Tests for acp_adapter.session — SessionManager and SessionState."""

import json
from types import SimpleNamespace
import pytest
from unittest.mock import MagicMock, patch

from acp_adapter.session import SessionManager, SessionState
from hermes_state import SessionDB


def _mock_agent():
    return MagicMock(name="MockAIAgent")


@pytest.fixture()
def manager():
    """SessionManager with a mock agent factory (avoids needing API keys)."""
    return SessionManager(agent_factory=_mock_agent)


# ---------------------------------------------------------------------------
# create / get
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_create_session_returns_state(self, manager):
        state = manager.create_session(cwd="/tmp/work")
        assert isinstance(state, SessionState)
        assert state.cwd == "/tmp/work"
        assert state.session_id
        assert state.history == []
        assert state.agent is not None

    def test_create_session_registers_task_cwd(self, manager, monkeypatch):
        calls = []
        monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda task_id, cwd: calls.append((task_id, cwd)))
        state = manager.create_session(cwd="/tmp/work")
        assert calls == [(state.session_id, "/tmp/work")]

    def test_session_ids_are_unique(self, manager):
        s1 = manager.create_session()
        s2 = manager.create_session()
        assert s1.session_id != s2.session_id

    def test_get_session(self, manager):
        state = manager.create_session()
        fetched = manager.get_session(state.session_id)
        assert fetched is state

    def test_get_nonexistent_session_returns_none(self, manager):
        assert manager.get_session("does-not-exist") is None


# ---------------------------------------------------------------------------
# fork
# ---------------------------------------------------------------------------


class TestForkSession:
    def test_fork_session_deep_copies_history(self, manager):
        original = manager.create_session()
        original.history.append({"role": "user", "content": "hello"})
        original.history.append({"role": "assistant", "content": "hi"})

        forked = manager.fork_session(original.session_id, cwd="/new")
        assert forked is not None

        # History should be equal in content
        assert len(forked.history) == 2
        assert forked.history[0]["content"] == "hello"

        # But a deep copy — mutating one doesn't affect the other
        forked.history.append({"role": "user", "content": "extra"})
        assert len(original.history) == 2
        assert len(forked.history) == 3

    def test_fork_session_has_new_id(self, manager):
        original = manager.create_session()
        forked = manager.fork_session(original.session_id)
        assert forked is not None
        assert forked.session_id != original.session_id

    def test_fork_nonexistent_returns_none(self, manager):
        assert manager.fork_session("bogus-id") is None


# ---------------------------------------------------------------------------
# list / cleanup / remove
# ---------------------------------------------------------------------------


class TestListAndCleanup:
    def test_list_sessions_empty(self, manager):
        assert manager.list_sessions() == []

    def test_list_sessions_returns_created(self, manager):
        s1 = manager.create_session(cwd="/a")
        s2 = manager.create_session(cwd="/b")
        listing = manager.list_sessions()
        ids = {s["session_id"] for s in listing}
        assert s1.session_id in ids
        assert s2.session_id in ids
        assert len(listing) == 2

    def test_cleanup_clears_all(self, manager):
        manager.create_session()
        manager.create_session()
        assert len(manager.list_sessions()) == 2
        manager.cleanup()
        assert manager.list_sessions() == []

    def test_remove_session(self, manager):
        state = manager.create_session()
        assert manager.remove_session(state.session_id) is True
        assert manager.get_session(state.session_id) is None
        # Removing again returns False
        assert manager.remove_session(state.session_id) is False


# ---------------------------------------------------------------------------
# persistence — sessions survive process restarts (via SessionDB)
# ---------------------------------------------------------------------------


class TestPersistence:
    """Verify that sessions are persisted to SessionDB and can be restored."""

    def test_create_session_writes_to_db(self, manager):
        state = manager.create_session(cwd="/project")
        db = manager._get_db()
        assert db is not None
        row = db.get_session(state.session_id)
        assert row is not None
        assert row["source"] == "acp"
        # cwd stored in model_config JSON
        mc = json.loads(row["model_config"])
        assert mc["cwd"] == "/project"

    def test_get_session_restores_from_db(self, manager):
        """Simulate process restart: create session, drop from memory, get again."""
        state = manager.create_session(cwd="/work")
        state.history.append({"role": "user", "content": "hello"})
        state.history.append({"role": "assistant", "content": "hi there"})
        manager.save_session(state.session_id)

        sid = state.session_id

        # Drop from in-memory store (simulates process restart).
        with manager._lock:
            del manager._sessions[sid]

        # get_session should transparently restore from DB.
        restored = manager.get_session(sid)
        assert restored is not None
        assert restored.session_id == sid
        assert restored.cwd == "/work"
        assert len(restored.history) == 2
        assert restored.history[0]["content"] == "hello"
        assert restored.history[1]["content"] == "hi there"
        # Agent should have been recreated.
        assert restored.agent is not None

    def test_save_session_updates_db(self, manager):
        state = manager.create_session()
        state.history.append({"role": "user", "content": "test"})
        manager.save_session(state.session_id)

        db = manager._get_db()
        messages = db.get_messages_as_conversation(state.session_id)
        assert len(messages) == 1
        assert messages[0]["content"] == "test"

    def test_remove_session_deletes_from_db(self, manager):
        state = manager.create_session()
        db = manager._get_db()
        assert db.get_session(state.session_id) is not None
        manager.remove_session(state.session_id)
        assert db.get_session(state.session_id) is None

    def test_cleanup_removes_all_from_db(self, manager):
        s1 = manager.create_session()
        s2 = manager.create_session()
        db = manager._get_db()
        assert db.get_session(s1.session_id) is not None
        assert db.get_session(s2.session_id) is not None
        manager.cleanup()
        assert db.get_session(s1.session_id) is None
        assert db.get_session(s2.session_id) is None

    def test_list_sessions_includes_db_only(self, manager):
        """Sessions only in DB (not in memory) appear in list_sessions."""
        state = manager.create_session(cwd="/db-only")
        sid = state.session_id

        # Drop from memory.
        with manager._lock:
            del manager._sessions[sid]

        listing = manager.list_sessions()
        ids = {s["session_id"] for s in listing}
        assert sid in ids

    def test_fork_restores_source_from_db(self, manager):
        """Forking a session that is only in DB should work."""
        original = manager.create_session()
        original.history.append({"role": "user", "content": "context"})
        manager.save_session(original.session_id)

        # Drop original from memory.
        with manager._lock:
            del manager._sessions[original.session_id]

        forked = manager.fork_session(original.session_id, cwd="/fork")
        assert forked is not None
        assert len(forked.history) == 1
        assert forked.history[0]["content"] == "context"
        assert forked.session_id != original.session_id

    def test_update_cwd_restores_from_db(self, manager):
        state = manager.create_session(cwd="/old")
        sid = state.session_id

        with manager._lock:
            del manager._sessions[sid]

        updated = manager.update_cwd(sid, "/new")
        assert updated is not None
        assert updated.cwd == "/new"

        # Should also be persisted in DB.
        db = manager._get_db()
        row = db.get_session(sid)
        mc = json.loads(row["model_config"])
        assert mc["cwd"] == "/new"

    def test_only_restores_acp_sessions(self, manager):
        """get_session should not restore non-ACP sessions from DB."""
        db = manager._get_db()
        # Manually create a CLI session in the DB.
        db.create_session(session_id="cli-session-123", source="cli", model="test")
        # Should not be found via ACP SessionManager.
        assert manager.get_session("cli-session-123") is None

    def test_sessions_searchable_via_fts(self, manager):
        """ACP sessions stored in SessionDB are searchable via FTS5."""
        state = manager.create_session()
        state.history.append({"role": "user", "content": "how do I configure nginx"})
        state.history.append({"role": "assistant", "content": "Here is the nginx config..."})
        manager.save_session(state.session_id)

        db = manager._get_db()
        results = db.search_messages("nginx")
        assert len(results) > 0
        session_ids = {r["session_id"] for r in results}
        assert state.session_id in session_ids

    def test_tool_calls_persisted(self, manager):
        """Messages with tool_calls should round-trip through the DB."""
        state = manager.create_session()
        state.history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc_1", "type": "function",
                            "function": {"name": "terminal", "arguments": "{}"}}],
        })
        state.history.append({
            "role": "tool",
            "content": "output here",
            "tool_call_id": "tc_1",
            "name": "terminal",
        })
        manager.save_session(state.session_id)

        # Drop from memory, restore from DB.
        with manager._lock:
            del manager._sessions[state.session_id]

        restored = manager.get_session(state.session_id)
        assert restored is not None
        assert len(restored.history) == 2
        assert restored.history[0].get("tool_calls") is not None
        assert restored.history[1].get("tool_call_id") == "tc_1"

    def test_restore_preserves_persisted_provider_snapshot(self, tmp_path, monkeypatch):
        """Restored ACP sessions should keep their original runtime provider."""
        runtime_choice = {"provider": "anthropic"}

        def fake_resolve_runtime_provider(requested=None, **kwargs):
            provider = requested or runtime_choice["provider"]
            return {
                "provider": provider,
                "api_mode": "anthropic_messages" if provider == "anthropic" else "chat_completions",
                "base_url": f"https://{provider}.example/v1",
                "api_key": f"{provider}-key",
                "command": None,
                "args": [],
            }

        def fake_agent(**kwargs):
            return SimpleNamespace(
                model=kwargs.get("model"),
                provider=kwargs.get("provider"),
                base_url=kwargs.get("base_url"),
                api_mode=kwargs.get("api_mode"),
            )

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "model": {"provider": runtime_choice["provider"], "default": "test-model"}
        })
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve_runtime_provider,
        )
        db = SessionDB(tmp_path / "state.db")

        with patch("run_agent.AIAgent", side_effect=fake_agent):
            manager = SessionManager(db=db)
            state = manager.create_session(cwd="/work")
            manager.save_session(state.session_id)

            with manager._lock:
                del manager._sessions[state.session_id]

            runtime_choice["provider"] = "openrouter"
            restored = manager.get_session(state.session_id)

        assert restored is not None
        assert restored.agent.provider == "anthropic"
        assert restored.agent.base_url == "https://anthropic.example/v1"
