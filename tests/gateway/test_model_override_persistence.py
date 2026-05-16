"""Test per-session model/reasoning override persistence (Issue #26570)."""

import json
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def state_db(tmp_path):
    """Create a SessionDB with a temp database."""
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "test_state.db")
    yield db
    db.close()


class TestModelOverridePersistence:
    """session_model_overrides table CRUD."""

    def test_set_and_get_override(self, state_db):
        state_db.set_model_override("sess_abc", {
            "model": "gpt-4o", "provider": "openai",
            "api_key": "sk-xxx", "base_url": "", "api_mode": "chat",
        })
        result = state_db.get_all_model_overrides()
        assert "sess_abc" in result
        assert result["sess_abc"]["model"] == "gpt-4o"
        assert result["sess_abc"]["provider"] == "openai"

    def test_update_override(self, state_db):
        state_db.set_model_override("sess_1", {"model": "a"})
        state_db.set_model_override("sess_1", {"model": "b"})
        result = state_db.get_all_model_overrides()
        assert result["sess_1"]["model"] == "b"

    def test_delete_override(self, state_db):
        state_db.set_model_override("sess_1", {"model": "a"})
        state_db.del_model_override("sess_1")
        result = state_db.get_all_model_overrides()
        assert "sess_1" not in result

    def test_delete_nonexistent(self, state_db):
        # Should not raise
        state_db.del_model_override("nonexistent")

    def test_get_all_empty(self, state_db):
        result = state_db.get_all_model_overrides()
        assert result == {}

    def test_multiple_overrides(self, state_db):
        for i in range(5):
            state_db.set_model_override(f"sess_{i}", {"model": f"model_{i}"})
        result = state_db.get_all_model_overrides()
        assert len(result) == 5
        assert result["sess_3"]["model"] == "model_3"

    def test_survives_restart(self, tmp_path):
        """Overrides survive closing and reopening the DB."""
        from hermes_state import SessionDB
        db_path = tmp_path / "restart_test.db"
        db1 = SessionDB(db_path=db_path)
        db1.set_model_override("sess_x", {"model": "claude-4", "provider": "anthropic"})
        db1.close()

        db2 = SessionDB(db_path=db_path)
        result = db2.get_all_model_overrides()
        db2.close()
        assert result["sess_x"]["model"] == "claude-4"


class TestReasoningOverridePersistence:
    """session_reasoning_overrides table CRUD."""

    def test_set_and_get(self, state_db):
        state_db.set_reasoning_override("sess_1", {"effort": "high", "budget_tokens": 1000})
        result = state_db.get_all_reasoning_overrides()
        assert result["sess_1"]["effort"] == "high"

    def test_delete(self, state_db):
        state_db.set_reasoning_override("sess_1", {"effort": "low"})
        state_db.del_reasoning_override("sess_1")
        assert "sess_1" not in state_db.get_all_reasoning_overrides()

    def test_survives_restart(self, tmp_path):
        from hermes_state import SessionDB
        db_path = tmp_path / "restart_test.db"
        db1 = SessionDB(db_path=db_path)
        db1.set_reasoning_override("sess_y", {"effort": "medium"})
        db1.close()

        db2 = SessionDB(db_path=db_path)
        result = db2.get_all_reasoning_overrides()
        db2.close()
        assert result["sess_y"]["effort"] == "medium"
