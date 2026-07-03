import sqlite3

from hermes_state import SessionDB
from agent.usage_pricing import CanonicalUsage


def test_session_moa_usage_persistence(tmp_path):
    """Test that MoA per-slot accounting records are persisted correctly."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    
    # Create a session first so foreign key constraint passes
    session_id = "test-moa-sess"
    db.ensure_session(session_id, "test session", model="code-hard")
    
    usage1 = CanonicalUsage(input_tokens=10, output_tokens=20, cache_read_tokens=5, cache_write_tokens=0, reasoning_tokens=0)
    usage2 = CanonicalUsage(input_tokens=100, output_tokens=200, cache_read_tokens=50, cache_write_tokens=0, reasoning_tokens=0)
    
    breakdown = [
        {
            "role": "reference",
            "slot_index": 0,
            "provider": "openrouter",
            "model": "anthropic/claude-opus",
            "usage": usage1,
            "cost_usd": 0.05
        },
        {
            "role": "aggregator",
            "slot_index": 1,
            "provider": "openai",
            "model": "gpt-5.5",
            "usage": usage2,
            "cost_usd": 0.15
        },
        {
            "role": "reference",
            "slot_index": 2,
            "provider": None,
            "model": None,
            "usage": None,
            "cost_usd": None
        }
    ]
    
    db.add_session_moa_usage(session_id, breakdown, preset="code-hard")
    
    # Verify the records were written correctly
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM session_moa_usage ORDER BY id").fetchall()
    assert len(rows) == 3
    
    # Reference
    assert rows[0]["session_id"] == session_id
    assert rows[0]["preset"] == "code-hard"
    assert rows[0]["role"] == "reference"
    assert rows[0]["slot_index"] == 0
    assert rows[0]["provider"] == "openrouter"
    assert rows[0]["model"] == "anthropic/claude-opus"
    assert rows[0]["input_tokens"] == 10
    assert rows[0]["output_tokens"] == 20
    assert rows[0]["cache_read_tokens"] == 5
    assert rows[0]["cost_usd"] == 0.05
    
    # Aggregator
    assert rows[1]["session_id"] == session_id
    assert rows[1]["preset"] == "code-hard"
    assert rows[1]["role"] == "aggregator"
    assert rows[1]["slot_index"] == 1
    assert rows[1]["provider"] == "openai"
    assert rows[1]["model"] == "gpt-5.5"
    assert rows[1]["input_tokens"] == 100
    assert rows[1]["output_tokens"] == 200
    assert rows[1]["cost_usd"] == 0.15

    # Defensive fallback for sparse/malformed entries: metadata stays safe and insertable.
    assert rows[2]["session_id"] == session_id
    assert rows[2]["preset"] == "code-hard"
    assert rows[2]["role"] == "reference"
    assert rows[2]["slot_index"] == 2
    assert rows[2]["provider"] == ""
    assert rows[2]["model"] == ""
    assert rows[2]["input_tokens"] == 0
    assert rows[2]["output_tokens"] == 0
    assert rows[2]["cache_read_tokens"] == 0
    assert rows[2]["cache_write_tokens"] == 0
    assert rows[2]["reasoning_tokens"] == 0
    assert rows[2]["cost_usd"] is None
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("PRAGMA table_info(session_moa_usage)")
        columns = [col[1] for col in cursor.fetchall()]
        assert "slot_index" in columns
        assert "prompt" not in columns
        assert "output" not in columns
        assert "content" not in columns

def test_add_session_moa_usage_empty(tmp_path):
    """Test that empty breakdown does not crash."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    session_id = "test-moa-sess"
    db.ensure_session(session_id, "test session", model="code-hard")
    
    db.add_session_moa_usage(session_id, [], preset="code-hard")
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM session_moa_usage").fetchall()
    assert len(rows) == 0
