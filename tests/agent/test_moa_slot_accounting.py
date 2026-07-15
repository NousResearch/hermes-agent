import sqlite3
from decimal import Decimal
from types import SimpleNamespace

from hermes_state import SessionDB
from agent.usage_pricing import CanonicalUsage


def test_session_moa_usage_persistence(tmp_path):
    """Test that MoA per-slot accounting records are persisted correctly."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)

    # Create a session first so foreign key constraint passes
    session_id = "test-moa-sess"
    db.ensure_session(session_id, "test session", model="code-hard")

    usage1 = CanonicalUsage(
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=5,
        cache_write_tokens=0,
        reasoning_tokens=0,
    )
    usage2 = CanonicalUsage(
        input_tokens=100,
        output_tokens=200,
        cache_read_tokens=50,
        cache_write_tokens=0,
        reasoning_tokens=0,
    )

    breakdown = [
        {
            "role": "reference",
            "slot_index": 0,
            "provider": "openrouter",
            "model": "anthropic/claude-opus",
            "usage": usage1,
            "cost_usd": Decimal("0.05"),
        },
        {
            "role": "aggregator",
            "slot_index": 1,
            "provider": "openai",
            "model": "gpt-5.5",
            "usage": usage2,
            "cost_usd": 0.15,
        },
        {
            "role": "reference",
            "slot_index": 2,
            "provider": None,
            "model": None,
            "usage": None,
            "cost_usd": None,
        },
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


def test_deleting_session_cascades_moa_usage(tmp_path):
    """Deleting a session must also delete its detailed MoA accounting rows."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    session_id = "test-moa-delete"
    db.ensure_session(session_id, "test session", model="review")
    db.add_session_moa_usage(
        session_id,
        [
            {
                "role": "reference",
                "slot_index": 0,
                "provider": "openrouter",
                "model": "anthropic/claude-opus",
                "usage": CanonicalUsage(input_tokens=10, output_tokens=2),
                "cost_usd": 0.05,
            }
        ],
        preset="review",
    )

    assert db.delete_session(session_id) is True

    with sqlite3.connect(db_path) as conn:
        session_count = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()[0]
        usage_count = conn.execute(
            "SELECT COUNT(*) FROM session_moa_usage WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
    assert session_count == 0
    assert usage_count == 0


def _moa_response(*, content, input_tokens, output_tokens, model):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


def _make_moa_agent(tmp_path, monkeypatch, db, session_id):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: review
  presets:
    review:
      reference_models:
        - provider: openrouter
          model: anthropic/claude-sonnet-4.6
      aggregator:
        provider: openrouter
        model: openai/gpt-5.5
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    def fake_call_llm(**kwargs):
        if kwargs["task"] == "moa_reference":
            return _moa_response(
                content="reference advice",
                input_tokens=11,
                output_tokens=3,
                model="anthropic/claude-sonnet-4.6",
            )
        return _moa_response(
            content="aggregator answer",
            input_tokens=29,
            output_tokens=7,
            model="openai/gpt-5.5",
        )

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    from run_agent import AIAgent

    return AIAgent(
        api_key="moa-virtual-provider",
        base_url="moa://local",
        model="review",
        provider="moa",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=["file"],
        max_iterations=1,
        session_db=db,
        session_id=session_id,
    )


def test_real_moa_conversation_persists_matching_aggregate_and_slot_usage(
    tmp_path, monkeypatch
):
    """The real MoA conversation path writes matching aggregate/detail usage."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    session_id = "moa-integration-success"
    agent = _make_moa_agent(tmp_path, monkeypatch, db, session_id)

    result = agent.run_conversation("solve this")

    assert result["final_response"] == "aggregator answer"
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        session = conn.execute(
            "SELECT input_tokens, output_tokens FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        rows = conn.execute(
            "SELECT role, input_tokens, output_tokens "
            "FROM session_moa_usage WHERE session_id = ? ORDER BY slot_index",
            (session_id,),
        ).fetchall()
        model_usage = conn.execute(
            "SELECT input_tokens, output_tokens FROM session_model_usage "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchone()

    assert session is not None
    assert model_usage is not None
    assert [
        (row["role"], row["input_tokens"], row["output_tokens"]) for row in rows
    ] == [
        ("reference", 11, 3),
        ("aggregator", 29, 7),
    ]
    assert session["input_tokens"] == sum(row["input_tokens"] for row in rows)
    assert session["output_tokens"] == sum(row["output_tokens"] for row in rows)
    assert model_usage["input_tokens"] == session["input_tokens"]
    assert model_usage["output_tokens"] == session["output_tokens"]


def test_real_moa_conversation_rolls_back_aggregate_when_slot_insert_fails(
    tmp_path, monkeypatch
):
    """A detailed-write failure cannot leave aggregate usage committed alone."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    session_id = "moa-integration-rollback"
    agent = _make_moa_agent(tmp_path, monkeypatch, db, session_id)

    db.ensure_session(session_id, "test session", model="review")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TRIGGER reject_moa_usage
            BEFORE INSERT ON session_moa_usage
            BEGIN
                SELECT RAISE(ABORT, 'forced detailed usage failure');
            END
            """
        )

    result = agent.run_conversation("solve this")

    assert result["final_response"] == "aggregator answer"
    with sqlite3.connect(db_path) as conn:
        session = conn.execute(
            "SELECT input_tokens, output_tokens, api_call_count "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        usage_count = conn.execute(
            "SELECT COUNT(*) FROM session_moa_usage WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        model_usage_count = conn.execute(
            "SELECT COUNT(*) FROM session_model_usage WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]

    assert session == (0, 0, 0)
    assert usage_count == 0
    assert model_usage_count == 0
