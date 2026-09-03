"""Turn-boundary fresh-session rollover persists a compact recovery pointer."""

import json
from pathlib import Path

from hermes_state import SessionDB
from session_rollover import RolloverPolicy, TurnBoundaryRollover


def test_rollover_is_pending_until_a_safe_next_turn_then_preserves_lineage(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("old", source="cli")
    db.append_message("old", "user", "first request")
    db.append_message("old", "assistant", "first answer")

    rollover = TurnBoundaryRollover(db)
    assert rollover.mark_pending("old", threshold_tokens=159_000) is True
    assert rollover.adopt_at_turn_boundary("old", active_work=True) is None

    new_session_id = rollover.adopt_at_turn_boundary("old", active_work=False)

    assert new_session_id
    old = db.get_session("old")
    new = db.get_session(new_session_id)
    assert old["end_reason"] == "turn_boundary_rollover"
    assert new["parent_session_id"] == "old"
    assert json.loads(new["model_config"])["turn_boundary_handoff"] == {
        "previous_session_id": "old",
        "recovery": "Use session_search to recover earlier details if needed.",
    }
    assert db.get_messages_as_conversation(new_session_id) == []


def test_ratio_policy_resolves_every_window_and_stays_below_compression() -> None:
    policy = RolloverPolicy.from_config(
        {
            "enabled": True,
            "ratio": 0.75,
            "safety_margin_tokens": 1_000,
        }
    )

    assert policy.resolve(context_length=32_000, compression_threshold=28_000) == 24_000
    assert policy.resolve(context_length=272_000, compression_threshold=250_000) == 204_000
    assert policy.resolve(context_length=900_000, compression_threshold=850_000) == 675_000


def test_policy_re_resolves_on_model_switch_and_expert_cap_never_crosses_compression() -> None:
    policy = RolloverPolicy.from_config(
        {
            "enabled": True,
            "ratio": 0.90,
            "threshold_tokens": 800_000,
            "safety_margin_tokens": 2_000,
        }
    )

    assert policy.resolve(context_length=900_000, compression_threshold=810_000) == 800_000
    assert policy.resolve(context_length=272_000, compression_threshold=230_000) == 228_000


def test_disabled_policy_never_resolves_a_trigger() -> None:
    assert RolloverPolicy.from_config({}).resolve(272_000, 240_000) is None


def test_adoption_is_exactly_once_and_preserves_gateway_identity(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(
        "old",
        source="telegram",
        model="provider/model",
        session_key="telegram:chat:1",
        user_id="user",
        chat_id="chat",
        chat_type="dm",
        thread_id="thread",
        cwd="/project",
        profile_name="default",
    )
    rollover = TurnBoundaryRollover(db)
    assert rollover.mark_pending("old", threshold_tokens=10) is True

    child = rollover.adopt_at_turn_boundary("old", active_work=False)
    assert child
    assert rollover.adopt_at_turn_boundary("old", active_work=False) is None

    row = db.get_session(child)
    assert row is not None
    assert {key: row[key] for key in ("source", "model", "session_key", "user_id", "chat_id", "chat_type", "thread_id", "cwd", "profile_name")} == {
        "source": "telegram", "model": "provider/model", "session_key": "telegram:chat:1",
        "user_id": "user", "chat_id": "chat", "chat_type": "dm", "thread_id": "thread",
        "cwd": "/project", "profile_name": "default",
    }
    assert db.get_messages_as_conversation(child) == []
