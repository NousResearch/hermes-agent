"""Turn-boundary fresh-session rollover persists a compact recovery pointer."""

import json
from dataclasses import replace
from pathlib import Path

from hermes_state import SessionDB
from session_rollover import (
    RolloverPolicy,
    TurnBoundaryRollover,
    allows_new_work,
    consume_handoff_note,
    mark_completed_turn,
)
from agent.session_lifecycle import LifecycleBudget, LifecycleState, evaluate_lifecycle


def test_oh_my_feed_runaway_fixture_drains_on_relative_budget_before_exhaustion() -> None:
    """The incident must reserve closeout work without an absolute token cap."""
    incident = LifecycleBudget(
        context_window_tokens=272_000,
        prompt_tokens=201_000,
        reserved_output_tokens=16_000,
        reserved_tool_result_tokens=12_000,
        reserved_checkpoint_tokens=8_000,
        max_iterations=150,
        iterations_used=148,
        api_calls=1_123,
        cache_read_tokens=142_130_176,
        compactions=16,
        in_flight_workers=2,
        closeout_iterations=2,
    )

    decision = evaluate_lifecycle(incident)

    assert decision.state is LifecycleState.DRAINING
    assert decision.accept_new_tools is False
    assert decision.accept_new_delegations is False
    assert decision.allow_in_flight_completion is True
    assert decision.context_utilization == 201_000 / 272_000
    # High historical counters are observability evidence, not a rollover rule.
    assert evaluate_lifecycle(replace(incident, api_calls=0, cache_read_tokens=0)).state is LifecycleState.DRAINING


def test_draining_parent_refuses_new_work_but_not_completion_delivery(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent", source="cli")
    db.patch_session_model_config("parent", {"turn_boundary_lifecycle": {"state": "draining"}})
    agent = type("Agent", (), {"_session_db": db, "session_id": "parent"})()

    assert allows_new_work(agent) is False


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


def test_policy_re_resolves_on_model_switch_without_fixed_token_decision() -> None:
    policy = RolloverPolicy.from_config(
        {
            "enabled": True,
            "ratio": 0.90,
            "threshold_tokens": 800_000,
            "safety_margin_tokens": 2_000,
        }
    )

    assert policy.resolve(context_length=900_000, compression_threshold=810_000) == 808_000
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


def test_child_preserves_parent_runtime_config_but_consumes_pending_marker(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_config = {
        "provider": "openrouter",
        "base_url": "https://example.invalid/v1",
        "api_mode": "responses",
        "reasoning_effort": "high",
        "request_timeout": 91,
        "gateway_runtime": {"profile": "nightly"},
        "yolo": True,
    }
    db.create_session("old", source="acp", model="provider/model", model_config=parent_config)
    rollover = TurnBoundaryRollover(db)
    assert rollover.mark_pending("old", threshold_tokens=10)

    child = rollover.adopt_at_turn_boundary("old", active_work=False)
    assert child

    child_row = db.get_session(child)
    assert child_row is not None
    child_config = json.loads(child_row["model_config"])
    for key, value in parent_config.items():
        assert child_config[key] == value
    assert "_turn_boundary_rollover_pending" not in child_config
    assert child_config["turn_boundary_handoff"]["previous_session_id"] == "old"


def test_mark_pending_merges_without_clobbering_concurrent_model_mutation(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("old", source="cli", model_config={"provider": "first"})
    original_patch = db.patch_session_model_config

    def patch_with_concurrent_change(session_id, patch):
        original_patch(session_id, {"reasoning_effort": "xhigh", "yolo": True})
        original_patch(session_id, patch)

    db.patch_session_model_config = patch_with_concurrent_change
    assert TurnBoundaryRollover(db).mark_pending("old", threshold_tokens=10)
    row = db.get_session("old")
    assert row is not None
    config = json.loads(row["model_config"])
    assert config["provider"] == "first"
    assert config["reasoning_effort"] == "xhigh"
    assert config["yolo"] is True
    assert config["_turn_boundary_rollover_pending"]["threshold_tokens"] == 10


def test_completed_turn_never_arms_after_final_persistence_failure(monkeypatch, tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("old", source="cli")

    class Agent:
        session_id = "old"
        _session_db = db
        context_compressor = type("Compressor", (), {
            "context_length": 1000,
            "threshold_tokens": 900,
            "last_prompt_tokens": 800,
        })()

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"session_rollover": {"enabled": True, "ratio": 0.75}},
    )
    result = {"completed": True, "cleanup_errors": ["persist_session: disk full"]}
    assert mark_completed_turn(Agent(), result) is False
    row = db.get_session("old")
    assert row is not None
    assert "_turn_boundary_rollover_pending" not in json.loads(row["model_config"] or "{}")


def test_completed_turn_reloads_active_profile_policy_and_live_fallback_budget(monkeypatch, tmp_path: Path) -> None:
    """No default-profile snapshot or pre-fallback window may arm a rollover."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("active-profile", source="cli")

    class Agent:
        session_id = "active-profile"
        _session_db = db
        context_compressor = type("Compressor", (), {
            "context_length": 2000,
            "threshold_tokens": 1800,
            "last_prompt_tokens": 1600,
        })()

    agent = Agent()
    active_profile = {"session_rollover": {"enabled": True, "ratio": 0.75}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: active_profile)
    assert mark_completed_turn(agent, {"completed": True})

    # The fallback's smaller live window now resolves to a new trigger and its
    # active-profile policy disables the next boundary rather than reusing the
    # first call's default config/window.
    row = db.get_session("active-profile")
    assert row is not None
    db.patch_session_model_config("active-profile", {"_turn_boundary_rollover_pending": None})
    active_profile["session_rollover"] = {"enabled": False, "ratio": 0.75}
    agent.context_compressor = type("FallbackCompressor", (), {
        "context_length": 800,
        "threshold_tokens": 700,
        "last_prompt_tokens": 650,
    })()
    assert mark_completed_turn(agent, {"completed": True}) is False
    row = db.get_session("active-profile")
    assert row is not None
    assert "_turn_boundary_rollover_pending" not in json.loads(row["model_config"] or "{}")


def test_handoff_note_is_visible_once_without_transcript_message(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("old", source="cli")
    rollover = TurnBoundaryRollover(db)
    assert rollover.mark_pending("old", threshold_tokens=10)
    child = rollover.adopt_at_turn_boundary("old", active_work=False)
    assert child
    agent = type("Agent", (), {"_session_db": db, "session_id": child})()

    first = consume_handoff_note(agent)
    assert "old" in first
    assert "session_search" in first
    assert consume_handoff_note(agent) == ""
    assert db.get_messages_as_conversation(child) == []
    row = db.get_session(child)
    assert row is not None
    assert "turn_boundary_handoff" not in json.loads(row["model_config"] or "{}")
