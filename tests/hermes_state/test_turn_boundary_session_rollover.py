"""Turn-boundary fresh-session rollover persists a compact recovery pointer."""

import json
from dataclasses import replace
from pathlib import Path
import threading

import pytest

from hermes_state import SessionDB
from session_rollover import (
    END_REASON,
    RECOVERY_END_REASON,
    RolloverPolicy,
    TurnBoundaryRollover,
    allows_new_work,
    allows_new_delegation,
    consume_handoff_note,
    mark_completed_turn,
)
from agent.session_lifecycle import LifecycleBudget, LifecycleState, evaluate_lifecycle


@pytest.fixture(autouse=True)
def enabled_rollover_policy(monkeypatch):
    """Direct persistence tests model an arm created while the opt-in is enabled."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"session_rollover": {"enabled": True, "ratio": 0.75}},
    )


# The exact production incident, session ``20260903_112822_b3a7b6``. These
# numbers are the regression fixture for Stage 2A recovery: 12h wall clock,
# 1,123 API calls, 142,130,176 cache-read tokens, 16 terminal compactions and
# an exhausted iteration budget.
OH_MY_FEED_INCIDENT = {
    "elapsed_seconds": 12 * 3600,
    "api_calls": 1_123,
    "cache_read_tokens": 142_130_176,
    "compactions": 16,
    "max_iterations": 150,
    "iterations_used": 150,
}


def test_oh_my_feed_incident_fixture_drains_on_iteration_closeout_reserve() -> None:
    """The exact incident counters must drain via the relative budget only."""
    incident = LifecycleBudget(
        context_window_tokens=272_000,
        # Context alone is still comfortable; the closeout reserve is what
        # must stop new work, so the fixture cannot pass by accident.
        prompt_tokens=120_000,
        reserved_output_tokens=16_000,
        reserved_tool_result_tokens=12_000,
        reserved_checkpoint_tokens=8_000,
        max_iterations=OH_MY_FEED_INCIDENT["max_iterations"],
        iterations_used=OH_MY_FEED_INCIDENT["iterations_used"],
        api_calls=OH_MY_FEED_INCIDENT["api_calls"],
        cache_read_tokens=OH_MY_FEED_INCIDENT["cache_read_tokens"],
        compactions=OH_MY_FEED_INCIDENT["compactions"],
        in_flight_workers=2,
        closeout_iterations=2,
    )

    decision = evaluate_lifecycle(incident)

    assert decision.state is LifecycleState.DRAINING
    assert decision.remaining_iterations == 0
    assert decision.reserved_headroom_tokens == 36_000
    # 142M cache-read and 16 compactions are evidence, never the rule: with a
    # fresh iteration budget the very same counters stay healthy.
    assert evaluate_lifecycle(replace(incident, iterations_used=1)).state is LifecycleState.HEALTHY


def test_completed_turn_records_recovery_status_without_arming_a_rollover(
    monkeypatch, tmp_path: Path
) -> None:
    """A healthy turn must still publish the fields doctor recovery reads."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("live", source="cli")

    class Agent:
        session_id = "live"
        _session_db = db
        _executing_tools = False
        _active_children = ()
        max_iterations = 150
        context_compressor = type("Compressor", (), {
            "context_length": 272_000,
            "threshold_tokens": 250_000,
            "last_prompt_tokens": 10_000,
        })()

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"session_rollover": {"enabled": True, "ratio": 0.75}},
    )
    assert mark_completed_turn(Agent(), {"completed": True, "api_calls": 3}) is False

    config = json.loads(db.get_session("live")["model_config"] or "{}")
    assert "_turn_boundary_rollover_pending" not in config
    status = config["turn_boundary_lifecycle"]
    assert status["state"] == "healthy"
    assert status["reserved_headroom_tokens"] == 0
    assert status["in_flight_workers"] == 0
    assert status["active_tool_call"] is False
    assert status["context_utilization"] == 10_000 / 272_000
    assert status["last_progress_at"] >= status["state_entered_at"] > 0


def test_zero_headroom_does_not_turn_closeout_iterations_into_a_fixed_drain() -> None:
    """Ratio-only rollout must not fence a healthy low-usage 272K model."""
    decision = evaluate_lifecycle(LifecycleBudget(
        context_window_tokens=272_000,
        prompt_tokens=2_000,
        reserved_output_tokens=0,
        reserved_tool_result_tokens=0,
        reserved_checkpoint_tokens=0,
        max_iterations=3,
        iterations_used=1,
        closeout_iterations=2,
    ))

    assert decision.state is LifecycleState.HEALTHY
    assert decision.accept_new_tools is True


def test_stalled_turn_does_not_advance_last_meaningful_progress(
    monkeypatch, tmp_path: Path
) -> None:
    """A completed turn that did no API work must not look like progress."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("live", source="cli")

    class Agent:
        session_id = "live"
        _session_db = db
        max_iterations = 150
        context_compressor = type("Compressor", (), {
            "context_length": 272_000,
            "threshold_tokens": 250_000,
            "last_prompt_tokens": 10_000,
        })()

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"session_rollover": {"enabled": True, "ratio": 0.75}},
    )
    agent = Agent()
    mark_completed_turn(agent, {"completed": True, "api_calls": 4})
    first = json.loads(db.get_session("live")["model_config"])["turn_boundary_lifecycle"]

    mark_completed_turn(agent, {"completed": True, "api_calls": 0})
    second = json.loads(db.get_session("live")["model_config"])["turn_boundary_lifecycle"]

    assert second["last_progress_at"] == first["last_progress_at"]
    assert second["updated_at"] >= first["updated_at"]


def test_doctor_recovery_request_is_idempotent_and_ends_with_explicit_reason(
    tmp_path: Path,
) -> None:
    """Exactly one rollover per idempotency key, with a doctor-specific reason."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("stalled", source="cli")
    rollover = TurnBoundaryRollover(db)

    assert rollover.request_recovery("stalled", idempotency_key="k1") == "armed"
    assert rollover.request_recovery("stalled", idempotency_key="k1") == "already_armed"
    assert rollover.request_recovery("stalled", idempotency_key="k2") == "already_armed"

    child = rollover.adopt_at_turn_boundary("stalled", active_work=False)
    assert child
    old = db.get_session("stalled")
    assert old["end_reason"] == RECOVERY_END_REASON
    handoff = json.loads(db.get_session(child)["model_config"])["turn_boundary_handoff"]
    assert handoff["previous_session_id"] == "stalled"
    assert handoff["recovery_key"] == "k1"
    # The consumed request cannot arm a second continuation.
    assert rollover.adopt_at_turn_boundary("stalled", active_work=False) is None


def test_concurrent_doctor_recovery_arms_one_key_and_child_binds_that_key(tmp_path: Path) -> None:
    """The check-and-arm transition is one write transaction, not read/patch/reread."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("stalled", source="cli", model_config={"provider": "openrouter"})
    start = threading.Barrier(3)
    results: dict[str, str | None] = {}

    def arm(key: str) -> None:
        start.wait()
        results[key] = TurnBoundaryRollover(db).request_recovery(
            "stalled", idempotency_key=key,
        )

    first = threading.Thread(target=arm, args=("k1",))
    second = threading.Thread(target=arm, args=("k2",))
    first.start()
    second.start()
    start.wait()
    first.join()
    second.join()

    winners = [key for key, outcome in results.items() if outcome == "armed"]
    assert len(winners) == 1
    winner = winners[0]
    config = json.loads(db.get_session("stalled")["model_config"])
    assert config["provider"] == "openrouter"
    assert config["_turn_boundary_rollover_pending"]["recovery_key"] == winner
    child = TurnBoundaryRollover(db).adopt_at_turn_boundary("stalled", active_work=False)
    assert child
    handoff = json.loads(db.get_session(child)["model_config"])["turn_boundary_handoff"]
    assert handoff["recovery_key"] == winner


def test_recovery_request_preserves_runtime_config_and_refuses_ended_sessions(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("live", source="cli", model_config={"provider": "openrouter"})
    rollover = TurnBoundaryRollover(db)
    assert rollover.request_recovery("live", idempotency_key="k1") == "armed"
    config = json.loads(db.get_session("live")["model_config"])
    assert config["provider"] == "openrouter"
    assert config["_turn_boundary_rollover_pending"]["recovery_key"] == "k1"

    db.create_session("done", source="cli")
    db.end_session("done", "user_exit")
    assert rollover.request_recovery("done", idempotency_key="k1") is None
    assert rollover.request_recovery("missing", idempotency_key="k1") is None


def test_core_armed_rollover_keeps_the_plain_turn_boundary_end_reason(
    tmp_path: Path,
) -> None:
    """Stage 1's own drain must stay distinguishable from doctor recovery."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("old", source="cli")
    rollover = TurnBoundaryRollover(db)
    assert rollover.mark_pending("old", threshold_tokens=10)

    child = rollover.adopt_at_turn_boundary("old", active_work=False)
    assert child
    assert db.get_session("old")["end_reason"] == END_REASON
    handoff = json.loads(db.get_session(child)["model_config"])["turn_boundary_handoff"]
    assert "recovery_key" not in handoff


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
    assert allows_new_delegation(agent) is False


def test_checkpoint_captures_repo_and_result_evidence_without_non_repo_git(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    non_repo = tmp_path / "not-a-repo"
    non_repo.mkdir()
    db.create_session("parent", source="cli", cwd=str(non_repo))
    agent = type("Agent", (), {"_session_db": db, "session_id": "parent", "cwd": str(non_repo)})()
    rollover = TurnBoundaryRollover(db)

    assert rollover.mark_pending(
        "parent", threshold_tokens=10,
        lifecycle={"state": "draining", "checkpoint": {"verification_evidence": ["tests/x"]}},
    )
    payload = json.loads(db.get_session("parent")["model_config"])["turn_boundary_lifecycle"]["checkpoint"]
    assert payload["worktree"] == str(non_repo)
    assert payload["branch"] is None
    assert payload["head"] is None
    assert payload["verification_evidence"] == ["tests/x"]


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
    handoff = json.loads(new["model_config"])["turn_boundary_handoff"]
    assert handoff["previous_session_id"] == "old"
    assert handoff["recovery"] == "Use session_search to recover earlier details if needed."
    assert handoff["idempotency_key"].startswith("turn-boundary:")
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


def test_adopted_child_starts_admissible_without_inheriting_parent_drain(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(
        "old",
        source="cli",
        model_config={
            "provider": "openrouter",
            "turn_boundary_lifecycle": {"state": "draining"},
        },
    )
    rollover = TurnBoundaryRollover(db)
    assert rollover.mark_pending("old", threshold_tokens=10)

    child = rollover.adopt_at_turn_boundary("old", active_work=False)

    assert child
    child_config = json.loads(db.get_session(child)["model_config"])
    assert child_config["provider"] == "openrouter"
    assert child_config["turn_boundary_lifecycle"]["state"] == "healthy"
    child_agent = type("Agent", (), {"_session_db": db, "session_id": child})()
    assert allows_new_work(child_agent) is True
    assert allows_new_delegation(child_agent) is True


def test_disabled_policy_does_not_publish_a_draining_lifecycle(monkeypatch, tmp_path: Path) -> None:
    """Opt-in rollover cannot change admission or doctor state while disabled."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("live", source="cli")

    class Agent:
        session_id = "live"
        _session_db = db
        max_iterations = 1
        context_compressor = type("Compressor", (), {
            "context_length": 272_000,
            "threshold_tokens": 250_000,
            "last_prompt_tokens": 2_000,
        })()

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"session_rollover": {"enabled": False}})
    assert mark_completed_turn(Agent(), {"completed": True, "api_calls": 1}) is False

    config = json.loads(db.get_session("live")["model_config"] or "{}")
    assert "_turn_boundary_rollover_pending" not in config
    assert "turn_boundary_lifecycle" not in config


def test_disabling_after_an_arm_cancels_it_before_adoption_and_reopens_admission(
    monkeypatch, tmp_path: Path,
) -> None:
    """Regression for the independent review's exact disabled-after-arm trace."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("live", source="cli")
    rollover = TurnBoundaryRollover(db)
    assert rollover.mark_pending(
        "live", threshold_tokens=10, lifecycle={"state": "draining"},
    )
    agent = type("Agent", (), {
        "session_id": "live", "_session_db": db,
        "context_compressor": type("Compressor", (), {
            "context_length": 1_000, "threshold_tokens": 900, "last_prompt_tokens": 800,
        })(),
    })()

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"session_rollover": {"enabled": False}},
    )
    assert mark_completed_turn(agent, {"completed": True, "api_calls": 1}) is False
    config = json.loads(db.get_session("live")["model_config"] or "{}")
    assert "_turn_boundary_rollover_pending" not in config
    assert "turn_boundary_lifecycle" not in config
    assert rollover.adopt_at_turn_boundary("live", active_work=False) is None
    assert db.get_session("live")["end_reason"] is None

    db.create_session(
        "stale", source="cli",
        model_config={"turn_boundary_lifecycle": {"state": "draining"}},
    )
    stale = type("Agent", (), {"session_id": "stale", "_session_db": db})()
    assert allows_new_work(stale) is True
    assert allows_new_delegation(stale) is True
    assert "turn_boundary_lifecycle" not in json.loads(
        db.get_session("stale")["model_config"] or "{}"
    )


def test_disabled_cleanup_preserves_concurrent_unrelated_model_config(tmp_path: Path) -> None:
    """Cancellation is one merge transaction and cannot clobber runtime settings."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(
        "live", source="cli", model_config={
            "provider": "openrouter",
            "_turn_boundary_rollover_pending": {"threshold_tokens": 10},
            "turn_boundary_lifecycle": {"state": "draining"},
        },
    )
    start = threading.Barrier(2)

    def mutate_runtime_config() -> None:
        start.wait()
        db.patch_session_model_config("live", {"reasoning_effort": "xhigh", "yolo": True})

    writer = threading.Thread(target=mutate_runtime_config)
    writer.start()
    start.wait()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"session_rollover": {"enabled": False}},
    )
    try:
        assert TurnBoundaryRollover(db).adopt_at_turn_boundary("live", active_work=False) is None
    finally:
        monkeypatch.undo()
    writer.join()

    config = json.loads(db.get_session("live")["model_config"] or "{}")
    assert config["provider"] == "openrouter"
    assert config["reasoning_effort"] == "xhigh"
    assert config["yolo"] is True
    assert "_turn_boundary_rollover_pending" not in config
    assert "turn_boundary_lifecycle" not in config


def test_enabled_policy_marker_fences_without_reloading_config(monkeypatch, tmp_path: Path) -> None:
    """Tool admission uses the persisted decision, not config I/O on every call."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(
        "draining", source="cli", model_config={
            "turn_boundary_lifecycle": {"state": "draining"},
            "_turn_boundary_rollover_policy": {"enabled": True},
        },
    )
    agent = type("Agent", (), {"session_id": "draining", "_session_db": db})()
    calls = 0

    def unexpected_config_read():
        nonlocal calls
        calls += 1
        raise AssertionError("admission reloaded config")

    monkeypatch.setattr("hermes_cli.config.load_config", unexpected_config_read)
    assert allows_new_work(agent) is False
    assert allows_new_delegation(agent) is False
    assert calls == 0


def test_disabled_policy_refuses_and_cleans_new_rollover_arms(monkeypatch, tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(
        "live", source="cli", model_config={
            "_turn_boundary_rollover_pending": {"threshold_tokens": 10},
            "turn_boundary_lifecycle": {"state": "draining"},
        },
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"session_rollover": {"enabled": False}},
    )

    rollover = TurnBoundaryRollover(db)
    assert rollover.mark_pending("live", threshold_tokens=10) is False
    assert rollover.request_recovery("live", idempotency_key="doctor") is None
    config = json.loads(db.get_session("live")["model_config"] or "{}")
    assert "_turn_boundary_rollover_pending" not in config
    assert "turn_boundary_lifecycle" not in config


def test_rollover_reasons_resolve_to_the_child_tip_and_preserve_arm_identity(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("core", source="cli")
    rollover = TurnBoundaryRollover(db)
    assert rollover.mark_pending("core", threshold_tokens=10)

    child = rollover.adopt_at_turn_boundary("core", active_work=False)

    assert child
    parent_config = json.loads(db.get_session("core")["model_config"])
    child_config = json.loads(db.get_session(child)["model_config"])
    assert parent_config["_turn_boundary_rollover_pending"]["idempotency_key"]
    assert child_config["turn_boundary_handoff"]["idempotency_key"] == parent_config["_turn_boundary_rollover_pending"]["idempotency_key"]
    assert db.get_compression_tip("core") == child
    assert db.resolve_resume_session_id("core") == child


def test_core_checkpoint_carries_available_result_evidence(monkeypatch, tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("old", source="cli", cwd=str(tmp_path))

    class Agent:
        session_id = "old"
        _session_db = db
        cwd = str(tmp_path)
        current_goal = "close review blockers"
        context_compressor = type("Compressor", (), {
            "context_length": 1_000,
            "threshold_tokens": 900,
            "last_prompt_tokens": 800,
        })()

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"session_rollover": {"enabled": True, "ratio": 0.75}},
    )
    assert mark_completed_turn(Agent(), {
        "completed": True,
        "api_calls": 1,
        "pending_workers": ["worker-1", "worker-2"],
        "changed_files": ["session_rollover.py"],
        "verification_evidence": ["pytest tests/hermes_state: passed"],
        "result_pointers": ["artifact://review/42"],
        "external_effects": ["local commit created"],
        "external_readback": ["git status clean"],
    })

    config = json.loads(db.get_session("old")["model_config"])
    assert config["turn_boundary_lifecycle"]["in_flight_workers"] == 2
    checkpoint = config["turn_boundary_lifecycle"]["checkpoint"]
    assert checkpoint["goal"] == "close review blockers"
    assert checkpoint["worktree"] == str(tmp_path)
    assert checkpoint["pending_workers"] == ["worker-1", "worker-2"]
    assert checkpoint["changed_files"] == ["session_rollover.py"]
    assert checkpoint["verification_evidence"] == ["pytest tests/hermes_state: passed"]
    assert checkpoint["result_pointers"] == ["artifact://review/42"]
    assert checkpoint["external_effects"] == ["local commit created"]
    assert checkpoint["external_readback"] == ["git status clean"]


def test_completed_turn_uses_turn_iterations_not_lineage_wide_api_calls(monkeypatch, tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("old", source="cli")

    class Agent:
        session_id = "old"
        _session_db = db
        max_iterations = 3
        context_compressor = type("Compressor", (), {
            "context_length": 272_000,
            "threshold_tokens": 250_000,
            "last_prompt_tokens": 2_000,
        })()

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"session_rollover": {
            "enabled": True, "ratio": 0.75,
            "reserved_checkpoint_tokens": 1,
        }},
    )
    assert mark_completed_turn(Agent(), {
        "completed": True,
        "turn_iterations": 0,
        "api_calls": 1_123,
    }) is False

    status = json.loads(db.get_session("old")["model_config"])["turn_boundary_lifecycle"]
    assert status["state"] == "healthy"
    assert status["api_calls"] == 1_123


def test_mark_pending_merges_without_clobbering_concurrent_model_mutation(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("old", source="cli", model_config={"provider": "first"})
    start = threading.Barrier(2)

    def mutate_runtime_config() -> None:
        start.wait()
        db.patch_session_model_config(
            "old", {"reasoning_effort": "xhigh", "yolo": True},
        )

    writer = threading.Thread(target=mutate_runtime_config)
    writer.start()
    start.wait()
    assert TurnBoundaryRollover(db).mark_pending("old", threshold_tokens=10)
    writer.join()
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


def test_disabled_adoption_cleans_an_active_parent_without_ending_it(
    monkeypatch, tmp_path: Path,
) -> None:
    """In-flight work delays adoption, never authoritative disabled cleanup."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(
        "live", source="cli", model_config={
            "provider": "openrouter",
            "unrelated": {"keep": 1},
        },
    )
    rollover = TurnBoundaryRollover(db)
    assert rollover.mark_pending(
        "live", threshold_tokens=10, lifecycle={"state": "draining"},
    )
    agent = type("Agent", (), {"session_id": "live", "_session_db": db})()

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"session_rollover": {"enabled": False}},
    )

    assert rollover.adopt_at_turn_boundary("live", active_work=True) is None

    row = db.get_session("live")
    assert row is not None
    assert row["ended_at"] is None
    assert row["end_reason"] is None
    config = json.loads(row["model_config"] or "{}")
    assert config["provider"] == "openrouter"
    assert config["unrelated"] == {"keep": 1}
    assert config["_turn_boundary_rollover_policy"] == {"enabled": False}
    assert "_turn_boundary_rollover_pending" not in config
    assert "turn_boundary_lifecycle" not in config
    assert allows_new_work(agent) is True
    assert allows_new_delegation(agent) is True


def test_disabled_active_cleanup_preserves_concurrent_config_and_parent_lease(
    monkeypatch, tmp_path: Path,
) -> None:
    """Disabled cleanup cannot clobber a concurrent write or move an active lease."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(
        "live", source="cli", model_config={
            "provider": "openrouter",
            "_turn_boundary_rollover_pending": {"threshold_tokens": 10},
            "turn_boundary_lifecycle": {"state": "draining"},
        },
    )
    holder = "pid=test:turn=active"
    assert db.try_acquire_session_turn_lease("live", holder, ttl_seconds=5)
    start = threading.Barrier(2)

    def mutate_runtime_config() -> None:
        start.wait()
        db.patch_session_model_config("live", {"reasoning_effort": "xhigh", "yolo": True})

    writer = threading.Thread(target=mutate_runtime_config)
    writer.start()
    start.wait()
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"session_rollover": {"enabled": False}},
    )
    assert TurnBoundaryRollover(db).adopt_at_turn_boundary("live", active_work=True) is None
    writer.join()

    row = db.get_session("live")
    assert row is not None
    assert row["ended_at"] is None
    config = json.loads(row["model_config"] or "{}")
    assert config["provider"] == "openrouter"
    assert config["reasoning_effort"] == "xhigh"
    assert config["yolo"] is True
    assert "_turn_boundary_rollover_pending" not in config
    assert "turn_boundary_lifecycle" not in config
    assert db.refresh_session_turn_lease("live", holder, ttl_seconds=5)
    db.release_session_turn_lease("live", holder)
