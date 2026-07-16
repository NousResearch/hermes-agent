"""Phase 8 RED contracts for same-window rehydrate.

These tests intentionally define the wished-for Phase 8 API and behavior before
implementation.  They must use only tmp_path/temp SessionDB state and must not
open or mutate the real user ~/.hermes/state.db.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest


RAW_CLOSED_TASK_NEEDLE = "CLOSED_TASK_A_RAW_TRANSCRIPT_NEEDLE_DO_NOT_REINJECT"
UNRELATED_TASK_NEEDLE = "UNRELATED_TASK_B_NEEDLE_DO_NOT_REINJECT"
SYNTHETIC_SECRET_NEEDLE = "SYNTHETIC_SECRET_TOKEN_PASSWORD_PRIVATE_BODY_NEEDLE"
METADATA_RAW_TRANSCRIPT_NEEDLE = "METADATA_ALLOWED_FIELD_RAW_TRANSCRIPT_NEEDLE_DO_NOT_LEAK"
METADATA_UNRELATED_AB_NEEDLE = "METADATA_ALLOWED_FIELD_UNRELATED_AB_NEEDLE_DO_NOT_LEAK"
METADATA_TOKEN_NEEDLE = "METADATA_ALLOWED_FIELD_TOKEN_NEEDLE_DO_NOT_LEAK"
METADATA_PASSWORD_NEEDLE = "METADATA_ALLOWED_FIELD_PASSWORD_NEEDLE_DO_NOT_LEAK"
METADATA_SECRET_NEEDLE = "METADATA_ALLOWED_FIELD_SECRET_NEEDLE_DO_NOT_LEAK"
METADATA_PRIVATE_BODY_NEEDLE = "METADATA_ALLOWED_FIELD_PRIVATE_BODY_NEEDLE_DO_NOT_LEAK"
SAFE_TASK_ID = "task-phase8-current"
SAFE_SESSION_ID = "phase8-temp-session"


def _isolated_hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    hermes_home = tmp_path / "isolated_hermes_home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _phase7_continuity_packet(**overrides: Any) -> Dict[str, Any]:
    packet: Dict[str, Any] = {
        "type": "context_health_compact.continuity_packet",
        "reason": "context_overflow_cannot_compress_further",
        "session_id": SAFE_SESSION_ID,
        "task_id": SAFE_TASK_ID,
        "message_count": 42,
        "approx_tokens": 199999,
        "rehydrate_status": "phase8_not_executed_requires_user_approval",
        "safety_boundary": (
            "raw transcript excluded; unrelated A/B task material excluded; "
            "secret, token, password, and private body content excluded"
        ),
        "raw_transcript_included": False,
        "unrelated_context_included": False,
        "secret_token_password_private_body_included": False,
    }
    packet.update(overrides)
    return packet


def _load_phase8_module():
    """Load the expected Phase 8 module and fail with the full contract.

    This should fail RED before implementation because production source
    `agent/context_health_rehydrate.py` has not been created yet.  The failure
    message documents the expected API so this is not a bare missing-import
    smoke test.
    """

    try:
        module = importlib.import_module("agent.context_health_rehydrate")
    except ModuleNotFoundError as exc:
        pytest.fail(
            "Phase 8 same-window rehydrate module is missing. Expected "
            "agent.context_health_rehydrate to expose planner/consumer APIs: "
            "build_same_window_rehydrate_plan(...), "
            "validate_continuity_packet(...), "
            "build_reentry_provider_payload(...), and "
            "apply_same_window_rehydrate_plan(...). This is the expected RED "
            "failure before Phase 8 implementation.",
            pytrace=False,
        )
    return module


def _assert_no_raw_needles(value: Any) -> None:
    serialized = json.dumps(value, sort_keys=True, default=str)
    assert RAW_CLOSED_TASK_NEEDLE not in serialized
    assert UNRELATED_TASK_NEEDLE not in serialized
    assert SYNTHETIC_SECRET_NEEDLE not in serialized
    assert "raw exception" not in serialized.lower()


def _unsafe_metadata_value_packet() -> Dict[str, Any]:
    return _phase7_continuity_packet(
        reason=METADATA_RAW_TRANSCRIPT_NEEDLE,
        session_id=METADATA_SECRET_NEEDLE,
        task_id=METADATA_PRIVATE_BODY_NEEDLE,
        rehydrate_status=METADATA_TOKEN_NEEDLE,
        safety_boundary=(
            f"{METADATA_PASSWORD_NEEDLE} {METADATA_UNRELATED_AB_NEEDLE}"
        ),
    )


def _assert_no_metadata_value_needles(value: Any) -> None:
    serialized = json.dumps(value, sort_keys=True, default=str)
    for needle in (
        METADATA_RAW_TRANSCRIPT_NEEDLE,
        METADATA_UNRELATED_AB_NEEDLE,
        METADATA_TOKEN_NEEDLE,
        METADATA_PASSWORD_NEEDLE,
        METADATA_SECRET_NEEDLE,
        METADATA_PRIVATE_BODY_NEEDLE,
    ):
        assert needle not in serialized


def test_phase7_continuity_packet_contract_builds_same_window_rehydrate_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 8 should consume Phase 7 continuity_packet into a dry-run plan."""

    _isolated_hermes_home(tmp_path, monkeypatch)
    module = _load_phase8_module()

    packet = _phase7_continuity_packet()
    plan = module.build_same_window_rehydrate_plan(
        continuity_packet=packet,
        session_id=SAFE_SESSION_ID,
        current_task_id=SAFE_TASK_ID,
        dry_run=True,
    )

    assert plan["status"] == "ready"
    assert plan["phase"] == "phase8_same_window_rehydrate"
    assert plan["continuity_packet_type"] == "context_health_compact.continuity_packet"
    assert plan["dry_run"] is True
    assert plan["mutates_state"] is False
    _assert_no_raw_needles(plan)


def test_same_window_rehydrate_plan_requires_no_new_process_and_no_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same-window rehydrate must not ask 병준 for new window/process or /clear."""

    _isolated_hermes_home(tmp_path, monkeypatch)
    module = _load_phase8_module()

    plan = module.build_same_window_rehydrate_plan(
        continuity_packet=_phase7_continuity_packet(),
        session_id=SAFE_SESSION_ID,
        current_task_id=SAFE_TASK_ID,
        dry_run=True,
    )

    assert plan["same_window"] is True
    assert plan["requires_new_process"] is False
    assert plan["requires_new_window"] is False
    assert plan["requires_new_session"] is False
    assert plan["requires_clear"] is False
    assert "/clear" not in json.dumps(plan, default=str)
    assert "/rehydrate" not in json.dumps(plan, default=str)


def test_continuity_packet_consumer_does_not_reinject_raw_transcript_or_private_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continuity consumer must keep raw transcript/unrelated/private data out."""

    _isolated_hermes_home(tmp_path, monkeypatch)
    module = _load_phase8_module()

    packet = _phase7_continuity_packet(
        unsafe_debug_payload=(
            f"{RAW_CLOSED_TASK_NEEDLE} {UNRELATED_TASK_NEEDLE} "
            f"{SYNTHETIC_SECRET_NEEDLE}"
        )
    )
    decision = module.validate_continuity_packet(packet)

    assert decision["status"] == "hold"
    assert decision["safe_hold"] is True
    assert decision["provider_call_allowed"] is False
    assert decision["raw_transcript_reinjected"] is False
    assert decision["unrelated_context_reinjected"] is False
    _assert_no_raw_needles(decision)


def test_temp_sessiondb_soft_archive_rehydrate_strategy_preserves_no_data_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 8 executor should use temp SessionDB only and preserve old rows."""

    hermes_home = _isolated_hermes_home(tmp_path, monkeypatch)
    temp_db_path = hermes_home / "temp_phase8_state.db"
    assert temp_db_path != Path.home() / ".hermes" / "state.db"

    from hermes_state import SessionDB

    db = SessionDB(db_path=temp_db_path)
    db.create_session(session_id=SAFE_SESSION_ID, source="cli", model="phase8-red")
    db.append_message(SAFE_SESSION_ID, role="user", content=RAW_CLOSED_TASK_NEEDLE)
    db.append_message(SAFE_SESSION_ID, role="assistant", content="old assistant answer")

    module = _load_phase8_module()
    plan = module.build_same_window_rehydrate_plan(
        continuity_packet=_phase7_continuity_packet(),
        session_id=SAFE_SESSION_ID,
        current_task_id=SAFE_TASK_ID,
        dry_run=False,
    )
    result = module.apply_same_window_rehydrate_plan(plan, session_db=db)

    active_messages = db.get_messages(SAFE_SESSION_ID)
    all_messages = db.get_messages(SAFE_SESSION_ID, include_inactive=True)

    assert result["same_session_id"] == SAFE_SESSION_ID
    assert result["deleted_rows"] == 0
    assert len(all_messages) >= 2
    assert any(RAW_CLOSED_TASK_NEEDLE in str(m.get("content")) for m in all_messages)
    assert not any(RAW_CLOSED_TASK_NEEDLE in str(m.get("content")) for m in active_messages)
    assert all("continuity_packet" in str(m.get("content")) for m in active_messages)
    _assert_no_raw_needles(result)


def test_rehydrate_provider_payload_uses_reentry_metadata_not_old_conversation_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In-memory conversation_history must not resurrect old transcript."""

    _isolated_hermes_home(tmp_path, monkeypatch)
    module = _load_phase8_module()

    old_conversation_history = [
        {"role": "user", "content": RAW_CLOSED_TASK_NEEDLE},
        {"role": "assistant", "content": UNRELATED_TASK_NEEDLE},
        {"role": "tool", "content": SYNTHETIC_SECRET_NEEDLE},
    ]
    plan = module.build_same_window_rehydrate_plan(
        continuity_packet=_phase7_continuity_packet(),
        session_id=SAFE_SESSION_ID,
        current_task_id=SAFE_TASK_ID,
        dry_run=True,
    )
    payload = module.build_reentry_provider_payload(
        plan=plan,
        previous_conversation_history=old_conversation_history,
    )

    assert payload["provider_visible"] is True
    assert payload["raw_previous_history_included"] is False
    assert payload["uses_working_context_packet"] is True
    assert payload["uses_reentry_metadata"] is True
    _assert_no_raw_needles(payload)


def test_invalid_or_unsafe_continuity_packet_fails_closed_without_leaking_exception_or_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsafe packet must HOLD before provider call and without raw leakage."""

    _isolated_hermes_home(tmp_path, monkeypatch)
    module = _load_phase8_module()

    unsafe_packet = {
        "type": "context_health_compact.continuity_packet",
        "session_id": SAFE_SESSION_ID,
        "task_id": SAFE_TASK_ID,
        "raw_transcript_included": True,
        "debug_exception": f"raw exception: {SYNTHETIC_SECRET_NEEDLE}",
        "raw_transcript": RAW_CLOSED_TASK_NEEDLE,
    }
    decision = module.validate_continuity_packet(unsafe_packet)

    assert decision["status"] == "hold"
    assert decision["safe_hold"] is True
    assert decision["provider_call_allowed"] is False
    assert decision["state_mutation_allowed"] is False
    assert decision["reason"] in {
        "unsafe_continuity_packet",
        "invalid_continuity_packet",
    }
    _assert_no_raw_needles(decision)


def test_allowed_metadata_field_values_with_raw_secret_private_material_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Allowed metadata keys must still reject unsafe raw/private values."""

    _isolated_hermes_home(tmp_path, monkeypatch)
    module = _load_phase8_module()

    decision = module.validate_continuity_packet(_unsafe_metadata_value_packet())

    assert decision["status"] == "hold"
    assert decision["safe_hold"] is True
    assert decision["provider_call_allowed"] is False
    assert decision["state_mutation_allowed"] is False
    _assert_no_metadata_value_needles(decision)
    _assert_no_metadata_value_needles(repr(decision))


def test_unsafe_metadata_values_do_not_promote_to_same_window_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsafe metadata values must not be promoted into a ready plan."""

    _isolated_hermes_home(tmp_path, monkeypatch)
    module = _load_phase8_module()

    plan = module.build_same_window_rehydrate_plan(
        continuity_packet=_unsafe_metadata_value_packet(),
        session_id=SAFE_SESSION_ID,
        current_task_id=SAFE_TASK_ID,
        dry_run=True,
    )

    assert plan["status"] == "hold"
    assert plan["safe_hold"] is True
    assert plan["provider_call_allowed"] is False
    assert plan["state_mutation_allowed"] is False
    assert plan.get("same_window") is True
    assert plan.get("requires_new_process") is False
    assert plan.get("requires_clear") is False
    _assert_no_metadata_value_needles(plan)
    _assert_no_metadata_value_needles(repr(plan))


def test_reentry_payload_does_not_reinject_unsafe_metadata_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provider payload must not expose unsafe metadata-field values."""

    _isolated_hermes_home(tmp_path, monkeypatch)
    module = _load_phase8_module()

    plan = module.build_same_window_rehydrate_plan(
        continuity_packet=_unsafe_metadata_value_packet(),
        session_id=SAFE_SESSION_ID,
        current_task_id=SAFE_TASK_ID,
        dry_run=True,
    )
    payload = module.build_reentry_provider_payload(
        plan=plan,
        previous_conversation_history=[
            {"role": "user", "content": METADATA_RAW_TRANSCRIPT_NEEDLE},
            {"role": "assistant", "content": METADATA_UNRELATED_AB_NEEDLE},
        ],
    )

    assert payload.get("status") == "hold"
    assert payload.get("safe_hold") is True
    assert payload.get("provider_call_allowed") is False
    _assert_no_metadata_value_needles(payload)
    _assert_no_metadata_value_needles(repr(payload))


def test_phase8_red_contract_does_not_include_gateway_update_survival_or_command_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 8 RED is planner/contract only, not command/gateway/update survival."""

    _isolated_hermes_home(tmp_path, monkeypatch)
    module = _load_phase8_module()

    plan = module.build_same_window_rehydrate_plan(
        continuity_packet=_phase7_continuity_packet(),
        session_id=SAFE_SESSION_ID,
        current_task_id=SAFE_TASK_ID,
        dry_run=True,
    )

    assert plan["scope"] == "planner_contract_only"
    assert plan["adds_cli_command"] is False
    assert plan["touches_gateway_update_survival"] is False
    assert plan["runtime_activation_required"] is False
    assert plan["allowed_files"] == ["tests/agent/test_same_window_rehydrate.py"]
