from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from torben_autonomy_ladder import (  # noqa: E402
    CATEGORIES,
    evaluate_dispatch,
    initial_state,
    load_config,
    record_auto_execution,
    record_clean_execution,
    record_error,
    record_gmail_trash_restore,
    write_json_atomic,
)


NOW = datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc)


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        tmp_path / "config" / "torben-autonomy-ladder.yaml",
        tmp_path / "state" / "torben-autonomy-ladder.json",
        tmp_path / "state" / "torben-autonomy-ladder-events.jsonl",
        tmp_path / "state" / "torben-oauth-scope-inventory.json",
        tmp_path / "state" / "torben-action-ledger.jsonl",
    )


def _write_config(path: Path, *, kill: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
schema: torben.autonomy-ladder-config.v1
autonomy_kill_switch: {str(kill).lower()}
event_log_path: state/torben-autonomy-ladder-events.jsonl
state_path: state/torben-autonomy-ladder.json
categories:
  gmail_archive:
    initial_rung: packet_only
    N_clean_required: 2
    max_per_run: 1
    max_per_day: 3
  gmail_trash:
    initial_rung: packet_only
    N_clean_required: 50
    max_per_run: 5
    max_per_day: 25
    auto_allowlist_classes:
      - expired_mfa_code
      - expired_security_code
  calendar_edit:
    initial_rung: packet_only
    N_clean_required: 2
    max_per_run: 1
    max_per_day: 3
  booking:
    initial_rung: packet_only
    N_clean_required: 2
    max_per_run: 1
    max_per_day: 1
  form_filing:
    initial_rung: packet_only
    N_clean_required: 2
    max_per_run: 1
    max_per_day: 1
  gtm_post:
    initial_rung: packet_only
    N_clean_required: 2
    max_per_run: 1
    max_per_day: 1
  payment_adjacent:
    initial_rung: packet_only
    N_clean_required: 2
    max_per_run: 1
    max_per_day: 1
""".lstrip(),
        encoding="utf-8",
    )


def _write_state_with_rung(config_path: Path, state_path: Path, category: str, rung: str) -> None:
    config = load_config(config_path)
    state = initial_state(config, now=NOW)
    state["categories"][category]["rung"] = rung
    write_json_atomic(state_path, state)


def _write_clean_scope_inventory(scope_path: Path) -> None:
    write_json_atomic(
        scope_path,
        {
            "schema": "torben.oauth-scope-inventory.v1",
            "status": "clean",
            "category_gates": {
                "gmail_archive": {"status": "clear", "reason": "no_work_account_write_scope_found"},
                "gmail_trash": {"status": "clear", "reason": "no_work_account_write_scope_found"},
                "calendar_edit": {"status": "clear", "reason": "no_work_account_write_scope_found"},
            },
        },
    )


def test_fresh_categories_initialize_packet_only(tmp_path: Path) -> None:
    config_path, state_path, _, scope_path, ledger_path = _paths(tmp_path)
    _write_config(config_path)

    for category in CATEGORIES:
        result = evaluate_dispatch(
            category=category,
            item_status="approved",
            config_path=config_path,
            state_path=state_path,
            scope_inventory_path=scope_path,
            ledger_path=ledger_path,
            now=NOW,
        )

        assert result["effective_rung"] == "packet_only"
        assert result["technical_auto_execution"] is False


def test_clean_count_marks_eligible_but_does_not_self_promote(tmp_path: Path) -> None:
    config_path, state_path, event_path, _, _ = _paths(tmp_path)
    _write_config(config_path)

    record_clean_execution(category="gmail_archive", config_path=config_path, state_path=state_path, event_log_path=event_path, now=NOW)
    second = record_clean_execution(category="gmail_archive", config_path=config_path, state_path=state_path, event_log_path=event_path, now=NOW)
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert second["promotion"]["status"] == "eligible_input_needed"
    assert state["categories"]["gmail_archive"]["promotion"]["manual_signal_required"] is True
    assert state["categories"]["gmail_archive"]["rung"] == "packet_only"


def test_error_demotes_one_rung_and_logs_before_next_dispatch(tmp_path: Path) -> None:
    config_path, state_path, event_path, scope_path, ledger_path = _paths(tmp_path)
    _write_config(config_path)
    _write_state_with_rung(config_path, state_path, "booking", "auto_within_caps")

    event = record_error(category="booking", error="provider_409", config_path=config_path, state_path=state_path, event_log_path=event_path, now=NOW)
    result = evaluate_dispatch(
        category="booking",
        item_status="approved",
        config_path=config_path,
        state_path=state_path,
        scope_inventory_path=scope_path,
        ledger_path=ledger_path,
        now=NOW,
    )

    assert event["from_rung"] == "auto_within_caps"
    assert event["to_rung"] == "approve_each"
    assert result["effective_rung"] == "approve_each"
    assert json.loads(event_path.read_text(encoding="utf-8").splitlines()[-1])["event"] == "demotion_on_error"


def test_auto_caps_enforced_and_overflow_requires_approval(tmp_path: Path) -> None:
    config_path, state_path, event_path, scope_path, ledger_path = _paths(tmp_path)
    _write_config(config_path)
    _write_state_with_rung(config_path, state_path, "gmail_trash", "auto_within_caps")
    _write_clean_scope_inventory(scope_path)
    record_auto_execution(category="gmail_trash", count=24, config_path=config_path, state_path=state_path, event_log_path=event_path, now=NOW)

    result = evaluate_dispatch(
        category="gmail_trash",
        item_status="approved",
        operation_class="expired_mfa_code",
        requested_count=3,
        config_path=config_path,
        state_path=state_path,
        scope_inventory_path=scope_path,
        ledger_path=ledger_path,
        now=NOW,
    )

    assert result["technical_auto_execution"] is True
    assert result["allowed_auto_count"] == 1
    assert result["approval_required_count"] == 2
    assert "cap_overflow" in result["reasons"]


def test_non_approved_items_are_refused_even_at_top_rung(tmp_path: Path) -> None:
    config_path, state_path, _, scope_path, ledger_path = _paths(tmp_path)
    _write_config(config_path)
    _write_state_with_rung(config_path, state_path, "gtm_post", "auto_within_caps")

    result = evaluate_dispatch(
        category="gtm_post",
        item_status="approval_required",
        config_path=config_path,
        state_path=state_path,
        scope_inventory_path=scope_path,
        ledger_path=ledger_path,
        now=NOW,
    )

    assert result["technical_auto_execution"] is False
    assert result["decision"] == "approve_each"
    assert result["reasons"] == ["status_approval_required"]


def test_gmail_trash_allowlist_and_restore_demote(tmp_path: Path) -> None:
    config_path, state_path, event_path, scope_path, ledger_path = _paths(tmp_path)
    _write_config(config_path)
    _write_state_with_rung(config_path, state_path, "gmail_trash", "auto_within_caps")
    _write_clean_scope_inventory(scope_path)

    denied = evaluate_dispatch(
        category="gmail_trash",
        item_status="approved",
        operation_class="newsletter",
        config_path=config_path,
        state_path=state_path,
        scope_inventory_path=scope_path,
        ledger_path=ledger_path,
        now=NOW,
    )
    restore = record_gmail_trash_restore(config_path=config_path, state_path=state_path, event_log_path=event_path, now=NOW)
    after = evaluate_dispatch(
        category="gmail_trash",
        item_status="approved",
        operation_class="expired_mfa_code",
        config_path=config_path,
        state_path=state_path,
        scope_inventory_path=scope_path,
        ledger_path=ledger_path,
        now=NOW,
    )

    assert denied["technical_auto_execution"] is False
    assert denied["reasons"] == ["gmail_trash_class_newsletter"]
    assert restore["to_rung"] == "approve_each"
    assert after["effective_rung"] == "approve_each"


def test_kill_switch_forces_packet_only_even_with_auto_state(tmp_path: Path) -> None:
    config_path, state_path, _, scope_path, ledger_path = _paths(tmp_path)
    _write_config(config_path)
    _write_state_with_rung(config_path, state_path, "gtm_post", "auto_within_caps")

    result = evaluate_dispatch(
        category="gtm_post",
        item_status="approved",
        config_path=config_path,
        state_path=state_path,
        scope_inventory_path=scope_path,
        ledger_path=ledger_path,
        env={"TORBEN_AUTONOMY_KILL": "1"},
        now=NOW,
    )

    assert result["effective_rung"] == "packet_only"
    assert result["technical_auto_execution"] is False
    assert result["reasons"] == ["global_kill_switch"]


def test_p0_9_scope_gate_pins_gmail_and_calendar_categories(tmp_path: Path) -> None:
    config_path, state_path, _, scope_path, ledger_path = _paths(tmp_path)
    _write_config(config_path)
    _write_state_with_rung(config_path, state_path, "gmail_archive", "auto_within_caps")
    write_json_atomic(
        scope_path,
        {
            "schema": "torben.oauth-scope-inventory.v1",
            "status": "type_1_findings",
            "category_gates": {
                "gmail_archive": {"status": "blocked_type_1", "floor": "packet_only", "reason": "work_account_google_write_scope"}
            },
        },
    )

    result = evaluate_dispatch(
        category="gmail_archive",
        item_status="approved",
        config_path=config_path,
        state_path=state_path,
        scope_inventory_path=scope_path,
        ledger_path=ledger_path,
        now=NOW,
    )

    assert result["effective_rung"] == "packet_only"
    assert result["reasons"] == ["p0_9_type_1_scope_gate"]


def test_p0_4_migration_gate_blocks_auto_execution(tmp_path: Path) -> None:
    config_path, state_path, _, scope_path, ledger_path = _paths(tmp_path)
    _write_config(config_path)
    _write_state_with_rung(config_path, state_path, "booking", "auto_within_caps")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps({"handle": "EA-20260705-001", "status": "approved", "migrated": True}) + "\n", encoding="utf-8")

    result = evaluate_dispatch(
        category="booking",
        item_status="approved",
        config_path=config_path,
        state_path=state_path,
        scope_inventory_path=scope_path,
        ledger_path=ledger_path,
        now=NOW,
    )

    assert result["effective_rung"] == "packet_only"
    assert result["migration_gate"]["approved_migrated_count"] == 1
