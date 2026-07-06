from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("HERMES_REPO_ROOT", "/Users/ericfreeman/.hermes/hermes-agent"))
NOW = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)


def _uv_python(code: str, *args: str) -> dict:
    env = dict(os.environ)
    env["HERMES_REPO_ROOT"] = str(REPO_ROOT)
    result = subprocess.run(
        ["uv", "run", "python", "-c", code, str(SCRIPTS_DIR), *args],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(result.stdout)


def _script_eval(tmp_path: Path, records: list[dict], *, now: datetime = NOW, apply: bool = True) -> dict:
    ledger = tmp_path / "torben-action-ledger.jsonl"
    with ledger.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    config = tmp_path / "torben-autonomy-ladder.yaml"
    state = tmp_path / "torben-autonomy-ladder.json"
    scope = tmp_path / "torben-oauth-scope-inventory.json"
    pending = tmp_path / "torben-pending-decisions.json"
    config.write_text(
        """
schema: torben.autonomy-ladder-config.v1
autonomy_kill_switch: false
event_log_path: events.jsonl
state_path: ladder-state.json
categories:
  gtm_post:
    initial_rung: packet_only
    N_clean_required: 10
    max_per_run: 1
    max_per_day: 1
  booking:
    initial_rung: packet_only
    N_clean_required: 10
    max_per_run: 1
    max_per_day: 1
""".lstrip(),
        encoding="utf-8",
    )
    state.write_text(
        json.dumps(
            {
                "schema": "torben.autonomy-ladder.v1",
                "generated_at": "2026-07-06T00:00:00Z",
                "categories": {
                    "gtm_post": {
                        "rung": "auto_within_caps",
                        "clean_approved_executions": 10,
                        "daily_auto_counts": {},
                        "promotion": {"status": "promoted", "manual_signal_required": True, "eligible_input_needed": False},
                        "updated_at": "2026-07-06T00:00:00Z",
                    },
                    "booking": {
                        "rung": "auto_within_caps",
                        "clean_approved_executions": 10,
                        "daily_auto_counts": {},
                        "promotion": {"status": "promoted", "manual_signal_required": True, "eligible_input_needed": False},
                        "updated_at": "2026-07-06T00:00:00Z",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    scope.write_text(json.dumps({"schema": "torben.oauth-scope-inventory.v1", "status": "clean", "category_gates": {}}), encoding="utf-8")
    code = r'''
import json
import sys
from datetime import datetime
from pathlib import Path

scripts_dir = Path(sys.argv[1])
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from torben_go_dark import evaluate_go_dark

payload = evaluate_go_dark(
    ledger_path=Path(sys.argv[2]),
    pending_decisions_path=Path(sys.argv[3]),
    ladder_config_path=Path(sys.argv[4]),
    ladder_state_path=Path(sys.argv[5]),
    scope_inventory_path=Path(sys.argv[6]),
    now=datetime.fromisoformat(sys.argv[7].replace("Z", "+00:00")),
    apply=sys.argv[8] == "1",
)
print(json.dumps(payload, sort_keys=True))
'''
    payload = _uv_python(code, str(ledger), str(pending), str(config), str(state), str(scope), now.isoformat().replace("+00:00", "Z"), "1" if apply else "0")
    payload["ledger_path"] = str(ledger)
    payload["pending_path"] = str(pending)
    return payload


def _record(handle: str, *, created_hours_ago: int, risk_class: str = "high", category: str = "gtm_post", go_dark: dict | None = None, **state) -> dict:
    created_at = (NOW - timedelta(hours=created_hours_ago)).isoformat().replace("+00:00", "Z")
    executor_state = {
        "category": category,
        "go_dark": go_dark or {"reping_after_hours": 24, "act_after_hours": 48, "expires_after_hours": 168},
        **state,
    }
    return {
        "handle": handle,
        "scope": "EA",
        "summary": f"Summary {handle}",
        "evidence_ids": [],
        "allowed_next_actions": ["approve", "discard"],
        "status": "approval_required",
        "risk_class": risk_class,
        "outbound_message_id": None,
        "created_at": created_at,
        "expires_at": None,
        "user_visible_summary": f"Visible {handle}",
        "executor_state": executor_state,
        "resolution_history": [],
    }


def _ledger_records(path: str) -> list[dict]:
    by_handle: dict[str, dict] = {}
    order: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["handle"] not in by_handle:
            order.append(record["handle"])
        by_handle[record["handle"]] = record
    return [by_handle[handle] for handle in order]


def test_go_dark_reping_happens_exactly_once(tmp_path: Path) -> None:
    payload = _script_eval(tmp_path, [_record("EA-20260706-001", created_hours_ago=25)])
    second = _script_eval(tmp_path, _ledger_records(payload["ledger_path"]))

    records = _ledger_records(payload["ledger_path"])
    history = records[0]["resolution_history"]
    assert len(payload["repings"]) == 1
    assert second["repings"] == []
    assert [item["status"] for item in history].count("go_dark_reping_sent") == 1


def test_low_risk_actionable_item_acts_with_undo_pointer_when_ladder_permits(tmp_path: Path) -> None:
    record = _record(
        "EA-20260706-002",
        created_hours_ago=49,
        risk_class="low",
        go_dark={"reping_after_hours": 24, "act_after_hours": 48, "expires_after_hours": 168, "reping_sent_at": "2026-07-05T12:00:00Z"},
        go_dark_actionable=True,
        undo_pointer="gmail:archive:thread-1",
    )
    payload = _script_eval(tmp_path, [record])

    records = _ledger_records(payload["ledger_path"])
    assert payload["actions"] == [{"category": "gtm_post", "handle": "EA-20260706-002", "undo_pointer": "gmail:archive:thread-1"}]
    assert records[0]["status"] == "executed"
    assert records[0]["resolution_history"][-1]["status"] == "go_dark_executed"


def test_high_risk_unanswered_item_rolls_into_pending_decisions(tmp_path: Path) -> None:
    record = _record(
        "EA-20260706-003",
        created_hours_ago=49,
        risk_class="high",
        go_dark={"reping_after_hours": 24, "act_after_hours": 48, "expires_after_hours": 168, "reping_sent_at": "2026-07-05T12:00:00Z"},
    )
    payload = _script_eval(tmp_path, [record])

    pending = json.loads(Path(payload["pending_path"]).read_text(encoding="utf-8"))
    records = _ledger_records(payload["ledger_path"])
    assert payload["actions"] == []
    assert pending[0]["handle"] == "EA-20260706-003"
    assert pending[0]["reason"] == "high_risk_or_not_go_dark_actionable"
    assert records[0]["status"] == "approval_required"


def test_expired_high_risk_item_logs_drop_not_act(tmp_path: Path) -> None:
    payload = _script_eval(tmp_path, [_record("EA-20260706-004", created_hours_ago=200, risk_class="high")])

    records = _ledger_records(payload["ledger_path"])
    assert payload["expirations"] == [{"handle": "EA-20260706-004", "outcome": "drop_not_act"}]
    assert records[0]["status"] == "expired"
    assert records[0]["resolution_history"][-1]["status"] == "go_dark_expired"


def test_protective_action_is_not_blocked_by_pending_input(tmp_path: Path) -> None:
    record = _record(
        "EA-20260706-005",
        created_hours_ago=25,
        risk_class="high",
        protective_action=True,
        operation="rollback",
    )
    payload = _script_eval(tmp_path, [record])

    records = _ledger_records(payload["ledger_path"])
    assert payload["protective_actions"] == [{"category": "gtm_post", "handle": "EA-20260706-005", "status": "executed"}]
    assert records[0]["status"] == "executed"
    assert records[0]["resolution_history"][-1]["status"] == "protective_action_executed"


def test_trash_booking_payment_never_go_dark_auto_act(tmp_path: Path) -> None:
    record = _record(
        "EA-20260706-006",
        created_hours_ago=49,
        risk_class="low",
        category="booking",
        go_dark={"reping_after_hours": 24, "act_after_hours": 48, "expires_after_hours": 168, "reping_sent_at": "2026-07-05T12:00:00Z"},
        go_dark_actionable=True,
        undo_pointer="booking:cancel:1",
    )
    payload = _script_eval(tmp_path, [record])

    assert payload["actions"] == []
    assert payload["pending_decisions"][0]["reason"] == "high_risk_or_not_go_dark_actionable"
