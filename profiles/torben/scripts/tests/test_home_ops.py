from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from torben_home_ops import build_home_ops_packet, infer_ladder_category
from torben_open_loops import LoopRow, add_loop


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(
    os.environ.get(
        "HERMES_REPO_ROOT",
        "/Users/eric/hermes-agent-torben-work" if Path("/Users/eric/hermes-agent-torben-work").exists() else "/Users/ericfreeman/.hermes/hermes-agent",
    )
)


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


def test_home_loop_becomes_generic_decision_packet() -> None:
    loop = LoopRow(1, "Fix garage keypad", "next-action", "eric", "", "home", "", "2026-07-06", "2026-07-06")

    packet = build_home_ops_packet(loop=loop, proposed_action="schedule locksmith service call", context="Keypad fails intermittently")

    assert packet["schema"] == "torben.home-ops.v1"
    assert packet["status"] == "packet_only"
    assert packet["category"] == "booking"
    assert packet["loop"]["id"] == "1"
    assert packet["approval_gate"]["requires_explicit_approval"] is True
    assert packet["approval_gate"]["allowed_without_approval"] is False
    assert packet["external_actions_taken"] == []
    assert "no external action without approved decision packet" in packet["blocked_actions"]


def test_non_home_admin_loop_is_refused() -> None:
    loop = LoopRow(2, "Draft GTM post", "next-action", "eric", "", "gtm", "", "2026-07-06", "2026-07-06")

    packet = build_home_ops_packet(loop=loop, proposed_action="draft post")

    assert packet["status"] == "refused"
    assert packet["reason"] == "loop_not_home_ops_domain"
    assert packet["external_actions_taken"] == []


def test_no_new_ladder_categories_are_allowed() -> None:
    assert infer_ladder_category("pay invoice") == "payment_adjacent"
    assert infer_ladder_category("submit paperwork") == "form_filing"
    with pytest.raises(ValueError, match="Unsupported ladder category"):
        infer_ladder_category("call plumber", explicit_category="home_ops")


def test_stage_packet_and_gate_blocks_until_approval(tmp_path: Path) -> None:
    loops = tmp_path / "loops.csv"
    ledger = tmp_path / "torben-action-ledger.jsonl"
    add_loop(path=loops, item="Fix garage keypad", domain="home")
    code = r'''
import json
import sys
from pathlib import Path
scripts_dir = Path(sys.argv[1])
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
from torben_decision_packet import resolve_decision_reply
from torben_home_ops import build_home_ops_packet, gate_external_action, load_loop_by_id, stage_home_ops_decision

loops = Path(sys.argv[2])
ledger = Path(sys.argv[3])
packet = build_home_ops_packet(
    loop=load_loop_by_id(loops, 1),
    proposed_action="schedule locksmith service call",
    context="Keypad fails intermittently",
)
stage = stage_home_ops_decision(ledger_path=ledger, packet=packet)
handle = stage["record"]["handle"]
before = gate_external_action(ledger_path=ledger, handle=handle, category=packet["category"])
approved = resolve_decision_reply(ledger_path=ledger, reply_text=f"approve option 1 {handle}")
after = gate_external_action(ledger_path=ledger, handle=handle, category=packet["category"])
print(json.dumps({"stage": stage, "before": before, "approved": approved, "after": after}, sort_keys=True))
'''
    payload = _uv_python(code, str(loops), str(ledger))

    assert payload["stage"]["record"]["status"] == "approval_required"
    assert payload["stage"]["record"]["executor_state"]["category"] == "booking"
    assert payload["before"]["status"] == "blocked"
    assert payload["before"]["allowed"] is False
    assert payload["before"]["reason"] == "explicit_approval_required"
    assert payload["approved"]["status"] == "approved"
    assert payload["after"]["status"] == "approved_to_execute"
    assert payload["after"]["allowed"] is True
