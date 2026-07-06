from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("HERMES_REPO_ROOT", "/Users/ericfreeman/.hermes/hermes-agent"))
NOW = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)


OPTIONS = [
    {"label": "Book the 10 AM slot", "upside": "fast", "downside": "early", "cost_time": "30m", "risk": "low"},
    {"label": "Ask for afternoon slots", "upside": "better fit", "downside": "delay", "cost_time": "5m", "risk": "low"},
]


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


def test_decision_packet_contains_required_sections() -> None:
    code = r'''
import json
import sys
from pathlib import Path
scripts_dir = Path(sys.argv[1])
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
from torben_decision_packet import PACKET_SECTIONS, render_decision_packet
options = json.loads(sys.argv[2])
packet = render_decision_packet(
    item="Choose dentist slot",
    context="Dentist emailed two possible windows.",
    options=options,
    recommendation="Book the 10 AM slot.",
    category="booking",
)
print(json.dumps({"packet": packet, "sections": PACKET_SECTIONS}))
'''
    payload = _uv_python(code, json.dumps(OPTIONS))
    packet = payload["packet"]

    for section in payload["sections"]:
        assert section in packet
    assert "approve option N / draft different / defer until [date] / drop" in packet
    assert "Nothing is sent, booked, bought, filed, posted, or otherwise executed" in packet


def test_stage_packet_writes_approval_required_record(tmp_path: Path) -> None:
    ledger = tmp_path / "torben-action-ledger.jsonl"
    code = r'''
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
scripts_dir = Path(sys.argv[1])
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
from torben_decision_packet import stage_decision_packet
payload = stage_decision_packet(
    ledger_path=Path(sys.argv[2]),
    loop_id=7,
    item="Choose dentist slot",
    context="Dentist emailed two possible windows.",
    options=json.loads(sys.argv[3]),
    recommendation="Book the 10 AM slot.",
    category="booking",
    risk_class="medium",
    now=datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc),
)
print(json.dumps(payload, sort_keys=True))
'''
    payload = _uv_python(code, str(ledger), json.dumps(OPTIONS))

    record = payload["record"]
    assert record["status"] == "approval_required"
    assert record["risk_class"] == "medium"
    assert record["executor_state"]["category"] == "booking"
    assert record["executor_state"]["loop_id"] == 7
    assert "blocked actions" in payload["packet"]


def test_approve_option_reply_moves_record_to_approved(tmp_path: Path) -> None:
    ledger = tmp_path / "torben-action-ledger.jsonl"
    code = r'''
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
scripts_dir = Path(sys.argv[1])
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
from torben_decision_packet import resolve_decision_reply, stage_decision_packet
stage = stage_decision_packet(
    ledger_path=Path(sys.argv[2]),
    loop_id=7,
    item="Choose dentist slot",
    context="Dentist emailed two possible windows.",
    options=json.loads(sys.argv[3]),
    recommendation="Book the 10 AM slot.",
    category="booking",
    risk_class="medium",
    now=datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc),
)
handle = stage["record"]["handle"]
resolved = resolve_decision_reply(
    ledger_path=Path(sys.argv[2]),
    reply_text=f"approve option 1 {handle}",
    now=datetime(2026, 7, 6, 12, 5, tzinfo=timezone.utc),
)
print(json.dumps(resolved, sort_keys=True))
'''
    payload = _uv_python(code, str(ledger), json.dumps(OPTIONS))

    assert payload["status"] == "approved"
    assert payload["selected_option"] == 1
    assert payload["record"]["status"] == "approved"
    assert payload["record"]["executor_state"]["selected_option"] == 1
    assert payload["record"]["resolution_history"][-1]["status"] == "approved"
