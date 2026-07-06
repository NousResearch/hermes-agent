from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("HERMES_REPO_ROOT", "/Users/ericfreeman/.hermes/hermes-agent"))


def _uv_python(code: str, *args: str) -> dict:
    env = dict(os.environ)
    env["HERMES_REPO_ROOT"] = str(REPO_ROOT)
    result = subprocess.run(
        ["uv", "run", "python", "-c", code, *args],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(result.stdout)


def _write_google_accounts_config(tmp_path: Path) -> Path:
    config = tmp_path / "google_accounts.yaml"
    config.write_text(
        "\n".join(
            [
                "accounts:",
                "  work:",
                "    alias: work",
                "    email: work@example.com",
                "    role: work",
                "    enabled: true",
                f"    token_path: {tmp_path / 'work-token.json'}",
                f"    client_secret_path: {tmp_path / 'work-client.json'}",
                "    scopes:",
                "      - https://www.googleapis.com/auth/calendar.events",
                "  personal:",
                "    alias: personal",
                "    email: personal@example.com",
                "    role: personal",
                "    enabled: true",
                f"    token_path: {tmp_path / 'personal-token.json'}",
                f"    client_secret_path: {tmp_path / 'personal-client.json'}",
                "    scopes:",
                "      - https://www.googleapis.com/auth/calendar.events",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def test_alignment_event_id_and_body_normalize_timezone_offsets() -> None:
    payload = _uv_python(
        r"""
import json
from hermes_cli.signal_coo import calendar_sync
candidate_utc = {
    "source_account": "work",
    "target_accounts": ["personal"],
    "start_at": "2026-07-06T13:00:00Z",
    "end_at": "2026-07-06T14:00:00Z",
    "evidence_ids": ["google-calendar:work:primary:event-1"],
}
candidate_offset = {
    **candidate_utc,
    "start_at": "2026-07-06T09:00:00-04:00",
    "end_at": "2026-07-06T10:00:00-04:00",
}
event_id = calendar_sync._alignment_event_id(candidate_offset, "personal")
body = calendar_sync._event_body(candidate_offset, "personal", event_id)
print(json.dumps({
    "same_id": event_id == calendar_sync._alignment_event_id(candidate_utc, "personal"),
    "start": body["start"],
    "end": body["end"],
}))
"""
    )

    assert payload["same_id"] is True
    assert payload["start"] == {"dateTime": "2026-07-06T13:00:00Z", "timeZone": "UTC"}
    assert payload["end"] == {"dateTime": "2026-07-06T14:00:00Z", "timeZone": "UTC"}


def test_same_event_key_create_delete_breaker_records_mutation(tmp_path: Path) -> None:
    config = _write_google_accounts_config(tmp_path)
    payload = _uv_python(
        r"""
import json
import sys
from hermes_cli.signal_coo import calendar_sync
config = sys.argv[1]
inserted = []
deleted = []
candidate = {
    "source_account": "work",
    "target_accounts": ["personal"],
    "summary": "Investor call",
    "start_at": "2026-07-06T09:00:00-04:00",
    "end_at": "2026-07-06T10:00:00-04:00",
    "evidence_ids": ["google-calendar:work:primary:event-1"],
}
expected_event_id = calendar_sync._alignment_event_id(candidate, "personal")
calendar_sync._read_token = lambda account: "access-token"
calendar_sync._google_insert_event = lambda account, token, event_body: inserted.append(event_body["id"]) or {"htmlLink": "x"}

def fake_list(account, token, *, time_min, time_max):
    if account.alias != "personal":
        return [], 1
    return [
        {
            "id": "torbenlegacyrawtimezone",
            "start": {"dateTime": "2026-07-06T13:00:00Z"},
            "end": {"dateTime": "2026-07-06T14:00:00Z"},
            "extendedProperties": {"private": {"torben_alignment": "true"}},
        }
    ], 1

calendar_sync._google_list_alignment_events = fake_list
calendar_sync._google_delete_event = lambda account, token, event_id: deleted.append(event_id)
sync = calendar_sync.sync_calendar_alignment_blocks(
    config_path=config,
    candidates=[candidate],
    source_events=[
        {
            "account_alias": "work",
            "calendar_summary": "Primary",
            "summary": "Investor call",
            "start_at": "2026-07-06T13:00:00Z",
            "end_at": "2026-07-06T14:00:00Z",
            "evidence_ids": ["google-calendar:work:primary:event-1"],
        }
    ],
    cleanup_stale=True,
    cleanup_window_start="2026-07-06T00:00:00Z",
    cleanup_window_end="2026-07-07T00:00:00Z",
)
print(json.dumps({"sync": sync, "inserted": inserted, "deleted": deleted, "expected_event_id": expected_event_id}))
""",
        str(config),
    )
    sync = payload["sync"]

    assert payload["inserted"] == [payload["expected_event_id"]]
    assert payload["deleted"] == []
    assert sync["external_mutations"] == 1
    assert sync["mutation_cap"] == 20
    assert sync["circuit_breakers"][0]["reason"] == "same_event_key_create_delete"
    assert sync["skipped"][-1]["reason"] == "circuit_breaker_same_event_key_create_delete"
    assert any(record["action"] == "circuit_breaker" for record in sync["mutation_records"])


def test_profile_calendar_mutation_audit_uses_policy_path_and_records_breakers(tmp_path: Path) -> None:
    home = tmp_path / "profile"
    (home / "config").mkdir(parents=True)
    (home / "state").mkdir()
    (home / "config" / "torben-automation-policy.yaml").write_text(
        """
ea:
  mutations:
    calendar_edit:
      audit_log_path: state/torben-calendar-mutation-audit.jsonl
""".lstrip(),
        encoding="utf-8",
    )
    payload = _uv_python(
        r"""
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
script_path = Path(sys.argv[1])
home = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("torben_calendar_alignment_audit_test", script_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
audit_path = module._calendar_mutation_audit_path(home)
sync = {
    "dry_run": False,
    "created": [],
    "deleted": [],
    "circuit_breakers": [{"reason": "same_event_key_create_delete", "event_key": "personal|a|b"}],
    "mutation_records": [{"action": "circuit_breaker", "event_key": "personal|a|b"}],
    "external_mutations": 0,
    "mutation_cap": 20,
}
module._append_mutation_audit(audit_path, sync, datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc))
rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
print(json.dumps({"audit_path": str(audit_path), "row": rows[0]}))
""",
        str(SCRIPTS_DIR / "torben_calendar_alignment_audit.py"),
        str(home),
    )

    assert Path(payload["audit_path"]) == home / "state" / "torben-calendar-mutation-audit.jsonl"
    assert payload["row"]["circuit_breakers"][0]["reason"] == "same_event_key_create_delete"
    assert payload["row"]["mutation_records"][0]["action"] == "circuit_breaker"
    assert payload["row"]["mutation_cap"] == 20
