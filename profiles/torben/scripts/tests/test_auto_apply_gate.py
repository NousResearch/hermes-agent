from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("HERMES_REPO_ROOT", "/Users/ericfreeman/.hermes/hermes-agent"))
SCRIPT = SCRIPTS_DIR / "torben_email_hygiene_auto_apply.py"


def _record(handle: str, *, status: str, operation: str, rung: str | None = None) -> dict:
    state = {
        "mutation_type": "gmail_hygiene",
        "hygiene_policy_version": 3,
        "operation": operation,
        "items": [],
    }
    if rung is not None:
        state["automation_rung"] = rung
    return {
        "handle": handle,
        "scope": "ea",
        "summary": f"{operation} fixture",
        "evidence_ids": [],
        "allowed_next_actions": ["approve_hygiene_apply"],
        "status": status,
        "risk_class": "low",
        "outbound_message_id": None,
        "created_at": "2026-07-05T12:00:00Z",
        "expires_at": None,
        "user_visible_summary": f"{operation} fixture",
        "executor_state": state,
        "resolution_history": [],
    }


def _write_home(tmp_path: Path, records: list[dict]) -> Path:
    home = tmp_path / "profile"
    state = home / "state"
    config = home / "config"
    state.mkdir(parents=True)
    config.mkdir(parents=True)
    (state / "torben-action-ledger.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (config / "google_accounts.yaml").write_text("accounts: {}\n", encoding="utf-8")
    (config / "torben-automation-policy.yaml").write_text(
        """
schema_version: 1
ea:
  mutations:
    email_archive_delete_label:
      enabled: true
      max_per_run: 10
      dry_run_required: false
      audit_log_path: state/torben-email-hygiene-audit.jsonl
""".lstrip(),
        encoding="utf-8",
    )
    return home


def _run_auto_apply(home: Path, *, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env["HERMES_REPO_ROOT"] = str(REPO_ROOT)
    env.pop("TORBEN_EMAIL_HYGIENE_AUTO_ALLOW_TRASH", None)
    env.pop("TORBEN_EMAIL_HYGIENE_AUTO_DISABLE_TRASH", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["uv", "run", "python", str(SCRIPT), "--dry-run", "--json"],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_approval_required_record_is_rejected(tmp_path: Path) -> None:
    home = _write_home(
        tmp_path,
        [_record("EA-20260705-001", status="approval_required", operation="archive")],
    )

    result = _run_auto_apply(home)
    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert payload["selected_handles"] == []
    assert payload["skipped_hygiene_handles"][0]["reason"] == "status_approval_required"


def test_unset_allow_trash_rejects_trash(tmp_path: Path) -> None:
    home = _write_home(
        tmp_path,
        [_record("EA-20260705-002", status="approved", operation="trash")],
    )

    result = _run_auto_apply(home)
    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert payload["allow_trash"] is False
    assert payload["selected_handles"] == []
    assert payload["skipped_hygiene_handles"][0]["reason"] == "operation_trash"


def test_approved_archive_is_selected_in_dry_run(tmp_path: Path) -> None:
    home = _write_home(
        tmp_path,
        [_record("EA-20260705-003", status="approved", operation="archive")],
    )

    result = _run_auto_apply(home)
    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert payload["allow_trash"] is False
    assert payload["selected_handles"] == ["EA-20260705-003"]
    assert payload["errors"] == []


def test_approved_record_below_auto_within_caps_is_rejected(tmp_path: Path) -> None:
    home = _write_home(
        tmp_path,
        [_record("EA-20260705-004", status="approved", operation="archive", rung="approve_each")],
    )

    result = _run_auto_apply(home)
    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert payload["selected_handles"] == []
    assert payload["skipped_hygiene_handles"][0]["reason"] == "rung_approve_each"
