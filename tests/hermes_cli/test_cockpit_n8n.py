from __future__ import annotations

import json
import socket
from pathlib import Path

from hermes_cli.cockpit_n8n import get_n8n_daily_checks_local_payload, get_n8n_daily_checks_payload


class FakeN8nClient:
    def __init__(self, executions_by_workflow: dict[str, dict]):
        self.executions_by_workflow = executions_by_workflow
        self.workflow_names = []
        self.execution_workflow_ids = []

    def list_workflows(self):
        return [
            {"id": "wf1", "name": "Morning Briefing"},
            {"id": "wf2", "name": "Workflow Health Daily Report"},
            {"id": "wf3", "name": "Auto-Remediation Monitor"},
            {"id": "wf4", "name": "Immich Nightly Sync Monitor"},
            {"id": "wf5", "name": "Obsidian Inbox Processor"},
            {"id": "wf6", "name": "Alexa Bring Sync"},
            {"id": "wf7", "name": "n8n Nightly Workflow Backup"},
        ]

    def latest_execution(self, workflow_id: str):
        self.execution_workflow_ids.append(workflow_id)
        return self.executions_by_workflow.get(workflow_id)


def test_n8n_checks_uses_live_latest_execution_status_and_timestamps():
    client = FakeN8nClient(
        {
            "wf1": {
                "id": "exec-secret-id-is-not-returned",
                "status": "success",
                "startedAt": "2026-05-14T07:00:00.000Z",
                "stoppedAt": "2026-05-14T07:00:12.000Z",
                "finished": True,
                "data": {"resultData": {"runData": {"Build Brief": [{}], "Send Discord": [{}]}}},
            },
        }
    )

    payload = get_n8n_daily_checks_payload(now=123.0, n8n_client=client)

    assert payload["source"] == "n8n_live_latest_execution"
    assert payload["live"] is True
    assert payload["fallback"] is False
    assert payload["generated_at"] == 123.0
    assert client.execution_workflow_ids == ["wf1", "wf2", "wf3", "wf4", "wf5", "wf6", "wf7"]

    check = payload["checks"][0]
    assert check["status"] == "success"
    assert check["last_run"] == "2026-05-14T07:00:00.000Z"
    assert check["last_completed"] == "2026-05-14T07:00:12.000Z"
    assert check["live_source"] == "live"
    assert check["execution_status"] == "success"
    assert "2 node(s) reported execution data" in check["output_summary"]
    assert "exec-secret-id" not in str(check)


def test_n8n_checks_matches_live_workflow_names_with_arrows():
    client = FakeN8nClient(
        {
            "wf6": {
                "status": "success",
                "startedAt": "2026-05-14T16:30:00.060Z",
                "stoppedAt": "2026-05-14T16:30:00.799Z",
                "finished": True,
                "data": {"resultData": {"runData": {"Find New Items": [{}]}}},
            },
        }
    )

    def list_workflows_with_arrow_name():
        return [
            {"id": "wf1", "name": "Morning Briefing"},
            {"id": "wf2", "name": "Workflow Health Daily Report"},
            {"id": "wf3", "name": "Auto-Remediation Monitor"},
            {"id": "wf4", "name": "Immich Nightly Sync Monitor"},
            {"id": "wf5", "name": "Obsidian Inbox Processor"},
            {"id": "wf6", "name": "Alexa → Bring Sync"},
            {"id": "wf7", "name": "n8n Nightly Workflow Backup"},
        ]

    client.list_workflows = list_workflows_with_arrow_name

    payload = get_n8n_daily_checks_payload(now=123.0, n8n_client=client)
    checks = {check["name"]: check for check in payload["checks"]}

    assert checks["Alexa Bring Sync"]["live_source"] == "live"
    assert checks["Alexa Bring Sync"]["execution_status"] == "success"
    assert checks["Alexa Bring Sync"]["last_completed"] == "2026-05-14T16:30:00.799Z"


def test_n8n_checks_live_failure_is_display_safe_and_sanitized():
    client = FakeN8nClient(
        {
            "wf2": {
                "status": "error",
                "startedAt": "2026-05-14T08:00:00Z",
                "stoppedAt": "2026-05-14T08:00:03Z",
                "data": {
                    "resultData": {
                        "lastNodeExecuted": "HTTP Request with token abcdef1234567890",
                        "error": {
                            "message": "POST https://hooks.example.test/webhook/abc failed for channel C1234567890 with Bearer secret-token-value credential My Discord Credential"
                        },
                    }
                },
            },
        }
    )

    payload = get_n8n_daily_checks_payload(now=123.0, n8n_client=client)
    failed = payload["checks"][1]

    assert failed["status"] == "error"
    assert failed["live_source"] == "live"
    assert failed["action_needed"] == "review latest n8n execution status"
    rendered = str(failed).lower()
    assert "https://" not in rendered
    assert "webhook" not in rendered
    assert "c1234567890" not in rendered
    assert "secret-token-value" not in rendered
    assert "credential my discord credential" not in rendered
    assert "[redacted-url]" in failed["output_summary"]
    assert "[redacted-channel]" in failed["output_summary"]


def test_n8n_checks_falls_back_to_fixture_when_live_api_fails():
    class FailingClient:
        def list_workflows(self):
            raise RuntimeError("n8n timeout: api key secret should stay server-side")

    payload = get_n8n_daily_checks_payload(now=123.0, n8n_client=FailingClient())

    assert payload["source"] == "fixture_bif_525_inventory"
    assert payload["live"] is False
    assert payload["fallback"] is True
    assert payload["stale"] is True
    assert payload["live_error"] == "n8n timeout: api key [redacted] should stay server-side"
    assert len(payload["checks"]) == 7
    assert {check["live_source"] for check in payload["checks"]} == {"fixture_fallback"}


def test_n8n_checks_timeout_uses_fixture_fallback():
    class TimeoutClient:
        def list_workflows(self):
            raise socket.timeout("timed out while connecting to token host")

    payload = get_n8n_daily_checks_payload(now=123.0, n8n_client=TimeoutClient())

    assert payload["fallback"] is True
    assert payload["live"] is False
    assert payload["stale"] is True
    assert "[redacted]" in payload["live_error"]


def test_n8n_checks_redacts_local_command_paths_and_traceback_details():
    client = FakeN8nClient(
        {
            "wf5": {
                "status": "error",
                "startedAt": "2026-05-14T09:00:00Z",
                "data": {
                    "resultData": {
                        "lastNodeExecuted": "/Users/marco/.local/bin/obsidian_inbox_sort.py",
                        "error": {
                            "message": "Command failed: /opt/homebrew/bin/python3 /Users/marco/.local/bin/obsidian_inbox_sort.py Traceback File \"/private/tmp/run.py\", line 4, in <module>"
                        },
                    }
                },
            }
        }
    )

    payload = get_n8n_daily_checks_payload(now=123.0, n8n_client=client)
    rendered = str(payload).lower()
    failed = payload["checks"][4]

    assert failed["status"] == "error"
    assert "command failed" not in rendered
    assert "traceback" not in rendered
    assert "/users" not in rendered
    assert "/opt" not in rendered
    assert "/private" not in rendered
    assert "obsidian_inbox_sort" not in rendered
    assert "[redacted-path]" in failed["output_summary"] or "execution error" in failed["output_summary"].lower()


def test_n8n_local_payload_uses_persisted_inventory_without_live_reads(_isolate_hermes_home):
    from hermes_cli.config import get_hermes_home

    state_dir = Path(get_hermes_home()) / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "n8n_workflow_inventory_summary.json").write_text(
        json.dumps(
            [
                {"name": "Morning Briefing", "errors": 0, "updatedAt": "2026-05-14T11:00:00Z", "id": "secret-id"},
                {"name": "Workflow Health Daily Report", "errors": 0, "updatedAt": "2026-05-14T11:05:00Z"},
                {"name": "Auto-Remediation Monitor", "errors": 0, "updatedAt": "2026-05-14T11:10:00Z"},
                {"name": "Immich Nightly Sync Monitor", "errors": 0, "updatedAt": "2026-05-14T11:15:00Z"},
                {"name": "Obsidian Inbox Processor", "errors": 0, "updatedAt": "2026-05-14T11:20:00Z"},
                {"name": "Alexa → Bring Sync", "errors": 2, "latest_error": "token=secret failed", "updatedAt": "2026-05-14T11:25:00Z"},
                {"name": "n8n Nightly Workflow Backup", "errors": 0, "updatedAt": "2026-05-14T11:30:00Z"},
            ]
        ),
        encoding="utf-8",
    )

    payload = get_n8n_daily_checks_local_payload(now=1778758200.0)

    assert payload["source"] == "local_n8n_inventory_summary"
    assert payload["live"] is False
    assert payload["fallback"] is False
    assert payload["stale"] is False
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["Morning Briefing"]["execution_status"] == "success"
    assert checks["Alexa Bring Sync"]["execution_status"] == "error"
    rendered = str(payload).lower()
    assert "secret-id" not in rendered
    assert "token=secret" not in rendered
    assert "[redacted]" in rendered


def test_n8n_checks_never_exposes_credential_or_payload_blobs_from_live_data():
    client = FakeN8nClient(
        {
            "wf5": {
                "status": "success",
                "startedAt": "2026-05-14T09:00:00Z",
                "data": {
                    "resultData": {
                        "runData": {
                            "Secret Node": [
                                {
                                    "data": {
                                        "main": [
                                            [
                                                {
                                                    "json": {
                                                        "token": "super-secret-token",
                                                        "webhookUrl": "https://hooks.example.test/webhook/super-secret",
                                                        "channelId": "C1234567890",
                                                        "credentialName": "Marco Discord Credential",
                                                        "large": "x" * 500,
                                                    }
                                                }
                                            ]
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                },
            }
        }
    )

    payload = get_n8n_daily_checks_payload(now=123.0, n8n_client=client)
    rendered = str(payload).lower()

    assert "super-secret-token" not in rendered
    assert "hooks.example" not in rendered
    assert "c1234567890" not in rendered
    assert "marco discord credential" not in rendered
    assert "xxxxx" not in rendered
    assert payload["checks"][4]["output_summary"] == "1 node(s) reported execution data."
