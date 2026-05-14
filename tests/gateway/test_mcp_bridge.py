from __future__ import annotations

import json

import pytest

from gateway import mcp_bridge
from gateway import mcp_bridge_tools


def _valid_payload() -> dict:
    return {
        "title": "Add focused local validation tests",
        "project": "hermes-agent",
        "mode": "local",
        "worktree_scope": {
            "path": "/tmp/hermes-worktree",
            "expected_branch": "task/example",
        },
        "task_contract": {
            "objective": "Implement focused validation around the local bridge without executing tasks.",
            "acceptance_criteria": ["records accepted task", "returns stored status"],
        },
        "allowed_actions": ["read repo files", "edit scoped bridge files", "run targeted tests"],
        "forbidden_actions": ["read_secret", "run_shell", "git_push", "docker_run"],
        "return_format": {"sections": ["summary", "tests"]},
    }


def test_submit_task_records_accepted_without_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    response = mcp_bridge.submit_task(_valid_payload())

    assert response["ok"] is True
    assert response["status"] == "accepted"
    status = mcp_bridge.get_task_status(response["task_id"])
    assert status["status"] == "accepted"
    assert status["execution"]["state"] == "not_executed"

    result = mcp_bridge.get_task_result(response["task_id"])
    assert result["result"] is None
    assert result["record"]["payload"]["title"] == "Add focused local validation tests"

    stored = list((tmp_path / "hermes" / "mcp_bridge_tasks").glob("mcp_*.json"))
    assert len(stored) == 1
    assert json.loads(stored[0].read_text())["status"] == "accepted"


def test_submit_task_refuses_missing_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload()
    payload.pop("worktree_scope")

    response = mcp_bridge.submit_task(payload)

    assert response["ok"] is False
    assert response["status"] == "refused"
    assert "repo_scope or worktree_scope" in response["refusal_reason"]


def test_submit_task_refuses_missing_forbidden_actions_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload()
    payload.pop("forbidden_actions")

    response = mcp_bridge.submit_task(payload)

    assert response["ok"] is False
    assert response["status"] == "refused"
    assert "missing required field(s): forbidden_actions" in response["refusal_reason"]
    status = mcp_bridge.get_task_status(response["task_id"])
    assert status["execution"]["state"] == "not_executed"


def test_submit_task_accepts_empty_forbidden_actions_list(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload()
    payload["forbidden_actions"] = []

    response = mcp_bridge.submit_task(payload)

    assert response["ok"] is True
    assert response["status"] == "accepted"
    status = mcp_bridge.get_task_status(response["task_id"])
    assert status["execution"]["state"] == "not_executed"


def test_submit_task_refuses_unsafe_allowed_actions(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload()
    payload["allowed_actions"] = ["run_shell", "read repo files"]

    response = mcp_bridge.submit_task(payload)

    assert response["ok"] is False
    assert response["status"] == "refused"
    assert "run_shell" in response["refusal_reason"]


def test_submit_task_refuses_complete_unsafe_contract_with_empty_forbidden_actions(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload()
    payload["forbidden_actions"] = []
    payload["task_contract"] = {
        "objective": "Use run_shell to mutate local files without invoking bridge dispatch.",
        "acceptance_criteria": ["unsafe capability is refused before execution"],
    }

    response = mcp_bridge.submit_task(payload)

    assert response["ok"] is False
    assert response["status"] == "refused"
    assert "run_shell" in response["refusal_reason"]
    assert "missing required field" not in response["refusal_reason"]
    status = mcp_bridge.get_task_status(response["task_id"])
    assert status["execution"]["state"] == "not_executed"


def test_submit_task_routes_risky_local_write_for_approval(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload()
    payload["task_contract"] = {
        "objective": "Use a direct write_file operation for a scoped local bridge test fixture.",
        "acceptance_criteria": ["task is routed for approval and remains unexecuted"],
    }
    payload["allowed_actions"] = ["direct write_file in scoped test fixture"]

    response = mcp_bridge.submit_task(payload)

    assert response["ok"] is True
    assert response["status"] == "approval_required"
    assert response["refusal_reason"] is None
    assert "local file writes" in response["approval_required_reason"]
    status = mcp_bridge.get_task_status(response["task_id"])
    assert status["execution"]["state"] == "approval_required"


@pytest.mark.parametrize(
    ("allowed_action", "reason_fragment"),
    [
        ("raw shell access", "raw shell"),
        ("read_secret", "secret"),
        ("read .env token files", "secret"),
        ("git_push", "git_push"),
        ("shopify_import", "shopify_import"),
        ("docker_run", "docker_run"),
        ("codex_direct", "Codex"),
    ],
)
def test_submit_task_refuses_required_unsafe_requests(
    tmp_path, monkeypatch, allowed_action, reason_fragment
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload()
    payload["allowed_actions"] = [allowed_action]

    response = mcp_bridge.submit_task(payload)

    assert response["ok"] is False
    assert response["status"] == "refused"
    assert reason_fragment in response["refusal_reason"]


@pytest.mark.parametrize(
    ("allowed_action", "reason_fragment"),
    [
        ("direct arbitrary write_file", "local file writes"),
        ("git reset --hard", "destructive git"),
        ("git push to task branch", "destructive git"),
        ("Shopify write products", "Shopify writes"),
        ("PROD deploy", "PROD actions"),
        ("openai_call", "OpenAI API"),
        ("network access", "network access"),
    ],
)
def test_submit_task_routes_required_risky_requests_for_approval(
    tmp_path, monkeypatch, allowed_action, reason_fragment
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload()
    payload["allowed_actions"] = [allowed_action]

    response = mcp_bridge.submit_task(payload)

    assert response["ok"] is True
    assert response["status"] == "approval_required"
    assert response["refusal_reason"] is None
    assert reason_fragment in response["approval_required_reason"]
    status = mcp_bridge.get_task_status(response["task_id"])
    assert status["execution"]["state"] == "approval_required"


def test_submit_task_does_not_refuse_forbidden_unsafe_actions(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload()
    payload["forbidden_actions"] = [
        "run_shell",
        "direct arbitrary write_file",
        "read_secret",
        "git reset",
        "docker_run",
        "openai_call",
    ]

    response = mcp_bridge.submit_task(payload)

    assert response["ok"] is True
    assert response["status"] == "accepted"
    status = mcp_bridge.get_task_status(response["task_id"])
    assert status["execution"]["state"] == "not_executed"


def test_submit_task_accepts_classifier_calibration_prohibitions(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload()
    payload["title"] = "Calibrate MCP bridge refusal classifier"
    payload["task_contract"] = {
        "objective": "Make the local classifier field-aware without weakening safety.",
        "acceptance_criteria": ["dangerous terms in prohibition fields do not cause refusal"],
        "safety_statement": "Do not expose direct command execution or raw shell access.",
    }
    payload["allowed_actions"] = ["read scoped bridge files", "edit classifier tests"]
    payload["forbidden_actions"] = [
        "run_shell",
        "raw shell access",
        "read .env token files",
        "git reset --hard",
        "Shopify write products",
        "restart services",
    ]
    payload["stop_conditions"] = [
        "Stop if direct command execution is requested.",
        "Stop before any service restart or submit_task reproduction.",
    ]
    payload["training_notes"] = (
        "Dangerous capabilities are examples of refused behavior only: docker_run, "
        "openai_call, git_push, and read_secret."
    )

    response = mcp_bridge.submit_task(payload)

    assert response["ok"] is True
    assert response["status"] == "accepted"


def test_submit_task_accepts_bounded_read_only_health_diagnostic(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload()
    payload["title"] = "Inspect MCP bridge health diagnostics"
    payload["task_contract"] = {
        "objective": (
            "Diagnose a local/public tunnel health issue for exact endpoint "
            "https://bridge.example.test and local origin http://127.0.0.1:8765."
        ),
        "acceptance_criteria": [
            "check process status only",
            "list MCP tools only",
            "inventory local status endpoints only",
        ],
    }
    payload["allowed_actions"] = [
        "read repo files",
        "check process status for local bridge only",
        "inventory exact endpoint https://bridge.example.test/health",
        "list_tools against exact origin http://127.0.0.1:8765/mcp",
    ]
    payload["forbidden_actions"] = [
        "submit_task",
        "settings changes",
        "tool changes",
        "restart services",
        "reload services",
        "mutations",
    ]
    payload["stop_conditions"] = [
        "Stop before any mutation, settings change, tool change, submit_task, restart, or reload.",
        "Stop if a host outside bridge.example.test or 127.0.0.1 is needed.",
    ]

    response = mcp_bridge.submit_task(payload)

    assert response["ok"] is True
    assert response["status"] == "accepted"


def test_submit_task_accepts_exact_scope_read_only_public_route_inventory_plan(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload()
    payload["title"] = "Inventory MCP public route exposure"
    payload["task_contract"] = {
        "objective": (
            "Build a read-only inventory plan for public MCP route exposure at exact origin "
            "https://bridge.example.test/mcp and local endpoint http://127.0.0.1:8765/mcp."
        ),
        "acceptance_criteria": [
            "list route names only",
            "record no setting changes",
            "do not submit_task against the live bridge",
        ],
    }
    payload["allowed_actions"] = [
        "read repo route definitions",
        "inventory exact endpoint https://bridge.example.test/mcp",
        "list_tools for exact local endpoint http://127.0.0.1:8765/mcp",
    ]
    payload["forbidden_actions"] = ["submit_task", "restart services", "reload services", "mutations"]
    payload["stop_conditions"] = [
        "Stop before any mutation, settings change, tool change, submit_task, restart, or reload."
    ]

    response = mcp_bridge.submit_task(payload)

    assert response["ok"] is True
    assert response["status"] in {"accepted", "approval_required"}
    assert response["refusal_reason"] is None


def test_submit_task_refuses_direct_command_execution_request(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload()
    payload["task_contract"] = {
        "objective": "Expose direct command execution for workers through the bridge.",
        "acceptance_criteria": ["raw shell access works"],
    }
    payload["allowed_actions"] = ["add direct command execution", "raw shell access"]

    response = mcp_bridge.submit_task(payload)

    assert response["ok"] is False
    assert "raw shell" in response["refusal_reason"] or "command execution" in response["refusal_reason"]


def test_submit_task_refuses_direct_public_mcp_tool_exposure(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload()
    payload["task_contract"] = {
        "objective": "Add public MCP tools for raw shell and direct write_file capabilities.",
        "acceptance_criteria": ["new public MCP tools are visible"],
    }
    payload["allowed_actions"] = ["add public MCP tools", "expose raw shell through MCP"]

    response = mcp_bridge.submit_task(payload)

    assert response["ok"] is False
    assert response["status"] == "refused"
    assert "MCP tool" in response["refusal_reason"] or "raw shell" in response["refusal_reason"]


def test_submit_task_refuses_direct_approval_bypass_request(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload()
    payload["task_contract"] = {
        "objective": "Bypass approval and restart the bridge process immediately.",
        "acceptance_criteria": ["approval gate is skipped"],
    }
    payload["allowed_actions"] = ["bypass approval", "restart bridge process"]

    response = mcp_bridge.submit_task(payload)

    assert response["ok"] is False
    assert response["status"] == "refused"
    assert "approval" in response["refusal_reason"]


@pytest.mark.parametrize(
    ("contract_text", "allowed_action", "reason_fragment"),
    [
        ("Read protected private configuration material from user config.", "read .env token files", "secret"),
        ("Add and remove MCP tools without exact approval.", "add MCP tools", "MCP tool"),
        ("Restart services while bypassing approval.", "bypass approval", "approval"),
    ],
)
def test_submit_task_refuses_required_safety_matrix(
    tmp_path, monkeypatch, contract_text, allowed_action, reason_fragment
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload()
    payload["task_contract"] = {
        "objective": contract_text,
        "acceptance_criteria": ["unsafe request is refused"],
    }
    payload["allowed_actions"] = [allowed_action]

    response = mcp_bridge.submit_task(payload)

    assert response["ok"] is False
    assert reason_fragment in response["refusal_reason"]


@pytest.mark.parametrize(
    ("contract_text", "allowed_action", "reason_fragment"),
    [
        ("Call network endpoints as needed to debug the bridge.", "network access", "network access"),
        ("Run destructive git cleanup with approval routing.", "git reset --hard", "destructive git"),
        ("Run destructive git cleanup without exact approval yet.", "git reset --hard", "destructive git"),
        ("Perform Shopify PROD writes with approval routing.", "Shopify PROD write", "PROD actions"),
        ("Perform Shopify PROD writes without exact approval yet.", "Shopify PROD write", "PROD actions"),
        ("Restart and reload services after approval.", "restart services", "restart"),
        ("Refresh tunnel process after approval.", "refresh bridge process", "restart/reload"),
        ("Reproduce submit_task against a staged task store after approval.", "submit_task reproduction", "submit_task"),
    ],
)
def test_submit_task_routes_required_safety_matrix_for_approval(
    tmp_path, monkeypatch, contract_text, allowed_action, reason_fragment
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload()
    payload["task_contract"] = {
        "objective": contract_text,
        "acceptance_criteria": ["unsafe request is routed for approval"],
    }
    payload["allowed_actions"] = [allowed_action]

    response = mcp_bridge.submit_task(payload)

    assert response["ok"] is True
    assert response["status"] == "approval_required"
    assert response["refusal_reason"] is None
    assert reason_fragment in response["approval_required_reason"]


def test_submit_task_refuses_broad_cross_repo_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload()
    payload["worktree_scope"] = "all repos under the user home"

    response = mcp_bridge.submit_task(payload)

    assert response["ok"] is False
    assert "cross-repo" in response["refusal_reason"]


def test_list_recent_tasks_is_local_and_limited(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    first = mcp_bridge.submit_task({**_valid_payload(), "title": "First local bridge task"})
    second = mcp_bridge.submit_task({**_valid_payload(), "title": "Second local bridge task"})

    listed = mcp_bridge.list_recent_tasks(limit=1)

    assert listed["ok"] is True
    assert len(listed["tasks"]) == 1
    assert listed["tasks"][0]["task_id"] in {first["task_id"], second["task_id"]}


def test_tool_facade_returns_errors_for_unknown_task(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    response = mcp_bridge_tools.get_task_status({"task_id": "mcp_missing"})

    assert response["ok"] is False
    assert "not found" in response["error"]


def test_mirror_task_result_updates_only_result_execution_and_updated_at(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    submitted = mcp_bridge.submit_task(_valid_payload())
    before = mcp_bridge.get_task_result(submitted["task_id"])["record"]

    mirrored = mcp_bridge.mirror_task_result(
        submitted["task_id"],
        "Final answer from Discord.",
        platform="discord",
    )

    assert mirrored is True
    after = mcp_bridge.get_task_result(submitted["task_id"])["record"]
    changed_keys = {key for key in after if after.get(key) != before.get(key)}
    assert changed_keys == {"status", "result", "execution", "updated_at"}
    assert after["payload"] == before["payload"]
    assert after["status"] == "completed"
    assert after["result"] == {
        "source": "discord_gateway_final_response",
        "platform": "discord",
        "response": "Final answer from Discord.",
    }
    assert after["execution"]["state"] == "completed"
    assert after["execution"]["result_mirrored_by"] == "Hermes"


@pytest.mark.parametrize(
    "response",
    [
        "HTTP 401 from provider while sending final response.",
        "token_invalidated",
        "token-revoked during delivery",
        "invalidated oauth credentials",
        "authentication token invalidated",
        "provider authentication failed",
        "no credentials stored for Discord provider",
        "refresh_token_reused",
    ],
)
def test_mirror_task_result_rejects_infrastructure_errors_on_fresh_record(
    tmp_path, monkeypatch, response
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    submitted = mcp_bridge.submit_task(_valid_payload())

    mirrored = mcp_bridge.mirror_task_result(submitted["task_id"], response, platform="discord")

    assert mirrored is False
    record = mcp_bridge.get_task_result(submitted["task_id"])["record"]
    assert record["status"] == "accepted"
    assert record["result"] is None
    assert record["execution"]["state"] == "not_executed"


@pytest.mark.parametrize(
    "response",
    [
        "HTTP 401",
        "token_invalidated",
        "provider authentication failed",
        "no credentials stored",
        "refresh_token_reused",
    ],
)
def test_mirror_task_result_infrastructure_errors_do_not_overwrite_completed_result(
    tmp_path, monkeypatch, response
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    submitted = mcp_bridge.submit_task(_valid_payload())
    assert mcp_bridge.mirror_task_result(
        submitted["task_id"],
        "SUCCESS: meaningful completed task report.",
        platform="discord",
    )
    before = mcp_bridge.get_task_result(submitted["task_id"])["record"]

    mirrored = mcp_bridge.mirror_task_result(submitted["task_id"], response, platform="discord")

    assert mirrored is False
    after = mcp_bridge.get_task_result(submitted["task_id"])["record"]
    assert after == before


@pytest.mark.parametrize(
    "response",
    [
        "SUCCESS: implemented source-only guard and targeted tests.",
        "BLOCKED: stopped because the requested file was outside the allowed scope.",
        "FAILED: validation found a real regression in the requested behavior.",
    ],
)
def test_mirror_task_result_preserves_normal_task_reports(tmp_path, monkeypatch, response):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    submitted = mcp_bridge.submit_task(_valid_payload())

    mirrored = mcp_bridge.mirror_task_result(submitted["task_id"], response, platform="discord")

    assert mirrored is True
    record = mcp_bridge.get_task_result(submitted["task_id"])["record"]
    assert record["status"] == "completed"
    assert record["execution"]["state"] == "completed"
    assert record["result"]["response"] == response


def test_resolve_task_id_from_text_matches_exact_mcp_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    submitted = mcp_bridge.submit_task(_valid_payload())

    resolved = mcp_bridge.resolve_task_id_from_text(
        f"please finish {submitted['task_id']} and report back"
    )

    assert resolved == submitted["task_id"]


def test_resolve_task_id_from_text_matches_unique_leading_code(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    submitted = mcp_bridge.submit_task(
        {**_valid_payload(), "title": "002BV Discord mirror implementation"}
    )

    resolved = mcp_bridge.resolve_task_id_from_text("Please work on 002BV.")

    assert resolved == submitted["task_id"]


def test_resolve_task_id_from_text_ambiguous_prefix_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    first = mcp_bridge.submit_task({**_valid_payload(), "title": "002BV first task"})
    second = mcp_bridge.submit_task({**_valid_payload(), "title": "002BV second task"})

    resolved = mcp_bridge.resolve_task_id_from_text("Please work on 002BV.")

    assert resolved is None
    assert mcp_bridge.get_task_result(first["task_id"])["result"] is None
    assert mcp_bridge.get_task_result(second["task_id"])["result"] is None


def test_resolve_task_id_from_text_ignores_refused_duplicate_when_one_active_match(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    title = "002EA define accepted read-only diagnostic task classes"
    refused_payload = {**_valid_payload(), "title": title}
    refused_payload.pop("allowed_actions")
    refused = mcp_bridge.submit_task(refused_payload)
    active_payload = {**_valid_payload(), "title": title}
    active_payload["task_contract"] = {
        "objective": "Use a direct write_file operation for a scoped local bridge test fixture.",
        "acceptance_criteria": ["task is routed for approval and remains unexecuted"],
    }
    active_payload["allowed_actions"] = ["direct write_file in scoped test fixture"]
    active = mcp_bridge.submit_task(active_payload)

    resolved = mcp_bridge.resolve_task_id_from_text(
        "Ю Одобрявам 002EA define accepted read-only diagnostic task classes"
    )

    assert refused["status"] == "refused"
    assert active["status"] == "approval_required"
    assert resolved == active["task_id"]


def test_resolve_task_id_from_text_multiple_non_refused_matches_fail_closed(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    first = mcp_bridge.submit_task({**_valid_payload(), "title": "002EA first active task"})
    second = mcp_bridge.submit_task({**_valid_payload(), "title": "002EA second active task"})
    refused_payload = {**_valid_payload(), "title": "002EA refused duplicate"}
    refused_payload.pop("allowed_actions")
    refused = mcp_bridge.submit_task(refused_payload)

    resolved = mcp_bridge.resolve_task_id_from_text("Please continue 002EA.")

    assert refused["status"] == "refused"
    assert resolved is None
    assert mcp_bridge.get_task_result(first["task_id"])["result"] is None
    assert mcp_bridge.get_task_result(second["task_id"])["result"] is None


def test_resolve_task_id_from_text_without_marker_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    submitted = mcp_bridge.submit_task({**_valid_payload(), "title": "002BV bridge task"})

    resolved = mcp_bridge.resolve_task_id_from_text("normal unrelated Discord chatter")

    assert resolved is None
    assert mcp_bridge.get_task_result(submitted["task_id"])["result"] is None


def test_resolve_task_id_from_text_prefers_exact_mcp_id_over_code(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    code_task = mcp_bridge.submit_task({**_valid_payload(), "title": "002DV-B2 code task"})
    exact_task = mcp_bridge.submit_task({**_valid_payload(), "title": "unrelated exact task"})

    resolved = mcp_bridge.resolve_task_id_from_text(
        f"Use 002DV-B2 but actually mirror {exact_task['task_id']}."
    )

    assert resolved == exact_task["task_id"]
    assert resolved != code_task["task_id"]


def test_resolve_task_id_from_text_matches_full_code_before_broad_ambiguity(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    mcp_bridge.submit_task({**_valid_payload(), "title": "002DV-A lifecycle helpers"})
    mcp_bridge.submit_task({**_valid_payload(), "title": "002DV-B1 private runner plan"})
    b2 = mcp_bridge.submit_task(
        {**_valid_payload(), "title": "002DV-B2 accepted read-only private runner source implementation tests"}
    )

    resolved = mcp_bridge.resolve_task_id_from_text(
        "Одобрявам 002DV-B2 accepted read-only private runner source implementation tests"
    )

    assert resolved == b2["task_id"]


def test_resolve_task_id_from_text_matches_longer_full_code(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    mcp_bridge.submit_task({**_valid_payload(), "title": "002DV-B2 accepted read-only private runner"})
    b2a = mcp_bridge.submit_task(
        {**_valid_payload(), "title": "002DV-B2-A commit accepted read-only private runner source/tests"}
    )

    resolved = mcp_bridge.resolve_task_id_from_text(
        "002DV-B2-A commit accepted read-only private runner source/tests"
    )

    assert resolved == b2a["task_id"]


def test_resolve_task_id_from_text_matches_hermes_bridge_full_code(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    dwa = mcp_bridge.submit_task(
        {
            **_valid_payload(),
            "title": "HERMES-BRIDGE-002DW-A Discord approval continuation task_id preservation",
        }
    )
    mcp_bridge.submit_task(
        {**_valid_payload(), "title": "HERMES-BRIDGE-002DW-B follow-up bridge task"}
    )

    resolved = mcp_bridge.resolve_task_id_from_text(
        "Approved HERMES-BRIDGE-002DW-A full-code resolver tests."
    )

    assert resolved == dwa["task_id"]


def test_resolve_task_id_from_text_keeps_broad_code_ambiguous(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    first = mcp_bridge.submit_task({**_valid_payload(), "title": "002DV-A lifecycle helpers"})
    second = mcp_bridge.submit_task({**_valid_payload(), "title": "002DV-B2 private runner"})

    resolved = mcp_bridge.resolve_task_id_from_text("Please continue 002DV.")

    assert resolved is None
    assert mcp_bridge.get_task_result(first["task_id"])["result"] is None
    assert mcp_bridge.get_task_result(second["task_id"])["result"] is None


def test_get_task_result_returns_mirrored_response(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    submitted = mcp_bridge.submit_task(_valid_payload())

    mcp_bridge.mirror_task_result(submitted["task_id"], "Delivered response.")

    result = mcp_bridge.get_task_result(submitted["task_id"])
    assert result["result"]["response"] == "Delivered response."


def test_mirror_helper_is_not_exposed_as_mcp_tool():
    tool_names = {schema["name"] for schema in mcp_bridge_tools.TOOL_SCHEMAS}
    handler_names = set(mcp_bridge_tools.TOOL_HANDLERS)
    expected_tool_names = {
        "submit_task",
        "get_task_status",
        "get_task_result",
        "list_recent_tasks",
    }

    assert tool_names == expected_tool_names
    assert handler_names == expected_tool_names
    assert "mirror_task_result" not in tool_names
    assert "resolve_task_id_from_text" not in tool_names
    assert "mirror_task_result" not in handler_names
    assert "resolve_task_id_from_text" not in handler_names
