import json
from pathlib import Path

from scripts.evals import team_qualification as runner


def _observation(secret: str) -> dict:
    route = {
        "requested_profile": "builder",
        "requested_provider": "provider-a",
        "requested_model": "model-a",
        "requested_reasoning_effort": "high",
        "resolved_provider": "provider-a",
        "resolved_model": "model-a",
        "resolved_reasoning_effort": "high",
        "transmitted_provider": "provider-a",
        "transmitted_model": "model-a",
        "transmitted_reasoning_effort": "high",
        "provider_reported_model": "model-a",
        "request_evidence_source": "request",
        "response_evidence_source": "response",
    }
    task_actions = [
        "create", "create", "start", "guide", "submit_review", "start",
        "request_changes", "submit_review", "start", "accept", "start",
    ]
    return {
        "selected_interface": "codex",
        "interface_version": "worker-interface-v1",
        "team_contract": "kanban-team-v1",
        "advertised_tools": [
            "worker_capabilities", "team_task", "wait_agent", "inspect_agent", "worker_control",
        ],
        "profiles_discovered": True,
        "public_tools_only": True,
        "team_actions": task_actions,
        "task_count": 2,
        "dependency_linked": True,
        "dependent_started_after_accept": True,
        "running_guidance": True,
        "initial_implementation_succeeded": True,
        "first_review_succeeded": True,
        "first_review_mentions_violet": True,
        "correction_reused_worker": True,
        "correction_linked_previous_run": True,
        "correction_retained_markers": True,
        "review_success_count": 2,
        "implementation_task_status": "done",
        "dependent_worker_succeeded": True,
        "route_agreement": True,
        "worker_tools_empty": True,
        "structured_tool_errors": 0,
        "all_terminal_acked": True,
        "authorization_violations": 0,
        "duplicate_owned_executions": 0,
        "parent_route": {
            "requested_provider": "parent-provider",
            "resolved_provider": "parent-provider",
            "transmitted_provider": "parent-provider",
            "provider_reported_model": "parent-model",
        },
        "worker_routes": [{"worker_ref": "worker:w1", "run_ref": "run:r1", **route}],
        "tool_calls": [{
            "sequence": 1, "advertised_tool": "worker_capabilities", "accepted": True,
        }],
        "task_lineage": [{
            "task_ref": "task:t1", "status": "done", "parent_refs": [],
        }],
        "worker_lineage": [{
            "worker_ref": "worker:w1", "profile": "builder",
            "runs": [{"run_ref": "run:r1", "status": "SUCCEEDED", "completion_ack": True}],
        }],
        "elapsed_seconds": 12.5,
        "token_usage": {"input_tokens": 100, "output_tokens": 20, "provider_body": secret},
        "cost": {"status": "unknown"},
        "error_type": None,
        "provider_body": secret,
    }


def _common(fixture: dict) -> dict:
    policy = fixture["execution_policy"]
    return {
        "candidate_sha": "a" * 40,
        "requested_interface": "codex",
        "limits": {
            "max_parent_iterations": policy["max_parent_iterations"],
            "max_child_iterations": policy["max_child_iterations"],
            "max_children": policy["max_children"],
            "max_seconds": policy["max_seconds"],
            "max_child_seconds": policy["max_child_seconds"],
            "max_parent_tokens": 4096,
        },
    }


def test_synthetic_success_builds_allowlisted_passing_report():
    fixture = runner.load_fixture()
    assert runner.REPO_ROOT == Path(runner.__file__).resolve().parents[2]
    secret = "synthetic-provider-body-must-not-appear"

    report = runner.build_report(fixture, _observation(secret), _common(fixture))

    assert report["scenario_passed"] is True
    assert [item["id"] for item in report["assertions"]] == [f"TQ-{index}" for index in range(1, 9)]
    assert all(item["passed"] for item in report["assertions"])
    assert set(fixture["required_receipt_fields"]).issubset(report)
    assert report["limits"]["max_parent_iterations"] == 32
    assert secret not in json.dumps(report)


def test_failed_semantic_assertion_is_preserved_without_relaxation():
    fixture = runner.load_fixture()
    observation = _observation("hidden")
    observation["first_review_mentions_violet"] = False

    report = runner.build_report(fixture, observation, _common(fixture))

    assert report["scenario_passed"] is False
    assert report["assertions"][3] == {
        "id": "TQ-4",
        "assertion": fixture["scenario"]["assertions"][3],
        "passed": False,
    }
    assert sum(not item["passed"] for item in report["assertions"]) == 1


def test_child_failure_is_sanitized_and_never_passes():
    fixture = runner.load_fixture()
    failed = runner._child_failure({}, "RuntimeError@synthetic:1")

    report = runner.build_report(fixture, failed, _common(fixture))

    assert report["scenario_passed"] is False
    assert report["error_type"] == "RuntimeError@synthetic:1"
    assert all(item["passed"] is False for item in report["assertions"])


def test_parent_uses_team_toolset_and_reports_only_observed_wire_identity():
    packet = {
        "parent_provider": "parent-provider",
        "parent_model": "configured-model",
        "requested_interface": "codex",
        "routes": [{
            "profile": "builder", "provider": "provider-a", "model": "model-a",
            "reasoning_effort": "high",
        }],
        "limits": {"max_child_iterations": 3, "max_child_seconds": 90, "max_children": 2},
    }

    config = runner._build_config(packet)
    assert list(runner.PARENT_TOOLSETS) == ["delegation", "kanban"]
    assert config["toolsets"] == list(runner.PARENT_TOOLSETS)
    assert runner._observed_transport_fields({
        "model": "wire-model",
        "reasoning": {"effort": "wire-effort"},
    }) == {
        "transmitted_model": "wire-model",
        "transmitted_reasoning_effort": "wire-effort",
    }
    assert runner._observed_transport_fields({
        "model": "outer-model",
        "reasoning_effort": "outer-effort",
        "extra_body": {"model": "wire-override", "reasoning": {"effort": "wire-override-effort"}},
    }) == {
        "transmitted_model": "wire-override",
        "transmitted_reasoning_effort": "wire-override-effort",
    }
    assert runner._observed_transport_fields({}) == {
        "transmitted_model": "unknown",
        "transmitted_reasoning_effort": "unknown",
    }


def test_v2_preserves_v1_assertions_and_applies_only_the_bounded_source_correction():
    v1_path = runner.FIXTURE_PATH.with_name("team-qualification-v1.json")
    v1 = runner.load_fixture(v1_path)
    v2 = runner.load_fixture()

    assert v1["execution_policy"]["max_parent_iterations"] == 24
    assert v2["execution_policy"]["max_parent_iterations"] == 32
    assert v2["execution_policy"]["minimum_expected_tool_operations"] == (
        1 + 2 + 4 + 1 + (5 * 3) + 2 + 1 + 1
    )
    assert v2["scenario"]["assertions"] == v1["scenario"]["assertions"]
    assert "targets set to a nonempty array" in v2["scenario"]["prompt_template"]
    assert v2["change_control"]["supersedes_fixture"] == "team-qualification-v1"
