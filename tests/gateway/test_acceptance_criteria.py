import json

from gateway.dev_control import clarifications, plan_artifacts
from gateway.dev_control.acceptance_criteria import (
    ALLOWED_VERIFICATION_COMMAND_SHAPES,
    allowlisted_command as shared_allowlisted_command,
    validate_and_downgrade_criteria,
)
from gateway.dev_control.acceptance_verification import allowlisted_command as verifier_allowlisted_command


def _fake_text_response(payload):
    message = type("Message", (), {"content": json.dumps(payload)})()
    choice = type("Choice", (), {"message": message})()
    return type("Response", (), {"choices": [choice]})()


def test_verification_allowlist_is_shared_with_verifier():
    assert verifier_allowlisted_command is shared_allowlisted_command
    assert shared_allowlisted_command("scripts/run_tests.sh tests/gateway/test_acceptance_verification.py")[0]
    assert not shared_allowlisted_command("python scripts/run_dev_signal_digest.py")[0]


def test_validate_and_downgrade_criteria_handles_live_failure_modes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    existing = repo / "tests/gateway/test_acceptance_verification.py"
    existing.parent.mkdir(parents=True)
    existing.write_text("def test_ok():\n    pass\n", encoding="utf-8")

    criteria = [
        {
            "statement": "SQL probe works.",
            "verification_method": "command",
            "verification_detail": "curl http://localhost:5667/v1/sql/query",
            "machine_checkable": True,
        },
        {
            "statement": "Failure loop test exists.",
            "verification_method": "test",
            "verification_detail": "scripts/run_tests.sh tests/observability/test_failure_loop.py",
            "machine_checkable": True,
        },
        {
            "statement": "Digest script can run.",
            "verification_method": "command",
            "verification_detail": "python scripts/run_dev_signal_digest.py",
            "machine_checkable": True,
        },
        {
            "statement": "Verifier tests still run.",
            "verification_method": "test",
            "verification_detail": "scripts/run_tests.sh tests/gateway/test_acceptance_verification.py",
            "machine_checkable": True,
        },
    ]

    validated, warnings = validate_and_downgrade_criteria(criteria, repo_roots=[str(repo)])

    assert [item["machine_checkable"] for item in validated] == [False, False, False, True]
    assert all(item["verification_method"] == "manual" for item in validated[:3])
    assert all(item.get("note") for item in validated[:3])
    assert "Command is not in the v1 verification allowlist" in validated[0]["note"]
    assert "Referenced path does not exist" in validated[1]["note"]
    assert "Command is not in the v1 verification allowlist" in validated[2]["note"]
    assert len(warnings) == 3


def test_clarified_brief_prompt_includes_allowlist_and_existing_path_instruction(monkeypatch):
    captured = {}

    def fake_call_llm(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _fake_text_response({
            "refined_vision": "Make criteria verifiable.",
            "goals": ["Constrain criteria."],
            "non_goals": [],
            "constraints": [],
            "assumptions": [],
            "acceptance_criteria": [
                {
                    "statement": "Criteria are generated.",
                    "verification_method": "manual",
                    "verification_detail": "Review manually.",
                    "machine_checkable": False,
                }
            ],
            "risk_notes": [],
            "open_questions": [],
            "suggested_next_action": "Review.",
        })

    monkeypatch.setattr(clarifications, "call_llm", fake_call_llm)
    clarifications._generate_clarified_brief_with_llm({
        "vision_brief": "Constrain acceptance criteria.",
        "project_context": {},
        "grounding_provenance": ["/repo/tests/gateway/test_acceptance_verification.py"],
    })

    system = captured["messages"][0]["content"]
    user = json.loads(captured["messages"][1]["content"])
    assert "MUST match one of these exact command shapes" in system
    assert "Reference only files/paths present in the provided repository grounding" in system
    for shape in ALLOWED_VERIFICATION_COMMAND_SHAPES:
        assert shape in system
    assert user["repository_grounding_paths"] == ["/repo/tests/gateway/test_acceptance_verification.py"]


def test_plan_artifact_prompt_includes_allowlist_and_existing_path_instruction(monkeypatch):
    captured = {}

    def fake_call_llm(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _fake_text_response(_plan_artifact_payload([{
            "statement": "Artifact is reviewed.",
            "verification_method": "manual",
            "verification_detail": "Review manually.",
            "machine_checkable": False,
        }]))

    monkeypatch.setattr(plan_artifacts, "call_llm", fake_call_llm)
    plan_artifacts._generate_artifact_with_llm(
        {
            "project_id": "OrynWorkspace",
            "vision_brief": "Build a plan.",
            "grounding_provenance": ["/repo/tests/gateway/test_acceptance_verification.py"],
        },
        previous=None,
        feedback=None,
    )

    system = captured["messages"][0]["content"]
    user = json.loads(captured["messages"][1]["content"])
    assert "MUST match one of these exact command shapes" in system
    assert "Reference only files/paths present in the provided repository grounding" in system
    for shape in ALLOWED_VERIFICATION_COMMAND_SHAPES:
        assert shape in system
    assert user["repository_grounding_paths"] == ["/repo/tests/gateway/test_acceptance_verification.py"]


def test_clarified_brief_post_validation_downgrades_bad_machine_criteria(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def fake_call_llm(**kwargs):
        return _fake_text_response({
            "refined_vision": "Build production feedback.",
            "goals": ["Use production signals."],
            "non_goals": [],
            "constraints": [],
            "assumptions": [],
            "acceptance_criteria": [
                {
                    "statement": "Laminar SQL can be queried.",
                    "verification_method": "command",
                    "verification_detail": "curl http://localhost:5667/v1/sql/query",
                    "machine_checkable": True,
                },
                {
                    "statement": "Verifier tests still run.",
                    "verification_method": "test",
                    "verification_detail": "scripts/run_tests.sh tests/observability/test_failure_loop.py",
                    "machine_checkable": True,
                },
            ],
            "risk_notes": [],
            "open_questions": [],
            "suggested_next_action": "Review.",
        })

    monkeypatch.setattr(clarifications, "call_llm", fake_call_llm)
    brief = clarifications._build_clarified_brief({
        "vision_brief": "Build production feedback.",
        "project_context": {"repositories": [{"path": str(repo)}]},
    })

    assert [item["machine_checkable"] for item in brief["acceptance_criteria"]] == [False, False]
    assert "Acceptance criteria downgraded" in brief["warning"]
    assert all(item.get("note") for item in brief["acceptance_criteria"])


def test_plan_artifact_post_validation_downgrades_bad_machine_criteria(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def fake_call_llm(**kwargs):
        return _fake_text_response(_plan_artifact_payload([
            {
                "statement": "Digest script can run.",
                "verification_method": "command",
                "verification_detail": "python scripts/run_dev_signal_digest.py",
                "machine_checkable": True,
            }
        ]))

    monkeypatch.setattr(plan_artifacts, "call_llm", fake_call_llm)
    artifact = plan_artifacts._generate_artifact(
        {
            "project_id": "OrynWorkspace",
            "vision_brief": "Build production feedback.",
            "project_context": {"repositories": [{"path": str(repo)}]},
        }
    )

    assert artifact["source"] == "llm"
    assert artifact["warning"]
    assert artifact["payload"]["warning"] == artifact["warning"]
    criterion = artifact["payload"]["acceptance_criteria"][0]
    assert criterion["machine_checkable"] is False
    assert criterion["verification_method"] == "manual"
    assert "Command is not in the v1 verification allowlist" in criterion["note"]


def _plan_artifact_payload(criteria):
    return {
        "title": "Verifiable Criteria Artifact",
        "overview": "Constrain plan acceptance criteria.",
        "product_intent": "Keep generated criteria compatible with verification.",
        "scope": ["Validate machine-checkable criteria."],
        "non_goals": ["Do not launch workers."],
        "assumptions": ["Verification remains advisory."],
        "user_workflow": ["Review the artifact."],
        "implementation_slices": [{"title": "Criteria validation", "description": "Downgrade invalid criteria."}],
        "validation_slices": [{"title": "Unit tests", "description": "Cover invalid and valid commands."}],
        "acceptance_criteria": criteria,
        "risks": [],
        "open_questions": [],
        "recommended_next_action": "Review.",
    }
