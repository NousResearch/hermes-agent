from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "schemas" / "repo-policy-v1.schema.json"
EXAMPLES = ROOT / "examples" / "repo-policy"

REQUIRED_TOP_LEVEL = {
    "version",
    "repo",
    "authority",
    "roles",
    "workflow",
    "branches",
    "gates",
    "runtime",
    "closeout",
}
REQUIRED_CLOSEOUT = {
    "Policy check",
    "Green 완료",
    "Yellow 대기",
    "Red 필요",
    "검증",
    "Git 상태",
    "Live 상태",
}
RUNTIME_CLOSEOUT = {
    "Gateway restart 필요",
    "Live runtime 반영됨",
    "대기열 포함됨",
}


def load_policy(name: str) -> dict:
    with (EXAMPLES / name).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def contract_errors(policy: dict) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_TOP_LEVEL - set(policy))
    if missing:
        errors.append(f"missing top-level keys: {missing}")

    if policy.get("version") != 1:
        errors.append("unsupported version")

    repo = policy.get("repo") or {}
    repo_class = repo.get("class")
    if repo_class not in {"product", "runtime_tooling", "other"}:
        errors.append("invalid repo.class")

    authority = policy.get("authority") or {}
    if authority.get("policy_source") != ".hermes/repo-policy.yaml":
        errors.append("authority.policy_source must be .hermes/repo-policy.yaml")

    roles = policy.get("roles") or {}
    workflow = policy.get("workflow") or {}
    branches = policy.get("branches") or {}
    runtime = policy.get("runtime") or {}
    closeout_sections = set((policy.get("closeout") or {}).get("required_sections") or [])

    missing_closeout = sorted(REQUIRED_CLOSEOUT - closeout_sections)
    if missing_closeout:
        errors.append(f"missing closeout sections: {missing_closeout}")

    if repo_class == "product":
        expected_workflow = {
            "canonical_landing": "develop",
            "work_done_means": "pushed_to_develop",
            "release_source": "develop",
            "live_apply": "deploy_gate",
        }
        for key, expected in expected_workflow.items():
            if workflow.get(key) != expected:
                errors.append(f"product workflow.{key} must be {expected}")
        if workflow.get("release_target") not in {"main", "master"}:
            errors.append("product workflow.release_target must be main or master")
        if branches.get("release_base") != workflow.get("release_target"):
            errors.append("product branches.release_base must match workflow.release_target")
        if roles.get("landing") != "develop":
            errors.append("product roles.landing must be develop")
        if branches.get("landing") != "develop":
            errors.append("product branches.landing must be develop")
        if roles.get("release") not in {"release_pr_to_main", "release_gate_to_release_target"}:
            errors.append("product roles.release must be release_pr_to_main or release_gate_to_release_target")
        if roles.get("live_apply") != "deploy_gate":
            errors.append("product roles.live_apply must be deploy_gate")

    if repo_class == "runtime_tooling":
        if workflow.get("canonical_landing") != "verified_landing":
            errors.append("runtime_tooling workflow.canonical_landing must be verified_landing")
        if workflow.get("live_apply") != "gateway_runtime_queue":
            errors.append("runtime_tooling workflow.live_apply must be gateway_runtime_queue")
        if roles.get("live_apply") != "gateway_runtime_queue":
            errors.append("runtime_tooling roles.live_apply must be gateway_runtime_queue")
        if runtime.get("live_apply_queue") != "required":
            errors.append("runtime_tooling runtime.live_apply_queue must be required")
        if runtime.get("restart_policy") not in {"queue_apply_one_restart", "manual_explicit_only"}:
            errors.append("runtime_tooling runtime.restart_policy must be explicit")
        missing_runtime_closeout = sorted(RUNTIME_CLOSEOUT - closeout_sections)
        if missing_runtime_closeout:
            errors.append(f"missing runtime closeout sections: {missing_runtime_closeout}")

    return errors


def test_schema_file_is_valid_json_and_documents_fail_closed_contract() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["version"]["const"] == 1
    assert schema["properties"]["authority"]["properties"]["policy_source"]["const"] == ".hermes/repo-policy.yaml"
    assert "Policy drift detected" in schema["description"]
    assert "workflow" in schema["required"]
    assert "work_done_means" in schema["properties"]["workflow"]["required"]


@pytest.mark.parametrize(
    "example",
    ["product-develop.yaml", "product-develop-master.yaml", "runtime-tooling-hermes-agent.yaml"],
)
def test_valid_examples_satisfy_repo_policy_contract(example: str) -> None:
    policy = load_policy(example)
    assert contract_errors(policy) == []


@pytest.mark.parametrize(
    ("example", "expected_error"),
    [
        ("invalid-product-main-landing.yaml", "product roles.landing must be develop"),
        ("invalid-runtime-missing-queue.yaml", "runtime_tooling roles.live_apply must be gateway_runtime_queue"),
        ("invalid-unsupported-version.yaml", "unsupported version"),
    ],
)
def test_invalid_examples_fail_in_expected_ways(example: str, expected_error: str) -> None:
    policy = load_policy(example)
    assert expected_error in contract_errors(policy)
