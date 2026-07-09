from pathlib import Path

import yaml

from hermes_cli.workflows_capabilities import require_implemented_primitives
from hermes_cli.workflows_spec import WorkflowSpec, validate_graph

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = [
    ROOT / "examples" / "workflows" / "code-change-review.yaml",
    ROOT / "examples" / "workflows" / "research-triage.yaml",
]
DOCS = ROOT / "website" / "docs" / "user-guide" / "features" / "workflows.md"
ALLOWED_CONTRACT_TYPES = {"string", "number", "boolean", "array", "object"}


def _load_spec(path: Path) -> WorkflowSpec:
    spec = WorkflowSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    validate_graph(spec)
    require_implemented_primitives(spec)
    return spec


def test_workflow_examples_validate_with_implemented_primitives() -> None:
    for path in EXAMPLES:
        _load_spec(path)


def test_workflow_examples_agent_tasks_have_enforced_result_contracts() -> None:
    for path in EXAMPLES:
        spec = _load_spec(path)
        agent_nodes = [node for node in spec.nodes.values() if node.type == "agent_task"]
        assert agent_nodes, f"{path} should contain agent_task examples"
        for node in agent_nodes:
            assert node.result_contract, f"{path}:{node.title or node.type} missing result_contract"
            for expected in node.result_contract.values():
                assert isinstance(expected, str)
                assert expected in ALLOWED_CONTRACT_TYPES or "|" in expected


def test_workflow_docs_cover_runtime_contract_privacy_and_limits() -> None:
    text = DOCS.read_text(encoding="utf-8")
    required = [
        "result_contract",
        "array",
        "object",
        "secrets",
        "redact",
        "Validation and deploy reject primitives",
        "send_message",
        "subworkflow",
        "webhook",
        "kanban_event",
    ]
    for marker in required:
        assert marker in text
