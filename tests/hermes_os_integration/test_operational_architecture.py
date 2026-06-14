import json
import os

from hermes_os_integration.architect_cli import BLOCKED, WARNING, main as architect_main
from hermes_os_integration.architecture_first import ArchitectureReviewRequest, create_grill_me_session, review_architecture
from hermes_os_integration.contracts import AgentRequest, RuntimeStatus
from hermes_os_integration.dashboard import (
    approvals_panel,
    architecture_score_panel,
    gap_panel,
    runtime_delegation_panel,
)
from hermes_os_integration.doc_generation import generate_missing_docs, roadmap_from_review, write_review_artifact
from hermes_os_integration.gates import (
    enforce_task_traceability,
    evaluate_task_execution,
    generate_traceable_tasks,
)
from hermes_os_integration.persistence import (
    LocalRepository,
    persist_agent_artifact,
    persist_approval,
    persist_decision,
    persist_grill_me_session,
    persist_review_report,
)
from hermes_os_integration.scanners import discover_projects, scan_project
from hermes_os_integration.wrapper import build_oneshot_command, build_runtime_prompt


def test_project_scanner_and_cli_json_output(tmp_path, capsys):
    projects = tmp_path / "projects"
    project = projects / "investing-system"
    docs = project / "docs"
    docs.mkdir(parents=True)
    (docs / "PROJECT.md").write_text("# Project\n", encoding="utf-8")
    (docs / "DOMAIN.md").write_text("# Domain\n", encoding="utf-8")

    discovered = discover_projects(str(projects))
    assert discovered[0]["project_id"] == "investing-system"

    scan = scan_project("investment-system", str(projects))
    assert scan.project_id == "investing-system"
    assert "PROJECT.md" in scan.present_documents
    assert scan.profile.canonical_name == "Investment System"

    exit_code = architect_main(["review", "investment-system", "--projects-root", str(projects), "--json"])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == WARNING
    assert output["project_id"] == "investing-system"
    assert "missing_documents" in output


def test_cli_block_on_critical(tmp_path, capsys):
    projects = tmp_path / "projects"
    (projects / "sample-project").mkdir(parents=True)

    exit_code = architect_main(["review", "sample-project", "--projects-root", str(projects), "--block-on-critical"])
    assert exit_code == BLOCKED
    assert "Architecture Review" in capsys.readouterr().out


def test_document_generation_and_review_artifact(tmp_path):
    request = ArchitectureReviewRequest(
        project_id="project-1",
        project_path=str(tmp_path),
        present_documents=[],
        completed_stages=[],
    )
    report = review_architecture(request)

    generated = generate_missing_docs(str(tmp_path), report)
    assert any(write.path.endswith("PROJECT.md") and write.status == "written" for write in generated.writes)
    assert (tmp_path / "docs" / "PROJECT.md").exists()

    skipped = generate_missing_docs(str(tmp_path), report)
    assert any(write.status == "skipped" for write in skipped.writes)

    artifact = write_review_artifact(str(tmp_path), report)
    assert artifact.status == "written"
    assert ".hermes" in artifact.path

    roadmap = roadmap_from_review(report)
    assert roadmap[0]["source"] == "architecture-review:project-1"


def test_dashboard_panels_and_execution_gates():
    report = review_architecture(ArchitectureReviewRequest(
        project_id="project-1",
        project_path="/tmp/project-1",
        present_documents=["PROJECT.md"],
        completed_stages=["business_system"],
    ))

    assert architecture_score_panel(report).data["score"] == report.architecture_score
    assert "missing_documents" in gap_panel(report).data
    assert approvals_panel([{"status": "pending"}], [{"task_id": "t1"}]).data["pending_count"] == 1
    assert runtime_delegation_panel(RuntimeStatus(False, "official-hermes-agent")).data["available"] is False

    blocked = evaluate_task_execution("coding", ["business_system"], dry_run=False)
    assert blocked.allowed is False
    assert "control_plane" in blocked.missing

    override = evaluate_task_execution("coding", ["business_system"], override={"approver": "hq", "reason": "urgent"})
    assert override.allowed is True
    assert override.audit["approver"] == "hq"

    trace = enforce_task_traceability({"title": "Do work"})
    assert trace.allowed is False

    tasks, error = generate_traceable_tasks({
        "spec_id": "spec-1",
        "summary": "review command",
        "acceptance_criteria": ["works"],
    }, True, "review-1")
    assert error is None
    assert tasks[0]["review_ref"] == "review-1"


def test_persistence_models(tmp_path):
    repository = LocalRepository(str(tmp_path))
    report = review_architecture(ArchitectureReviewRequest(
        project_id="project-1",
        project_path=str(tmp_path),
        present_documents=[],
        completed_stages=[],
    ))

    persist_review_report(repository, report)
    persist_grill_me_session(repository, create_grill_me_session("project-1"))
    persist_decision(repository, "decision-1", {"project_id": "project-1", "decision": "go"})
    persist_approval(repository, "approval-1", {"status": "approved"})
    persist_agent_artifact(repository, "artifact-1", {"schema": "ResearchReport"})

    assert repository.get("review-reports", "project-1")["project_id"] == "project-1"
    assert repository.latest("decisions")["decision"] == "go"
    assert len(repository.list("agent-artifacts")) == 1


def test_runtime_oneshot_command_and_prompt():
    request = AgentRequest(
        task_id="task-1",
        project_id="project-1",
        agent_kind="research",
        prompt="Research this.",
        working_directory="/tmp",
        context={"ticker": "AAPL"},
    )

    command = build_oneshot_command("/bin/hermes-agent", request)
    prompt = build_runtime_prompt(request)

    assert command[0] == "/bin/hermes-agent"
    assert command[1] == "--oneshot"
    assert "Research this." in command[2]
    assert "ticker: AAPL" in prompt
    assert "Agents are workers, not owners." in prompt
