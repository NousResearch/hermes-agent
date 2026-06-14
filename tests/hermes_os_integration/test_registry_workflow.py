from hermes_os_integration.market_research_architecture import MARKET_RESEARCH_ARCHITECTURE
from hermes_os_integration.registry import get_agent, select_agent_kind
from hermes_os_integration.workflows import CheckpointedWorkflow


def test_registry_contains_core_agents():
    assert get_agent("research").runtime_provider == "official-hermes-agent"
    assert get_agent("testing").allowed_tools
    assert select_agent_kind("docs") == "documentation"


def test_checkpointed_workflow_runs_two_dry_run_steps():
    workflow = CheckpointedWorkflow("workflow-1", ["research", "review"])
    run = workflow.execute({
        "task_id": "task-1",
        "project_id": "project-1",
        "prompt": "Run workflow",
        "working_directory": "/tmp",
        "dry_run": True,
    })

    assert [checkpoint.step for checkpoint in run.checkpoints] == ["research", "review"]
    assert run.latest_step() == "review"


def test_market_research_architecture_keeps_hermes_os_control_plane():
    assert MARKET_RESEARCH_ARCHITECTURE["control_plane"] == "Hermes OS"
    assert "research-agent" in MARKET_RESEARCH_ARCHITECTURE["pipeline"]
