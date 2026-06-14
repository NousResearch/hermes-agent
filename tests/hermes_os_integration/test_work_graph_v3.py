import json

from hermes_os_integration.dashboard import (
    agent_assignment_panel,
    dependency_block_panel,
    execution_validation_panel,
    work_graph_summary_panel,
)
from hermes_os_integration.persistence import (
    LocalRepository,
    persist_runtime_usage,
    persist_score_history,
    persist_work_graph,
)
from hermes_os_integration.plan_cli import main as plan_main
from hermes_os_integration.workspace_control import (
    blocker_approval_summary,
    build_workspace_summary,
    main as workspace_main,
    workspace_dashboard_panel,
)
from hermes_os_integration.review_loops import (
    ScheduledReview,
    autonomous_review_policy,
    run_review_loop,
    score_history_record,
)
from hermes_os_integration.work_graph import (
    Dependency,
    ExecutionResult,
    WorkGraph,
    WorkGraphNode,
    build_execution_queue,
    compile_work_graph,
    deserialize_work_graph,
    ingest_execution_result,
    read_architecture_artifacts,
    resolve_dependencies,
    save_work_graph,
    serialize_work_graph,
)
from hermes_os_integration.templates import (
    TemplateCompiler,
    TemplateLoader,
    TemplateRegistry,
    base_project_template,
)


def _project(tmp_path, name="project-1", docs=None):
    project = tmp_path / "projects" / name
    docs_dir = project / "docs"
    docs_dir.mkdir(parents=True)
    for doc in docs or ["PROJECT.md", "DOMAIN.md", "WORKFLOWS.md", "DASHBOARD.md", "METRICS.md", "APPROVALS.md", "AGENTS.md"]:
        (docs_dir / doc).write_text("# " + doc + "\n", encoding="utf-8")
    return project


def test_compile_work_graph_serializes_and_persists(tmp_path):
    project = _project(tmp_path, docs=["PROJECT.md", "DOMAIN.md", "WORKFLOWS.md"])

    graph = compile_work_graph("project-1", str(tmp_path / "projects"))
    payload = serialize_work_graph(graph)
    restored, error = deserialize_work_graph(payload)
    path = save_work_graph(str(project), graph)

    assert error is None
    assert restored.project_id == "project-1"
    assert "workgraph.json" in path
    assert graph.findings


def test_architecture_reader_reports_missing_docs(tmp_path):
    project = _project(tmp_path, docs=["PROJECT.md"])
    artifacts, missing = read_architecture_artifacts(str(project))

    assert "PROJECT.md" in artifacts
    assert "DOMAIN.md" in missing


def test_dependency_resolver_queue_and_ingestion():
    graph = WorkGraph(
        project_id="p1",
        nodes=[
            WorkGraphNode("a", "task", "A", "p1"),
            WorkGraphNode("b", "task", "B", "p1"),
        ],
        dependencies=[Dependency("a", "b", "a before b")],
    )

    resolution = resolve_dependencies(graph)
    queue = build_execution_queue(graph)
    updated = ingest_execution_result(graph, ExecutionResult("a", "completed"))

    assert resolution["ordered"] == ["a", "b"]
    assert queue[0]["node_id"] == "a"
    assert updated.nodes[0].status == "completed"


def test_cycle_detection_and_dashboard_panels():
    graph = WorkGraph(
        project_id="p1",
        nodes=[
            WorkGraphNode("a", "task", "A", "p1"),
            WorkGraphNode("b", "task", "B", "p1", status="blocked"),
        ],
        dependencies=[Dependency("a", "b"), Dependency("b", "a")],
    )

    resolution = resolve_dependencies(graph)
    assert resolution["cycles"] == ["a", "b"]
    assert work_graph_summary_panel(graph).data["blocked_count"] == 1
    assert dependency_block_panel(resolution).data["cycles"]
    assert execution_validation_panel(graph).data["node_statuses"]["blocked"] == 1
    assert agent_assignment_panel(graph).data["assignments_by_agent"] == {}


def test_plan_cli_json_and_write(tmp_path, capsys):
    project = _project(tmp_path, docs=["PROJECT.md", "DOMAIN.md", "WORKFLOWS.md"])

    exit_code = plan_main(["project-1", "--projects-root", str(tmp_path / "projects"), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert data["project_id"] == "project-1"

    plan_main(["project-1", "--projects-root", str(tmp_path / "projects"), "--write"])
    assert (project / "workgraph.json").exists()


def test_workspace_summary_cli_and_dashboard(tmp_path, capsys):
    _project(tmp_path, name="project-a", docs=["PROJECT.md"])
    _project(tmp_path, name="project-b", docs=["PROJECT.md", "DOMAIN.md"])

    summary = build_workspace_summary(str(tmp_path / "projects"))
    panel = workspace_dashboard_panel(summary)
    grouped = blocker_approval_summary(summary)
    exit_code = workspace_main(["--projects-root", str(tmp_path / "projects"), "--json"])

    assert len(summary.projects) == 2
    assert panel.data["project_count"] == 2
    assert grouped["highest_risk"]
    assert exit_code == 2
    assert "project-a" in capsys.readouterr().out


def test_review_loop_score_history_policy_and_persistence(tmp_path):
    schedule = ScheduledReview(schedule="nightly", project_scope=["project-1"])
    results = run_review_loop([{
        "project_id": "project-1",
        "project_path": str(tmp_path),
        "present_documents": ["PROJECT.md"],
        "completed_stages": ["business_system"],
    }], mode="proposal")
    history = score_history_record("project-1", 75, "review-1")
    repository = LocalRepository(str(tmp_path))

    persist_score_history(repository, "score-1", history)
    persist_runtime_usage(repository, "usage-1", {"agent_kind": "research", "status": "completed"})
    persist_work_graph(repository, "project-1", WorkGraph(project_id="project-1"))

    assert schedule.outputs
    assert results[0]["requires_approval"] is True
    assert autonomous_review_policy("write", high_risk_write=True)["allowed"] is False
    assert repository.latest("score-history")["score"] == 75
    assert repository.get("work-graphs", "project-1")["project_id"] == "project-1"


def test_template_engine_compiles_domain_neutral_graph():
    registry = TemplateRegistry()
    template = base_project_template()
    registered, error = registry.register(template)
    loaded = TemplateLoader().load_dict({
        "template_id": "custom",
        "name": "Custom",
        "nodes": [{"id": "n1", "type": "task", "title": "Do Work"}],
    })
    graph, compile_error = TemplateCompiler().compile(loaded, "project-1")

    assert error is None
    assert registered.template_id == "base-project"
    assert registry.get("base-project") is not None
    assert compile_error is None
    assert graph.nodes[0].id == "n1"
