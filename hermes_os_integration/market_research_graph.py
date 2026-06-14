"""Generic market research work graph templates."""

from .work_graph import Dependency, WorkGraph, WorkGraphNode, assign_agents, generate_validation_rules


def market_research_work_graph(project_id: str = "market-research"):
    nodes = [
        WorkGraphNode("market:research", "task", "Market Research", project_id, metadata={"task_type": "market"}),
        WorkGraphNode("market:evidence", "task", "Evidence Collection", project_id, metadata={"task_type": "research"}),
        WorkGraphNode("market:validation", "validation_result", "Evidence Validation", project_id),
        WorkGraphNode("market:experiment", "task", "Experiment", project_id, metadata={"task_type": "experiment"}),
        WorkGraphNode("market:decision", "approval", "Decision Record", project_id),
        WorkGraphNode("market:portfolio", "task", "Portfolio Assignment", project_id, metadata={"task_type": "review"}),
    ]
    dependencies = [
        Dependency("market:research", "market:evidence", "research informs evidence"),
        Dependency("market:evidence", "market:validation", "evidence must validate"),
        Dependency("market:validation", "market:experiment", "validated evidence unlocks experiment"),
        Dependency("market:experiment", "market:decision", "experiment informs decision"),
        Dependency("market:decision", "market:portfolio", "approved decision unlocks portfolio assignment"),
    ]
    return WorkGraph(
        project_id=project_id,
        nodes=nodes,
        dependencies=dependencies,
        assignments=assign_agents(nodes),
        validation_results=generate_validation_rules(nodes),
    )


def evidence_quality_node(project_id: str, source_count: int, confidence: float):
    return WorkGraphNode(
        id="market:evidence-quality",
        type="metric",
        title="Evidence Quality",
        project_id=project_id,
        metadata={"source_count": source_count, "confidence": confidence},
    )


def experiment_node(project_id: str, hypothesis: str, status: str = "planned"):
    return WorkGraphNode(
        id="market:experiment:" + hypothesis.lower().replace(" ", "-"),
        type="task",
        title="Experiment: " + hypothesis,
        project_id=project_id,
        status=status,
        metadata={"task_type": "experiment", "hypothesis": hypothesis},
    )


def market_dashboard_metrics(graph: WorkGraph):
    decision_nodes = [node for node in graph.nodes if node.type == "approval"]
    evidence_nodes = [node for node in graph.nodes if "evidence" in node.id]
    completed = [node for node in graph.nodes if node.status == "completed"]
    return {
        "decision_rate": len(decision_nodes) / max(1, len(graph.nodes)),
        "evidence_quality_nodes": len(evidence_nodes),
        "portfolio_impact_nodes": len([node for node in graph.nodes if "portfolio" in node.id]),
        "completion_rate": len(completed) / max(1, len(graph.nodes)),
    }
