"""Domain-neutral template engine for compiling reusable work graph patterns."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .errors import VALIDATION_ERROR, adapter_error
from .work_graph import Dependency, WorkGraph, WorkGraphNode, assign_agents, generate_validation_rules


@dataclass(frozen=True)
class TemplateDefinition:
    template_id: str
    name: str
    nodes: List[Dict[str, object]]
    dependencies: List[Dict[str, str]] = field(default_factory=list)
    metrics: List[Dict[str, object]] = field(default_factory=list)


class TemplateRegistry:
    def __init__(self):
        self._templates: Dict[str, TemplateDefinition] = {}

    def register(self, template: TemplateDefinition):
        _, error = TemplateValidator().validate(template)
        if error:
            return None, error
        self._templates[template.template_id] = template
        return template, None

    def get(self, template_id: str):
        return self._templates.get(template_id)

    def list(self):
        return sorted(self._templates.values(), key=lambda template: template.template_id)


class TemplateLoader:
    def load_dict(self, payload: Dict[str, object]):
        return TemplateDefinition(
            template_id=str(payload.get("template_id", "")),
            name=str(payload.get("name", "")),
            nodes=list(payload.get("nodes", [])),
            dependencies=list(payload.get("dependencies", [])),
            metrics=list(payload.get("metrics", [])),
        )


class TemplateValidator:
    def validate(self, template: TemplateDefinition):
        missing = []
        if not template.template_id:
            missing.append("template_id")
        if not template.name:
            missing.append("name")
        if not template.nodes:
            missing.append("nodes")
        for index, node in enumerate(template.nodes):
            for key in ["id", "type", "title"]:
                if not node.get(key):
                    missing.append("nodes[%s].%s" % (index, key))
        if missing:
            return None, adapter_error(VALIDATION_ERROR, "Invalid template: " + ", ".join(missing))
        return template, None


class TemplateCompiler:
    def compile(self, template: TemplateDefinition, project_id: str):
        _, error = TemplateValidator().validate(template)
        if error:
            return None, error
        nodes = [
            WorkGraphNode(
                id=str(node["id"]),
                type=str(node["type"]),
                title=str(node["title"]),
                project_id=project_id,
                metadata=dict(node.get("metadata", {})),
            )
            for node in template.nodes
        ]
        dependencies = [
            Dependency(
                source_id=str(dependency["source_id"]),
                target_id=str(dependency["target_id"]),
                reason=str(dependency.get("reason", "template dependency")),
            )
            for dependency in template.dependencies
        ]
        metric_nodes = [
            WorkGraphNode(
                id=str(metric["id"]),
                type="metric",
                title=str(metric["title"]),
                project_id=project_id,
                metadata=dict(metric.get("metadata", {})),
            )
            for metric in template.metrics
        ]
        all_nodes = nodes + metric_nodes
        return WorkGraph(
            project_id=project_id,
            nodes=all_nodes,
            dependencies=dependencies,
            assignments=assign_agents(all_nodes),
            validation_results=generate_validation_rules(all_nodes),
            findings=[],
        ), None


def base_project_template():
    return TemplateDefinition(
        template_id="base-project",
        name="Base Project",
        nodes=[
            {"id": "architecture", "type": "epic", "title": "Architecture"},
            {"id": "workflow-design", "type": "workflow", "title": "Workflow Design"},
            {"id": "dashboard-design", "type": "task", "title": "Dashboard Design"},
            {"id": "approval-model", "type": "approval", "title": "Approval Model"},
        ],
        dependencies=[
            {"source_id": "architecture", "target_id": "workflow-design", "reason": "architecture before workflows"},
            {"source_id": "workflow-design", "target_id": "dashboard-design", "reason": "workflows before dashboards"},
            {"source_id": "dashboard-design", "target_id": "approval-model", "reason": "dashboard before automation"},
        ],
        metrics=[
            {"id": "template-completeness", "title": "Template Completeness"},
        ],
    )
