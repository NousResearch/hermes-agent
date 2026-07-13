"""Domain-neutral template engine for compiling reusable work graph patterns."""

import json
import os
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
    version: str = "1"
    min_hermes_os_version: str = "1"
    source_path: str = ""


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

    def load_path(self, path: str):
        loaded, errors = TemplateLoader().load_path(path)
        for template in loaded:
            _, error = self.register(template)
            if error:
                errors.append(error.message)
        return loaded, errors


class TemplateLoader:
    def load_dict(self, payload: Dict[str, object]):
        return TemplateDefinition(
            template_id=str(payload.get("template_id", "")),
            name=str(payload.get("name", "")),
            nodes=list(payload.get("nodes", [])),
            dependencies=list(payload.get("dependencies", [])),
            metrics=list(payload.get("metrics", [])),
            version=str(payload.get("version", "1")),
            min_hermes_os_version=str(payload.get("min_hermes_os_version", "1")),
            source_path=str(payload.get("source_path", "")),
        )

    def load_file(self, path: str):
        with open(path, "r", encoding="utf-8") as handle:
            if path.endswith(".json"):
                payload = json.load(handle)
            elif path.endswith((".yaml", ".yml")):
                import yaml

                payload = yaml.safe_load(handle) or {}
            else:
                raise ValueError("Unsupported template file type: " + path)
        if isinstance(payload, dict) and isinstance(payload.get("templates"), list):
            return [self.load_dict({**item, "source_path": path}) for item in payload["templates"]]
        if isinstance(payload, list):
            return [self.load_dict({**item, "source_path": path}) for item in payload]
        if isinstance(payload, dict):
            return [self.load_dict({**payload, "source_path": path})]
        raise ValueError("Template payload must be an object or list: " + path)

    def load_path(self, path: str):
        if os.path.isdir(path):
            templates = []
            errors = []
            for file_name in sorted(os.listdir(path)):
                if not file_name.endswith((".json", ".yaml", ".yml")):
                    continue
                try:
                    templates.extend(self.load_file(os.path.join(path, file_name)))
                except Exception as exc:
                    errors.append(str(exc))
            return templates, errors
        try:
            return self.load_file(path), []
        except Exception as exc:
            return [], [str(exc)]


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
        if not template_compatible(template):
            return None, adapter_error(VALIDATION_ERROR, "Template requires newer Hermes OS: " + template.template_id)
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


def template_registry_paths(project_path: str = "", workspace_path: str = "", user_home: str = ""):
    paths = []
    if project_path:
        paths.append(os.path.join(project_path, ".hermes", "templates"))
    if workspace_path:
        paths.append(os.path.join(workspace_path, ".hermes", "templates"))
    if user_home:
        paths.append(os.path.join(user_home, ".hermes", "templates"))
    return paths


def discover_templates(paths: List[str]):
    loader = TemplateLoader()
    templates = []
    diagnostics = []
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        loaded, errors = loader.load_path(path)
        templates.extend(loaded)
        diagnostics.extend({"path": path, "error": error} for error in errors)
    return templates, diagnostics


def validate_template_file(path: str):
    templates, errors = TemplateLoader().load_path(path)
    diagnostics = [{"path": path, "error": error} for error in errors]
    validator = TemplateValidator()
    for template in templates:
        _, error = validator.validate(template)
        if error:
            diagnostics.append({
                "path": template.source_path or path,
                "template_id": template.template_id,
                "error": error.message,
            })
    return {"valid": not diagnostics, "diagnostics": diagnostics, "templates": templates}


def template_compatible(template: TemplateDefinition, hermes_os_version: str = "1"):
    try:
        return int(str(template.min_hermes_os_version).split(".", 1)[0]) <= int(str(hermes_os_version).split(".", 1)[0])
    except ValueError:
        return False
