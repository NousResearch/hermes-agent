"""Workspace project scanners for architecture-first reviews."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .architecture_first import REQUIRED_PROJECT_DOCS, existing_project_review_targets


PROJECT_ALIASES = {}


@dataclass(frozen=True)
class ProjectProfile:
    project_id: str
    canonical_name: str
    aliases: List[str]
    expected_metrics: List[str] = field(default_factory=list)
    review_hints: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProjectScan:
    project_id: str
    project_path: str
    present_documents: List[str]
    missing_documents: List[str]
    completed_stages: List[str]
    evidence: Dict[str, List[str]]
    profile: Optional[ProjectProfile] = None


def workspace_projects_root(start_path: Optional[str] = None):
    current = os.path.abspath(start_path or os.getcwd())
    while True:
        candidate = os.path.join(current, "projects")
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.join(os.path.expanduser("~"), "Workspace", "projects")
        current = parent


def project_profiles():
    return {}


def resolve_project_path(project: str, projects_root: Optional[str] = None):
    project = str(project)
    if os.path.isdir(project):
        return os.path.abspath(project)
    root = projects_root or workspace_projects_root()
    project_name = PROJECT_ALIASES.get(project, project)
    candidate = os.path.join(root, project_name)
    if os.path.isdir(candidate):
        return os.path.abspath(candidate)
    return os.path.abspath(candidate)


def discover_projects(projects_root: Optional[str] = None):
    root = projects_root or workspace_projects_root()
    if not os.path.isdir(root):
        return []
    projects = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path) or name.startswith("."):
            continue
        profile = project_profiles().get(name)
        projects.append({
            "project_id": name,
            "project_path": path,
            "profile": profile,
        })
    return projects


def scan_document_coverage(project_path: str):
    present = []
    search_roots = [project_path, os.path.join(project_path, "docs")]
    for doc in REQUIRED_PROJECT_DOCS:
        if any(os.path.exists(os.path.join(root, doc)) for root in search_roots):
            present.append(doc)
    missing = [doc for doc in REQUIRED_PROJECT_DOCS if doc not in present]
    return present, missing


def scan_architecture_evidence(project_path: str):
    evidence = {
        "domain_models": [],
        "workflows": [],
        "dashboards": [],
        "metrics": [],
        "approval_gates": [],
        "agents": [],
    }
    probes = {
        "domain_models": ["DOMAIN.md", "schema", "model"],
        "workflows": ["WORKFLOWS.md", "workflow"],
        "dashboards": ["DASHBOARD.md", "dashboard"],
        "metrics": ["METRICS.md", "metric"],
        "approval_gates": ["APPROVALS.md", "approval"],
        "agents": ["AGENTS.md", "agent"],
    }
    for root, dirs, files in os.walk(project_path):
        dirs[:] = [name for name in dirs if name not in {".git", "node_modules", ".venv", "__pycache__"}]
        rel_root = os.path.relpath(root, project_path)
        for file_name in files:
            lowered = file_name.lower()
            rel_path = file_name if rel_root == "." else os.path.join(rel_root, file_name)
            for category, tokens in probes.items():
                if any(token.lower() in lowered or token.lower() in rel_path.lower() for token in tokens):
                    evidence[category].append(rel_path)
    return evidence


def infer_completed_stages(present_documents: List[str], evidence: Dict[str, List[str]]):
    stages = []
    if "PROJECT.md" in present_documents:
        stages.extend(["business_system", "control_plane"])
    if "DOMAIN.md" in present_documents or evidence.get("domain_models"):
        stages.append("domain_models")
    if "WORKFLOWS.md" in present_documents or evidence.get("workflows"):
        stages.append("workflows")
    if "DASHBOARD.md" in present_documents or evidence.get("dashboards"):
        stages.append("dashboards")
    if "METRICS.md" in present_documents or evidence.get("metrics"):
        stages.append("metrics")
    if "APPROVALS.md" in present_documents or evidence.get("approval_gates"):
        stages.append("approval_gates")
    if "AGENTS.md" in present_documents or evidence.get("agents"):
        stages.append("agents")
    return stages


def scan_project(project: str, projects_root: Optional[str] = None):
    project_path = resolve_project_path(project, projects_root)
    project_id = os.path.basename(project_path)
    present, missing = scan_document_coverage(project_path)
    evidence = scan_architecture_evidence(project_path) if os.path.isdir(project_path) else {}
    completed = infer_completed_stages(present, evidence)
    profile = project_profiles().get(project_id) or ProjectProfile(
        project_id=project_id,
        canonical_name=project_id,
        aliases=[project_id],
        expected_metrics=[],
        review_hints=[],
    )
    return ProjectScan(
        project_id=project_id,
        project_path=project_path,
        present_documents=present,
        missing_documents=missing,
        completed_stages=completed,
        evidence=evidence,
        profile=profile,
    )
