"""Hermes Git Projects dashboard plugin API.

Routes are mounted by the dashboard at /api/plugins/hermes-git-projects/.
The plugin keeps all state under get_hermes_home()/hermes-git-projects.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from hermes_constants import get_hermes_home

router = APIRouter()

PLUGIN_NAME = "hermes-git-projects"
SCAN_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "venv", ".venv", "env", ".env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build",
    "target", ".next", ".nuxt", ".vite", "coverage", ".cache",
}
MAX_SCAN_DEPTH = 4
MAX_SCAN_REPOS = 160
GIT_TIMEOUT = 45

DEFAULT_SUGGESTED_SKILLS: List[Dict[str, Any]] = [
    {
        "name": "zeo-development-superpowers",
        "label": "Zeo Development Superpowers",
        "reason": "Use for Zeo-style implementation discipline: inspect, preserve architecture, verify, and report clearly.",
        "default": True,
        "triggers": ["implementation", "dashboard", "feature", "project"],
    },
    {
        "name": "systematic-debugging",
        "label": "Systematic Debugging",
        "reason": "Use for bugs, regressions, failures, errors, broken builds, and unclear root causes.",
        "default": True,
        "triggers": ["bug", "error", "failure", "regression", "broken", "traceback"],
    },
    {
        "name": "test-driven-development",
        "label": "Test-Driven Development",
        "reason": "Use when behavior changes need tests or acceptance checks before/alongside implementation.",
        "default": True,
        "triggers": ["feature", "test", "acceptance", "behavior", "bug"],
    },
    {
        "name": "requesting-code-review",
        "label": "Requesting Code Review",
        "reason": "Use for non-trivial code changes before finalizing or pushing work.",
        "default": True,
        "triggers": ["review", "refactor", "security", "quality", "changes"],
    },
    {
        "name": "github-pr-workflow",
        "label": "GitHub PR Workflow",
        "reason": "Use for GitHub-hosted repos when the work should land on a branch and become a PR.",
        "default": False,
        "triggers": ["github", "pull request", "branch", "push", "merge"],
    },
    {
        "name": "github-code-review",
        "label": "GitHub Code Review",
        "reason": "Use when reviewing or preparing GitHub pull requests.",
        "default": False,
        "triggers": ["github", "pr", "review", "diff"],
    },
    {
        "name": "github-repo-management",
        "label": "GitHub Repo Management",
        "reason": "Use for clone/fork/remote/release repository operations.",
        "default": False,
        "triggers": ["clone", "remote", "repo", "release", "fork"],
    },
    {
        "name": "writing-plans",
        "label": "Writing Plans",
        "reason": "Use when the issue needs a scoped implementation plan before execution.",
        "default": False,
        "triggers": ["plan", "architecture", "multi-step", "scope"],
    },
    {
        "name": "subagent-driven-development",
        "label": "Subagent Driven Development",
        "reason": "Use when work benefits from specialist agents, parallel inspection, or staged reviews.",
        "default": False,
        "triggers": ["subagent", "parallel", "specialist", "multi-agent"],
    },
]

class ProjectImportRequest(BaseModel):
    repo_url: str = Field(..., min_length=3)
    branch: Optional[str] = None

class SourceControlRequest(BaseModel):
    action: str
    branch: Optional[str] = None

class ProjectIssueCreate(BaseModel):
    title: str = Field(..., min_length=1)
    body: str = ""
    kind: str = "issue"
    severity: str = "medium"
    labels: List[str] = Field(default_factory=list)
    assignee: Optional[str] = None
    parent_task_ids: List[str] = Field(default_factory=list)
    selected_skills: Optional[List[str]] = None
    recommended_branch: Optional[str] = None

class SuggestedSkill(BaseModel):
    name: str
    label: Optional[str] = None
    reason: str = ""
    default: bool = False
    triggers: List[str] = Field(default_factory=list)

class SuggestedSkillsUpdate(BaseModel):
    skills: List[SuggestedSkill]


def _base_dir() -> Path:
    path = get_hermes_home() / PLUGIN_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path

def _repos_dir() -> Path:
    path = _base_dir() / "repos"
    path.mkdir(parents=True, exist_ok=True)
    return path

def _issues_path() -> Path:
    return _base_dir() / "issues.json"

def _skills_path() -> Path:
    return _base_dir() / "suggested-skills.json"

def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)

def _read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def _load_issues() -> Dict[str, Any]:
    data = _read_json(_issues_path(), {"issues": []})
    if not isinstance(data, dict) or not isinstance(data.get("issues"), list):
        return {"issues": []}
    return data

def _save_issues(data: Dict[str, Any]) -> None:
    _atomic_write_json(_issues_path(), data)

def _load_suggested_skills() -> List[Dict[str, Any]]:
    data = _read_json(_skills_path(), None)
    if not data:
        _atomic_write_json(_skills_path(), {"skills": DEFAULT_SUGGESTED_SKILLS})
        return list(DEFAULT_SUGGESTED_SKILLS)
    skills = data.get("skills") if isinstance(data, dict) else None
    if not isinstance(skills, list):
        return list(DEFAULT_SUGGESTED_SKILLS)
    cleaned = []
    for item in skills:
        if isinstance(item, dict) and item.get("name"):
            cleaned.append({
                "name": str(item["name"]).strip(),
                "label": item.get("label") or str(item["name"]).replace("-", " ").title(),
                "reason": item.get("reason", ""),
                "default": bool(item.get("default")),
                "triggers": [str(t) for t in item.get("triggers", []) if t],
            })
    return cleaned or list(DEFAULT_SUGGESTED_SKILLS)

def _git(repo: Path, args: List[str], timeout: int = GIT_TIMEOUT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(repo), text=True, capture_output=True, timeout=timeout
    )

def _git_out(repo: Path, args: List[str], default: str = "") -> str:
    try:
        proc = _git(repo, args)
    except Exception:
        return default
    if proc.returncode != 0:
        return default
    return proc.stdout.strip()

def _git_checked(repo: Path, args: List[str], timeout: int = GIT_TIMEOUT) -> str:
    try:
        proc = _git(repo, args, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail=f"git {' '.join(args)} timed out") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "git command failed").strip()[-1200:]
        raise HTTPException(status_code=400, detail=detail)
    return proc.stdout.strip()

def _is_git_repo(path: Path) -> bool:
    return (path / ".git").exists() and _git_out(path, ["rev-parse", "--is-inside-work-tree"]) == "true"

def _validate_repo_url(url: str) -> str:
    clean = url.strip()
    if not clean:
        raise HTTPException(status_code=400, detail="repo_url is required")
    allowed = (
        clean.startswith("https://")
        or clean.startswith("http://")
        or clean.startswith("ssh://")
        or re.match(r"^[\w.-]+@[\w.-]+:[\w./~:-]+(?:\.git)?$", clean)
    )
    if not allowed:
        raise HTTPException(status_code=400, detail="Only HTTPS, HTTP, SSH, or git@host:path repository URLs are supported")
    if any(token in clean for token in ["\n", "\r", "\x00"]):
        raise HTTPException(status_code=400, detail="Invalid repository URL")
    return clean

def _repo_slug(url: str) -> str:
    base = url.rstrip("/").split("/")[-1]
    if ":" in base and not base.startswith("http"):
        base = base.split(":")[-1]
    if base.endswith(".git"):
        base = base[:-4]
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip(".-_") or "repo"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{digest}"

def _project_id(path: Path) -> str:
    return "gp_" + hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:14]

def _github_info(remote: str) -> Dict[str, Optional[str]]:
    if not remote:
        return {"host": None, "owner": None, "repo": None, "url": None, "issues_url": None, "new_issue_url": None}
    owner = repo = None
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", remote)
    if not m:
        m = re.match(r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$", remote)
    if m:
        owner, repo = m.group(1), m.group(2)
        url = f"https://github.com/{owner}/{repo}"
        return {"host": "github.com", "owner": owner, "repo": repo, "url": url, "issues_url": f"{url}/issues", "new_issue_url": f"{url}/issues/new"}
    return {"host": None, "owner": None, "repo": None, "url": None, "issues_url": None, "new_issue_url": None}

def _ahead_behind(repo: Path) -> Dict[str, int]:
    upstream = _git_out(repo, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], "")
    if not upstream:
        return {"ahead": 0, "behind": 0}
    counts = _git_out(repo, ["rev-list", "--left-right", "--count", f"{upstream}...HEAD"], "0\t0")
    parts = counts.replace("\t", " ").split()
    try:
        behind, ahead = int(parts[0]), int(parts[1])
    except Exception:
        behind = ahead = 0
    return {"ahead": ahead, "behind": behind}

def _detect_stack(repo: Path) -> List[str]:
    markers = {
        "pyproject.toml": "Python", "requirements.txt": "Python", "manage.py": "Django",
        "package.json": "Node/React", "vite.config.ts": "Vite", "vite.config.js": "Vite",
        "tailwind.config.js": "Tailwind", "tailwind.config.ts": "Tailwind", "docker-compose.yml": "Docker",
        "Dockerfile": "Docker", "go.mod": "Go", "Cargo.toml": "Rust", "pom.xml": "Java/Maven",
    }
    found: List[str] = []
    for name, label in markers.items():
        if (repo / name).exists() and label not in found:
            found.append(label)
    return found

def _source_control(repo: Path) -> Dict[str, Any]:
    branch = _git_out(repo, ["branch", "--show-current"], "detached")
    branches = [b.strip().lstrip("* ").strip() for b in _git_out(repo, ["branch", "--format", "%(refname:short)"], "").splitlines() if b.strip()]
    remote = _git_out(repo, ["config", "--get", "remote.origin.url"], "")
    status = _git_out(repo, ["status", "--short"], "")
    latest = _git_out(repo, ["log", "-1", "--pretty=format:%h %s"], "")
    upstream = _git_out(repo, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], "")
    ab = _ahead_behind(repo)
    return {
        "branch": branch,
        "branches": branches,
        "remote": remote,
        "dirty": bool(status.strip()),
        "status": status.splitlines()[:80],
        "latest_commit": latest,
        "upstream": upstream,
        **ab,
        "github": _github_info(remote),
    }

def _issue_records_for(project_id: str) -> List[Dict[str, Any]]:
    return [i for i in _load_issues().get("issues", []) if i.get("project_id") == project_id]

def _scan_repo(repo: Path) -> Dict[str, Any]:
    sc = _source_control(repo)
    pid = _project_id(repo)
    return {
        "id": pid,
        "name": repo.name,
        "path": str(repo),
        "description": _read_description(repo),
        "stack": _detect_stack(repo),
        "source_control": sc,
        "issues": _issue_records_for(pid),
        "issue_count": len(_issue_records_for(pid)),
        "ready": True,
    }

def _read_description(repo: Path) -> str:
    for name in ("README.md", "README.rst", "README.txt"):
        p = repo / name
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    s = line.strip(" #\t")
                    if s:
                        return s[:220]
            except Exception:
                pass
    return "Imported Git project ready for Hermes work."

def _discover_repos() -> List[Path]:
    roots = [_repos_dir()]
    cwd = Path.cwd()
    for candidate in (cwd, cwd.parent):
        if candidate.exists() and candidate not in roots:
            roots.append(candidate)
    repos: List[Path] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        stack: List[tuple[Path, int]] = [(root, 0)]
        while stack and len(repos) < MAX_SCAN_REPOS:
            path, depth = stack.pop()
            if path.name in SCAN_SKIP_DIRS:
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            if _is_git_repo(path):
                repos.append(path)
                continue
            if depth >= MAX_SCAN_DEPTH:
                continue
            try:
                children = [c for c in path.iterdir() if c.is_dir() and c.name not in SCAN_SKIP_DIRS]
            except Exception:
                continue
            stack.extend((c, depth + 1) for c in children)
    return repos

def _projects() -> List[Dict[str, Any]]:
    projects = []
    for repo in _discover_repos():
        try:
            projects.append(_scan_repo(repo))
        except Exception as exc:
            projects.append({"id": _project_id(repo), "name": repo.name, "path": str(repo), "ready": False, "error": str(exc)})
    return sorted(projects, key=lambda p: (not str(p.get("path", "")).startswith(str(_repos_dir())), p.get("name", "").lower()))

def _find_project(project_id: str) -> Dict[str, Any]:
    for project in _projects():
        if project.get("id") == project_id:
            return project
    raise HTTPException(status_code=404, detail="Project not found")

def _branch_slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._/-]+", "-", text.lower()).strip("-/.")
    value = re.sub(r"[-/]{2,}", "-", value)
    return value[:80] or "issue"

def _infer_skills(issue: ProjectIssueCreate, project: Dict[str, Any]) -> List[str]:
    if issue.selected_skills is not None:
        return _dedupe(issue.selected_skills)
    text = " ".join([issue.title, issue.body, issue.kind, " ".join(issue.labels)]).lower()
    selected: List[str] = []
    for skill in _load_suggested_skills():
        name = skill.get("name")
        if not name:
            continue
        triggers = [str(t).lower() for t in skill.get("triggers", [])]
        if skill.get("default") or any(t and t in text for t in triggers):
            selected.append(name)
    gh = (((project.get("source_control") or {}).get("github") or {}).get("url"))
    if gh:
        selected.append("github-pr-workflow")
    return _dedupe(selected)

def _dedupe(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        clean = str(item).strip()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out

def _create_kanban_task(project: Dict[str, Any], issue: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from hermes_cli import kanban_db
        conn = kanban_db.connect()
        try:
            body = issue.get("body") or ""
            body += "\n\nHermes Git Projects issue metadata:\n"
            body += f"- Project: {project.get('name')}\n- Workspace: {project.get('path')}\n- Issue id: {issue.get('id')}\n"
            if issue.get("recommended_branch"):
                body += f"- Recommended branch: {issue.get('recommended_branch')}\n"
            task_id = kanban_db.create_task(
                conn,
                title=f"Resolve {project.get('name')}: {issue.get('title')}",
                body=body.strip(),
                assignee=issue.get("assignee"),
                created_by="hermes-git-projects",
                workspace_kind="dir",
                workspace_path=project.get("path"),
                parents=issue.get("parent_task_ids") or (),
                skills=issue.get("selected_skills") or (),
                # Kanban currently accepts "running" or "blocked" as explicit
                # initial statuses. With no parents, "running" creates an
                # immediately claimable task; with parents, create_task moves it
                # to todo until dependencies are complete.
                initial_status="running",
                idempotency_key=f"hermes-git-projects:{issue.get('id')}",
            )
            return {"todo_id": task_id}
        finally:
            conn.close()
    except Exception as exc:
        return {"todo_error": str(exc)}

@router.get("/summary")
async def summary() -> Dict[str, Any]:
    return {
        "plugin": PLUGIN_NAME,
        "description": "Import Git repos as Hermes projects, keep local clones, log issues into Kanban todos, and manage source-control actions.",
        "storage": {"base": str(_base_dir()), "repos": str(_repos_dir()), "issues": str(_issues_path()), "suggested_skills": str(_skills_path())},
        "projects": _projects(),
        "suggested_skills": _load_suggested_skills(),
    }

@router.post("/import")
async def import_project(body: ProjectImportRequest) -> Dict[str, Any]:
    url = _validate_repo_url(body.repo_url)
    target = _repos_dir() / _repo_slug(url)
    if target.exists() and not _is_git_repo(target):
        raise HTTPException(status_code=409, detail=f"Target exists but is not a git repo: {target}")
    if not target.exists():
        args = ["clone"]
        if body.branch:
            args += ["--branch", body.branch]
        args += [url, str(target)]
        try:
            proc = subprocess.run(["git", *args], text=True, capture_output=True, timeout=180)
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="git clone timed out") from exc
        if proc.returncode != 0:
            raise HTTPException(status_code=400, detail=(proc.stderr or proc.stdout or "git clone failed")[-1200:])
    else:
        _git_checked(target, ["fetch", "--all", "--prune"], timeout=120)
        if body.branch:
            _git_checked(target, ["checkout", body.branch])
    return {"ok": True, "project": _scan_repo(target), "message": "Repository imported and scanned"}

@router.post("/projects/{project_id}/source-control")
async def source_control(project_id: str, body: SourceControlRequest) -> Dict[str, Any]:
    project = _find_project(project_id)
    repo = Path(project["path"])
    action = body.action.strip().lower()
    if action == "fetch":
        output = _git_checked(repo, ["fetch", "--all", "--prune"], timeout=120)
    elif action == "pull":
        output = _git_checked(repo, ["pull", "--ff-only"], timeout=120)
    elif action == "push":
        branch = _git_out(repo, ["branch", "--show-current"])
        if not branch:
            raise HTTPException(status_code=400, detail="Cannot push while HEAD is detached")
        output = _git_checked(repo, ["push", "-u", "origin", branch], timeout=180)
    elif action == "checkout":
        if not body.branch:
            raise HTTPException(status_code=400, detail="branch is required for checkout")
        output = _git_checked(repo, ["checkout", body.branch])
    elif action == "create_branch":
        if not body.branch:
            raise HTTPException(status_code=400, detail="branch is required for create_branch")
        output = _git_checked(repo, ["checkout", "-b", body.branch])
    else:
        raise HTTPException(status_code=400, detail="action must be one of fetch, pull, push, checkout, create_branch")
    return {"ok": True, "action": action, "output": output, "project": _scan_repo(repo)}

@router.post("/projects/{project_id}/issues")
async def create_issue(project_id: str, body: ProjectIssueCreate) -> Dict[str, Any]:
    project = _find_project(project_id)
    now = int(time.time())
    issue_id = "gpi_" + hashlib.sha1(f"{project_id}:{body.title}:{now}".encode("utf-8")).hexdigest()[:12]
    branch = body.recommended_branch or f"issue/{_branch_slug(body.title)}"
    selected_skills = _infer_skills(body, project)
    issue: Dict[str, Any] = {
        "id": issue_id,
        "project_id": project_id,
        "title": body.title.strip(),
        "body": body.body.strip(),
        "kind": body.kind.strip() or "issue",
        "severity": body.severity.strip() or "medium",
        "labels": _dedupe(body.labels),
        "assignee": body.assignee,
        "parent_task_ids": _dedupe(body.parent_task_ids),
        "selected_skills": selected_skills,
        "recommended_branch": branch,
        "workspace_path": project.get("path"),
        "created_at": now,
        "updated_at": now,
    }
    issue.update(_create_kanban_task(project, issue))
    data = _load_issues()
    data.setdefault("issues", []).insert(0, issue)
    _save_issues(data)
    return {"ok": True, "issue": issue, "project": _scan_repo(Path(project["path"]))}

@router.get("/suggested-skills")
async def suggested_skills() -> Dict[str, Any]:
    return {"skills": _load_suggested_skills(), "path": str(_skills_path())}

@router.put("/suggested-skills")
async def update_suggested_skills(body: SuggestedSkillsUpdate) -> Dict[str, Any]:
    cleaned = []
    for skill in body.skills:
        name = skill.name.strip()
        if not name:
            continue
        cleaned.append({
            "name": name,
            "label": skill.label or name.replace("-", " ").title(),
            "reason": skill.reason,
            "default": skill.default,
            "triggers": _dedupe(skill.triggers),
        })
    _atomic_write_json(_skills_path(), {"skills": cleaned})
    return {"ok": True, "skills": cleaned, "path": str(_skills_path())}
