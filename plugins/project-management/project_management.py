"""Autonomous project bootstrap helpers for Hermes.

The plugin provides `/createproject` (and `/newproject`) plus archive/delete
helpers. Project creation follows the user's agreed operating model:

- intake may ask questions;
- after intake, use defaults and continue autonomously;
- Beads is the canonical issue graph;
- Kanban is the agent execution board;
- Serena is local CLI-first code intelligence;
- Context7 docs are captured locally when available;
- designer creates a mockup-image prompt/reference before UI implementation;
- scope-ledger items are expanded into Beads issues and audited before completion.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_cli import kanban_db as kb

PROJECTS_ROOT_ENV = "HERMES_PROJECTS_ROOT"
PROJECT_ORCHESTRATOR_ENV = "HERMES_PROJECT_ORCHESTRATOR"
PROJECT_DESIGNER_ENV = "HERMES_PROJECT_DESIGNER"
PROJECT_DEVELOPER_ENV = "HERMES_PROJECT_DEVELOPER"
STATE_ROOT_ENV = "HERMES_PROJECT_MANAGEMENT_HOME"
PENDING_TTL_SECONDS_ENV = "HERMES_PROJECT_PENDING_TTL_SECONDS"
BEADS_BIN_ENV = "HERMES_BEADS_BIN"
SERENA_BIN_ENV = "HERMES_SERENA_BIN"
CONTEXT7_BIN_ENV = "HERMES_CONTEXT7_BIN"
WIKI_PATH_ENV = "WIKI_PATH"
OBSIDIAN_VAULT_PATH_ENV = "OBSIDIAN_VAULT_PATH"

DEFAULT_ORCHESTRATOR_PROFILE = "default"
DEFAULT_DESIGNER_PROFILE = "design"
DEFAULT_DEVELOPER_PROFILE = "fullstack"
DEFAULT_WIKI_DIR_NAME = "project-memory"
STATE_DIR_NAME = "project-management"
PENDING_DIR_NAME = "pending"
PROJECT_MANIFEST = ".hermes-project.json"
DEFAULT_PENDING_TTL_SECONDS = 24 * 60 * 60

REQUIRED_FIELDS = ["project_name", "project_type", "goal"]
OPTIONAL_FIELDS = [
    "tech_stack", "target_users", "auth", "database", "integrations",
    "deployment", "constraints", "success_criteria", "avoid", "must_haves",
    "modules", "features", "pages", "screens", "data_entities", "user_roles",
    "workflows", "known_list_items", "design_style", "non_goals",
]

_LABEL_ALIASES = {
    "project name": "project_name", "project": "project_name", "name": "project_name",
    "type": "project_type", "project type": "project_type", "kind": "project_type",
    "goal": "goal", "purpose": "goal",
    "tech stack": "tech_stack", "technology stack": "tech_stack", "stack": "tech_stack",
    "technologies": "tech_stack", "language": "tech_stack", "languages": "tech_stack",
    "target users": "target_users", "users": "target_users", "audience": "target_users",
    "auth": "auth", "authentication": "auth", "login": "auth",
    "database": "database", "db": "database", "data": "database",
    "integrations": "integrations", "integration": "integrations",
    "deployment": "deployment", "deploy": "deployment", "hosting": "deployment",
    "constraints": "constraints", "constraint": "constraints",
    "success criteria": "success_criteria", "success": "success_criteria",
    "what to avoid": "avoid", "avoid": "avoid",
    "must-haves": "must_haves", "must haves": "must_haves", "requirements": "must_haves",
    "modules": "modules", "module list": "modules", "program modules": "modules",
    "features": "features", "feature list": "features",
    "pages": "pages", "page list": "pages",
    "screens": "screens", "screen list": "screens",
    "entities": "data_entities", "data entities": "data_entities", "models": "data_entities",
    "roles": "user_roles", "user roles": "user_roles",
    "workflows": "workflows", "flows": "workflows", "user flows": "workflows",
    "known list items": "known_list_items", "list items": "known_list_items", "items": "known_list_items",
    "design style": "design_style", "visual style": "design_style",
    "non goals": "non_goals", "non-goals": "non_goals",
}

_SERENA_LANG_MAP = {
    "python": "python", "py": "python", "fastapi": "python", "django": "python", "flask": "python",
    "typescript": "typescript", "ts": "typescript", "react": "typescript", "reactjs": "typescript",
    "nextjs": "typescript", "next.js": "typescript", "vue": "typescript", "svelte": "typescript",
    "javascript": "javascript", "js": "javascript", "node": "javascript", "nodejs": "javascript", "node.js": "javascript",
    "go": "go", "golang": "go", "rust": "rust", "java": "java", "kotlin": "kotlin",
    "swift": "swift", "dart": "dart", "flutter": "dart", "ruby": "ruby", "php": "php",
}

_DOC_LIBRARY_HINTS = {
    "react": "/reactjs/react.dev",
    "nextjs": "/vercel/next.js",
    "next.js": "/vercel/next.js",
    "prisma": "/prisma/docs",
    "postgres": "/postgres/postgres",
    "postgresql": "/postgres/postgres",
    "fastapi": "/tiangolo/fastapi",
    "tailwind": "/tailwindlabs/tailwindcss.com",
    "tailwindcss": "/tailwindlabs/tailwindcss.com",
}


@dataclass
class OperationResult:
    ok: bool
    message: str
    data: dict[str, Any]


def projects_root() -> Path:
    raw = os.environ.get(PROJECTS_ROOT_ENV)
    return Path(raw).expanduser() if raw else Path.home() / "projects"


def archived_projects_root() -> Path:
    return projects_root() / "archived"


def deleted_projects_root() -> Path:
    return projects_root() / "deleted"


def project_memory_root() -> Path:
    raw = os.environ.get(WIKI_PATH_ENV) or os.environ.get(OBSIDIAN_VAULT_PATH_ENV)
    return Path(raw).expanduser() if raw else Path.home() / DEFAULT_WIKI_DIR_NAME


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _append_log(path: Path, entry: str) -> None:
    existing = _read_text(path)
    path.write_text(existing.rstrip() + "\n\n" + entry.strip() + "\n", encoding="utf-8")


def _project_wikilink(slug: str) -> str:
    return f"[[projects/{slug}]]"


def _ensure_project_memory_wiki() -> Path:
    wiki = project_memory_root()
    for sub in [
        wiki / "projects",
        wiki / "references",
        wiki / "raw" / "briefs",
        wiki / "raw" / "screenshots",
        wiki / "raw" / "transcripts",
        wiki / "raw" / "assets",
        wiki / "_archive",
    ]:
        sub.mkdir(parents=True, exist_ok=True)

    today = _today()
    schema = wiki / "SCHEMA.md"
    if not schema.exists():
        schema.write_text(textwrap.dedent(f"""\
            # Project Memory Schema

            ## Domain
            Durable project memory for Hermes-managed projects. This wiki tracks project briefs, role routing, design references, implementation context, and links to Beads, Serena, and Codebase Memory MCP.

            ## Conventions
            - Main index: `index.md`.
            - One project page per project: `projects/<project-slug>.md`.
            - Reference workflows live in `references/`.
            - Raw source material lives in `raw/` and should not be edited after capture.
            - Every project page must be listed in `index.md` and every update should be appended to `log.md`.

            ## Memory Layers
            - Project Memory Wiki: human-readable markdown source of truth.
            - Beads: canonical task graph inside each project workspace.
            - Serena: local CLI-first code intelligence.
            - Codebase Memory MCP: optional per-repo code graph/architecture memory.
            - Hermes persistent memory: compact stable preferences and environment facts only.
        """), encoding="utf-8")

    index = wiki / "index.md"
    if not index.exists():
        index.write_text(textwrap.dedent(f"""\
            # Project Memory Index

            > Durable index for Hermes-managed project memory. Read this first before creating or updating project memory.
            > Last updated: {today} | Total project pages: 0

            ## Active Projects

            ## Planning Projects

            ## Archived Projects

            ## Reference Workflows

            - [[references/hermes-project-workflow]] — Hermes project bootstrap workflow.
            - [[references/design-mockup-workflow]] — Designer-owned mockup prompt/image reference workflow.
            - [[references/profile-routing]] — Role routing defaults.
            - [[references/codebase-memory]] — Codebase Memory MCP companion layer.
        """), encoding="utf-8")

    log = wiki / "log.md"
    if not log.exists():
        log.write_text(textwrap.dedent(f"""\
            # Project Memory Log

            > Append-only chronological record of project-memory actions.

            ## [{today}] create | Project memory wiki initialized
            - Created by Hermes project-management plugin.
        """), encoding="utf-8")

    references = {
        "hermes-project-workflow.md": "# Hermes Project Workflow\n\nProject workspaces are bootstrapped by the Hermes project-management plugin. See [[references/design-mockup-workflow]], [[references/profile-routing]], and [[references/codebase-memory]].\n",
        "design-mockup-workflow.md": "# Design Mockup Workflow\n\nDesigner creates `docs/design.md` with the mockup prompt, image path/URL, visual rationale, responsive states, accessibility notes, and handoff guidance before frontend/UI implementation.\n",
        "profile-routing.md": "# Profile Routing\n\nDefault roles: `pm` for coordination, `design` for mockup/UI specs, `fullstack` for implementation.\n",
        "codebase-memory.md": "# Codebase Memory MCP\n\nUse Codebase Memory MCP as an optional per-repository code graph. It complements but does not replace project memory pages.\n",
    }
    for filename, content in references.items():
        ref_path = wiki / "references" / filename
        if not ref_path.exists():
            ref_path.write_text(content, encoding="utf-8")
    return wiki


def _project_memory_page(payload: dict[str, str], project_dir: Path, slug: str, orch: str, tech_tokens: list[str], issues: dict[str, str], board_slug: str, decisions: list[str]) -> str:
    today = _today()
    designer = os.environ.get(PROJECT_DESIGNER_ENV, DEFAULT_DESIGNER_PROFILE)
    developer = os.environ.get(PROJECT_DEVELOPER_ENV, DEFAULT_DEVELOPER_PROFILE)
    decision_lines = "\n".join(f"- {d}" for d in decisions) or "- No default assumptions recorded."
    tech_lines = "\n".join(f"- {t}" for t in tech_tokens) or "- Unspecified"
    issue_lines = "\n".join(f"- {k}: `{v}`" for k, v in sorted(issues.items())) or "- No Beads issues recorded during bootstrap."
    return textwrap.dedent(f"""\
        ---
        title: {payload['project_name']}
        created: {today}
        updated: {today}
        type: project
        status: active
        project_slug: {slug}
        workspace: {project_dir}
        profiles: [{orch}, {designer}, {developer}]
        beads: {project_dir / '.beads'}
        serena: {project_dir / '.serena' / 'project.yml'}
        codebase_memory_project: {slug}
        ---

        # {payload['project_name']}

        ## Summary
        {payload['goal']}

        ## Workspace Links
        - Workspace: `{project_dir}`
        - AGENTS: `{project_dir / 'AGENTS.md'}`
        - Project brief: `{project_dir / 'project-brief.md'}`
        - Tech stack: `{project_dir / 'tech-stack.md'}`
        - Decisions: `{project_dir / 'decisions.md'}`
        - Scope ledger: `{project_dir / 'scope-ledger.md'}`
        - Scope JSON: `{project_dir / 'scope.json'}`
        - Completion audit: `{project_dir / 'completion-audit.md'}`

        ## Agent Roles
        - PM / orchestrator: `{orch}`
        - Designer: `{designer}`
        - Developer: `{developer}`

        ## Design Memory
        - Design source of truth: `{project_dir / 'docs' / 'design.md'}`
        - Mockup prompt: stored in `docs/design.md`
        - Mockup image path/URL: stored in `docs/design.md` after design work
        - Workflow: [[references/design-mockup-workflow]]

        ## Code Memory
        - Serena project: `{project_dir / '.serena' / 'project.yml'}`
        - Codebase Memory MCP project: `{slug}`
        - Codebase Memory reference: [[references/codebase-memory]]

        ## Beads / Task Memory
        - Beads directory: `{project_dir / '.beads'}`
        - Kanban board: `{board_slug}`
        {issue_lines}

        ## Tech Stack
        {tech_lines}

        ## Key Decisions
        {decision_lines}

        ## Open Questions
        - TBD

        ## Cross References
        - [[index]]
        - [[references/hermes-project-workflow]]
        - [[references/profile-routing]]
    """)


def _update_project_memory(payload: dict[str, str], project_dir: Path, slug: str, orch: str, tech_tokens: list[str], issues: dict[str, str], board_slug: str, decisions: list[str]) -> Path:
    wiki = _ensure_project_memory_wiki()
    today = _today()
    page = wiki / "projects" / f"{slug}.md"
    page.write_text(_project_memory_page(payload, project_dir, slug, orch, tech_tokens, issues, board_slug, decisions), encoding="utf-8")

    index = wiki / "index.md"
    index_text = _read_text(index)
    link = _project_wikilink(slug)
    entry = f"- {link} — active; workspace `{project_dir}`; design `{project_dir / 'docs' / 'design.md'}`; codebase memory `{slug}`."
    if link not in index_text:
        marker = "## Active Projects"
        if marker in index_text:
            index_text = index_text.replace(marker, marker + "\n\n" + entry, 1)
        else:
            index_text += "\n\n## Active Projects\n\n" + entry + "\n"
    index_text = re.sub(r"Last updated: \d{4}-\d{2}-\d{2}", f"Last updated: {today}", index_text)
    total = len(list((wiki / "projects").glob("*.md")))
    index_text = re.sub(r"Total project pages: \d+", f"Total project pages: {total}", index_text)
    index.write_text(index_text.rstrip() + "\n", encoding="utf-8")

    _append_log(wiki / "log.md", textwrap.dedent(f"""\
        ## [{today}] create | {payload['project_name']}
        - Created project memory page: `projects/{slug}.md`
        - Workspace: `{project_dir}`
        - Design reference: `{project_dir / 'docs' / 'design.md'}`
        - Codebase Memory MCP project: `{slug}`
    """))
    return page



def _state_root() -> Path:
    raw = os.environ.get(STATE_ROOT_ENV)
    if raw:
        return Path(raw).expanduser()
    hermes_home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    return hermes_home / STATE_DIR_NAME


def _pending_root() -> Path:
    return _state_root() / PENDING_DIR_NAME


def _ensure_dirs() -> None:
    projects_root().mkdir(parents=True, exist_ok=True)
    archived_projects_root().mkdir(parents=True, exist_ok=True)
    deleted_projects_root().mkdir(parents=True, exist_ok=True)
    _pending_root().mkdir(parents=True, exist_ok=True)


def _slugify_name(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        raise ValueError("project name is required")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    slug = re.sub(r"_+", "_", slug).strip("._-").lower()
    if not slug:
        raise ValueError("project name is required")
    return slug


def project_path_from_name(name: str) -> Path:
    return projects_root() / _slugify_name(name)


def archived_project_path_from_name(name: str) -> Path:
    return archived_projects_root() / _slugify_name(name)


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _manifest_path(project_dir: Path) -> Path:
    return project_dir / PROJECT_MANIFEST


def _load_manifest(project_dir: Path) -> Optional[dict[str, Any]]:
    return _read_json(_manifest_path(project_dir))


def _save_manifest(project_dir: Path, payload: dict[str, Any]) -> None:
    _write_json(_manifest_path(project_dir), payload)


def _session_key(event: Any, gateway: Any = None) -> str:
    source = getattr(event, "source", None)
    if gateway is not None and source is not None:
        resolver = getattr(gateway, "_session_key_for_source", None)
        if callable(resolver):
            try:
                key = resolver(source)
                if key:
                    return str(key)
            except Exception:
                pass
    platform = getattr(getattr(source, "platform", None), "value", None) or "?"
    chat_id = getattr(source, "chat_id", None) or "?"
    user_id = getattr(source, "user_id", None) or "?"
    return f"{platform}:{chat_id}:{user_id}"


def _pending_path(session_key: str) -> Path:
    import hashlib
    digest = hashlib.sha256(session_key.encode("utf-8", "ignore")).hexdigest()
    return _pending_root() / f"{digest}.json"


def _pending_ttl_seconds() -> int:
    try:
        return max(60, int(os.environ.get(PENDING_TTL_SECONDS_ENV, DEFAULT_PENDING_TTL_SECONDS)))
    except Exception:
        return DEFAULT_PENDING_TTL_SECONDS


def _is_pending_stale(state: dict[str, Any], *, now: Optional[int] = None) -> bool:
    ts = int(state.get("updated_at") or state.get("created_at") or 0)
    return not ts or ((int(time.time()) if now is None else now) - ts) > _pending_ttl_seconds()


def pending_state(session_key: str) -> Optional[dict[str, Any]]:
    state = _read_json(_pending_path(session_key))
    if not state:
        return None
    if _is_pending_stale(state):
        clear_pending_state(session_key)
        return None
    return state


def save_pending_state(session_key: str, state: dict[str, Any]) -> None:
    state = dict(state)
    state["session_key"] = session_key
    state["updated_at"] = int(time.time())
    _write_json(_pending_path(session_key), state)


def clear_pending_state(session_key: str) -> None:
    try:
        _pending_path(session_key).unlink(missing_ok=True)
    except Exception:
        pass


def cancel_pending_state(session_key: str) -> bool:
    path = _pending_path(session_key)
    existed = path.exists()
    clear_pending_state(session_key)
    return existed


def _normalize_label(label: str) -> Optional[str]:
    key = re.sub(r"[^a-z0-9]+", " ", str(label).strip().lower()).strip()
    return _LABEL_ALIASES.get(key)


def parse_project_answers(text: str) -> dict[str, str]:
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            out: dict[str, str] = {}
            for key, value in parsed.items():
                norm = _normalize_label(str(key))
                if norm and value is not None:
                    out[norm] = str(value).strip()
            return out
    out = {}
    for raw in re.split(r"[\n\r]+", text):
        line = raw.strip().lstrip("-•*").strip()
        if not line or ":" not in line:
            continue
        label, value = line.split(":", 1)
        norm = _normalize_label(label)
        if norm:
            out[norm] = value.strip()
    # Support the common slash form: /createproject --name ToDoist --type web app
    out.update({k: v for k, v in _parse_flag_args(text).items() if v})
    return out


def _parse_flag_args(raw: str) -> dict[str, str]:
    tokens = re.findall(r"--([a-zA-Z][\w-]*)\s+(?:\"([^\"]+)\"|'([^']+)'|([^\n\r]+?))(?=\s+--|$)", raw or "")
    out: dict[str, str] = {}
    key_map = {"name": "project_name", "type": "project_type", "stack": "tech_stack"}
    for key, dq, sq, bare in tokens:
        norm = key_map.get(key.lower().replace("-", "_"), key.lower().replace("-", "_"))
        if norm in REQUIRED_FIELDS or norm in OPTIONAL_FIELDS:
            out[norm] = (dq or sq or bare or "").strip()
    return out


def _format_missing(fields: Iterable[str]) -> str:
    labels = {
        "project_name": "Project Name", "project_type": "Project Type", "goal": "Goal",
        "tech_stack": "Tech Stack", "target_users": "Target Users", "auth": "Auth",
        "database": "Database", "deployment": "Deployment", "integrations": "Integrations",
        "constraints": "Constraints", "success_criteria": "Success Criteria",
        "avoid": "What to Avoid", "must_haves": "Must-Haves",
    }
    return ", ".join(labels.get(f, f) for f in fields)


def _format_prompt(initial_name: Optional[str] = None) -> str:
    name_line = f"Project Name: {initial_name}" if initial_name else "Project Name: ..."
    return "\n".join([
        "## New project intake",
        "",
        "Answer as much as you can. Leave unknown items blank. The PM will use sensible defaults after this intake and will not ask more user questions unless the project is unsafe or impossible.",
        "",
        "```",
        name_line,
        "Project Type: web app | website | mobile app | API | desktop app | automation | other",
        "Goal: ...",
        "Target Users: ...",
        "Must-Haves: ...",
        "Modules: ... (preserve every module in the list)",
        "Features: ...",
        "Pages: ...",
        "Screens: ...",
        "User Roles: ...",
        "Workflows: ...",
        "Data Entities: ...",
        "Auth: yes/no/unspecified",
        "Database: yes/no/unspecified",
        "Integrations: ...",
        "Tech Stack: ...",
        "Deployment: ...",
        "Design Style: ...",
        "Constraints: ...",
        "Success Criteria: ...",
        "What to Avoid: ...",
        "Non-Goals: ...",
        "```",
        "",
        "Important: if you provide a numbered or comma-separated list, every item will become a scope-ledger entry and a Beads-tracked deliverable.",
    ])


def _parse_tech_stack(raw: str) -> list[str]:
    return [t.strip().strip("\"'") for t in re.split(r"[,/|+\n]+", raw or "") if t.strip().strip("\"'")]


def _split_scope_items(raw: Any) -> list[str]:
    """Split user-provided scope lists while preserving every requested item."""
    if raw is None:
        return []
    if isinstance(raw, list):
        parts = [str(x).strip() for x in raw]
    else:
        text = str(raw).strip()
        if not text:
            return []
        # Prefer line/numbered-list splitting when the user gave an explicit list.
        normalized = re.sub(r"\r\n?", "\n", text)
        normalized = re.sub(r"(?<!\d)\b\d+[\).]\s+", "\n", normalized)
        normalized = re.sub(r"\s+[•*]\s+", "\n", normalized)
        if "\n" in normalized:
            parts = re.split(r"\n+|;", normalized)
        else:
            parts = re.split(r",|;|\s+\|\s+", normalized)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        item = re.sub(r"^[-•*\s]+", "", str(part).strip()).strip(" .")
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _scope_sources(payload: dict[str, str]) -> list[tuple[str, str]]:
    return [
        ("module", "modules"), ("feature", "features"), ("page", "pages"),
        ("screen", "screens"), ("workflow", "workflows"), ("role", "user_roles"),
        ("data", "data_entities"), ("deliverable", "known_list_items"),
    ]


def _build_scope_items(payload: dict[str, str]) -> list[dict[str, str]]:
    """Create the canonical scope ledger from explicit lists and must-haves."""
    items: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for kind, field in _scope_sources(payload):
        for item in _split_scope_items(payload.get(field, "")):
            key = (kind, item.lower())
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "id": f"SCOPE-{len(items) + 1:03d}",
                "kind": kind,
                "title": item,
                "source_field": field,
                "status": "planned",
                "beads_issue": "",
                "acceptance": f"{item} is implemented, wired into the app, and verified with an automated or smoke check.",
            })
    if not items:
        for item in _split_scope_items(payload.get("must_haves", "")):
            key = ("deliverable", item.lower())
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "id": f"SCOPE-{len(items) + 1:03d}",
                "kind": "deliverable",
                "title": item,
                "source_field": "must_haves",
                "status": "planned",
                "beads_issue": "",
                "acceptance": f"{item} is delivered and verified.",
            })
    if not items:
        items.append({
            "id": "SCOPE-001",
            "kind": "project",
            "title": payload.get("goal", "Deliver the requested project"),
            "source_field": "goal",
            "status": "planned",
            "beads_issue": "",
            "acceptance": "The project goal is delivered with tests or smoke checks.",
        })
    return items


def _scope_markdown(scope_items: list[dict[str, str]]) -> str:
    rows = ["# Scope Ledger", "", "Every requested deliverable must map to Beads and pass verification.", ""]
    rows.append("| ID | Kind | Title | Beads Issue | Acceptance | Status |")
    rows.append("|---|---|---|---|---|---|")
    for item in scope_items:
        title = item["title"].replace("|", "\\|")
        acceptance = item["acceptance"].replace("|", "\\|")
        rows.append(f"| {item['id']} | {item['kind']} | {title} | {item.get('beads_issue') or 'TBD'} | {acceptance} | {item['status']} |")
    rows.extend([
        "",
        "## Completion Rule",
        "The project is not complete until every scope item has a closed Beads issue, implementation evidence, and passing verification.",
    ])
    return "\n".join(rows) + "\n"


def _decision_matrix_markdown() -> str:
    return textwrap.dedent("""\
        # Decision Matrix

        After intake, agents must keep working without user approval unless the project is unsafe, illegal, or impossible.

        | Situation | Decision |
        |---|---|
        | Missing implementation detail | Choose a sensible default and record it. |
        | Missing visual detail | Designer chooses and records it in `docs/design.md`. |
        | Missing technical detail | Developer chooses and records it in `decisions.md` or Beads. |
        | Ambiguous feature behavior | Build the simplest useful version and record the assumption. |
        | Tool failure | Create a fix issue and continue other ready work. |
        | Agent needs input | Ask the correct agent, not the user. |
        | Scope item missing from Beads | Create a Beads issue before implementation continues. |
        | Requested list has many items | Preserve every item; never build only examples. |
        | Final audit finds gaps | Create missing Beads issues and reopen delivery. |
    """)


def _default_stack_for_type(project_type: str) -> str:
    key = (project_type or "").lower()
    if "mobile" in key:
        return "React Native, TypeScript"
    if "api" in key:
        return "Python, FastAPI, PostgreSQL"
    if "desktop" in key:
        return "Python, PySide6"
    if "site" in key and "web app" not in key:
        return "Next.js, React, TypeScript, Tailwind CSS"
    return "Next.js, React, TypeScript, PostgreSQL, Prisma, Tailwind CSS"


def _apply_defaults(payload: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    out = {k: str(v).strip() for k, v in dict(payload).items() if str(v).strip()}
    decisions: list[str] = []
    ptype = out.get("project_type", "web app")
    if "project_type" not in out:
        out["project_type"] = ptype
        decisions.append("Assumed project type `web app` because none was provided during intake.")
    if not out.get("tech_stack"):
        out["tech_stack"] = _default_stack_for_type(ptype)
        decisions.append(f"Assumed tech stack `{out['tech_stack']}` for project type `{ptype}`.")
    defaults = {
        "target_users": "general users",
        "auth": "not included unless required by a generated feature task",
        "database": "PostgreSQL if persistent data is needed by the app model",
        "integrations": "none specified",
        "deployment": "local development first; production target to be selected by deployment automation",
        "constraints": "keep scope focused; avoid approval gates after intake",
        "success_criteria": "project scaffolds, tasks, docs, design brief, Beads, Serena, and Kanban initialize successfully",
        "avoid": "approval-based stalls; undocumented assumptions; task cards waiting for human confirmation",
        "must_haves": "autonomous execution after intake; Beads canonical tasks; Serena code intelligence; Context7 docs; designer mockup prompt before UI implementation",
        "modules": "",
        "features": "",
        "pages": "",
        "screens": "",
        "data_entities": "",
        "user_roles": "",
        "workflows": "",
        "known_list_items": "",
        "design_style": "clean, modern, accessible, responsive interface",
        "non_goals": "none specified",
    }
    for key, value in defaults.items():
        if not out.get(key):
            out[key] = value
            decisions.append(f"Defaulted `{key}` to `{value}`.")
    return out, decisions


def _serena_language_ids(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokens:
        key = tok.lower().replace(" ", "").replace("-", "")
        lang = _SERENA_LANG_MAP.get(key) or _SERENA_LANG_MAP.get(tok.lower())
        if lang and lang not in seen:
            seen.add(lang)
            out.append(lang)
    return out


def _run(cmd: list[str], *, cwd: Path, timeout: int = 120) -> tuple[bool, str]:
    try:
        result = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout, check=True)
        return True, result.stdout.strip()
    except FileNotFoundError:
        return False, f"command not found: {cmd[0]}"
    except subprocess.CalledProcessError as exc:
        return False, f"exit {exc.returncode}: {(exc.stdout or '').strip()}"
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s: {' '.join(cmd)}"


def _git_init(project_dir: Path) -> tuple[bool, str]:
    return _run(["git", "init"], cwd=project_dir, timeout=60)


def _run_beads_init(project_dir: Path) -> tuple[bool, str]:
    bd = os.environ.get(BEADS_BIN_ENV, "bd")
    ok, msg = _run([bd, "init", "--quiet"], cwd=project_dir, timeout=60)
    if ok and not (project_dir / ".beads").exists():
        return False, "bd init exited successfully but no .beads directory was created"
    return ok, msg or "bd init completed"


def _bd_create(project_dir: Path, title: str, *, issue_type: str = "task", priority: str = "2", description: str = "", parent: str = "", deps: list[str] | None = None, assignee: str = "") -> tuple[bool, str, str]:
    bd = os.environ.get(BEADS_BIN_ENV, "bd")
    cmd = [bd, "create", title, "--type", issue_type, "--priority", priority, "--json"]
    if description:
        cmd += ["--description", description]
    if parent:
        cmd += ["--parent", parent]
    if assignee:
        cmd += ["--assignee", assignee]
    for dep in deps or []:
        cmd += ["--deps", dep]
    ok, output = _run(cmd, cwd=project_dir, timeout=60)
    issue_id = ""
    if ok:
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict):
                issue_id = str(parsed.get("id") or parsed.get("issue_id") or parsed.get("ID") or "")
        except Exception:
            m = re.search(r"[a-z]+-[A-Za-z0-9]+", output)
            issue_id = m.group(0) if m else ""
    return ok, issue_id, output


def _create_beads_plan(project_dir: Path, payload: dict[str, str], decisions: list[str], orchestrator_profile: str, scope_items: list[dict[str, str]]) -> tuple[dict[str, str], list[str], list[dict[str, str]]]:
    warnings: list[str] = []
    issues: dict[str, str] = {}
    ok, epic_id, output = _bd_create(
        project_dir,
        f"epic: create {payload['project_name']}",
        issue_type="epic",
        priority="0",
        assignee=orchestrator_profile,
        description="Canonical project epic. Beads owns task state; Kanban mirrors execution. No approval gates after intake.",
    )
    if not ok or not epic_id:
        return issues, [f"bd create epic failed: {output}"], scope_items
    issues["epic"] = epic_id

    def create(key: str, title: str, desc: str, deps: list[str] | None = None, issue_type: str = "task", priority: str = "2", assignee: str | None = None) -> None:
        dep_args = [f"blocks:{dep}" for dep in (deps or []) if dep]
        ok2, iid, out = _bd_create(
            project_dir, title, issue_type=issue_type, priority=priority,
            description=desc, parent=epic_id, deps=dep_args, assignee=assignee or orchestrator_profile,
        )
        if ok2 and iid:
            issues[key] = iid
        else:
            warnings.append(f"bd create `{title}` failed: {out}")

    designer_profile = os.environ.get(PROJECT_DESIGNER_ENV, DEFAULT_DESIGNER_PROFILE)
    developer_profile = os.environ.get(PROJECT_DEVELOPER_ENV, DEFAULT_DEVELOPER_PROFILE)

    create("scaffold", "bootstrap workspace scaffold", "Create base folders, README, project brief, decisions, docs layout, and git repo.", priority="1")
    create("docs", "fetch and curate Context7 stack docs", "Fetch relevant Context7 documentation and save concise references under docs/context7/.", deps=[issues.get("scaffold", "")], priority="1")
    create("serena", "initialize Serena code intelligence", "Create/index Serena project and record usage notes in docs/serena.md.", deps=[issues.get("scaffold", "")], priority="1")
    create("plan", "generate autonomous implementation task graph", "Turn the brief into concrete autonomous implementation tasks. Use defaults and record assumptions; do not wait for approval.", deps=[issues.get("docs", ""), issues.get("serena", "")], priority="1")
    create("design_mockup", "craft mockup website design prompt and reference", "Designer turns the project details into a high-quality prompt for a mockup website/app design graphic, generates or attaches the resulting image when image generation is available, and saves the prompt plus visual guidance under docs/design.md. UI/UX implementation must use this visual reference.", deps=[issues.get("plan", "")], priority="1", assignee=designer_profile)
    create("app_shell", "build initial app shell", "Create the initial project shell for the selected stack. Use docs/design.md and any mockup image reference for UI/UX decisions. Verify with automated checks.", deps=[issues.get("plan", ""), issues.get("design_mockup", "")], issue_type="feature", assignee=developer_profile)
    create("data_model", "define first-pass data model", "Implement the minimal data model required by the project goal. Verify with tests or schema checks.", deps=[issues.get("plan", "")], issue_type="feature", assignee=developer_profile)
    create("tests", "add automated verification", "Add smoke tests/build/lint checks so completion is automated rather than approval-based.", deps=[issues.get("app_shell", ""), issues.get("data_model", "")], assignee=developer_profile)
    scope_issue_ids: list[str] = []
    scope_parent = issues.get("plan", "") or epic_id
    for item in scope_items:
        title = f"deliver {item['id']}: {item['title']}"
        desc = textwrap.dedent(f"""\
            Scope item: {item['id']}
            Kind: {item['kind']}
            Source field: {item['source_field']}

            Required deliverable:
            {item['title']}

            Acceptance gate:
            {item['acceptance']}

            Rules:
            - Do not close this issue until this exact item is implemented or explicitly blocked with evidence.
            - Add or update tests/smoke checks for this item.
            - Link implementation notes in the close reason.
        """)
        ok3, iid, out3 = _bd_create(
            project_dir, title, issue_type="feature", priority="1",
            description=desc, parent=epic_id, deps=[f"blocks:{scope_parent}"], assignee=developer_profile,
        )
        if ok3 and iid:
            item["beads_issue"] = iid
            issues[item["id"]] = iid
            scope_issue_ids.append(iid)
        else:
            warnings.append(f"bd create scope item `{item['title']}` failed: {out3}")

    create("scope_audit", "run final scope completion audit", "Compare scope-ledger.md and scope.json against closed Beads issues, implementation evidence, and verification output. Reopen or create issues for any missing item.", deps=scope_issue_ids or [issues.get("tests", "")], assignee=orchestrator_profile, priority="0")
    create("deploy", "prepare deployment path", "Add deployment notes/scripts for the default target after tests are green and the final scope audit is complete.", deps=[issues.get("tests", ""), issues.get("scope_audit", "")], assignee=developer_profile)
    return issues, warnings, scope_items


def _yaml_scalar(value: Any) -> str:
    """Return a JSON-compatible quoted scalar, valid in YAML 1.2."""
    return json.dumps(str(value), ensure_ascii=False)


def _write_serena_project_yml(project_dir: Path, project_name: str, lang_ids: list[str]) -> Path:
    serena_dir = project_dir / ".serena"
    serena_dir.mkdir(parents=True, exist_ok=True)
    yml = serena_dir / "project.yml"
    langs = lang_ids or ["typescript"]
    ignored = ["**/.git/**", "**/node_modules/**", "**/dist/**", "**/build/**", "**/__pycache__/**", "**/.next/**"]
    content = ["# Generated by Hermes project-management plugin", f"project_name: {_yaml_scalar(project_name)}", "languages:"]
    content += [f"  - {_yaml_scalar(lang)}" for lang in langs]
    content += ["read_only: false", "ignored_paths:"]
    content += [f"  - {_yaml_scalar(pat)}" for pat in ignored]
    content += ["ignore_all_files_in_gitignore: true"]
    yml.write_text("\n".join(content) + "\n", encoding="utf-8")
    return yml


def _run_serena_index(project_dir: Path) -> tuple[bool, str]:
    serena = os.environ.get(SERENA_BIN_ENV, "serena")
    ok, msg = _run([serena, "project", "index", str(project_dir)], cwd=project_dir, timeout=300)
    if ok:
        return True, msg or "serena project index completed"
    # The locally documented command may not be available in every Serena version;
    # a generated project.yml still gives agents project-aware setup instructions.
    return False, msg


def _fetch_context7_docs(tokens: list[str], docs_dir: Path) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    produced: list[str] = []
    docs_dir.mkdir(parents=True, exist_ok=True)
    ctx7 = os.environ.get(CONTEXT7_BIN_ENV) or shutil.which("ctx7")
    npx = shutil.which("npx")
    if not ctx7 and not npx:
        warnings.append("Context7 fetch skipped: neither ctx7 nor npx is available in PATH")
        return produced, warnings
    for tech in tokens:
        key = tech.lower().replace(" ", "").replace("-", "")
        lib_id = _DOC_LIBRARY_HINTS.get(key) or _DOC_LIBRARY_HINTS.get(tech.lower())
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", tech).strip("_").lower() or "library"
        dest = docs_dir / f"{safe}.md"
        try:
            if lib_id:
                cmd = ([ctx7, "docs", lib_id, "setup patterns quickstart examples"] if ctx7 else [npx, "ctx7", "docs", lib_id, "setup patterns quickstart examples"])
                ok, out = _run(cmd, cwd=docs_dir, timeout=120)
            else:
                cmd = ([ctx7, "library", tech] if ctx7 else [npx, "ctx7", "library", tech])
                ok, out = _run(cmd, cwd=docs_dir, timeout=120)
            if ok and out.strip():
                body = out.strip()
                if len(body) > 12000:
                    body = body[:12000] + "\n\n[Truncated to 12k chars for agent readability.]"
                dest.write_text(f"# {tech} Context7 Notes\n\nLibrary hint: `{lib_id or 'resolved via search'}`\n\n```text\n{body}\n```\n", encoding="utf-8")
                produced.append(str(dest))
            else:
                warnings.append(f"Context7 docs for {tech} skipped: {out}")
        except Exception as exc:
            warnings.append(f"Context7 docs for {tech} skipped: {exc}")
    return produced, warnings


def _write_project_files(project_dir: Path, payload: dict[str, str], decisions: list[str], tech_tokens: list[str], docs_fetched: list[str], bd_ok: bool, serena_ok: bool, scope_items: list[dict[str, str]]) -> None:
    project_name = payload["project_name"]
    project_dir.joinpath("docs", "context7").mkdir(parents=True, exist_ok=True)
    (project_dir / "README.md").write_text(f"# {project_name}\n\n{payload['goal']}\n", encoding="utf-8")
    (project_dir / "project-brief.md").write_text(textwrap.dedent(f"""\
        # Project Brief

        - **Project**: {project_name}
        - **Type**: {payload['project_type']}
        - **Goal**: {payload['goal']}
        - **Target Users**: {payload['target_users']}
        - **Must-Haves**: {payload['must_haves']}
        - **Constraints**: {payload['constraints']}
        - **Success Criteria**: {payload['success_criteria']}
        - **What to Avoid**: {payload['avoid']}
        - **Auth**: {payload['auth']}
        - **Database**: {payload['database']}
        - **Integrations**: {payload['integrations']}
        - **Deployment**: {payload['deployment']}
        - **Tech Stack**: {payload['tech_stack']}
        - **Modules**: {payload.get('modules', '')}
        - **Features**: {payload.get('features', '')}
        - **Pages**: {payload.get('pages', '')}
        - **Screens**: {payload.get('screens', '')}
        - **User Roles**: {payload.get('user_roles', '')}
        - **Workflows**: {payload.get('workflows', '')}
        - **Data Entities**: {payload.get('data_entities', '')}
        - **Design Style**: {payload.get('design_style', '')}
        - **Non-Goals**: {payload.get('non_goals', '')}
    """), encoding="utf-8")
    (project_dir / "scope-ledger.md").write_text(_scope_markdown(scope_items), encoding="utf-8")
    (project_dir / "scope.json").write_text(json.dumps(scope_items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (project_dir / "decision-matrix.md").write_text(_decision_matrix_markdown(), encoding="utf-8")
    (project_dir / "completion-audit.md").write_text(textwrap.dedent("""\
        # Completion Audit

        The PM owns this file at final audit.

        ## Audit Checklist
        - [ ] Every `scope.json` item has a Beads issue.
        - [ ] Every scope Beads issue is closed or has a justified blocker.
        - [ ] Tests, build, lint, or smoke checks passed.
        - [ ] UI work follows `docs/design.md`.
        - [ ] Missing work created new Beads issues before project close.
    """), encoding="utf-8")
    (project_dir / "tech-stack.md").write_text("# Tech Stack\n\n" + "\n".join(f"- {t}" for t in tech_tokens) + "\n", encoding="utf-8")
    (project_dir / "decisions.md").write_text("# Decisions and Defaults\n\n" + ("\n".join(f"- {d}" for d in decisions) if decisions else "- No defaults were needed.") + "\n", encoding="utf-8")
    if decisions:
        (project_dir / "assumptions.md").write_text("# Assumptions\n\n" + "\n".join(f"- {d}" for d in decisions) + "\n", encoding="utf-8")
    (project_dir / "docs" / "serena.md").write_text(textwrap.dedent(f"""\
        # Serena Usage

        Serena is the local code-intelligence layer for this project.

        - Project file: `.serena/project.yml`
        - Re-index after larger edits: `serena project index .`
        - Status at bootstrap: {'indexed' if serena_ok else 'project file generated; index command should be retried if needed'}

        Use Serena for semantic code search, symbol lookup, debugging, and navigation.
    """), encoding="utf-8")
    (project_dir / "docs" / "design.md").write_text(textwrap.dedent(f"""\
        # Design Mockup Reference

        The UI/UX designer owns this file before frontend implementation begins.

        ## Required designer step
        1. Read `project-brief.md`, `scope-ledger.md`, `scope.json`, `tech-stack.md`, `decisions.md`, `decision-matrix.md`, and this project's `AGENTS.md`.
        2. Craft a polished image-generation prompt for a mockup website/app design graphic based on the project details.
        3. Generate or attach the resulting mockup image when image generation is available.
        4. Save the final prompt, image path/URL, design rationale, responsive states, accessibility notes, and implementation guidance here.
        5. File follow-up Beads tickets for any UI/UX features implied by the mockup.

        ## Prompt template
        ```text
        Create a mockup image of a modern professionally designed {payload['project_type']} for {payload['goal']}.
        Audience: {payload['target_users']}.
        Must include: {payload['must_haves']}.
        Scope ledger: cover every module/page/workflow family in `scope-ledger.md`.
        Visual direction: clean SaaS-grade landing/app UI, strong hero section, clear navigation, rounded cards, polished dashboard or product preview, tasteful iconography, readable typography, accessible contrast, realistic forms and controls, responsive layout cues, and a cohesive color palette.
        Avoid: {payload['avoid']}.
        ```

        ## Implementation contract
        - Frontend/UI tickets must use the selected mockup image as the primary visual reference.
        - If the generated mockup conflicts with constraints in `project-brief.md`, document the tradeoff here and prefer the brief.
        - Do not block on user approval; make a tasteful default choice and record it.
    """), encoding="utf-8")
    (project_dir / "task-index.md").write_text(textwrap.dedent(f"""\
        # Task Index

        Beads is canonical for issue state. Kanban mirrors executable agent work.

        - Beads status at bootstrap: {'initialized' if bd_ok else 'initialization warning; retry `bd init` if needed'}
        - Context7 docs fetched: {len(docs_fetched)}
        - No approval gates are allowed after intake.
        - Scope items tracked: {len(scope_items)}
    """), encoding="utf-8")
    (project_dir / "AGENTS.md").write_text(_agents_md(payload, decisions, docs_fetched, bd_ok, serena_ok), encoding="utf-8")


def _agents_md(payload: dict[str, str], decisions: list[str], docs_fetched: list[str], bd_ok: bool, serena_ok: bool) -> str:
    docs_lines = "\n".join(f"- `{Path(p).as_posix()}`" for p in docs_fetched) or "- No docs fetched; use Context7 before library-specific implementation."
    decision_lines = "\n".join(f"- {d}" for d in decisions) or "- No default assumptions recorded."
    return textwrap.dedent(f"""\
        # AGENTS.md — {payload['project_name']}

        ## Operating Rules
        - Initial intake is complete. Do not ask the user more questions unless the workflow is unrecoverably blocked.
        - Use defaults and document assumptions rather than pausing.
        - Beads is canonical for issue state.
        - Kanban is the execution board for agents.
        - Serena is the code-intelligence layer.
        - Context7 docs in `docs/context7/` should be read before stack-specific implementation.
        - `scope-ledger.md` and `scope.json` are the canonical deliverable list.
        - Every scope item must map to at least one Beads issue.
        - Do not mark the project complete while any scope item is missing, unverified, or not closed.
        - If a requested list contains many items, build every item. Do not build examples only.
        - Designer must create `docs/design.md` with a mockup-image prompt and resulting image reference before UI/UX feature implementation.
        - Frontend/UI implementation must use `docs/design.md` and the mockup image as the visual source of truth.
        - No task may wait for user approval after intake.
        - Agent-to-agent review is allowed, but lack of review must not block progress; record the review path and continue.
        - Reviews must be automated where possible: tests, lint, build, smoke checks, static analysis.
        - Failures create fix tasks; they do not create approval waits.

        ## Project Summary
        - Project: {payload['project_name']}
        - Type: {payload['project_type']}
        - Goal: {payload['goal']}
        - Stack: {payload['tech_stack']}

        ## Defaults / Decisions
        {decision_lines}

        ## Context7 References
        {docs_lines}

        ## Beads
        Status: {'initialized' if bd_ok else 'needs retry'}

        Recommended commands:
        ```bash
        bd ready --json
        bd update <id> --claim
        bd close <id> --reason "Completed and verified" --json
        ```

        ## Serena
        Status: {'indexed' if serena_ok else 'project config generated; index may need retry'}

        Recommended commands:
        ```bash
        serena project index .
        serena project health-check .
        ```

        ## Design Mockup Workflow
        - Design source file: `docs/design.md`
        - Designer task: transform the project brief into a polished image-generation prompt, generate or attach the mockup image when available, and record the visual system.
        - Developer task: implement UI/UX features from the design reference; if code realities require changes, update Beads and document the deviation rather than waiting for approval.
    """)


def _create_kanban_tasks(project_dir: Path, board_slug: str, payload: dict[str, str], issues: dict[str, str], orchestrator_profile: str) -> dict[str, str]:
    created: dict[str, str] = {}
    task_defs = [
        ("scaffold", "bootstrap workspace scaffold", []),
        ("docs", "fetch and curate Context7 stack docs", ["scaffold"]),
        ("serena", "initialize Serena code intelligence", ["scaffold"]),
        ("plan", "generate autonomous implementation task graph", ["docs", "serena"]),
        ("design_mockup", "craft mockup website design prompt and reference", ["plan"]),
        ("app_shell", "build initial app shell", ["plan", "design_mockup"]),
        ("data_model", "define first-pass data model", ["plan"]),
        ("tests", "add automated verification", ["app_shell", "data_model"]),
        ("scope_audit", "run final scope completion audit", ["tests"]),
        ("deploy", "prepare deployment path", ["tests", "scope_audit"]),
    ]
    assignees = {
        "design_mockup": os.environ.get(PROJECT_DESIGNER_ENV, DEFAULT_DESIGNER_PROFILE),
        "app_shell": os.environ.get(PROJECT_DEVELOPER_ENV, DEFAULT_DEVELOPER_PROFILE),
        "data_model": os.environ.get(PROJECT_DEVELOPER_ENV, DEFAULT_DEVELOPER_PROFILE),
        "tests": os.environ.get(PROJECT_DEVELOPER_ENV, DEFAULT_DEVELOPER_PROFILE),
        "deploy": os.environ.get(PROJECT_DEVELOPER_ENV, DEFAULT_DEVELOPER_PROFILE),
    }
    with kb.connect(board=board_slug) as conn:
        for key, title, parent_keys in task_defs:
            parents = [created[p] for p in parent_keys if p in created]
            body = textwrap.dedent(f"""\
                ## {title}

                Project: {payload['project_name']}
                Beads issue: {issues.get(key, 'not-created')}
                Workspace: {project_dir}

                Rules:
                - Execute autonomously.
                - Do not wait for user approval.
                - Use Serena for code navigation and debugging.
                - Read relevant Context7 docs before stack-specific work.
                - For UI/UX work, read `docs/design.md` and use its mockup image reference as the visual source of truth.
                - Verify with automated checks.
                - Check `scope-ledger.md` before closing work.
                - If blocked by a technical failure, create/record a fix task.
            """)
            task_id = kb.create_task(
                conn, title=title, body=body, assignee=assignees.get(key, orchestrator_profile),
                created_by="project-management", workspace_kind="dir", workspace_path=str(project_dir),
                priority=1 if key in {"scaffold", "docs", "serena", "plan", "design_mockup"} else 2,
                parents=parents,
                # Keep task skill attachments limited to universally available workers.
                # Serena and Context7 guidance is carried in the task body so tasks
                # do not fail if a runtime lacks those optional skill bundles.
                skills=["kanban-worker", "beads-workflow"],
                idempotency_key=f"project-management:{board_slug}:{key}",
            )
            created[key] = task_id
    return created


def create_project(*, payload: dict[str, str], initial_name: Optional[str] = None, orchestrator_profile: Optional[str] = None) -> OperationResult:
    _ensure_dirs()
    payload = dict(payload)
    if initial_name and not payload.get("project_name"):
        payload["project_name"] = initial_name
    missing = [field for field in REQUIRED_FIELDS if not str(payload.get(field, "")).strip()]
    if missing:
        return OperationResult(False, f"Missing required fields: {_format_missing(missing)}", {"missing": missing})

    payload, decisions = _apply_defaults(payload)
    project_name = payload["project_name"]
    slug = _slugify_name(project_name)
    project_dir = projects_root() / slug
    if project_dir.exists():
        return OperationResult(False, f"Project `{slug}` already exists.", {"project_path": str(project_dir)})

    warnings: list[str] = []
    board_slug: Optional[str] = None
    try:
        project_dir.mkdir(parents=True, exist_ok=False)
        git_ok, git_msg = _git_init(project_dir)
        if not git_ok:
            warnings.append(f"git init: {git_msg}")

        tech_tokens = _parse_tech_stack(payload["tech_stack"])
        scope_items = _build_scope_items(payload)
        docs_fetched, doc_warnings = _fetch_context7_docs(tech_tokens, project_dir / "docs" / "context7")
        warnings.extend(doc_warnings)

        lang_ids = _serena_language_ids(tech_tokens)
        _write_serena_project_yml(project_dir, project_name, lang_ids)

        bd_ok, bd_msg = _run_beads_init(project_dir)
        if not bd_ok:
            warnings.append(f"bd init: {bd_msg}")

        orch = (orchestrator_profile or os.environ.get(PROJECT_ORCHESTRATOR_ENV) or DEFAULT_ORCHESTRATOR_PROFILE).strip()
        issues: dict[str, str] = {}
        if bd_ok:
            issues, issue_warnings, scope_items = _create_beads_plan(project_dir, payload, decisions, orch, scope_items)
            warnings.extend(issue_warnings)

        board_meta = kb.create_board(slug, name=project_name, description=payload["goal"])
        board_slug = board_meta.get("slug") or slug
        kanban_tasks = _create_kanban_tasks(project_dir, board_slug, payload, issues, orch)

        serena_ok, serena_msg = _run_serena_index(project_dir)
        if not serena_ok:
            warnings.append(f"serena index: {serena_msg}")

        _write_project_files(project_dir, payload, decisions, tech_tokens, docs_fetched, bd_ok, serena_ok, scope_items)

        project_memory_page = _update_project_memory(payload, project_dir, slug, orch, tech_tokens, issues, board_slug, decisions)

        manifest = {
            "project_name": project_name,
            "slug": slug,
            "project_path": str(project_dir),
            "created_at": int(time.time()),
            "status": "active",
            "board_slugs": [board_slug],
            "orchestrator_profile": orch,
            "project_memory_page": str(project_memory_page),
            "project_memory_root": str(project_memory_root()),
            "codebase_memory_project": slug,
            "tech_stack": tech_tokens,
            "serena_languages": lang_ids,
            "beads_issues": issues,
            "scope_items": scope_items,
            "scope_count": len(scope_items),
            "kanban_tasks": kanban_tasks,
            "decisions": decisions,
            "answers": {k: payload.get(k, "") for k in REQUIRED_FIELDS + OPTIONAL_FIELDS},
        }
        _save_manifest(project_dir, manifest)

        return OperationResult(True, "Project created.", {
            "project_name": project_name, "slug": slug, "project_path": str(project_dir),
            "board_slug": board_slug, "orchestrator_profile": orch,
            "tech_stack": tech_tokens, "serena_languages": lang_ids,
            "docs_fetched": docs_fetched, "bd_ok": bd_ok, "serena_ok": serena_ok,
            "project_memory_page": str(project_memory_page),
            "project_memory_root": str(project_memory_root()),
            "codebase_memory_project": slug,
            "beads_issues": issues, "scope_items": scope_items, "scope_count": len(scope_items), "kanban_tasks": kanban_tasks,
            "warnings": warnings, "decisions": decisions,
        })
    except Exception as exc:
        shutil.rmtree(project_dir, ignore_errors=True)
        if board_slug:
            try:
                kb.remove_board(board_slug, archive=False)
            except Exception:
                pass
        return OperationResult(False, f"Failed to create project: {exc}", {"error": str(exc)})


def _list_dirs(root: Path) -> list[Path]:
    return [c for c in sorted(root.iterdir(), key=lambda p: p.name.lower()) if c.is_dir()] if root.exists() else []


def list_active_projects() -> list[dict[str, str]]:
    _ensure_dirs()
    out = []
    for child in _list_dirs(projects_root()):
        if child.name in {"archived", "deleted"} or child.name.startswith("."):
            continue
        manifest = _load_manifest(child) or {}
        if manifest.get("status") == "deleted":
            continue
        out.append({"name": manifest.get("project_name") or child.name, "slug": child.name, "path": str(child), "archived": False})
    return out


def _find_project_dir(name: str) -> Optional[Path]:
    slug = _slugify_name(name)
    for candidate in [projects_root() / slug, archived_projects_root() / slug]:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _board_slugs_for_manifest(manifest: dict[str, Any], fallback: str) -> list[str]:
    raw = manifest.get("board_slugs")
    return [str(x) for x in raw if str(x).strip()] if isinstance(raw, list) and raw else [fallback]


def _remove_board_everywhere(slug: str, *, archive: bool) -> list[dict[str, Any]]:
    try:
        return [kb.remove_board(slug, archive=archive)]
    except Exception as exc:
        return [{"slug": slug, "error": str(exc), "archive": archive}]


def archive_project(name: str) -> OperationResult:
    _ensure_dirs()
    project_dir = _find_project_dir(name)
    if not project_dir:
        return OperationResult(False, f"Project `{name}` not found.", {})
    if project_dir.parent == archived_projects_root():
        return OperationResult(True, f"Project `{project_dir.name}` is already archived.", {"project_path": str(project_dir)})
    manifest = _load_manifest(project_dir) or {}
    slug = manifest.get("slug") or project_dir.name
    board_results: list[dict[str, Any]] = []
    for board in _board_slugs_for_manifest(manifest, slug):
        board_results.extend(_remove_board_everywhere(board, archive=True))
    target = archived_projects_root() / project_dir.name
    if target.exists():
        target = archived_projects_root() / f"{project_dir.name}-{int(time.time())}"
    shutil.move(str(project_dir), str(target))
    manifest.update({"status": "archived", "archived_at": int(time.time()), "project_path": str(target)})
    _save_manifest(target, manifest)
    return OperationResult(True, "Project archived.", {"project_name": manifest.get("project_name") or project_dir.name, "slug": slug, "project_path": str(target), "board_results": board_results})


def delete_project(name: str) -> OperationResult:
    _ensure_dirs()
    project_dir = _find_project_dir(name)
    if not project_dir:
        return OperationResult(False, f"Project `{name}` not found.", {})
    manifest = _load_manifest(project_dir) or {}
    slug = manifest.get("slug") or project_dir.name
    board_results: list[dict[str, Any]] = []
    for board in _board_slugs_for_manifest(manifest, slug):
        board_results.extend(_remove_board_everywhere(board, archive=True))
    target = deleted_projects_root() / project_dir.name
    if target.exists():
        target = deleted_projects_root() / f"{project_dir.name}-{int(time.time())}"
    shutil.move(str(project_dir), str(target))
    manifest.update({"status": "deleted", "deleted_at": int(time.time()), "project_path": str(target)})
    _save_manifest(target, manifest)
    return OperationResult(True, "Project moved to trash.", {"project_name": manifest.get("project_name") or project_dir.name, "slug": slug, "project_path": str(target), "board_results": board_results})


def _parse_delete_confirmation_args(raw_args: str) -> tuple[str, str]:
    text = (raw_args or "").strip()
    match = re.search(r"(?:^|\s)--confirm(?:=|\s+)(\S+)\s*$", text)
    if not match:
        return text, ""
    return text[:match.start()].strip(), match.group(1).strip()


def _delete_confirmation_result(raw_args: str) -> OperationResult:
    name, confirmation = _parse_delete_confirmation_args(raw_args)
    if not name:
        return OperationResult(False, "Project name is required before delete confirmation.", {})
    project_dir = _find_project_dir(name)
    if not project_dir:
        return OperationResult(False, f"Project `{name}` not found.", {})
    manifest = _load_manifest(project_dir) or {}
    slug = manifest.get("slug") or project_dir.name
    project_name = manifest.get("project_name") or project_dir.name
    if confirmation:
        if confirmation != slug:
            return OperationResult(False, f"Delete confirmation mismatch. To move this project to trash, run `/deleteproject {slug} --confirm {slug}`.", {"project_name": project_name, "slug": slug, "project_path": str(project_dir)})
        return delete_project(name)
    return OperationResult(False, f"Delete confirmation required. This will move `{project_name}` to trash and archive its Kanban boards. Run `/deleteproject {slug} --confirm {slug}` to confirm.", {"project_name": project_name, "slug": slug, "project_path": str(project_dir)})


def begin_new_project(session_key: str, initial_name: Optional[str] = None) -> dict[str, Any]:
    state = {"action": "newproject", "stage": "awaiting_intake", "initial_name": initial_name, "answers": {}, "created_at": int(time.time())}
    save_pending_state(session_key, state)
    return state


def _schedule_send(gateway: Any, event: Any, message: str) -> None:
    source = getattr(event, "source", None)
    if source is None:
        return
    adapter = getattr(gateway, "adapters", {}).get(getattr(source, "platform", None))
    chat_id = getattr(source, "chat_id", None)
    if adapter is None or not chat_id:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(adapter.send(chat_id, message))


def _build_new_project_reply(initial_name: Optional[str]) -> str:
    return "## New project intake started\n\n" + _format_prompt(initial_name)


def _build_choice_reply(kind: str, projects: list[dict[str, str]]) -> str:
    if not projects:
        return f"No active projects found for /{kind}."
    command = f"/{kind}project"
    instruction = (
        "Run `/deleteproject <slug>` to review the target, then confirm with `/deleteproject <slug> --confirm <slug>`."
        if kind == "delete"
        else f"Run `{command} <slug>` to {kind} a project."
    )
    return "\n".join([f"## {kind.title()} project", "", instruction, ""] + [f"- `{p['slug']}`" for p in projects])


def handle_gateway_command(event: Any, gateway: Any) -> Optional[str]:
    raw_text = (getattr(event, "text", "") or "").strip()
    command = (getattr(event, "get_command", lambda: "")() or "").lstrip("/").strip().lower()
    args = (getattr(event, "get_command_args", lambda: "")() or "").strip()
    session_key = _session_key(event, gateway)
    if not raw_text:
        return None

    if command in {"newproject", "createproject"}:
        parsed = parse_project_answers(args)
        if args and not parsed:
            parsed = _parse_flag_args(args) or {"project_name": args.strip()}
        initial_name = parsed.get("project_name") or None
        missing = [f for f in REQUIRED_FIELDS if not str(parsed.get(f, "")).strip()]
        if not missing:
            result = create_project(payload=parsed, initial_name=initial_name)
            _schedule_send(gateway, event, _render_operation_result("newproject", result))
            if result.ok:
                clear_pending_state(session_key)
            return "skip"
        state = begin_new_project(session_key, initial_name=initial_name)
        state["answers"] = parsed
        save_pending_state(session_key, state)
        _schedule_send(gateway, event, _build_new_project_reply(initial_name))
        return "skip"

    if command == "cancelproject":
        cancelled = cancel_pending_state(session_key)
        _schedule_send(gateway, event, "Current project draft cancelled." if cancelled else "No pending project draft to cancel.")
        return "skip"

    if command == "deleteproject":
        if args:
            result = _delete_confirmation_result(args)
            _schedule_send(gateway, event, _render_operation_result("deleteproject", result))
            return "skip"
        _schedule_send(gateway, event, _build_choice_reply("delete", list_active_projects()))
        return "skip"

    if command == "archiveproject":
        if args:
            result = archive_project(args)
            _schedule_send(gateway, event, _render_operation_result("archiveproject", result))
            return "skip"
        _schedule_send(gateway, event, _build_choice_reply("archive", list_active_projects()))
        return "skip"

    pending = pending_state(session_key)
    if not pending or raw_text.startswith("/"):
        return None
    if pending.get("action") == "newproject":
        answers = dict(pending.get("answers") or {})
        answers.update(parse_project_answers(raw_text))
        if pending.get("initial_name") and not answers.get("project_name"):
            answers["project_name"] = pending.get("initial_name")
        missing = [f for f in REQUIRED_FIELDS if not str(answers.get(f, "")).strip()]
        if missing:
            pending["answers"] = answers
            save_pending_state(session_key, pending)
            _schedule_send(gateway, event, "I still need: " + _format_missing(missing) + "\n\n" + _format_prompt(pending.get("initial_name")))
            return "skip"
        result = create_project(payload=answers, initial_name=pending.get("initial_name"))
        if result.ok:
            clear_pending_state(session_key)
        _schedule_send(gateway, event, _render_operation_result("newproject", result))
        return "skip"
    return None


def _render_operation_result(kind: str, result: OperationResult) -> str:
    if not result.ok:
        return f"{kind}: {result.message}"
    data = result.data
    if kind == "newproject":
        warnings = data.get("warnings") or []
        warn_block = ("\n\n**Bootstrap warnings:**\n" + "\n".join(f"- {w}" for w in warnings)) if warnings else ""
        decisions = data.get("decisions") or []
        decision_block = ("\n\n**Defaults recorded:**\n" + "\n".join(f"- {d}" for d in decisions[:8])) if decisions else ""
        return (
            "## Project created\n\n"
            f"- Project: `{data.get('project_name')}`\n"
            f"- Folder: `{data.get('project_path')}`\n"
            f"- Board: `{data.get('board_slug')}`\n"
            f"- Beads issues: {len(data.get('beads_issues') or {})}\n"
            f"- Scope items: {data.get('scope_count') or len(data.get('scope_items') or [])}\n"
            f"- Kanban tasks: {len(data.get('kanban_tasks') or {})}\n"
            f"- Tech stack: {', '.join(data.get('tech_stack') or []) or 'unspecified'}\n"
            f"- Beads: {'✅' if data.get('bd_ok') else '⚠️'}\n"
            f"- Serena index: {'✅' if data.get('serena_ok') else '⚠️'}\n"
            "\nAutonomous execution rules were written to `AGENTS.md`; scope gates were written to `scope-ledger.md` and `completion-audit.md`."
            f"{decision_block}{warn_block}"
        )
    if kind == "archiveproject":
        return f"## Project archived\n\n- Project: `{data.get('project_name')}`\n- Archived path: `{data.get('project_path')}`"
    if kind == "deleteproject":
        return f"## Project moved to trash\n\n- Project: `{data.get('project_name')}`\n- Trash path: `{data.get('project_path')}`\n- Slug: `{data.get('slug')}`"
    return result.message


def handle_direct_command(kind: str, raw_args: str, *, orchestrator_profile: Optional[str] = None) -> str:
    raw_args = (raw_args or "").strip()
    if kind in {"newproject", "createproject"}:
        payload = parse_project_answers(raw_args)
        if raw_args and not payload:
            payload = _parse_flag_args(raw_args) or {"project_name": raw_args}
        result = create_project(payload=payload, initial_name=payload.get("project_name"), orchestrator_profile=orchestrator_profile)
        if result.ok:
            return _render_operation_result("newproject", result)
        return result.message + "\n\n" + _format_prompt(payload.get("project_name"))
    if kind == "archiveproject":
        return _render_operation_result(kind, archive_project(raw_args)) if raw_args else _build_choice_reply("archive", list_active_projects())
    if kind == "deleteproject":
        return _render_operation_result(kind, _delete_confirmation_result(raw_args)) if raw_args else _build_choice_reply("delete", list_active_projects())
    if kind == "cancelproject":
        return "Project draft cancellation is handled in gateway chats. Use /cancelproject there."
    return "Unsupported command"
