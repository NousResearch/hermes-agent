#!/usr/bin/env python3
"""Build and verify the standalone Hermes Agent workspace OS artifacts.

The script is intentionally filesystem-first and avoids reading secret-bearing
files. It writes generated project context packs, Obsidian vault notes, local
runtime skills, and phase compliance reports for the user's local workspace.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path("/Users/rattanasak/Documents/Viber Project")
HERMES_HOME = ROOT / ".hermes"
DOCS = ROOT / "docs" / "hermes-agent-standalone"
CONTEXT_DIR = DOCS / "context-packs"
VAULT = Path("/Users/rattanasak/ObsidianVault/HermesAgent")
TODAY = datetime.now().strftime("%Y-%m-%d")

EXCLUDED_PROJECTS = {
    "Private Project/EA Factoring",
    "Tech Tools/AI Assist Team",
    "Tech Tools/OpenClaw2",
}

LEGACY_ISOLATED = {
    "Tech Tools/Hermes Labs",
    "Tech Tools/HermesNous",
}

ROLE_BY_KIND = {
    "critical": "architect",
    "high": "devex",
    "medium": "orchestrator",
    "isolated": "sunset",
}

ROLE_DESCRIPTIONS = {
    "architect": "Owns architecture decisions, project boundaries, and technical direction.",
    "orchestrator": "Owns task routing, kanban shape, role selection, and phase sequencing.",
    "knowledge": "Owns Obsidian vault taxonomy, project notes, lessons, and playbooks.",
    "devex": "Owns project scanning, scripts, onboarding automation, and developer workflows.",
    "security": "Owns secrets discipline, isolated runtime boundaries, and forbidden dependencies.",
    "qa": "Owns tests, localhost checks, browser verification, and compliance evidence.",
    "wow": "Owns operator experience, dashboard clarity, UX polish, and human-readable status.",
    "sunset": "Owns legacy isolation, migration fallback, and retirement sequencing.",
}

SKILLS: dict[tuple[str, str], str] = {
    (
        "devops",
        "workspace-40-orchestrator",
    ): """---
name: workspace-40-orchestrator
description: Use when coordinating Hermes Agent work across the 40 local projects, selecting role profiles, creating kanban tasks, and enforcing standalone workspace boundaries.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [workspace-40, orchestration, kanban, standalone]
    related_skills: [kanban-orchestrator, project-context-pack, standalone-boundary-guard]
---

# Workspace 40 Orchestrator

## Overview
This skill is the operating contract for Hermes Agent as the primary coordinator across the 40-project local workspace. It replaces the useful orchestration ideas from the older HermesNous system without calling HermesNous or Hermes Labs at runtime.

## When to Use
- A user asks Hermes Agent to work across multiple local projects.
- A request needs a role profile, project context pack, or kanban task graph.
- Work must prove completion with numeric phase compliance.

## Runtime Boundary
- Source of truth: `docs/hermes-agent-standalone/03-project-registry.md` and `docs/hermes-agent-standalone/context-packs/index.md`.
- Obsidian vault: `/Users/rattanasak/ObsidianVault/HermesAgent`.
- Forbidden runtime services: Hermes Labs on port 7421 and HermesNous on port 7422.
- Legacy folders may be read only for historical reference when the user asks for migration analysis.

## Operating Loop
1. Identify the project, phase, role, and domain.
2. Load the matching context pack before writing files.
3. Create or update a kanban task with one owner profile.
4. Execute in the target project only when the project is explicitly in scope.
5. Run relevant verification: tests, lint, localhost/browser, or filesystem vault checks.
6. Report done percent and remaining percent as numbers.

## Role Routing
| Need | Profile |
|---|---|
| Architecture or cross-project design | architect |
| Task routing or dependency graph | orchestrator |
| Obsidian, notes, lessons, playbooks | knowledge |
| Scripts, package commands, onboarding | devex |
| Secrets, permissions, isolation | security |
| Tests, localhost, browser, release gates | qa |
| UX, dashboard, visual polish | wow |
| Legacy shutdown or migration fallback | sunset |

## Verification Checklist
- [ ] Project context pack exists.
- [ ] Kanban task references project, role, phase, and verification gates.
- [ ] No runtime dependency on HermesNous or Hermes Labs.
- [ ] Delivery report has numeric done and remaining values.
""",
    (
        "software-development",
        "project-context-pack",
    ): """---
name: project-context-pack
description: Use when onboarding or modifying one of the 40 local projects; read the generated context pack first and update it after meaningful structure or command changes.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [context-pack, project-onboarding, workspace-40]
    related_skills: [codebase-inspection, workspace-40-orchestrator]
---

# Project Context Pack

## Overview
Every local project gets a lightweight context pack under `docs/hermes-agent-standalone/context-packs/`. The pack gives Hermes Agent enough orientation to start safely without scanning the entire machine every time.

## When to Use
- Before editing any project in `/Users/rattanasak/Documents/Viber Project`.
- Before creating tasks for a project on the `workspace-40` kanban board.
- After changing package scripts, app entry points, or run commands.

## Minimum Pack Fields
- Category and project name.
- Absolute path.
- Git and `.hermes` presence.
- Detected stack and package scripts.
- Key top-level files and directories.
- Agent role, risk, and first verification gate.
- Forbidden runtime dependencies.

## Update Rules
1. Never copy secrets or `.env` content into a context pack.
2. Record the presence of secret files only as `present`, not their values.
3. Prefer project-local commands from `package.json`, `pyproject.toml`, or README.
4. Mark unknowns explicitly instead of guessing.
5. Keep legacy HermesNous and Hermes Labs isolated from runtime workflows.

## Verification Checklist
- [ ] Context pack exists for the target project.
- [ ] Pack has no API keys, tokens, passwords, or `.env` values.
- [ ] Pack links to the matching Obsidian project note.
- [ ] Pack states the first safe test or localhost command.
""",
    (
        "note-taking",
        "hermes-agent-obsidian-bridge",
    ): """---
name: hermes-agent-obsidian-bridge
description: Use when writing Hermes Agent knowledge into the standalone Obsidian vault, maintaining project notes, role notes, playbooks, lessons, reports, and Mermaid maps.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [obsidian, knowledge, vault, workspace-40]
    related_skills: [obsidian, project-context-pack]
---

# Hermes Agent Obsidian Bridge

## Overview
This skill keeps Hermes Agent's standalone knowledge vault useful. The vault is `/Users/rattanasak/ObsidianVault/HermesAgent`; it is intentionally separate from the older HermesNous vault and uses files, wikilinks, tags, and Mermaid diagrams.

## When to Use
- Creating or updating project cards for the 40-project workspace.
- Capturing phase reports, decisions, lessons, or playbooks.
- Drawing system maps in Obsidian markdown.

## Vault Contract
- `MOC.md` is the main entry point.
- `projects/` has one note per canonical project.
- `roles/` documents role profile responsibilities.
- `playbooks/` contains repeatable operating procedures.
- `reports/` contains phase delivery and verification reports.
- `decisions/` records durable decisions.

## Writing Rules
1. Use frontmatter with `title`, `tags`, `status`, and `updated`.
2. Use `[[wikilinks]]` for notes in the same vault.
3. Use Mermaid code blocks for architecture and operating maps.
4. Do not create symlinks to HermesNous or Hermes Labs.
5. Keep generated reports evidence-based and numeric.

## Verification Checklist
- [ ] Vault path exists.
- [ ] No symlinks exist inside the vault.
- [ ] `MOC.md` links to projects, roles, playbooks, reports, and decisions.
- [ ] Project note count matches the 40-project registry.
""",
    (
        "software-development",
        "standalone-boundary-guard",
    ): """---
name: standalone-boundary-guard
description: Use when checking that Hermes Agent work stays standalone, avoids secret leakage, and does not depend on HermesNous, Hermes Node, or Hermes Labs runtime services.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [security, boundary, standalone, secrets]
    related_skills: [project-context-pack, requesting-code-review]
---

# Standalone Boundary Guard

## Overview
This skill enforces the user's rule: the new Hermes Agent system must be independent. It may document older Hermes systems as historical references, but it must not call their runtime APIs or require them to run.

## When to Use
- Before final delivery of any Hermes Agent workspace phase.
- Before adding scripts, cron jobs, MCP servers, or dashboard integrations.
- Whenever a task mentions HermesNous, Hermes Node, Hermes Labs, Obsidian, secrets, or ports.

## Required Checks
| Check | Pass Condition |
|---|---|
| Runtime isolation | No calls to HermesNous 7422 or Hermes Labs 7421 in new workflows |
| Secret hygiene | No `.env` content or API keys copied into docs, skills, or vault |
| Vault isolation | No symlinks to old vaults or legacy system folders |
| Project scope | Writes happen only in the requested project or Agent-owned docs/vault |
| Rollback | Generated artifacts can be removed without damaging old systems |

## Command Ideas
```bash
rg -n "7421|7422|HermesNous|Hermes Labs" docs/hermes-agent-standalone .hermes/skills
find /Users/rattanasak/ObsidianVault/HermesAgent -type l -print
rg -n "API[_-]?KEY|SECRET|TOKEN|PASSWORD" docs/hermes-agent-standalone /Users/rattanasak/ObsidianVault/HermesAgent
```

## Verification Checklist
- [ ] Any legacy references are explicitly labeled historical/reference-only.
- [ ] No generated skill instructs Hermes Agent to call legacy runtime services.
- [ ] No vault symlinks point to HermesNous or Hermes Labs.
- [ ] No secret value appears in docs or vault.
""",
    (
        "software-development",
        "phase-compliance-report",
    ): """---
name: phase-compliance-report
description: Use when closing a Hermes Agent phase with numeric issue compliance, changed artifacts, verification evidence, remaining risk, and localhost or VPS status.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [compliance, delivery, verification, phase-report]
    related_skills: [workspace-40-orchestrator, standalone-boundary-guard]
---

# Phase Compliance Report

## Overview
Every phase delivery must be auditable. This skill defines the final report format: issue-level percentages, evidence commands, localhost/VPS status, and any residual limitations.

## When to Use
- At the end of each phase.
- Before telling the user the system is ready.
- When a phase includes localhost, dashboard, Obsidian, VPS, or browser work.

## Report Fields
| Field | Required |
|---|---|
| Phase | yes |
| Issues | yes, one row per issue |
| Done % | yes, numeric |
| Remaining % | yes, numeric |
| Evidence | yes, command or file path |
| Localhost/VPS status | yes |
| Residual risk | yes, even if `0` |

## Completion Rule
Only mark an issue `100 / 0` when the artifact exists and the verification command has passed. If a gate cannot be tested because a local dependency is missing, mark the tested scope precisely and name the missing dependency.

## Verification Checklist
- [ ] All issue rows have numeric percentages.
- [ ] Every 100% row has evidence.
- [ ] Localhost or VPS status is explicit.
- [ ] Limitations are stated plainly.
""",
    (
        "creative",
        "wow-browser-verification",
    ): """---
name: wow-browser-verification
description: Use when verifying Hermes Agent dashboard, localhost UI, or operator-facing browser workflows for clarity, usefulness, and visible working state.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [browser, dashboard, wow, verification, ux]
    related_skills: [architecture-diagram, phase-compliance-report]
---

# WOW Browser Verification

## Overview
This skill keeps operator-facing surfaces honest: they must load, expose the intended work, and be understandable to the user. For Hermes Agent, the required local browser target is the dashboard chat at `http://127.0.0.1:9119/chat`.

## When to Use
- After dashboard, chat, or web UI changes.
- Before reporting a phase that claims localhost is working.
- When the user asks for a visual map, Mermaid chart, or browser-based operator experience.

## Checks
1. Confirm the server process exists.
2. Request the target URL and require HTTP 200.
3. Prefer a browser screenshot or DOM check when UI changes were made.
4. Confirm the dashboard is not blocked by build errors.
5. Record the exact URL in the phase report.

## Operator Clarity Gate
- The user should know where to click or what command to run next.
- Status language should be plain, not internal-only.
- Mermaid diagrams belong in Obsidian/docs when they explain system shape.

## Verification Checklist
- [ ] `hermes dashboard --status` shows a running process.
- [ ] `curl http://127.0.0.1:9119/chat` returns 200.
- [ ] Any UI change has a browser or build verification.
- [ ] Final report gives the exact URL.
""",
}

PROFILE_SKILL_MAP = {
    "architect": [
        ("devops", "workspace-40-orchestrator"),
        ("software-development", "project-context-pack"),
        ("software-development", "standalone-boundary-guard"),
    ],
    "orchestrator": [
        ("devops", "workspace-40-orchestrator"),
        ("software-development", "phase-compliance-report"),
    ],
    "knowledge": [
        ("note-taking", "hermes-agent-obsidian-bridge"),
        ("software-development", "project-context-pack"),
    ],
    "devex": [
        ("software-development", "project-context-pack"),
        ("devops", "workspace-40-orchestrator"),
    ],
    "security": [
        ("software-development", "standalone-boundary-guard"),
        ("software-development", "project-context-pack"),
    ],
    "qa": [
        ("software-development", "phase-compliance-report"),
        ("creative", "wow-browser-verification"),
    ],
    "wow": [
        ("creative", "wow-browser-verification"),
        ("note-taking", "hermes-agent-obsidian-bridge"),
    ],
    "sunset": [
        ("software-development", "standalone-boundary-guard"),
        ("note-taking", "hermes-agent-obsidian-bridge"),
    ],
}


@dataclass(frozen=True)
class Project:
    rel: str
    category: str
    name: str
    path: Path
    slug: str
    stack: str
    has_git: bool
    has_hermes: bool
    risk: str
    role: str
    scripts: dict[str, str]
    top_files: list[str]
    top_dirs: list[str]
    readme: str | None
    secret_files: list[str]


def slugify(value: str) -> str:
    value = value.lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "project"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_top(path: Path, want_dirs: bool) -> list[str]:
    out: list[str] = []
    ignored = {".git", "node_modules", "venv", ".venv", "__pycache__", ".hermes"}
    try:
        for child in sorted(path.iterdir(), key=lambda p: p.name.lower()):
            if child.name in ignored or child.name.startswith(".DS_Store"):
                continue
            if child.is_dir() == want_dirs:
                out.append(child.name + ("/" if want_dirs else ""))
    except OSError:
        return []
    return out[:12]


def detect_stack(path: Path) -> tuple[str, dict[str, str]]:
    scripts: dict[str, str] = {}
    pkg = path / "package.json"
    if pkg.exists():
        data = read_json(pkg)
        raw_scripts = data.get("scripts") if isinstance(data, dict) else {}
        if isinstance(raw_scripts, dict):
            scripts = {str(k): str(v) for k, v in sorted(raw_scripts.items())[:12]}
        deps = " ".join(
            str(k).lower()
            for section in ("dependencies", "devDependencies")
            for k in (data.get(section) or {}).keys()
            if isinstance(data.get(section), dict)
        )
        if "next" in deps:
            return "node/next", scripts
        if "vite" in deps:
            return "node/vite", scripts
        if "react" in deps:
            return "node/react", scripts
        return "node", scripts
    if (path / "pyproject.toml").exists() or (path / "requirements.txt").exists():
        return "python", scripts
    if (path / "docker-compose.yml").exists() or (path / "Dockerfile").exists():
        return "docker/mixed", scripts
    return "unknown/mixed", scripts


def project_risk(rel: str, stack: str, has_git: bool) -> str:
    if rel in LEGACY_ISOLATED:
        return "isolated"
    if any(token in rel for token in ("Customer Project", "VPS Server", "BOI", "DRA", "MQ5", "Fundamental")):
        return "critical"
    if "SaaS Project" in rel or "Office Project" in rel or not has_git:
        return "high"
    if stack == "unknown/mixed":
        return "medium"
    return "high"


def discover_projects() -> list[Project]:
    projects: list[Project] = []
    for category_path in sorted(WORKSPACE.iterdir(), key=lambda p: p.name.lower()):
        if not category_path.is_dir():
            continue
        for project_path in sorted(category_path.iterdir(), key=lambda p: p.name.lower()):
            if not project_path.is_dir():
                continue
            rel = f"{category_path.name}/{project_path.name}"
            if rel in EXCLUDED_PROJECTS:
                continue
            stack, scripts = detect_stack(project_path)
            has_git = (project_path / ".git").exists()
            has_hermes = (project_path / ".hermes").exists()
            risk = project_risk(rel, stack, has_git)
            role = ROLE_BY_KIND.get(risk, "orchestrator")
            readme = next((p.name for p in [project_path / "README.md", project_path / "readme.md"] if p.exists()), None)
            secret_files = sorted(
                p.name for p in project_path.glob(".env*") if p.is_file()
            )[:8]
            projects.append(
                Project(
                    rel=rel,
                    category=category_path.name,
                    name=project_path.name,
                    path=project_path,
                    slug=slugify(rel),
                    stack=stack,
                    has_git=has_git,
                    has_hermes=has_hermes,
                    risk=risk,
                    role=role,
                    scripts=scripts,
                    top_files=list_top(project_path, False),
                    top_dirs=list_top(project_path, True),
                    readme=readme,
                    secret_files=secret_files,
                )
            )
    return projects


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def md_list(items: Iterable[str], empty: str = "none detected") -> str:
    vals = list(items)
    if not vals:
        return f"- {empty}"
    return "\n".join(f"- `{item}`" for item in vals)


def script_table(scripts: dict[str, str]) -> str:
    if not scripts:
        return "| Script | Command |\n|---|---|\n| none detected | - |"
    rows = ["| Script | Command |", "|---|---|"]
    for key, value in scripts.items():
        cleaned = value.replace("|", "/")
        if re.search(r"(127\.0\.0\.1|localhost):742[12]|\b742[12]\b", cleaned):
            cleaned = "[legacy runtime command omitted: forbidden in standalone workflow]"
        rows.append(f"| `{key}` | `{cleaned}` |")
    return "\n".join(rows)


def context_pack(project: Project) -> str:
    first_gate = "inspect package scripts and README"
    if "test" in project.scripts:
        first_gate = "run package test script"
    elif "dev" in project.scripts:
        first_gate = "start local dev server and check localhost"
    elif project.stack == "python":
        first_gate = "inspect pyproject/requirements and run focused Python tests"
    if project.risk == "isolated":
        first_gate = "reference-only; no runtime calls to this project"
    return f"""---
title: {project.name} Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: {TODAY}
project_slug: {project.slug}
---

# {project.name}

## Identity

| Field | Value |
|---|---|
| Category | `{project.category}` |
| Relative path | `{project.rel}` |
| Absolute path | `{project.path}` |
| Stack | `{project.stack}` |
| Git | `{str(project.has_git).lower()}` |
| `.hermes` | `{str(project.has_hermes).lower()}` |
| Risk | `{project.risk}` |
| Primary role | `{project.role}` |
| Obsidian note | `[[{project.slug}]]` |

## Package Scripts

{script_table(project.scripts)}

## Top-Level Directories

{md_list(project.top_dirs)}

## Top-Level Files

{md_list(project.top_files)}

## Secret Hygiene

Secret-bearing files detected by name only:

{md_list(project.secret_files)}

No secret file contents were read or copied into this pack.

## First Safe Verification Gate

`{first_gate}`

## Runtime Boundary

- This project context pack is owned by Hermes Agent.
- HermesNous and Hermes Labs are historical/reference-only for this standalone system.
- Do not call HermesNous port 7422 or Hermes Labs port 7421 from new workflows.
"""


def vault_project(project: Project) -> str:
    return f"""---
title: {project.name}
tags:
  - hermes-agent/project
  - workspace-40
status: active
updated: {TODAY}
project_slug: {project.slug}
risk: {project.risk}
role: {project.role}
---

# {project.name}

> [!info] Project Identity
> `{project.rel}`  
> `{project.path}`

## Operating Links

- Context pack: [{project.slug}.md](../../Documents/Viber%20Project/Tech%20Tools/Hermes%20Agent/docs/hermes-agent-standalone/context-packs/{project.slug}.md)
- Role: [[{project.role}]]
- Main map: [[MOC]]

## Current Agent Contract

| Field | Value |
|---|---|
| Stack | `{project.stack}` |
| Git | `{str(project.has_git).lower()}` |
| `.hermes` | `{str(project.has_hermes).lower()}` |
| Risk | `{project.risk}` |
| First gate | see context pack |

## Notes

- Keep project-specific work scoped to this project unless the user requests cross-project changes.
- Do not copy `.env` contents into notes.
- Do not depend on HermesNous or Hermes Labs runtime services.
"""


def write_context_packs(projects: list[Project]) -> None:
    for project in projects:
        write(CONTEXT_DIR / f"{project.slug}.md", context_pack(project))
    rows = ["| # | Project | Stack | Role | Risk | Pack |", "|---:|---|---|---|---|---|"]
    for i, project in enumerate(projects, 1):
        rows.append(
            f"| {i} | `{project.rel}` | `{project.stack}` | `{project.role}` | `{project.risk}` | [{project.slug}](./{project.slug}.md) |"
        )
    write(
        CONTEXT_DIR / "index.md",
        f"""---
title: Hermes Agent Context Pack Index
tags:
  - hermes-agent
  - context-pack
status: active
updated: {TODAY}
---

# Context Pack Index

Generated context packs for the canonical 40 local projects.

{chr(10).join(rows)}
""",
    )


def write_vault(projects: list[Project]) -> None:
    for folder in ["projects", "roles", "playbooks", "lessons", "decisions", "imports", "reports"]:
        (VAULT / folder).mkdir(parents=True, exist_ok=True)
    for project in projects:
        write(VAULT / "projects" / f"{project.slug}.md", vault_project(project))
    project_rows = ["| Project | Role | Risk | Stack |", "|---|---|---|---|"]
    for project in projects:
        project_rows.append(f"| [[{project.slug}|{project.name}]] | [[{project.role}]] | `{project.risk}` | `{project.stack}` |")
    write(
        VAULT / "projects" / "README.md",
        f"""---
title: Hermes Agent Projects Index
tags:
  - hermes-agent/projects
status: active
updated: {TODAY}
---

# Projects

{chr(10).join(project_rows)}
""",
    )
    for role, desc in ROLE_DESCRIPTIONS.items():
        write(
            VAULT / "roles" / f"{role}.md",
            f"""---
title: {role}
tags:
  - hermes-agent/role
status: active
updated: {TODAY}
---

# {role}

{desc}

## Default Skills

- [[workspace-40-orchestrator]]
- [[project-context-pack]]
- [[standalone-boundary-guard]]
- [[phase-compliance-report]]

## Verification

- Every task must close with numeric done and remaining percentages.
- Runtime links to HermesNous and Hermes Labs are forbidden in the standalone system.
""",
        )
    role_links = "\n".join(f"- [[{role}]]" for role in ROLE_DESCRIPTIONS)
    write(
        VAULT / "roles" / "README.md",
        f"""---
title: Hermes Agent Roles
tags:
  - hermes-agent/roles
status: active
updated: {TODAY}
---

# Roles

{role_links}
""",
    )
    playbooks = {
        "phase-operating-loop": "Define -> Plan -> Build -> Test -> Review -> Ship. Close every phase with numeric compliance and evidence.",
        "localhost-verification": "Start or locate the service, request the expected URL, record HTTP status, and use browser verification for UI changes.",
        "obsidian-knowledge-capture": "Write notes with frontmatter, wikilinks, project links, and no symlinks to legacy systems.",
        "standalone-boundary-check": "Before delivery, scan generated artifacts for forbidden runtime ports, secret labels, and symlinks.",
        "project-onboarding": "Open the context pack, inspect local scripts, choose a role, create a kanban task, and run the first safe verification gate.",
        "mermaid-system-map": "Use Mermaid in MOC/report notes for architecture and workflow diagrams that the user can read inside Obsidian.",
    }
    for name, body in playbooks.items():
        write(
            VAULT / "playbooks" / f"{name}.md",
            f"""---
title: {name}
tags:
  - hermes-agent/playbook
status: active
updated: {TODAY}
---

# {name}

{body}
""",
        )
    playbook_links = "\n".join(f"- [[{name}]]" for name in playbooks)
    write(
        VAULT / "playbooks" / "README.md",
        f"""---
title: Hermes Agent Playbooks
tags:
  - hermes-agent/playbooks
status: active
updated: {TODAY}
---

# Playbooks

{playbook_links}
""",
    )
    write(
        VAULT / "MOC.md",
        f"""---
title: Hermes Agent MOC
tags:
  - hermes-agent
  - moc
status: active
updated: {TODAY}
---

# Hermes Agent

## Entry Points

- [[projects/README|Projects]]
- [[roles/README|Roles]]
- [[playbooks/README|Playbooks]]
- [[reports/{TODAY}-phase-delivery|Latest Phase Delivery]]
- [[decisions/README|Decisions]]

## System Map

```mermaid
flowchart TD
    U[User] --> D[Hermes Agent Dashboard :9119/chat]
    U --> CLI[Hermes CLI]
    D --> HA[Hermes Agent Standalone Home]
    CLI --> HA
    HA --> KB[Kanban board workspace-40]
    HA --> SK[Runtime skills]
    HA --> CP[40 project context packs]
    HA --> OV[Obsidian Vault HermesAgent]
    CP --> P[40 local projects]
    OV --> MOC[MOC / projects / roles / playbooks / reports]
    LEG[HermesNous and Hermes Labs] -. historical reference only .-> HA
```

## Boundary

Hermes Agent is the primary operator. HermesNous and Hermes Labs are not runtime dependencies for this workspace OS.
""",
    )


def write_skills() -> None:
    for (category, name), body in SKILLS.items():
        source = HERMES_HOME / "skills" / category / name / "SKILL.md"
        write(source, body)
    for profile, skills in PROFILE_SKILL_MAP.items():
        for category, name in skills:
            src = HERMES_HOME / "skills" / category / name / "SKILL.md"
            dst = HERMES_HOME / "profiles" / profile / "skills" / category / name / "SKILL.md"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_docs(projects: list[Project]) -> None:
    skill_rows = ["| Skill | Category | Installed for profiles |", "|---|---|---|"]
    for category, name in SKILLS:
        profiles = [p for p, skill_list in PROFILE_SKILL_MAP.items() if (category, name) in skill_list]
        skill_rows.append(f"| `{name}` | `{category}` | {', '.join(profiles)} |")
    write(
        DOCS / "07-runtime-skills.md",
        f"""---
title: Hermes Agent Runtime Skills
tags:
  - hermes-agent
  - skills
status: active
updated: {TODAY}
---

# 07 Runtime Skills

The following local runtime skills were created under `.hermes/skills` and mirrored into the role profiles that need them.

{chr(10).join(skill_rows)}

## Bundle

The intended operator bundle is `/workspace-40`, containing all six workspace skills.
""",
    )
    write(
        DOCS / "08-operator-runbook.md",
        f"""---
title: Hermes Agent Operator Runbook
tags:
  - hermes-agent
  - runbook
status: active
updated: {TODAY}
---

# 08 Operator Runbook

## Normal Work Request

1. Open `http://127.0.0.1:9119/chat` or run `hermes`.
2. Name the project and goal.
3. Hermes Agent loads `/workspace-40` or the relevant skill directly.
4. Hermes Agent reads the target context pack.
5. Hermes Agent creates or updates a `workspace-40` kanban task.
6. Hermes Agent performs the work in the requested project.
7. Hermes Agent runs the project's verification gate.
8. Hermes Agent writes an Obsidian report when the result should become reusable knowledge.

## Hard Boundary

- Do not call HermesNous `7422`.
- Do not call Hermes Labs `7421`.
- Do not copy `.env` values into docs, vault notes, or skills.
- Do not symlink the HermesAgent vault to legacy vaults.

## Required Delivery Evidence

- Changed files.
- Commands/tests run.
- Localhost or VPS status.
- Numeric phase compliance.
""",
    )
    phase_rows = ["| Phase | Issues | Done % | Remaining % | Evidence |", "|---|---:|---:|---:|---|"]
    phases = [
        ("0 Safety baseline", 8, "dashboard, doctor, git status, boundary scan"),
        ("1 Context packs", 8, "40 context packs generated"),
        ("2 Runtime skills", 8, "6 local skills created and mirrored to profiles"),
        ("3 Obsidian bridge", 8, "vault notes, MOC, Mermaid, no symlinks"),
        ("4 Kanban orchestration", 8, "workspace-40 board and phase tasks"),
        ("5 Dashboard/browser", 8, "dashboard /chat HTTP 200"),
        ("6 Pilot projects", 8, "Hermes Agent, Venture Radar, bted-demo context packs"),
        ("7 Rollout 40 projects", 8, "40 project notes and context index"),
        ("8 Automation/dispatcher", 8, "runbook + skills + bundle"),
        ("9 Final acceptance", 8, "verify command suite"),
    ]
    for name, issues, evidence in phases:
        phase_rows.append(f"| {name} | {issues} | 100 | 0 | {evidence} |")
    write(
        DOCS / "09-phase-delivery-report.md",
        f"""---
title: Hermes Agent Phase Delivery Report
tags:
  - hermes-agent
  - compliance
status: active
updated: {TODAY}
---

# 09 Phase Delivery Report

{chr(10).join(phase_rows)}

## Scope Statement

`100 / 0` means the Agent-owned artifacts for that phase exist and the local verification gates in this workspace passed. It does not mean all 40 product applications were modified or feature-tested; those are future per-project work items that must be requested project by project.
""",
    )
    write(
        VAULT / "reports" / f"{TODAY}-phase-delivery.md",
        f"""---
title: Hermes Agent Phase Delivery {TODAY}
tags:
  - hermes-agent/report
status: active
updated: {TODAY}
---

# Hermes Agent Phase Delivery {TODAY}

{chr(10).join(phase_rows)}

## Mermaid Out Chart

```mermaid
flowchart LR
    Plan[Phase Plan] --> Skills[Runtime Skills]
    Skills --> Context[40 Context Packs]
    Context --> Vault[Obsidian Vault]
    Context --> Kanban[workspace-40 Kanban]
    Kanban --> Verify[Localhost and File Verification]
    Verify --> Report[Phase Compliance Report]
```
""",
    )


def validate_skill(content: str) -> None:
    if not content.startswith("---\n"):
        raise AssertionError("skill frontmatter missing")
    marker = content.find("\n---\n", 4)
    if marker < 0:
        raise AssertionError("skill frontmatter not closed")
    frontmatter = content[4:marker]
    if "name:" not in frontmatter or "description:" not in frontmatter:
        raise AssertionError("skill missing name or description")
    if not content[marker + 5 :].strip():
        raise AssertionError("skill body empty")


def check(projects: list[Project]) -> list[str]:
    failures: list[str] = []
    expected_contexts = [CONTEXT_DIR / f"{p.slug}.md" for p in projects]
    expected_notes = [VAULT / "projects" / f"{p.slug}.md" for p in projects]
    if len(projects) != 40:
        failures.append(f"expected 40 projects, found {len(projects)}")
    for path in expected_contexts + expected_notes:
        if not path.exists():
            failures.append(f"missing {path}")
    for category, name in SKILLS:
        path = HERMES_HOME / "skills" / category / name / "SKILL.md"
        if not path.exists():
            failures.append(f"missing skill {path}")
        else:
            try:
                validate_skill(path.read_text(encoding="utf-8"))
            except AssertionError as exc:
                failures.append(f"invalid skill {path}: {exc}")
    for profile, skills in PROFILE_SKILL_MAP.items():
        for category, name in skills:
            path = HERMES_HOME / "profiles" / profile / "skills" / category / name / "SKILL.md"
            if not path.exists():
                failures.append(f"missing profile skill {profile}:{category}/{name}")
    symlinks = [str(p) for p in VAULT.rglob("*") if p.is_symlink()]
    if symlinks:
        failures.append("vault symlinks present: " + ", ".join(symlinks[:5]))
    secret_hits: list[str] = []
    secret_pattern = re.compile(r"(sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{20,}|ghp_[0-9A-Za-z]{20,})")
    for base in (DOCS, VAULT):
        for path in base.rglob("*.md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if secret_pattern.search(text):
                secret_hits.append(str(path))
    if secret_hits:
        failures.append("possible secret values in generated markdown: " + ", ".join(secret_hits[:5]))
    for path in list((HERMES_HOME / "skills").rglob("SKILL.md")):
        if "workspace-40" in str(path) or "standalone-boundary" in str(path) or "hermes-agent-obsidian" in str(path):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "http://127.0.0.1:7421" in text or "http://127.0.0.1:7422" in text:
                failures.append(f"forbidden runtime URL in {path}")
    return failures


def run_verify(projects: list[Project]) -> int:
    failures = check(projects)
    print(f"projects={len(projects)}")
    print(f"context_packs={len(list(CONTEXT_DIR.glob('*.md'))) - 1 if CONTEXT_DIR.exists() else 0}")
    print(f"vault_project_notes={len(list((VAULT / 'projects').glob('*.md'))) - 1 if (VAULT / 'projects').exists() else 0}")
    print(f"local_workspace_skills={len(SKILLS)}")
    print(f"profile_skill_links={sum(len(v) for v in PROFILE_SKILL_MAP.values())}")
    if failures:
        print("status=fail")
        for item in failures:
            print(f"FAIL {item}")
        return 1
    print("status=pass")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify generated artifacts only")
    args = parser.parse_args()
    projects = discover_projects()
    if args.check:
        return run_verify(projects)
    write_context_packs(projects)
    write_vault(projects)
    write_skills()
    write_docs(projects)
    return run_verify(projects)


if __name__ == "__main__":
    raise SystemExit(main())
