#!/usr/bin/env python3
"""Operate the repo-local multi-AI workflow files.

This CLI is intentionally file-first. It creates and updates project-local
Markdown state so any AI tool can continue the same work without shared chat
memory or a central service.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


PROTOCOL_README = """# Multi-AI Workflow

This folder contains the project-local operating protocol for coordinating
multiple AI tools. Start with `AGENTS.md`, then read `.hermes/context.md`,
`.hermes/active.md`, `.hermes/decisions.md`, `.hermes/issues/`, and
`.hermes/handoff.md`.

Use `scripts/multi_ai_workflow.py comply --project .` to summarize current
issue completion.
"""

AGENTS_TEMPLATE = """# Project AI Instructions

This project uses the HermesAgent multi-AI workflow.

Before substantial work:

1. Read this file.
2. Read `.hermes/context.md`, `.hermes/active.md`, `.hermes/decisions.md`.
3. Read `.hermes/issues/` and claim an issue before implementation.
4. Update `.hermes/handoff.md` before handing work to another AI.
5. If Opus 4.8 writes a plan, run `scripts/multi_ai_workflow.py route --project . --plan-file <plan>` to choose the next AI executor.

Prompt Shortcut Registry:

- `/Users/rattanasak/ObsidianVault/HermesAgent/ai-context/prompt-shortcut-registry.md`

## Closeout Protocol

Before saying work is complete:

1. Run the listed verification commands.
2. Check localhost and VPS only when the issue says they apply.
3. Report the evidence, remaining risk, and one recommended next step.

Do not read or copy `.env` values, secrets, runtime databases, logs, or cache
output into workflow files.
"""

CLAUDE_TEMPLATE = """@AGENTS.md

## Claude Code

Use this file as the Claude adapter. Follow the shared multi-AI workflow in
`AGENTS.md` and use `.hermes/issues/` for tracked work.
"""

QWEN_TEMPLATE = """# Qwen Adapter

Read `AGENTS.md` before work. Use `.hermes/issues/` for tracked work and
`.hermes/handoff.md` for continuation notes.
"""

GEMINI_TEMPLATE = """@AGENTS.md

# Gemini Adapter

Use the shared project instructions and keep handoff state in `.hermes/`.
"""

CURSOR_RULE_TEMPLATE = """---
description: Multi-AI workflow rules.
alwaysApply: true
---

- Read `AGENTS.md` before substantial work.
- Use `.hermes/issues/` as the local issue registry.
- Update `.hermes/handoff.md` before handing work to another AI.
- Do not claim localhost or VPS success unless a real command checked it.
"""

HERMES_CONTEXT_TEMPLATE = """# Project Context

This project uses the multi-AI workflow. Keep durable project facts here.
"""

HERMES_ACTIVE_TEMPLATE = """# Active State

No active issue has been claimed yet.
"""

HERMES_DECISIONS_TEMPLATE = """# Decisions

Record durable workflow and architecture decisions here.
"""

HERMES_HANDOFF_TEMPLATE = """# Multi-AI Handoff

task:
issue_id:
phase:
latest_state:
next_agent:
next_step:

## Verification

verification_run:
localhost_result:
vps_result:
remaining_risk:
"""

ISSUES_README_TEMPLATE = """# Multi-AI Issues

This folder is the local issue registry for AI handoffs.

Rules:

- One tracked issue per file.
- Use `docs/multi-ai-workflow/templates/issue.md`.
- Keep `done_percent` and `remaining_percent` numeric.
- Do not store secrets or `.env` contents here.
"""

PLANS_README_TEMPLATE = """# Opus Plans

Put Opus 4.8 planning output here when the plan should be routed to another AI.

Recommended flow:

1. Opus writes a plan file here.
2. Run `python3 scripts/multi_ai_workflow.py route --project . --plan-file .hermes/plans/<file>.md --write`.
3. Review `.hermes/routes/<file>.json`.
4. Use the generated handoff prompt with Codex App, Qwen on Cursor, or Gemini on Antigravity.
"""

ROUTES_README_TEMPLATE = """# AI Route Recommendations

This folder stores routing recommendations generated from Opus plan files.

Each JSON file records the ranked executor choices, the primary recommendation,
and a handoff prompt for the recommended AI tool.
"""

ISSUE_TEMPLATE = """# Multi-AI Issue Template

issue_id:
phase:
title:
owner_role:
assigned_ai:
reviewer_ai:
worktree_path:
branch:

## Goal

goal:

## Scope

scope:
out_of_scope:

## Boundaries

files_allowed:
files_forbidden:
secrets_policy:

## Done Criteria

done_when:

## Verification

verify_commands:
localhost_check:
vps_check:
evidence:

## Status

status:
done_percent:
remaining_percent:
blocker:

## Handoff

next_agent:
next_step:
handoff_required:
"""

HANDOFF_TEMPLATE = """# Multi-AI Handoff Template

task:
issue_id:
phase:
latest_state:
next_agent:
next_step:

## Work Context

worktree_path:
branch:
files_changed:
files_to_avoid:

## Verification

verification_run:
localhost_result:
vps_result:
remaining_risk:

## Decision Log

decisions_made:
decisions_needed:

## Continue Prompt

prompt_for_next_ai:
"""

OPUS_PLAN_TEMPLATE = """# Opus Plan Template

planner_ai: Opus 4.8
task:
phase:
project:

## Goal

Describe the planned outcome.

## Work Type

Choose the closest words that apply:

- backend
- frontend
- tests
- scripts
- refactor
- browser
- UX
- documentation
- research
- deployment

## Recommended By Opus

If Opus already has a preference, write it here. The router will still score all options.

preferred_executor:

## Implementation Plan

Write the plan here.

## Verification

List required tests, localhost checks, VPS checks, or file checks.
"""

AI_PAIR_README_TEMPLATE = """# AI Pair Jobs

This folder stores active `Use AI Pair` state.

Rules:

- One pair job per issue folder.
- Coder may edit only after owner-approved plan.
- Reviewer is read-only by default.
- Do not store secrets, token values, runtime databases, logs, or cache output.
"""

PAIR_STATE_TEMPLATE = {
    "version": 1,
    "status": "pair_selected",
    "issue_id": "",
    "task": "",
    "coder_ai": "",
    "reviewer_ai": "",
    "reviewer_mode": "read_only",
    "branch": "",
    "gitlab_host": "",
    "vps_status": "waiting-runtime",
    "created_at": "",
    "updated_at": "",
    "retry_count": 0,
    "max_review_rounds": 2,
}

CODER_PLAN_TEMPLATE = """# Coder Plan

issue_id:
task:
coder_ai:
branch:
status: plan_requested

## Scope

scope:
out_of_scope:
files_likely_touched:

## Plan

implementation_steps:

## Verification

commands:
localhost_check:
vps_check:

## Owner Approval

approved_by_owner: no
approval_note:
"""

CODER_BRIEF_TEMPLATE = """# Coder Brief

issue_id:
task:
coder_ai:
branch:

## Summary

diff_summary:
files_changed:

## Verification

commands_run:
results:
localhost_result:
vps_result:

## Review Request

review_focus:
known_risks:
out_of_scope:
"""

REVIEW_PACKET_TEMPLATE = """# Review Packet

issue_id:
reviewer_ai:
reviewer_mode: read_only
decision_expected: pass | changes_requested | blocked

## Inputs

approved_plan:
coder_brief:
diff_summary:
verification_evidence:

## Rules

- Reviewer must not edit files.
- Reviewer checks only this packet unless owner approves more context.
- Reviewer returns pass, changes_requested, or blocked.
"""

REVIEW_RESULT_TEMPLATE = """# Review Result

issue_id:
reviewer_ai:
reviewer_mode: read_only
decision:

## Summary

summary:

## Findings

findings:

## Required Changes

required_changes:

## Evidence Checked

evidence_checked:

## Owner Attention

should_owner_inspect_manually:
"""

GITLAB_GATE_TEMPLATE = """# GitLab Gate

issue_id:
gitlab_host:
project_path:
merge_request_url:
pipeline_url:
pipeline_status:

## Evidence

diff_source:
artifacts_checked:
ci_result:

## Merge Rule

owner_merge_required: yes
auto_merge_allowed: no
"""

FIELD_RE = re.compile(r"^([A-Za-z0-9_]+):\s*(.*)$")
SCRIPT_DIR = Path(__file__).resolve().parent

EXECUTOR_PROFILES = {
    "codex_app": {
        "tool": "Codex App",
        "best_for": "repo changes, Python/CLI/backend work, tests, git workflows, code review, security-sensitive implementation",
        "signals": (
            "codex",
            "python",
            "pytest",
            "cli",
            "script",
            "scripts",
            "backend",
            "api",
            "test",
            "tests",
            "git",
            "worktree",
            "refactor",
            "security",
            "review",
            "repository",
            "code",
            "implementation",
        ),
    },
    "qwen_cursor": {
        "tool": "Qwen on Cursor",
        "best_for": "editor-driven coding, frontend changes, TypeScript/React components, UI polish inside Cursor",
        "signals": (
            "qwen",
            "cursor",
            "react",
            "typescript",
            "tsx",
            "frontend",
            "ui",
            "component",
            "components",
            "style",
            "styling",
            "css",
            "form",
            "forms",
            "editor",
            "incremental",
        ),
    },
    "gemini_antigravity": {
        "tool": "Gemini on Antigravity",
        "best_for": "large-context app exploration, browser/UX validation, multimodal review, research-heavy checks",
        "signals": (
            "gemini",
            "antigravity",
            "browser",
            "ux",
            "multimodal",
            "large-context",
            "large context",
            "research",
            "explore",
            "exploration",
            "inspect",
            "flow",
            "flows",
            "compare",
            "validation",
            "app",
        ),
    },
}


def _project_root(project: str | Path) -> Path:
    return Path(project).expanduser().resolve()


def _write_if_needed(path: Path, text: str, force: bool) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def init_project(project: str | Path, force: bool = False) -> dict[str, Any]:
    root = _project_root(project)
    files = {
        root / "AGENTS.md": AGENTS_TEMPLATE,
        root / "CLAUDE.md": CLAUDE_TEMPLATE,
        root / "QWEN.md": QWEN_TEMPLATE,
        root / "GEMINI.md": GEMINI_TEMPLATE,
        root / ".cursor" / "rules" / "multi-ai-workflow.mdc": CURSOR_RULE_TEMPLATE,
        root / ".hermes" / "context.md": HERMES_CONTEXT_TEMPLATE,
        root / ".hermes" / "active.md": HERMES_ACTIVE_TEMPLATE,
        root / ".hermes" / "decisions.md": HERMES_DECISIONS_TEMPLATE,
        root / ".hermes" / "handoff.md": HERMES_HANDOFF_TEMPLATE,
        root / ".hermes" / "issues" / "README.md": ISSUES_README_TEMPLATE,
        root / ".hermes" / "ai-pair" / "README.md": AI_PAIR_README_TEMPLATE,
        root / ".hermes" / "plans" / "README.md": PLANS_README_TEMPLATE,
        root / ".hermes" / "routes" / "README.md": ROUTES_README_TEMPLATE,
        root / "docs" / "multi-ai-workflow" / "README.md": PROTOCOL_README,
        root / "docs" / "multi-ai-workflow" / "templates" / "issue.md": ISSUE_TEMPLATE,
        root / "docs" / "multi-ai-workflow" / "templates" / "handoff.md": HANDOFF_TEMPLATE,
        root / "docs" / "multi-ai-workflow" / "templates" / "opus-plan.md": OPUS_PLAN_TEMPLATE,
        root / "docs" / "multi-ai-workflow" / "templates" / "ai-pair" / "coder-brief.md": CODER_BRIEF_TEMPLATE,
        root / "docs" / "multi-ai-workflow" / "templates" / "ai-pair" / "review-result.md": REVIEW_RESULT_TEMPLATE,
    }
    created: list[str] = []
    skipped: list[str] = []
    for path, text in files.items():
        if _write_if_needed(path, text, force):
            created.append(str(path))
        else:
            skipped.append(str(path))
    return {
        "project": str(root),
        "created": created,
        "skipped": skipped,
        "created_count": len(created),
        "skipped_count": len(skipped),
    }


def _issue_path(project: str | Path, issue_id: str) -> Path:
    safe_id = issue_id.strip()
    if not safe_id or "/" in safe_id or safe_id in {".", ".."}:
        raise ValueError("issue_id must be a non-empty filename-safe value")
    return _project_root(project) / ".hermes" / "issues" / f"{safe_id}.md"


def _ai_pair_path(project: str | Path, issue_id: str) -> Path:
    safe_id = issue_id.strip()
    if not safe_id or "/" in safe_id or safe_id in {".", ".."}:
        raise ValueError("issue_id must be a non-empty filename-safe value")
    return _project_root(project) / ".hermes" / "ai-pair" / safe_id


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _join_commands(commands: list[str]) -> str:
    return "; ".join(command for command in commands if command.strip())


def _render_issue(
    *,
    issue_id: str,
    phase: str,
    title: str,
    owner_role: str,
    assigned_ai: str,
    goal: str,
    scope: str,
    out_of_scope: str,
    verify_commands: list[str],
    localhost_check: str,
    vps_check: str,
    branch: str,
    worktree_path: str,
    reviewer_ai: str,
) -> str:
    verify = _join_commands(verify_commands)
    return f"""# {title}

issue_id: {issue_id}
phase: {phase}
title: {title}
owner_role: {owner_role}
assigned_ai: {assigned_ai}
reviewer_ai: {reviewer_ai}
worktree_path: {worktree_path}
branch: {branch}

## Goal

goal: {goal}

## Scope

scope: {scope}
out_of_scope: {out_of_scope}

## Boundaries

files_allowed:
files_forbidden: .env files, runtime databases, logs, caches, unrelated dirty files
secrets_policy: Do not read or copy secret values.

## Done Criteria

done_when: Verification evidence is recorded and remaining_percent is 0.

## Verification

verify_commands: {verify}
localhost_check: {localhost_check}
vps_check: {vps_check}
evidence:

## Status

status: open
done_percent: 0
remaining_percent: 100
blocker:

## Handoff

next_agent:
next_step:
handoff_required: yes
"""


def create_issue(
    *,
    project: str | Path,
    issue_id: str,
    phase: str,
    title: str,
    owner_role: str,
    assigned_ai: str,
    goal: str,
    scope: str,
    out_of_scope: str,
    verify_commands: list[str],
    localhost_check: str,
    vps_check: str,
    branch: str,
    worktree_path: str,
    reviewer_ai: str,
    force: bool = False,
) -> Path:
    path = _issue_path(project, issue_id)
    if path.exists() and not force:
        raise FileExistsError(f"Issue already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _render_issue(
            issue_id=issue_id,
            phase=phase,
            title=title,
            owner_role=owner_role,
            assigned_ai=assigned_ai,
            goal=goal,
            scope=scope,
            out_of_scope=out_of_scope,
            verify_commands=verify_commands,
            localhost_check=localhost_check,
            vps_check=vps_check,
            branch=branch,
            worktree_path=worktree_path,
            reviewer_ai=reviewer_ai,
        ),
        encoding="utf-8",
    )
    return path


def parse_issue_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = FIELD_RE.match(line)
        if match:
            fields[match.group(1)] = match.group(2)
    return fields


def update_issue_fields(path: str | Path, updates: dict[str, str]) -> Path:
    issue_path = Path(path)
    lines = issue_path.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    updated_lines: list[str] = []
    for line in lines:
        match = FIELD_RE.match(line)
        if match and match.group(1) in updates:
            key = match.group(1)
            updated_lines.append(f"{key}: {updates[key]}")
            seen.add(key)
        else:
            updated_lines.append(line)
    for key, value in updates.items():
        if key not in seen:
            updated_lines.append(f"{key}: {value}")
    issue_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    return issue_path


def claim_issue(
    *,
    project: str | Path,
    issue_id: str,
    assigned_ai: str,
    branch: str,
    worktree_path: str,
) -> Path:
    path = _issue_path(project, issue_id)
    if not path.exists():
        raise FileNotFoundError(f"Issue not found: {path}")
    return update_issue_fields(
        path,
        {
            "assigned_ai": assigned_ai,
            "branch": branch,
            "worktree_path": worktree_path,
            "status": "claimed",
        },
    )


def update_issue_status(
    *,
    project: str | Path,
    issue_id: str,
    status: str,
    done_percent: str,
    remaining_percent: str,
    evidence: str,
) -> Path:
    path = _issue_path(project, issue_id)
    if not path.exists():
        raise FileNotFoundError(f"Issue not found: {path}")
    return update_issue_fields(
        path,
        {
            "status": status,
            "done_percent": done_percent,
            "remaining_percent": remaining_percent,
            "evidence": evidence,
        },
    )


def _run_command(cmd: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": cmd,
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "output": proc.stdout,
    }


def _git_status_short(root: Path) -> str:
    result = _run_command(["git", "status", "--short", "--untracked-files=no"], root)
    return result["output"].strip()


def _safe_branch_slug(issue_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", issue_id.strip()).strip("-").lower()
    if not slug:
        raise ValueError("issue_id must produce a non-empty branch slug")
    return slug


def propose_ai_pair_branch(
    *,
    project: str | Path,
    issue_id: str,
    task: str,
) -> dict[str, Any]:
    root = _project_root(project)
    status = _git_status_short(root)
    branch = f"ai-pair/{_safe_branch_slug(issue_id)}"
    dirty = bool(status)
    if dirty:
        return {
            "project": str(root),
            "ok": False,
            "dirty": True,
            "status": status,
            "branch": branch,
            "task": task,
            "requires_owner_approval": True,
            "reason": "worktree has uncommitted changes; owner must approve before creating or switching branch",
        }
    return {
        "project": str(root),
        "ok": True,
        "dirty": False,
        "status": "",
        "branch": branch,
        "task": task,
        "requires_owner_approval": True,
        "reason": "clean worktree; branch proposal is ready for owner approval",
    }


def _fill_template_fields(text: str, values: dict[str, str]) -> str:
    rendered = text
    for key, value in values.items():
        rendered = rendered.replace(f"{key}:", f"{key}: {value}", 1)
    return rendered


def _agent_env_name(ai_name: str) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "_", ai_name).strip("_").upper()
    return f"HERMES_AI_PAIR_{key}_COMMAND"


def resolve_ai_pair_agent_command(ai_name: str, explicit_command: str = "") -> dict[str, Any]:
    env_name = _agent_env_name(ai_name)
    command = explicit_command.strip() or os.environ.get(env_name, "").strip()
    if command:
        executable = shlex.split(command)[0]
        return {
            "ok": shutil.which(executable) is not None or Path(executable).exists(),
            "ai": ai_name,
            "env_name": env_name,
            "command": command,
            "reason": "configured command found",
        }
    return {
        "ok": False,
        "ai": ai_name,
        "env_name": env_name,
        "command": "",
        "reason": f"missing runnable adapter command; set {env_name}",
    }


def create_ai_pair_job(
    *,
    project: str | Path,
    issue_id: str,
    task: str,
    coder_ai: str,
    reviewer_ai: str,
    branch: str,
    gitlab_host: str,
    force: bool = False,
) -> dict[str, Any]:
    root = _project_root(project)
    pair_dir = _ai_pair_path(root, issue_id)
    if pair_dir.exists() and not force:
        raise FileExistsError(f"AI Pair job already exists: {pair_dir}")
    pair_dir.mkdir(parents=True, exist_ok=True)
    now = _now_iso()
    state = dict(PAIR_STATE_TEMPLATE)
    state.update(
        {
            "issue_id": issue_id,
            "task": task,
            "coder_ai": coder_ai,
            "reviewer_ai": reviewer_ai,
            "branch": branch,
            "gitlab_host": gitlab_host,
            "created_at": now,
            "updated_at": now,
        }
    )
    (pair_dir / "pair-state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    values = {
        "issue_id": issue_id,
        "task": task,
        "coder_ai": coder_ai,
        "reviewer_ai": reviewer_ai,
        "branch": branch,
        "gitlab_host": gitlab_host,
    }
    files = {
        "coder-plan.md": CODER_PLAN_TEMPLATE,
        "coder-brief.md": CODER_BRIEF_TEMPLATE,
        "review-packet.md": REVIEW_PACKET_TEMPLATE,
        "review-result.md": REVIEW_RESULT_TEMPLATE,
        "gitlab-gate.md": GITLAB_GATE_TEMPLATE,
        "handoff.md": HANDOFF_TEMPLATE,
    }
    for filename, template in files.items():
        (pair_dir / filename).write_text(
            _fill_template_fields(template, values),
            encoding="utf-8",
        )
    return {
        "project": str(root),
        "issue_id": issue_id,
        "pair_dir": str(pair_dir),
        "status": state["status"],
        "coder_ai": coder_ai,
        "reviewer_ai": reviewer_ai,
        "branch": branch,
        "gitlab_host": gitlab_host,
    }


def _load_ai_pair_state(project: str | Path, issue_id: str) -> tuple[Path, dict[str, Any]]:
    pair_dir = _ai_pair_path(project, issue_id)
    state_path = pair_dir / "pair-state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"AI Pair state not found: {state_path}")
    return pair_dir, json.loads(state_path.read_text(encoding="utf-8"))


def build_ai_pair_coder_plan_prompt(state: dict[str, Any]) -> str:
    return f"""Use AI Pair

You are the coder AI: {state.get("coder_ai", "")}
Reviewer AI: {state.get("reviewer_ai", "")} in read-only mode.

Task:
{state.get("task", "")}

Branch/worktree:
{state.get("branch", "")}

Rules:
- Do not edit files.
- Do not write code.
- Do not create commits.
- Read context only as needed.
- Return a coder plan for owner approval before implementation.

Return the plan in this structure:

# Coder Plan

issue_id: {state.get("issue_id", "")}
task: {state.get("task", "")}
coder_ai: {state.get("coder_ai", "")}
branch: {state.get("branch", "")}
status: plan_ready_for_owner

## Scope

scope:
out_of_scope:
files_likely_touched:

## Plan

implementation_steps:

## Verification

commands:
localhost_check:
vps_check:

## Owner Approval

approved_by_owner: no
approval_note:
"""


def run_ai_pair_coder_plan(
    *,
    project: str | Path,
    issue_id: str,
    execute: bool = False,
    coder_command: str = "",
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    root = _project_root(project)
    pair_dir, state = _load_ai_pair_state(root, issue_id)
    prompt = build_ai_pair_coder_plan_prompt(state)
    prompt_path = pair_dir / "coder-plan-prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")

    resolved = resolve_ai_pair_agent_command(str(state.get("coder_ai", "")), coder_command)
    if not resolved["ok"]:
        state["status"] = "blocked_missing_coder_runtime"
        state["runtime_error"] = resolved["reason"]
        _write_ai_pair_state(pair_dir, state)
        blocker_path = pair_dir / "automation-blocker.md"
        blocker_path.write_text(
            "\n".join(
                [
                    "# AI Pair Automation Blocked",
                    "",
                    f"status: {state['status']}",
                    f"coder_ai: {state.get('coder_ai', '')}",
                    f"required_env: {resolved['env_name']}",
                    f"reason: {resolved['reason']}",
                    "",
                    "This job must not fall back to manual prompt forwarding.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return {
            "ok": False,
            "executed": False,
            "status": state["status"],
            "reason": resolved["reason"],
            "required_env": resolved["env_name"],
            "prompt_path": str(prompt_path),
            "blocker_path": str(blocker_path),
        }

    if not execute:
        return {
            "ok": True,
            "executed": False,
            "status": state.get("status"),
            "command": resolved["command"],
            "prompt_path": str(prompt_path),
        }

    completed = subprocess.run(
        shlex.split(resolved["command"]),
        input=prompt,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(root),
        timeout=timeout_seconds,
        check=False,
    )
    output_path = pair_dir / "coder-plan.raw.md"
    output_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        state["status"] = "blocked_coder_runtime_failed"
        state["runtime_error"] = completed.stderr.strip()
        _write_ai_pair_state(pair_dir, state)
        return {
            "ok": False,
            "executed": True,
            "status": state["status"],
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
            "output_path": str(output_path),
        }

    (pair_dir / "coder-plan.md").write_text(completed.stdout, encoding="utf-8")
    state["status"] = "coder_plan_ready_for_owner"
    state.pop("runtime_error", None)
    _write_ai_pair_state(pair_dir, state)
    return {
        "ok": True,
        "executed": True,
        "status": state["status"],
        "returncode": completed.returncode,
        "coder_plan_path": str(pair_dir / "coder-plan.md"),
        "output_path": str(output_path),
    }


# ทะเบียนที่นั่ง AI ทั้งหมด (bundle_id ตรวจจาก /Applications จริง · ไม่เดา · 2026-06-07)
# ทุกตัวรับบทไหนก็ได้ (coach/coder/reviewer) ตามที่เจ้าของกำหนดใน role_map
DESKTOP_SEAT_PROFILES = {
    "claude": {
        "bundle_id": "com.anthropic.claudefordesktop",
        "app_name": "Claude",
        "aliases": ("claude", "claude code", "claude-code", "claude app", "claude desktop"),
    },
    "codex": {
        "bundle_id": "com.openai.codex",
        "app_name": "Codex",
        "aliases": ("codex", "codex app"),
    },
    "cursor": {
        "bundle_id": "com.todesktop.230313mzl4w4u92",
        "app_name": "Cursor",
        "aliases": ("cursor", "cursor app"),
    },
    # Qwen ใช้งานผ่าน Cursor (Qwen extension) → แชร์แอป Cursor ตัวเดียวกัน
    "qwen": {
        "bundle_id": "com.todesktop.230313mzl4w4u92",
        "app_name": "Cursor",
        "aliases": ("qwen", "qwen code", "qwen extension", "cursor/qwen", "cursor qwen", "cursor-qwen"),
    },
    # Gemini ใช้งานบน Antigravity (IDE ของ Google)
    "gemini": {
        "bundle_id": "com.google.antigravity",
        "app_name": "Antigravity",
        "aliases": ("gemini", "antigravity", "gemini antigravity"),
    },
    # Grok ใช้งานผ่าน grok CLI ใน Terminal ของ macOS
    "grok": {
        "bundle_id": "com.apple.Terminal",
        "app_name": "Terminal",
        "aliases": ("grok", "grok cli", "grok terminal"),
    },
    "manus": {
        "bundle_id": "im.manus.desktop",
        "app_name": "Manus",
        "aliases": ("manus", "manus app", "manus desktop"),
    },
}


def _desktop_profile_for_ai(ai_name: str) -> dict[str, Any]:
    normalized = ai_name.strip().lower()
    for seat_id, profile in DESKTOP_SEAT_PROFILES.items():
        if normalized == seat_id or normalized in profile["aliases"]:
            result = dict(profile)
            result["seat_id"] = seat_id
            return result
    return {
        "seat_id": re.sub(r"[^a-z0-9]+", "-", normalized).strip("-") or "unknown",
        "bundle_id": "",
        "app_name": ai_name,
        "aliases": (),
    }


def _git_branch(root: Path) -> str:
    completed = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip()


def build_desktop_seat_registry(
    *,
    project: str | Path,
    coach_ai: str,
    coder_ai: str,
    reviewer_ai: str,
) -> dict[str, Any]:
    root = _project_root(project)
    return {
        "version": 1,
        "project": str(root),
        "project_name": root.name,
        "branch": _git_branch(root),
        "roles": {
            "coach": {"ai": coach_ai, **_desktop_profile_for_ai(coach_ai)},
            "coder": {"ai": coder_ai, **_desktop_profile_for_ai(coder_ai)},
            "reviewer": {"ai": reviewer_ai, **_desktop_profile_for_ai(reviewer_ai)},
        },
    }


def evaluate_desktop_seats(
    *,
    registry: dict[str, Any],
    apps: list[dict[str, Any]],
    windows_by_pid: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    project_name = str(registry.get("project_name", ""))
    branch = str(registry.get("branch", ""))
    app_by_bundle = {app.get("bundle_id"): app for app in apps}
    roles: dict[str, Any] = {}
    all_ok = True

    for role, seat in registry["roles"].items():
        bundle_id = seat.get("bundle_id", "")
        app = app_by_bundle.get(bundle_id, {})
        pid = int(app.get("pid") or 0)
        running = bool(app.get("running")) and pid > 0
        windows = windows_by_pid.get(pid, []) if running else []
        usable_windows = [
            window
            for window in windows
            if int(window.get("bounds", {}).get("width", 0)) >= 700
            and int(window.get("bounds", {}).get("height", 0)) >= 500
        ]
        matching_windows = [
            window
            for window in windows
            if project_name and project_name.lower() in str(window.get("title", "")).lower()
        ]
        branch_windows = [
            window
            for window in windows
            if branch and branch.lower() in str(window.get("title", "")).lower()
        ]
        ok = running and (bool(matching_windows) or bool(branch_windows))
        if not ok:
            all_ok = False
        roles[role] = {
            "ai": seat.get("ai", ""),
            "seat_id": seat.get("seat_id", ""),
            "bundle_id": bundle_id,
            "pid": pid,
            "running": running,
            "usable_window_count": len(usable_windows),
            "project_window_count": len(matching_windows),
            "branch_window_count": len(branch_windows),
            "ok": ok,
            "blocker": ""
            if ok
            else "seat is not bound to the target project/worktree window",
        }

    return {
        "ok": all_ok,
        "project": registry.get("project", ""),
        "project_name": project_name,
        "branch": branch,
        "roles": roles,
    }


def _cua_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return json.loads(completed.stdout)


def audit_ai_pair_desktop(
    *,
    project: str | Path,
    issue_id: str,
    coach_ai: str = "",
    coder_ai: str = "",
    reviewer_ai: str = "",
    cua_driver: str = "cua-driver",
) -> dict[str, Any]:
    root = _project_root(project)
    pair_dir, state = _load_ai_pair_state(root, issue_id)
    registry = build_desktop_seat_registry(
        project=root,
        coach_ai=coach_ai or str(state.get("coach_ai", "")),
        coder_ai=coder_ai or str(state.get("coder_ai", "")),
        reviewer_ai=reviewer_ai or str(state.get("reviewer_ai", "")),
    )
    apps_payload = _cua_json([cua_driver, "call", "list_apps", "{}"])
    apps = list(apps_payload.get("apps", []))
    windows_by_pid: dict[int, list[dict[str, Any]]] = {}
    for role in registry["roles"].values():
        app = next((item for item in apps if item.get("bundle_id") == role.get("bundle_id")), None)
        pid = int((app or {}).get("pid") or 0)
        if pid > 0 and pid not in windows_by_pid:
            windows_payload = _cua_json([cua_driver, "call", "list_windows", json.dumps({"pid": pid})])
            windows_by_pid[pid] = list(windows_payload.get("windows", []))

    report = evaluate_desktop_seats(registry=registry, apps=apps, windows_by_pid=windows_by_pid)
    (pair_dir / "desktop-seat-registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (pair_dir / "desktop-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    state["status"] = "desktop_audit_ready" if report["ok"] else "blocked_desktop_seat_binding"
    state["desktop_audit_ok"] = report["ok"]
    _write_ai_pair_state(pair_dir, state)
    return report


def _ai_pair_role_name(state: dict[str, Any], role: str) -> str:
    if role == "coach":
        return str(state.get("coach_ai", ""))
    if role == "coder":
        return str(state.get("coder_ai", ""))
    if role == "reviewer":
        return str(state.get("reviewer_ai", ""))
    return role


def _load_desktop_audit(pair_dir: Path) -> dict[str, Any]:
    audit_path = pair_dir / "desktop-audit.json"
    if not audit_path.exists():
        return {"ok": False, "roles": {}}
    return json.loads(audit_path.read_text(encoding="utf-8"))


def prepare_ai_pair_desktop_handoff(
    *,
    project: str | Path,
    issue_id: str,
    role: str,
    phase: str,
    prompt_text: str,
) -> dict[str, Any]:
    root = _project_root(project)
    pair_dir, state = _load_ai_pair_state(root, issue_id)
    normalized_role = role.strip().lower()
    if normalized_role not in {"coach", "coder", "reviewer"}:
        raise ValueError("role must be one of: coach, coder, reviewer")
    normalized_phase = re.sub(r"[^a-zA-Z0-9_.-]+", "-", phase.strip()).strip("-") or "handoff"
    audit = _load_desktop_audit(pair_dir)
    role_audit = dict(audit.get("roles", {}).get(normalized_role, {}))
    seat_ok = bool(role_audit.get("ok"))
    status = "ready_for_desktop_send" if seat_ok else "queued_waiting_for_seat"

    handoff_dir = pair_dir / "desktop-handoffs"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    queue_path = pair_dir / "desktop-handoff-queue.json"
    if queue_path.exists():
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    else:
        queue = {"version": 1, "issue_id": issue_id, "items": []}
    sequence = len(queue.get("items", [])) + 1
    handoff_id = f"{sequence:03d}-{normalized_phase}-{normalized_role}"
    prompt_path = handoff_dir / f"{handoff_id}-prompt.md"
    response_path = handoff_dir / f"{handoff_id}-response.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    item = {
        "id": handoff_id,
        "role": normalized_role,
        "phase": normalized_phase,
        "ai": role_audit.get("ai") or _ai_pair_role_name(state, normalized_role),
        "status": status,
        "seat_ok": seat_ok,
        "prompt_path": str(prompt_path),
        "response_path": str(response_path),
        "created_at": _now_iso(),
    }
    queue.setdefault("items", []).append(item)
    queue["updated_at"] = _now_iso()
    queue_path.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    state["status"] = "desktop_handoff_ready" if seat_ok else "desktop_handoff_queued"
    state["last_desktop_handoff_id"] = handoff_id
    state["desktop_handoff_queue_ok"] = True
    state.pop("runtime_error", None)
    _write_ai_pair_state(pair_dir, state)
    return {
        "ok": True,
        "status": status,
        "seat_ok": seat_ok,
        "handoff_id": handoff_id,
        "queue_path": str(queue_path),
        "prompt_path": str(prompt_path),
        "response_path": str(response_path),
    }


def _review_decision(review_text: str) -> str:
    decision = _markdown_field_value(review_text, "decision").strip().lower()
    normalized = decision.replace(" ", "_").replace("-", "_")
    if normalized in {"pass", "passed", "approved", "approve"}:
        return "pass"
    if normalized in {"changes_requested", "change_requested", "needs_changes", "failed"}:
        return "changes_requested"
    if normalized == "blocked":
        return "blocked"
    return normalized or "blocked"


def record_ai_pair_review_result(
    *,
    project: str | Path,
    issue_id: str,
    review_text: str,
) -> dict[str, Any]:
    pair_dir, state = _load_ai_pair_state(project, issue_id)
    decision = _review_decision(review_text)
    result_path = pair_dir / "review-result.md"
    result_path.write_text(review_text, encoding="utf-8")
    if decision == "pass":
        status = "review_passed"
    elif decision == "changes_requested":
        status = "changes_requested_to_coder"
        state["retry_count"] = int(state.get("retry_count") or 0) + 1
    else:
        status = "review_blocked"
    state["status"] = status
    state["last_review_decision"] = decision
    _write_ai_pair_state(pair_dir, state)
    return {
        "ok": decision in {"pass", "changes_requested"},
        "decision": decision,
        "status": status,
        "review_result_path": str(result_path),
        "retry_count": int(state.get("retry_count") or 0),
    }


def _markdown_field_value(text: str, field: str) -> str:
    prefix = f"{field}:"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return ""


def _write_ai_pair_state(pair_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _now_iso()
    (pair_dir / "pair-state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_ai_pair_review_gate(
    *,
    pair_dir: Path,
    state: dict[str, Any],
    plan_text: str,
    brief_text: str,
    diff_summary: str,
    verification_evidence: str,
) -> None:
    errors: list[str] = []
    approval = _markdown_field_value(plan_text, "approved_by_owner").lower()
    if approval not in {"yes", "true", "approved"}:
        errors.append("coder-plan.md must include approved_by_owner: yes")

    for field in ("diff_summary", "files_changed", "commands_run", "results", "review_focus"):
        if not _markdown_field_value(brief_text, field):
            errors.append(f"coder-brief.md missing required value: {field}:")

    if not diff_summary.strip():
        errors.append("diff_summary argument must not be empty")
    if not verification_evidence.strip():
        errors.append("verification_evidence argument must not be empty")

    if errors:
        state["status"] = "blocked_missing_review_gate"
        state["review_gate_errors"] = errors
        _write_ai_pair_state(pair_dir, state)
        raise ValueError("AI Pair review gate blocked: " + "; ".join(errors))


def render_ai_pair_review_packet(
    *,
    project: str | Path,
    issue_id: str,
    diff_summary: str,
    verification_evidence: str,
) -> Path:
    pair_dir, state = _load_ai_pair_state(project, issue_id)
    plan_text = (pair_dir / "coder-plan.md").read_text(encoding="utf-8")
    brief_text = (pair_dir / "coder-brief.md").read_text(encoding="utf-8")
    _validate_ai_pair_review_gate(
        pair_dir=pair_dir,
        state=state,
        plan_text=plan_text,
        brief_text=brief_text,
        diff_summary=diff_summary,
        verification_evidence=verification_evidence,
    )
    packet = f"""# Review Packet

issue_id: {issue_id}
reviewer_ai: {state.get("reviewer_ai", "")}
reviewer_mode: read_only
decision_expected: pass | changes_requested | blocked

## Inputs

approved_plan:
{plan_text.strip()}

coder_brief:
{brief_text.strip()}

diff_summary:
{diff_summary}

verification_evidence:
{verification_evidence}

## Rules

- Reviewer must not edit files.
- Reviewer checks only this packet unless owner approves more context.
- Reviewer returns pass, changes_requested, or blocked.
"""
    packet_path = pair_dir / "review-packet.md"
    packet_path.write_text(packet, encoding="utf-8")
    state["status"] = "review_packet_ready"
    state.pop("review_gate_errors", None)
    _write_ai_pair_state(pair_dir, state)
    return packet_path


def gitlab_gate_dry_run(
    *,
    project: str | Path,
    issue_id: str,
    project_path: str,
    merge_request_iid: str,
    token_env: str = "GITLAB_TOKEN",
) -> dict[str, Any]:
    pair_dir, state = _load_ai_pair_state(project, issue_id)
    gitlab_host = str(state.get("gitlab_host", ""))
    endpoint_host = gitlab_host.rstrip("/")
    encoded_project = project_path.replace("/", "%2F")
    endpoints = {
        "merge_request": f"{endpoint_host}/api/v4/projects/{encoded_project}/merge_requests/{merge_request_iid}",
        "diffs": f"{endpoint_host}/api/v4/projects/{encoded_project}/merge_requests/{merge_request_iid}/diffs",
        "pipelines": f"{endpoint_host}/api/v4/projects/{encoded_project}/merge_requests/{merge_request_iid}/pipelines",
    }
    gate_path = pair_dir / "gitlab-gate.md"
    gate_path.write_text(
        f"""# GitLab Gate

issue_id: {issue_id}
gitlab_host: {gitlab_host}
project_path: {project_path}
merge_request_iid: {merge_request_iid}
token_env: {token_env}
executed: no

## Dry Run Endpoints

merge_request: {endpoints["merge_request"]}
diffs: {endpoints["diffs"]}
pipelines: {endpoints["pipelines"]}

## Secret Policy

Token values must never be written to workflow files or AI prompts.
""",
        encoding="utf-8",
    )
    return {
        "project": str(_project_root(project)),
        "issue_id": issue_id,
        "gitlab_host": gitlab_host,
        "project_path": project_path,
        "merge_request_iid": merge_request_iid,
        "token_env": token_env,
        "endpoints": endpoints,
        "executed": False,
        "gate_path": str(gate_path),
    }


def create_worktree(
    *,
    project: str | Path,
    issue_id: str,
    assigned_ai: str,
    branch: str,
    worktree_path: str | Path,
    execute: bool = False,
) -> dict[str, Any]:
    root = _project_root(project)
    target = Path(worktree_path).expanduser()
    if not target.is_absolute():
        target = (root / target).resolve()
    cmd = ["git", "worktree", "add", str(target), "-b", branch]
    result: dict[str, Any] = {
        "project": str(root),
        "issue_id": issue_id,
        "assigned_ai": assigned_ai,
        "branch": branch,
        "worktree_path": str(target),
        "command": cmd,
        "executed": execute,
        "returncode": None,
        "output": "",
    }
    if execute:
        run_result = _run_command(cmd, root)
        result.update(run_result)
        if run_result["returncode"] == 0:
            claim_issue(
                project=root,
                issue_id=issue_id,
                assigned_ai=assigned_ai,
                branch=branch,
                worktree_path=str(target),
            )
    return result


def github_issue_sync(
    *,
    project: str | Path,
    issue_id: str,
    execute: bool = False,
) -> dict[str, Any]:
    root = _project_root(project)
    path = _issue_path(root, issue_id)
    if not path.exists():
        raise FileNotFoundError(f"Issue not found: {path}")
    fields = parse_issue_fields(path)
    title = fields.get("title", issue_id)
    body = path.read_text(encoding="utf-8")
    cmd = ["gh", "issue", "create", "--title", title, "--body", body]
    result: dict[str, Any] = {
        "project": str(root),
        "issue_id": issue_id,
        "command": cmd,
        "body": body,
        "executed": execute,
        "returncode": None,
        "output": "",
    }
    if execute:
        run_result = _run_command(cmd, root)
        result.update(run_result)
    return result


def write_handoff(
    *,
    project: str | Path,
    task: str,
    issue_id: str,
    phase: str,
    latest_state: str,
    next_agent: str,
    next_step: str,
    verification_run: str,
    localhost_result: str,
    vps_result: str,
    remaining_risk: str,
) -> Path:
    root = _project_root(project)
    path = root / ".hermes" / "handoff.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# Multi-AI Handoff

task: {task}
issue_id: {issue_id}
phase: {phase}
latest_state: {latest_state}
next_agent: {next_agent}
next_step: {next_step}

## Verification

verification_run: {verification_run}
localhost_result: {localhost_result}
vps_result: {vps_result}
remaining_risk: {remaining_risk}
""",
        encoding="utf-8",
    )
    return path


def _normalized_words(text: str) -> list[str]:
    lowered = text.lower()
    return re.findall(r"[a-z0-9.+#-]+", lowered)


def recommend_executor(plan_text: str) -> dict[str, Any]:
    words = _normalized_words(plan_text)
    joined = " ".join(words)
    ranked: list[dict[str, Any]] = []
    preferred_hint = ""
    for line in plan_text.splitlines():
        if line.lower().startswith("preferred_executor:"):
            preferred_hint = line.split(":", 1)[1].strip().lower()
            break

    for executor_id, profile in EXECUTOR_PROFILES.items():
        matched: list[str] = []
        score = 0
        for signal in profile["signals"]:
            signal_text = str(signal).lower()
            if " " in signal_text:
                if signal_text in joined:
                    matched.append(signal_text)
                    score += 2
            elif signal_text in words:
                matched.append(signal_text)
                score += 2
        if preferred_hint and (
            preferred_hint in executor_id
            or preferred_hint in str(profile["tool"]).lower()
            or str(profile["tool"]).lower() in preferred_hint
        ):
            matched.append("preferred_executor")
            score += 4
        ranked.append(
            {
                "id": executor_id,
                "tool": profile["tool"],
                "score": score,
                "suitability_percent": 0.0,
                "best_for": profile["best_for"],
                "matched_signals": matched,
            }
        )

    ranked.sort(key=lambda item: (-item["score"], item["id"]))
    top_score = ranked[0]["score"] if ranked else 0
    for item in ranked:
        item["suitability_percent"] = (
            round((item["score"] / top_score) * 100, 2) if top_score else 0.0
        )
    primary = ranked[0]
    return {
        "primary": primary,
        "alternates": ranked[1:],
        "ranked": ranked,
        "reason": (
            f"เลือก {primary['tool']} เพราะสัญญาณในแผนตรงกับงาน: "
            f"{', '.join(primary['matched_signals']) if primary['matched_signals'] else 'fallback default'}"
        ),
    }


def build_handoff_prompt(plan_text: str, recommendation: dict[str, Any]) -> str:
    primary = recommendation["primary"]
    return (
        f"Prompt for {primary['tool']}\n\n"
        "คุณได้รับงานต่อจาก Opus 4.8 ซึ่งเป็น planner หลัก\n"
        "ให้ทำตามแผนด้านล่างแบบ issue-driven และต้องรัน verification ก่อนสรุปว่างานเสร็จ\n\n"
        f"Recommended executor: {primary['tool']}\n"
        f"Why: {recommendation['reason']}\n\n"
        "Opus plan:\n"
        f"{plan_text.strip()}\n"
    )


def route_plan_file(
    *,
    project: str | Path,
    plan_file: str | Path,
    write: bool = False,
) -> dict[str, Any]:
    root = _project_root(project)
    plan_path = Path(plan_file).expanduser()
    if not plan_path.is_absolute():
        plan_path = (root / plan_path).resolve()
    plan_text = plan_path.read_text(encoding="utf-8")
    recommendation = recommend_executor(plan_text)
    handoff_prompt = build_handoff_prompt(plan_text, recommendation)
    result: dict[str, Any] = {
        "project": str(root),
        "plan_file": str(plan_path),
        "recommendation": recommendation,
        "handoff_prompt": handoff_prompt,
        "output_path": "",
        "written": False,
    }
    if write:
        routes_dir = root / ".hermes" / "routes"
        routes_dir.mkdir(parents=True, exist_ok=True)
        output_path = routes_dir / f"{plan_path.stem}.json"
        output_path.write_text(json.dumps(result | {"output_path": str(output_path), "written": True}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["output_path"] = str(output_path)
        result["written"] = True
    return result


def complete_issue_for_review(
    *,
    project: str | Path,
    issue_id: str,
    completed_by: str,
    evidence: str,
    review_ai: str = "Opus 4.8",
) -> dict[str, Any]:
    root = _project_root(project)
    issue_path = _issue_path(root, issue_id)
    if not issue_path.exists():
        raise FileNotFoundError(f"Issue not found: {issue_path}")
    fields = parse_issue_fields(issue_path)
    update_issue_fields(
        issue_path,
        {
            "status": "ready_for_opus_review",
            "done_percent": "100",
            "remaining_percent": "0",
            "evidence": evidence,
            "next_agent": review_ai,
            "next_step": "Review implementation against the original plan and verification evidence.",
        },
    )
    request_dir = root / ".hermes" / "review-requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    request_path = request_dir / f"{issue_id}.md"
    request_path.write_text(
        f"""# Review Request - {issue_id}

issue_id: {issue_id}
title: {fields.get("title", issue_id)}
completed_by: {completed_by}
review_ai: {review_ai}
status: pending
created_at: {datetime.now().isoformat(timespec="seconds")}
owner_notification: {completed_by} marked {issue_id} ready for {review_ai} review.

## Evidence

{evidence}

## Review Checklist

- Compare implementation with the Opus plan.
- Verify the evidence commands are real and sufficient.
- If accepted, update issue status to `verified`.
- If changes are needed, update issue status to `changes_requested` and write concrete next steps.
""",
        encoding="utf-8",
    )
    return {
        "issue_path": str(issue_path),
        "review_request_path": str(request_path),
        "owner_notification": f"{completed_by} marked {issue_id} ready for {review_ai} review.",
    }


def _percent(value: str) -> float:
    try:
        return float(value.strip() or "0")
    except ValueError:
        return 0.0


def build_comply_report(project: str | Path) -> dict[str, Any]:
    root = _project_root(project)
    issue_dir = root / ".hermes" / "issues"
    issues: list[dict[str, Any]] = []
    for path in sorted(issue_dir.glob("*.md")) if issue_dir.exists() else []:
        if path.name == "README.md":
            continue
        fields = parse_issue_fields(path)
        issues.append(
            {
                "issue_id": fields.get("issue_id", path.stem),
                "phase": fields.get("phase", ""),
                "title": fields.get("title", path.stem),
                "assigned_ai": fields.get("assigned_ai", ""),
                "status": fields.get("status", ""),
                "done_percent": _percent(fields.get("done_percent", "0")),
                "remaining_percent": _percent(fields.get("remaining_percent", "100")),
                "path": str(path),
            }
        )
    count = len(issues)
    done = round(sum(issue["done_percent"] for issue in issues) / count, 2) if count else 0.0
    remaining = round(sum(issue["remaining_percent"] for issue in issues) / count, 2) if count else 0.0
    return {
        "project": str(root),
        "summary": {
            "issue_count": count,
            "done_percent": done,
            "remaining_percent": remaining,
        },
        "issues": issues,
    }


def list_review_requests(project: str | Path) -> dict[str, Any]:
    root = _project_root(project)
    review_dir = root / ".hermes" / "review-requests"
    items: list[dict[str, Any]] = []
    for path in sorted(review_dir.glob("*.md")) if review_dir.exists() else []:
        fields = parse_issue_fields(path)
        items.append(
            {
                "issue_id": fields.get("issue_id", path.stem),
                "review_ai": fields.get("review_ai", ""),
                "status": fields.get("status", ""),
                "path": str(path),
            }
        )
    pending = [item for item in items if item["status"] in {"", "pending"}]
    return {
        "pending_count": len(pending),
        "total_count": len(items),
        "items": items,
    }


def _is_completed_issue(path: Path) -> bool:
    fields = parse_issue_fields(path)
    return fields.get("status") in {"verified", "reviewed", "closed"} and _percent(
        fields.get("remaining_percent", "100")
    ) == 0


def compact_workflow(project: str | Path, execute: bool = False) -> dict[str, Any]:
    root = _project_root(project)
    hermes = root / ".hermes"
    archive_dir = hermes / "archive"
    archive_name = f"workflow-{datetime.now().strftime('%Y-%m')}.md"
    archive_path = archive_dir / archive_name
    candidates: list[Path] = []

    issue_dir = hermes / "issues"
    for path in sorted(issue_dir.glob("*.md")) if issue_dir.exists() else []:
        if path.name != "README.md" and _is_completed_issue(path):
            candidates.append(path)
    for folder in (hermes / "plans", hermes / "routes"):
        for path in sorted(folder.glob("*")) if folder.exists() else []:
            if path.is_file() and path.name != "README.md":
                candidates.append(path)

    if execute and candidates:
        archive_dir.mkdir(parents=True, exist_ok=True)
        chunks = [f"# Workflow Archive {datetime.now().strftime('%Y-%m')}\n"]
        if archive_path.exists():
            chunks.append(archive_path.read_text(encoding="utf-8"))
            chunks.append("\n")
        for path in candidates:
            rel = path.relative_to(root)
            chunks.append(f"\n## {rel}\n\n")
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
            chunks.append("\n")
        archive_path.write_text("".join(chunks), encoding="utf-8")
        for path in candidates:
            path.unlink()

    return {
        "project": str(root),
        "executed": execute,
        "archive_path": str(archive_path),
        "archived_count": len(candidates),
        "candidates": [str(path) for path in candidates],
    }


def render_comply_report(report: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    lines = [
        f"Project: {report['project']}",
        (
            "Summary: "
            f"{report['summary']['issue_count']} issues, "
            f"{report['summary']['done_percent']}% done, "
            f"{report['summary']['remaining_percent']}% remaining"
        ),
        "",
        "| Phase | Issue | Assigned AI | Status | Done % | Remaining % |",
        "|---|---|---|---|---:|---:|",
    ]
    for issue in report["issues"]:
        lines.append(
            "| {phase} | {issue_id} | {assigned_ai} | {status} | {done:g} | {remaining:g} |".format(
                phase=issue["phase"],
                issue_id=issue["issue_id"],
                assigned_ai=issue["assigned_ai"],
                status=issue["status"],
                done=issue["done_percent"],
                remaining=issue["remaining_percent"],
            )
        )
    return "\n".join(lines) + "\n"


def _load_checker_module():
    path = SCRIPT_DIR / "multi_ai_workflow_check.py"
    spec = importlib.util.spec_from_file_location("_multi_ai_workflow_check_runtime", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load checker module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_status_payload(project: str | Path) -> dict[str, Any]:
    root = _project_root(project)
    checker = _load_checker_module()
    return {
        "project": str(root),
        "readiness": checker.inspect_project(root),
        "comply": build_comply_report(root),
        "review_requests": list_review_requests(root),
    }


class _StatusHandler(BaseHTTPRequestHandler):
    project: Path

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            readiness = build_status_payload(self.project)["readiness"]
            self._json(200 if readiness["ok"] else 503, readiness)
            return
        if self.path == "/comply":
            self._json(200, build_comply_report(self.project))
            return
        if self.path == "/status":
            self._json(200, build_status_payload(self.project))
            return
        self._json(404, {"ok": False, "error": "not found"})


def serve_status(project: str | Path, host: str, port: int) -> None:
    root = _project_root(project)
    handler = type("StatusHandler", (_StatusHandler,), {"project": root})
    server = HTTPServer((host, port), handler)
    print(f"Serving multi-AI workflow status on http://{host}:{port}", flush=True)
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Bootstrap workflow files.")
    init_parser.add_argument("--project", default=".")
    init_parser.add_argument("--force", action="store_true")
    init_parser.add_argument("--format", choices=("text", "json"), default="text")

    issue_parser = subparsers.add_parser("issue", help="Manage workflow issues.")
    issue_subparsers = issue_parser.add_subparsers(dest="issue_command", required=True)

    create_parser = issue_subparsers.add_parser("create", help="Create an issue file.")
    create_parser.add_argument("--project", default=".")
    create_parser.add_argument("--issue-id", required=True)
    create_parser.add_argument("--phase", required=True)
    create_parser.add_argument("--title", required=True)
    create_parser.add_argument("--owner-role", required=True)
    create_parser.add_argument("--assigned-ai", default="unassigned")
    create_parser.add_argument("--reviewer-ai", default="")
    create_parser.add_argument("--goal", required=True)
    create_parser.add_argument("--scope", required=True)
    create_parser.add_argument("--out-of-scope", default="")
    create_parser.add_argument("--verify-command", action="append", default=[])
    create_parser.add_argument("--localhost-check", default="not applicable")
    create_parser.add_argument("--vps-check", default="not applicable")
    create_parser.add_argument("--branch", default="")
    create_parser.add_argument("--worktree-path", default="")
    create_parser.add_argument("--force", action="store_true")

    claim_parser = issue_subparsers.add_parser("claim", help="Claim an existing issue.")
    claim_parser.add_argument("--project", default=".")
    claim_parser.add_argument("--issue-id", required=True)
    claim_parser.add_argument("--assigned-ai", required=True)
    claim_parser.add_argument("--branch", required=True)
    claim_parser.add_argument("--worktree-path", required=True)

    update_parser = issue_subparsers.add_parser("update", help="Update issue status.")
    update_parser.add_argument("--project", default=".")
    update_parser.add_argument("--issue-id", required=True)
    update_parser.add_argument("--status", required=True)
    update_parser.add_argument("--done-percent", required=True)
    update_parser.add_argument("--remaining-percent", required=True)
    update_parser.add_argument("--evidence", required=True)

    complete_parser = issue_subparsers.add_parser("complete", help="Mark an issue ready for Opus review.")
    complete_parser.add_argument("--project", default=".")
    complete_parser.add_argument("--issue-id", required=True)
    complete_parser.add_argument("--completed-by", required=True)
    complete_parser.add_argument("--evidence", required=True)
    complete_parser.add_argument("--review-ai", default="Opus 4.8")
    complete_parser.add_argument("--format", choices=("text", "json"), default="text")

    comply_parser = subparsers.add_parser("comply", help="Summarize issue completion.")
    comply_parser.add_argument("--project", default=".")
    comply_parser.add_argument("--format", choices=("text", "json"), default="text")

    compact_parser = subparsers.add_parser("compact", help="Archive completed workflow files.")
    compact_parser.add_argument("--project", default=".")
    compact_parser.add_argument("--execute", action="store_true")
    compact_parser.add_argument("--format", choices=("text", "json"), default="text")

    ai_pair_parser = subparsers.add_parser("ai-pair", help="Manage Use AI Pair workflow state.")
    ai_pair_subparsers = ai_pair_parser.add_subparsers(dest="ai_pair_command", required=True)

    ai_pair_branch = ai_pair_subparsers.add_parser("branch", help="Propose an AI Pair branch.")
    ai_pair_branch.add_argument("--project", default=".")
    ai_pair_branch.add_argument("--issue-id", required=True)
    ai_pair_branch.add_argument("--task", required=True)

    ai_pair_init = ai_pair_subparsers.add_parser("init", help="Create an AI Pair job state folder.")
    ai_pair_init.add_argument("--project", default=".")
    ai_pair_init.add_argument("--issue-id", required=True)
    ai_pair_init.add_argument("--task", required=True)
    ai_pair_init.add_argument("--coder-ai", required=True)
    ai_pair_init.add_argument("--reviewer-ai", required=True)
    ai_pair_init.add_argument("--branch", required=True)
    ai_pair_init.add_argument("--gitlab-host", required=True)
    ai_pair_init.add_argument("--force", action="store_true")
    ai_pair_init.add_argument("--format", choices=("text", "json"), default="text")

    ai_pair_run = ai_pair_subparsers.add_parser("run", help="Run an AI Pair automation phase.")
    ai_pair_run.add_argument("phase", choices=("coder-plan",))
    ai_pair_run.add_argument("--project", default=".")
    ai_pair_run.add_argument("--issue-id", required=True)
    ai_pair_run.add_argument("--execute", action="store_true")
    ai_pair_run.add_argument("--coder-command", default="")
    ai_pair_run.add_argument("--timeout-seconds", type=int, default=300)
    ai_pair_run.add_argument("--format", choices=("text", "json"), default="text")

    ai_pair_review_result = ai_pair_subparsers.add_parser(
        "review-result", help="Record an AI Pair reviewer result and update retry state."
    )
    ai_pair_review_result.add_argument("--project", default=".")
    ai_pair_review_result.add_argument("--issue-id", required=True)
    ai_pair_review_result.add_argument("--review-file", default="")
    ai_pair_review_result.add_argument("--review-text", default="")
    ai_pair_review_result.add_argument("--format", choices=("text", "json"), default="text")

    ai_pair_desktop = ai_pair_subparsers.add_parser("desktop", help="Audit AI Pair desktop app seats.")
    ai_pair_desktop_subparsers = ai_pair_desktop.add_subparsers(
        dest="ai_pair_desktop_command", required=True
    )
    ai_pair_desktop_audit = ai_pair_desktop_subparsers.add_parser(
        "audit", help="Check Claude/Cursor/Codex app seats against the target project."
    )
    ai_pair_desktop_audit.add_argument("--project", default=".")
    ai_pair_desktop_audit.add_argument("--issue-id", required=True)
    ai_pair_desktop_audit.add_argument("--coach-ai", default="")
    ai_pair_desktop_audit.add_argument("--coder-ai", default="")
    ai_pair_desktop_audit.add_argument("--reviewer-ai", default="")
    ai_pair_desktop_audit.add_argument("--cua-driver", default="cua-driver")
    ai_pair_desktop_audit.add_argument("--format", choices=("text", "json"), default="text")
    ai_pair_desktop_handoff = ai_pair_desktop_subparsers.add_parser(
        "handoff", help="Queue a desktop handoff prompt for a specific AI role."
    )
    ai_pair_desktop_handoff.add_argument("--project", default=".")
    ai_pair_desktop_handoff.add_argument("--issue-id", required=True)
    ai_pair_desktop_handoff.add_argument("--role", choices=("coach", "coder", "reviewer"), required=True)
    ai_pair_desktop_handoff.add_argument("--phase", required=True)
    ai_pair_desktop_handoff.add_argument("--prompt-file", default="")
    ai_pair_desktop_handoff.add_argument("--prompt-text", default="")
    ai_pair_desktop_handoff.add_argument("--format", choices=("text", "json"), default="text")

    worktree_parser = subparsers.add_parser("worktree", help="Manage issue worktrees.")
    worktree_subparsers = worktree_parser.add_subparsers(dest="worktree_command", required=True)
    worktree_create = worktree_subparsers.add_parser("create", help="Create and claim a git worktree.")
    worktree_create.add_argument("--project", default=".")
    worktree_create.add_argument("--issue-id", required=True)
    worktree_create.add_argument("--assigned-ai", required=True)
    worktree_create.add_argument("--branch", required=True)
    worktree_create.add_argument("--worktree-path", required=True)
    worktree_create.add_argument("--execute", action="store_true")
    worktree_create.add_argument("--format", choices=("text", "json"), default="text")

    github_parser = subparsers.add_parser("github", help="Sync workflow issues to GitHub.")
    github_subparsers = github_parser.add_subparsers(dest="github_command", required=True)
    github_create = github_subparsers.add_parser("issue", help="Create a GitHub issue from a local issue.")
    github_create.add_argument("--project", default=".")
    github_create.add_argument("--issue-id", required=True)
    github_create.add_argument("--execute", action="store_true")
    github_create.add_argument("--format", choices=("text", "json"), default="text")

    handoff_parser = subparsers.add_parser("handoff", help="Write project handoff state.")
    handoff_subparsers = handoff_parser.add_subparsers(dest="handoff_command", required=True)
    handoff_write = handoff_subparsers.add_parser("write", help="Write .hermes/handoff.md.")
    handoff_write.add_argument("--project", default=".")
    handoff_write.add_argument("--task", required=True)
    handoff_write.add_argument("--issue-id", required=True)
    handoff_write.add_argument("--phase", required=True)
    handoff_write.add_argument("--latest-state", required=True)
    handoff_write.add_argument("--next-agent", required=True)
    handoff_write.add_argument("--next-step", required=True)
    handoff_write.add_argument("--verification-run", required=True)
    handoff_write.add_argument("--localhost-result", required=True)
    handoff_write.add_argument("--vps-result", required=True)
    handoff_write.add_argument("--remaining-risk", required=True)

    status_parser = subparsers.add_parser("status", help="Print read-only project status.")
    status_parser.add_argument("--project", default=".")
    status_parser.add_argument("--format", choices=("json",), default="json")

    route_parser = subparsers.add_parser("route", help="Recommend the next AI executor for an Opus plan.")
    route_parser.add_argument("--project", default=".")
    route_parser.add_argument("--plan-file", required=True)
    route_parser.add_argument("--write", action="store_true")
    route_parser.add_argument("--format", choices=("text", "json"), default="text")

    serve_parser = subparsers.add_parser("serve", help="Serve read-only status on localhost.")
    serve_parser.add_argument("--project", default=".")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)

    if args.command == "init":
        result = init_project(args.project, force=args.force)
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(
                f"Initialized {result['project']}: "
                f"{result['created_count']} created, {result['skipped_count']} skipped"
            )
        return 0

    if args.command == "issue" and args.issue_command == "create":
        path = create_issue(
            project=args.project,
            issue_id=args.issue_id,
            phase=args.phase,
            title=args.title,
            owner_role=args.owner_role,
            assigned_ai=args.assigned_ai,
            goal=args.goal,
            scope=args.scope,
            out_of_scope=args.out_of_scope,
            verify_commands=args.verify_command,
            localhost_check=args.localhost_check,
            vps_check=args.vps_check,
            branch=args.branch,
            worktree_path=args.worktree_path,
            reviewer_ai=args.reviewer_ai,
            force=args.force,
        )
        print(str(path))
        return 0

    if args.command == "issue" and args.issue_command == "claim":
        path = claim_issue(
            project=args.project,
            issue_id=args.issue_id,
            assigned_ai=args.assigned_ai,
            branch=args.branch,
            worktree_path=args.worktree_path,
        )
        print(str(path))
        return 0

    if args.command == "issue" and args.issue_command == "update":
        path = update_issue_status(
            project=args.project,
            issue_id=args.issue_id,
            status=args.status,
            done_percent=args.done_percent,
            remaining_percent=args.remaining_percent,
            evidence=args.evidence,
        )
        print(str(path))
        return 0

    if args.command == "issue" and args.issue_command == "complete":
        result = complete_issue_for_review(
            project=args.project,
            issue_id=args.issue_id,
            completed_by=args.completed_by,
            evidence=args.evidence,
            review_ai=args.review_ai,
        )
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(result["owner_notification"])
            print(result["review_request_path"])
        return 0

    if args.command == "comply":
        print(render_comply_report(build_comply_report(args.project), args.format), end="")
        return 0

    if args.command == "compact":
        result = compact_workflow(args.project, execute=args.execute)
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            mode = "executed" if result["executed"] else "dry-run"
            print(f"{mode}: {result['archived_count']} files -> {result['archive_path']}")
        return 0

    if args.command == "ai-pair" and args.ai_pair_command == "branch":
        print(
            json.dumps(
                propose_ai_pair_branch(project=args.project, issue_id=args.issue_id, task=args.task),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "ai-pair" and args.ai_pair_command == "init":
        result = create_ai_pair_job(
            project=args.project,
            issue_id=args.issue_id,
            task=args.task,
            coder_ai=args.coder_ai,
            reviewer_ai=args.reviewer_ai,
            branch=args.branch,
            gitlab_host=args.gitlab_host,
            force=args.force,
        )
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"{result['status']}: {result['pair_dir']}")
        return 0

    if args.command == "ai-pair" and args.ai_pair_command == "run":
        result = run_ai_pair_coder_plan(
            project=args.project,
            issue_id=args.issue_id,
            execute=args.execute,
            coder_command=args.coder_command,
            timeout_seconds=args.timeout_seconds,
        )
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"{result['status']}: {result.get('reason', result.get('coder_plan_path', ''))}")
        return 0 if result["ok"] else 2

    if args.command == "ai-pair" and args.ai_pair_command == "review-result":
        review_text = args.review_text
        if args.review_file:
            review_path = Path(args.review_file).expanduser()
            if not review_path.is_absolute():
                review_path = (_project_root(args.project) / review_path).resolve()
            review_text = review_path.read_text(encoding="utf-8")
        if not review_text.strip():
            raise ValueError("--review-text or --review-file is required")
        result = record_ai_pair_review_result(
            project=args.project,
            issue_id=args.issue_id,
            review_text=review_text,
        )
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"{result['status']}: {result['decision']}")
        return 0 if result["ok"] else 2

    if args.command == "ai-pair" and args.ai_pair_command == "desktop":
        if args.ai_pair_desktop_command == "audit":
            result = audit_ai_pair_desktop(
                project=args.project,
                issue_id=args.issue_id,
                coach_ai=args.coach_ai,
                coder_ai=args.coder_ai,
                reviewer_ai=args.reviewer_ai,
                cua_driver=args.cua_driver,
            )
            if args.format == "json":
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                status = "OK" if result["ok"] else "BLOCKED"
                print(f"Desktop seat audit: {status}")
                for role, detail in result["roles"].items():
                    print(
                        f"- {role}: {detail['ai']} running={detail['running']} "
                        f"usable_windows={detail['usable_window_count']} "
                        f"project_windows={detail['project_window_count']} ok={detail['ok']}"
                    )
            return 0 if result["ok"] else 2
        prompt_text = args.prompt_text
        if args.prompt_file:
            prompt_path = Path(args.prompt_file).expanduser()
            if not prompt_path.is_absolute():
                prompt_path = (_project_root(args.project) / prompt_path).resolve()
            prompt_text = prompt_path.read_text(encoding="utf-8")
        if not prompt_text.strip():
            raise ValueError("--prompt-text or --prompt-file is required")
        result = prepare_ai_pair_desktop_handoff(
            project=args.project,
            issue_id=args.issue_id,
            role=args.role,
            phase=args.phase,
            prompt_text=prompt_text,
        )
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"{result['status']}: {result['prompt_path']}")
        return 0

    if args.command == "worktree" and args.worktree_command == "create":
        result = create_worktree(
            project=args.project,
            issue_id=args.issue_id,
            assigned_ai=args.assigned_ai,
            branch=args.branch,
            worktree_path=args.worktree_path,
            execute=args.execute,
        )
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            mode = "executed" if result["executed"] else "dry-run"
            print(f"{mode}: {' '.join(result['command'])}")
            if result["returncode"] is not None:
                print(f"returncode: {result['returncode']}")
        return 0 if result["returncode"] in (None, 0) else int(result["returncode"])

    if args.command == "github" and args.github_command == "issue":
        result = github_issue_sync(
            project=args.project,
            issue_id=args.issue_id,
            execute=args.execute,
        )
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            mode = "executed" if result["executed"] else "dry-run"
            print(f"{mode}: {' '.join(result['command'])}")
            if result["returncode"] is not None:
                print(f"returncode: {result['returncode']}")
        return 0 if result["returncode"] in (None, 0) else int(result["returncode"])

    if args.command == "handoff" and args.handoff_command == "write":
        path = write_handoff(
            project=args.project,
            task=args.task,
            issue_id=args.issue_id,
            phase=args.phase,
            latest_state=args.latest_state,
            next_agent=args.next_agent,
            next_step=args.next_step,
            verification_run=args.verification_run,
            localhost_result=args.localhost_result,
            vps_result=args.vps_result,
            remaining_risk=args.remaining_risk,
        )
        print(str(path))
        return 0

    if args.command == "status":
        print(json.dumps(build_status_payload(args.project), ensure_ascii=False, indent=2))
        return 0

    if args.command == "route":
        result = route_plan_file(
            project=args.project,
            plan_file=args.plan_file,
            write=args.write,
        )
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            primary = result["recommendation"]["primary"]
            print(f"Recommended: {primary['tool']} ({primary['id']})")
            print(f"Reason: {result['recommendation']['reason']}")
            print("")
            print("Ranked options:")
            for option in result["recommendation"]["ranked"]:
                signals = ", ".join(option["matched_signals"]) or "none"
                print(
                    f"- {option['tool']}: {option['suitability_percent']}% "
                    f"(score {option['score']}; {signals})"
                )
            if result["written"]:
                print(f"Written: {result['output_path']}")
        return 0

    if args.command == "serve":
        serve_status(args.project, args.host, args.port)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
