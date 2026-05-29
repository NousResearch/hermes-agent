
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised through isolated subprocess smoke tests
    class _MinimalYaml:
        """Tiny YAML subset for go-workflow's generated state files.

        Supports the mapping/list/scalar shape emitted by `safe_dump` below. It
        is intentionally not a general YAML parser; install PyYAML for arbitrary
        YAML syntax.
        """

        @staticmethod
        def safe_load(text: str) -> Any:
            lines = [line.rstrip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
            if not lines:
                return None

            def scalar(value: str) -> Any:
                value = value.strip()
                if value in {"", "null", "~"}:
                    return None
                if value == "true":
                    return True
                if value == "false":
                    return False
                if value == "[]":
                    return []
                if value == "{}":
                    return {}
                if value.startswith('"') and value.endswith('"'):
                    return json.loads(value)
                if value.startswith("'") and value.endswith("'"):
                    return value[1:-1].replace("''", "'")
                try:
                    return int(value)
                except ValueError:
                    return value

            def parse_block(index: int, indent: int) -> tuple[Any, int]:
                if index >= len(lines):
                    return {}, index
                stripped = lines[index][indent:]
                if stripped == "-" or stripped.startswith("- "):
                    result: list[Any] = []
                    while index < len(lines):
                        line = lines[index]
                        current_indent = len(line) - len(line.lstrip(" "))
                        marker = line[indent:]
                        if current_indent != indent or not (marker == "-" or marker.startswith("- ")):
                            break
                        item = "" if marker == "-" else marker[2:]
                        if not item:
                            value, index = parse_block(index + 1, indent + 2)
                            result.append(value)
                        elif re.match(r"^[A-Za-z0-9_ -]+:\s*", item):
                            key, rest = item.split(":", 1)
                            mapping: dict[str, Any] = {key.strip(): scalar(rest) if rest.strip() else None}
                            index += 1
                            while index < len(lines):
                                child = lines[index]
                                child_indent = len(child) - len(child.lstrip(" "))
                                if child_indent <= indent:
                                    break
                                if child_indent == indent + 2 and not child[child_indent:].startswith("- "):
                                    child_key, child_rest = child[child_indent:].split(":", 1)
                                    if child_rest.strip():
                                        mapping[child_key.strip()] = scalar(child_rest)
                                        index += 1
                                    else:
                                        value, index = parse_block(index + 1, child_indent + 2)
                                        mapping[child_key.strip()] = value
                                else:
                                    break
                            result.append(mapping)
                        else:
                            result.append(scalar(item))
                            index += 1
                    return result, index
                mapping: dict[str, Any] = {}
                while index < len(lines):
                    line = lines[index]
                    current_indent = len(line) - len(line.lstrip(" "))
                    if current_indent != indent or ":" not in line[current_indent:]:
                        break
                    key, rest = line[current_indent:].split(":", 1)
                    if rest.strip():
                        mapping[key.strip()] = scalar(rest)
                        index += 1
                    else:
                        value, index = parse_block(index + 1, indent + 2)
                        mapping[key.strip()] = value
                return mapping, index

            parsed, _ = parse_block(0, 0)
            return parsed

        @staticmethod
        def safe_dump(data: Any, sort_keys: bool = False, allow_unicode: bool = True) -> str:
            def scalar(value: Any) -> str:
                if value is None:
                    return "null"
                if value is True:
                    return "true"
                if value is False:
                    return "false"
                if isinstance(value, (int, float)):
                    return str(value)
                text = str(value)
                if not text or text.strip() != text or text.lower() in {"null", "true", "false", "~"} or any(ch in text for ch in ":#[]{}\n\"'"):
                    return json.dumps(text, ensure_ascii=not allow_unicode)
                return text

            def dump(value: Any, indent: int = 0) -> list[str]:
                prefix = " " * indent
                if isinstance(value, dict):
                    items = value.items()
                    if sort_keys:
                        items = sorted(items)
                    lines: list[str] = []
                    for key, item in items:
                        if isinstance(item, list) and not item:
                            lines.append(f"{prefix}{key}: []")
                        elif isinstance(item, dict) and not item:
                            lines.append(f"{prefix}{key}: {{}}")
                        elif isinstance(item, (dict, list)):
                            lines.append(f"{prefix}{key}:")
                            lines.extend(dump(item, indent + 2))
                        else:
                            lines.append(f"{prefix}{key}: {scalar(item)}")
                    return lines
                if isinstance(value, list):
                    if not value:
                        return [f"{prefix}[]"]
                    lines = []
                    for item in value:
                        if isinstance(item, dict):
                            lines.append(f"{prefix}-")
                            lines.extend(dump(item, indent + 2))
                        elif isinstance(item, list):
                            lines.append(f"{prefix}-")
                            lines.extend(dump(item, indent + 2))
                        else:
                            lines.append(f"{prefix}- {scalar(item)}")
                    return lines
                return [f"{prefix}{scalar(value)}"]

            return "\n".join(dump(data)) + "\n"

    yaml = _MinimalYaml()

VALID_STATUSES = {"todo", "ready", "claimed", "active", "waiting", "review", "done", "cancelled"}
FLOW_DIR = Path(".go-workflow")
HERMES_COMPAT_DIR = Path(".hermes")
TASKS_PATH = FLOW_DIR / "tasks.yaml"
GOALS_PATH = FLOW_DIR / "goals.yaml"
CONFIG_PATH = FLOW_DIR / "config.yaml"
GATES_PATH = FLOW_DIR / "gates.yaml"
RUNS_DIR = FLOW_DIR / "runs"
RUNTIME_DIR = FLOW_DIR / "runtime"
PROJECT_DOC_PATH = FLOW_DIR / "project.md"
PROMPTS_DIR = FLOW_DIR / "prompts"
SKILLS_DIR = FLOW_DIR / "skills" / "go-workflow"

PHASES = [
    ("setup", "SETUP", "Establish repository context before planning or editing."),
    ("plan", "PLAN", "Turn intent and repo state into bounded requirements and acceptance checks."),
    ("route-claim", "ROUTE/CLAIM", "Select exactly one executable task and write an exclusive handoff."),
    ("build", "BUILD", "Change only the claimed task's allowed modify scope."),
    ("verify", "VERIFY", "Run the task and repository checks that prove the change."),
    ("docs-ledger", "DOCS/LEDGER", "Update docs, task state, evidence, and run ledgers."),
    ("devil", "DEVIL", "Run adversarial review for risky or multi-file changes."),
    ("antislop", "ANTISLOP", "Remove sloppy artifacts before shipping."),
    ("ship", "SHIP", "Commit, push or PR, and report final git evidence."),
]

PHASE_CONTRACTS = {
    "setup": {
        "inputs": ["AGENTS.md", ".go-workflow/config.yaml", ".go-workflow/goals.yaml", ".go-workflow/tasks.yaml", ".go-workflow/gates.yaml", "git status"],
        "outputs": ["current repo state", "available task queue", "known dirty/untracked files"],
        "allowed_mutations": ["none; read-only unless explicit setup/hygiene task"],
        "required_evidence": ["validation command output", "git status summary"],
        "stop_conditions": ["missing workflow files", "dirty state that overlaps intended task", "unknown repo contract"],
        "handoff": "Repo context is explicit enough for PLAN without guessing.",
    },
    "plan": {
        "inputs": ["claimed or requested task", "goals.yaml", "tasks.yaml", "project docs"],
        "outputs": ["requirement/acceptance mapping", "docs impact", "verification plan"],
        "allowed_mutations": ["tasks.yaml/goals.yaml only when the task is queue planning"],
        "required_evidence": ["task has acceptance", "task has verification", "dependencies known"],
        "stop_conditions": ["ambiguous source of truth", "missing acceptance", "unresolved dependency"],
        "handoff": "A bounded task is ready to be claimed or executed.",
    },
    "route-claim": {
        "inputs": ["ready task", "dependency status", "agent id"],
        "outputs": ["exclusive claim", "handoff markdown", "lease metadata"],
        "allowed_mutations": [".go-workflow/tasks.yaml claim fields", ".go-workflow/runs/*-handoff.md"],
        "required_evidence": ["claim written", "handoff path", "scope read/modify listed"],
        "stop_conditions": ["task not ready", "dependencies not done", "overlapping active claim"],
        "handoff": "Worker reads handoff and stays inside scope.modify.",
    },
    "build": {
        "inputs": ["handoff", "scope.modify", "acceptance criteria"],
        "outputs": ["focused code/docs changes"],
        "allowed_mutations": ["only paths listed in scope.modify"],
        "required_evidence": ["changed path list", "scope compliance note"],
        "stop_conditions": ["needed file outside scope", "unrelated dirty file", "unsafe data loss risk"],
        "handoff": "Changes are ready for mechanical verification.",
    },
    "verify": {
        "inputs": ["task verification list", "repo gates", "changed files"],
        "outputs": ["command results", "failure fixes or blocked status"],
        "allowed_mutations": ["fixes needed to satisfy acceptance within scope"],
        "required_evidence": ["task verification output", "repo validation output"],
        "stop_conditions": ["failing required check", "missing dependency", "unreproducible result"],
        "handoff": "All required checks have explicit pass/fail evidence.",
    },
    "docs-ledger": {
        "inputs": ["accepted changes", "docs policy", "evidence"],
        "outputs": ["updated docs", "updated tasks.yaml/tasks.md", "run ledger"],
        "allowed_mutations": ["docs/update paths", ".go-workflow/tasks.yaml", "tasks.md", ".go-workflow/runs/*-ledger.md"],
        "required_evidence": ["docs changed or checked/no-change reason", "ledger path", "task evidence"],
        "stop_conditions": ["docs drift", "missing evidence", "human cockpit stale"],
        "handoff": "Task record and docs describe what changed and why.",
    },
    "devil": {
        "inputs": ["diff", "task acceptance", "verification evidence", "docs decision"],
        "outputs": ["adversarial verdict", "fix list or approval"],
        "allowed_mutations": ["review notes; fixes only after returning to BUILD/VERIFY"],
        "required_evidence": ["verdict A/B/C/D or explicit not-required reason"],
        "stop_conditions": ["verdict C/D", "unreviewed risky auth/data/scheduling change"],
        "handoff": "Either safe-to-ship verdict or concrete fixes.",
    },
    "antislop": {
        "inputs": ["diff", "docs", "final report draft"],
        "outputs": ["cleaned comments/docs/tasks", "secret/slop scan result"],
        "allowed_mutations": ["small cleanup inside already allowed paths"],
        "required_evidence": ["diff check", "no placeholders/secrets note"],
        "stop_conditions": ["placeholder TODOs", "AI-ish filler", "secret-like strings"],
        "handoff": "Change set is clean enough to ship.",
    },
    "ship": {
        "inputs": ["verified diff", "ledger", "git status", "commit policy"],
        "outputs": ["commit/PR/push evidence", "final branch state"],
        "allowed_mutations": ["git index/history for relevant files only"],
        "required_evidence": ["commit or PR link", "push/ahead-behind status", "final git status"],
        "stop_conditions": ["failing checks", "unrelated staged files", "behind remote without safe rebase"],
        "handoff": "Final user report includes tasks, commits, verification, docs, and open risks.",
    },
}


SUPPORT_SKILLS = {
    "audit": {
        "name": "AUDIT",
        "summary": "Run selectable quality gates for code, tests, architecture, API, performance, accessibility, UX, and security-style checks.",
        "inputs": ["claimed task acceptance", "diff or target path", "verification output", "docs impact", "selected audit profiles"],
        "outputs": ["profile-specific findings", "severity/ship verdict", "required fixes or explicit pass evidence"],
        "allowed_mutations": ["review notes only; fixes must return to BUILD/VERIFY before finish"],
        "required_evidence": ["audit_profile_selected", "audit_evidence_recorded"],
        "stop_conditions": ["unreviewed risky code path", "security/data/auth/perf concern without disposition", "missing selected audit evidence"],
        "commands": ["code", "test", "arch", "api", "perf", "a11y", "ux", "security", "ship", "diff", "re-audit"],
    },
    "git": {
        "name": "GIT ROUTER",
        "summary": "Keep repository state safe: status hygiene, selective staging, ship, CI repair, PR/check inspection, and clean return to main.",
        "inputs": ["git status", "claimed task scope.modify", "commit policy", "remote/upstream state", "CI/PR state when present"],
        "outputs": ["clean or explained git state", "focused commit/PR", "CI/check evidence", "main/default-branch return evidence"],
        "allowed_mutations": ["git index/history for in-scope paths only", "branches/PRs when the task commit policy allows it"],
        "required_evidence": ["git_status_clean_or_explained", "selective_staging_used", "ci_or_pr_status_checked"],
        "stop_conditions": ["unrelated dirty files would be mixed", "behind remote without safe rebase", "failing required checks", "cannot return to main/default branch"],
        "commands": ["status", "stage", "commit", "ship", "fix-ci", "pr-checks", "return-main"],
    },
    "release": {
        "name": "RELEASE",
        "summary": "Separate release/version/tag/notes behavior from generic docs-ledger work and make final publish evidence explicit.",
        "inputs": ["completed task ledgers", "version policy", "release scope", "git/PR status", "explicit publish flag for external side effects"],
        "outputs": ["version bump or no-release rationale", "tag/release notes plan", "publish URL or dry-run evidence"],
        "allowed_mutations": ["configured version files", "release notes", "tags/releases only when explicitly requested"],
        "required_evidence": ["release_scope_confirmed", "version_or_no_release_recorded", "final_ship_evidence_recorded"],
        "stop_conditions": ["verification/docs/devil/antislop gates not passed", "unclear version bump", "implicit GitHub release side effect"],
        "commands": ["dry-run", "version", "notes", "tag", "publish"],
    },
    "interview": {
        "name": "INTERVIEW",
        "summary": "Ambiguity gate for work that cannot be safely planned without a small number of targeted questions.",
        "inputs": ["user request", "repo contract", "known tasks/goals", "ambiguity score"],
        "outputs": ["minimum clarifying questions", "assumptions", "blocked/waiting task state when needed"],
        "allowed_mutations": ["tasks.yaml status/notes only when recording a real waiting state"],
        "required_evidence": ["ambiguity_assessed", "questions_minimized_or_assumptions_recorded"],
        "stop_conditions": ["choice changes implementation path materially", "missing credentials", "risk of data loss"],
        "commands": ["score", "ask", "assume", "block"],
    },
    "cancel": {
        "name": "CANCEL",
        "summary": "Gracefully stop, supersede, or cancel obsolete workflow tasks without corrupting queue or git state.",
        "inputs": ["task id or scope to cancel", "current claim status", "dirty files", "superseding decision"],
        "outputs": ["cancelled/superseded task state", "rollback or preserved handoff", "clear open-risk note"],
        "allowed_mutations": ["task status/notes", "ledger note", "targeted revert only when explicitly safe"],
        "required_evidence": ["cancel_scope_identified", "state_preserved_or_reverted", "next_action_recorded"],
        "stop_conditions": ["unclear whether user wants cancel vs pause", "uncommitted valuable work would be lost", "claimed task by another agent"],
        "commands": ["cancel-task", "supersede", "rollback-claim", "preserve-notes"],
    },
    "better": {
        "name": "BETTER",
        "summary": "Post-run improvement loop: turn a completed task/session into better workflow rules, tasks, tests, docs, or skill updates.",
        "inputs": ["completed ledger", "verification failures", "review feedback", "user correction", "repo contract"],
        "outputs": ["improvement tasks", "skill/doc patches", "tests or gates to prevent recurrence"],
        "allowed_mutations": ["tasks.yaml/goals.yaml", "docs", "skills", "tests when the improvement task is claimed"],
        "required_evidence": ["improvement_source_recorded", "reusable_change_captured_or_declined"],
        "stop_conditions": ["one-off task progress being mistaken for durable knowledge", "unverified process change"],
        "commands": ["reflect", "add-task", "patch-skill", "add-gate", "decline"],
    },
}


def support_skill_md(skill_id: str, contract: dict[str, Any]) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items)
    commands = ", ".join(f"`{item}`" for item in contract["commands"])
    return f"""---
name: go-workflow-{skill_id}
description: {contract['name']} support skill for repo-contained go-workflow runs. {contract['summary']}
version: 1.0.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, support, {skill_id}]
---

# go-workflow support skill: {contract['name']}

## Purpose

{contract['summary']}

## Commands / modes

{commands}

## Inputs

{bullets(contract['inputs'])}

## Outputs

{bullets(contract['outputs'])}

## Allowed mutations

{bullets(contract['allowed_mutations'])}

## Required evidence

{bullets(contract['required_evidence'])}

## Failure / stop conditions

{bullets(contract['stop_conditions'])}

## Handoff contract

Record required evidence with `python3 scripts/gate.py --task-id <TASK-ID> --phase {skill_id} --evidence key=value` before depending on this support gate.
"""


def phase_skill_md(phase_id: str, phase_name: str, summary: str) -> str:
    c = PHASE_CONTRACTS[phase_id]
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items)
    return f"""---
name: go-workflow-{phase_id}
description: {phase_name} phase for repo-contained go-workflow runs. {summary}
version: 1.0.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, phase, {phase_id}]
---

# go-workflow phase: {phase_name}

## Purpose

{summary}

## Inputs

{bullets(c['inputs'])}

## Outputs

{bullets(c['outputs'])}

## Allowed mutations

{bullets(c['allowed_mutations'])}

## Required evidence

{bullets(c['required_evidence'])}

## Failure / stop conditions

{bullets(c['stop_conditions'])}

## Handoff contract

{c['handoff']}
"""


def root_skill_md() -> str:
    phase_lines = "\n".join(f"- `go-workflow-{pid}` — {name}: {summary}" for pid, name, summary in PHASES)
    support_lines = "\n".join(f"- `go-workflow-{sid}` — {c['name']}: {c['summary']}" for sid, c in SUPPORT_SKILLS.items())
    return f"""---
name: go-workflow
description: Use when operating a repo-contained GO workflow with goals.yaml, tasks.yaml, gates, claim/finish lifecycle, phase skills, and agent prompts for Hermes, Codex, Claude, and Gemini.
version: 1.1.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, repo-local, orchestration, tasks, goals, agents, phase-gates]
---

# go-workflow Skill Bundle

## Core Rule

The repository is the source of truth. Chat state, Kanban boards, and model memory are runtime aids only.

## Workflow

SETUP → PLAN → ROUTE/CLAIM → BUILD → VERIFY → DOCS/LEDGER → DEVIL → ANTISLOP → SHIP

## Phase skills

{phase_lines}

## Support skills / routers

{support_lines}

Load the root skill first, then the phase or support skill matching the current gate/router when you need detailed inputs, outputs, allowed mutations, evidence, stop conditions, and handoff rules.

## Entry Points

```bash
python3 scripts/next_task.py --validate
python3 scripts/next_task.py --list --limit 5
python3 scripts/next_task.py --claim --agent <hermes|codex|claude|gemini|name>
python3 scripts/finish_task.py <TASK-ID> --evidence "<path-or-command>" --agent <name>
python3 scripts/gate.py --task-id <TASK-ID> --phase verify --evidence "task_verification_run=pytest passed"
```

Every workflow trigger must show the upcoming task preview before claiming or executing work. Use `python3 scripts/next_task.py --list --limit 5` and include that short list in the agent's first checkpoint/report.

## Rules

- Read `AGENTS.md` first, then `.go-workflow/config.yaml`, `goals.yaml`, `tasks.yaml`, and `gates.yaml`.
- Claim exactly one `ready` task before editing.
- Stay inside `scope.modify` unless the task is updated first.
- Do not claim `todo`, `waiting`, `review`, `done`, or `cancelled` work.
- Do not finish without evidence.
- For risky/multi-file changes, run a devil review before ship.
- Commit one task at a time unless tasks are inseparable.
"""



def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def yaml_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return yaml.safe_load(path.read_text()) or default


def yaml_save(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def load_tasks() -> dict[str, Any]:
    if not TASKS_PATH.exists():
        raise SystemExit(f"missing {TASKS_PATH}; run: go-workflow init --project <id>")
    data = yaml_load(TASKS_PATH, {})
    data.setdefault("version", 1)
    data.setdefault("project", "unknown")
    data.setdefault("tasks", [])
    return data


def save_tasks(data: dict[str, Any]) -> None:
    yaml_save(TASKS_PATH, data)


def load_goals() -> dict[str, Any]:
    data = yaml_load(GOALS_PATH, {})
    data.setdefault("version", 1)
    data.setdefault("project", "unknown")
    data.setdefault("goals", [])
    return data


def save_goals(data: dict[str, Any]) -> None:
    yaml_save(GOALS_PATH, data)


def task_by_id(data: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    return next((task for task in data.get("tasks", []) if task.get("id") == task_id), None)


def goal_ids(goals: dict[str, Any]) -> set[str]:
    return {goal.get("id") for goal in goals.get("goals", []) if goal.get("id")}


def normalize_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def default_config(project: str, name: str | None) -> dict[str, Any]:
    return {
        "version": 1,
        "name": "go-workflow",
        "project": {"id": project, "name": name or project},
        "paths": {
            "root": ".go-workflow",
            "goals": ".go-workflow/goals.yaml",
            "tasks": ".go-workflow/tasks.yaml",
            "gates": ".go-workflow/gates.yaml",
            "runs": ".go-workflow/runs",
            "prompts": ".go-workflow/prompts",
            "skills": ".go-workflow/skills",
            "runtime": ".go-workflow/runtime",
        },
        "workflow": ["setup", "plan", "route_claim", "build", "verify", "docs_ledger", "devil", "antislop", "ship"],
        "agents": ["hermes", "codex", "claude", "gemini"],
        "policy": {
            "repo_is_source_of_truth": True,
            "claim_before_edit": True,
            "finish_requires_evidence": True,
            "commit_per_task": True,
            "parallel_requires_non_overlapping_modify_scope": True,
        },
    }


def project_doc(project: str, name: str | None) -> str:
    return f"""# go-workflow Project Contract — {name or project}

This repository uses go-workflow as its local source of truth for agentic work.

## Canonical files

- `.go-workflow/config.yaml` — policy and paths.
- `.go-workflow/goals.yaml` — durable outcomes.
- `.go-workflow/tasks.yaml` — machine-readable task queue.
- `.go-workflow/gates.yaml` — phase gates.
- `.go-workflow/prompts/` — model-specific prompts.
- `.go-workflow/skills/go-workflow/SKILL.md` — portable root skill bundle.
- `.go-workflow/skills/go-workflow-<phase>/SKILL.md` — phase skills for setup, plan, route-claim, build, verify, docs-ledger, devil, antislop, and ship.
- `.go-workflow/skills/go-workflow-<support>/SKILL.md` — support routers for audit, git, release, interview, cancel, and better.
- `.go-workflow/runtime/go_workflow/` — embedded runtime used by `scripts/` wrappers, so target repos can run without pip install.
- `.hermes/skills/go-workflow*/SKILL.md` — compatibility view for Hermes-style skill discovery.
- `scripts/next_task.py`, `scripts/finish_task.py`, and `scripts/gate.py` — stable self-contained repo entrypoints.

## Workflow

SETUP → PLAN → ROUTE/CLAIM → BUILD → VERIFY → DOCS/LEDGER → DEVIL → ANTISLOP → SHIP
"""


def default_gates() -> dict[str, Any]:
    return {
        "version": 1,
        "gates": {
            "plan": ["goal_exists", "task_has_acceptance", "task_has_verification", "dependencies_known"],
            "route_claim": ["status_ready", "dependencies_done", "exclusive_claim_written"],
            "build": ["modify_scope_respected", "no_unrelated_dirty_files"],
            "verify": ["task_verification_run", "repo_validation_passes"],
            "docs_ledger": ["evidence_recorded", "ledger_written", "human_cockpit_updated_if_present"],
            "devil": ["required_for_risky_or_multifile_changes"],
            "audit": ["audit_profile_selected", "audit_evidence_recorded"],
            "git": ["git_status_clean_or_explained", "selective_staging_used", "ci_or_pr_status_checked"],
            "release": ["release_scope_confirmed", "version_or_no_release_recorded", "final_ship_evidence_recorded"],
            "antislop": ["diff_check_clean"],
            "ship": ["git_status_reported", "commit_or_pr_linked"],
        },
        "phase_evidence_required": {
            "plan": ["task_has_acceptance", "task_has_verification", "dependencies_known"],
            "route_claim": ["exclusive_claim_written"],
            "build": ["modify_scope_respected"],
            "verify": ["task_verification_run", "repo_validation_passes"],
            "docs_ledger": ["evidence_recorded", "ledger_written"],
            "devil": ["devil_review_or_not_required"],
            "audit": ["audit_profile_selected", "audit_evidence_recorded"],
            "git": ["git_status_clean_or_explained", "selective_staging_used", "ci_or_pr_status_checked"],
            "release": ["release_scope_confirmed", "version_or_no_release_recorded", "final_ship_evidence_recorded"],
            "antislop": ["diff_check_clean"],
            "ship": ["git_status_reported", "commit_or_pr_linked"],
        },
    }


def skill_md() -> str:
    return root_skill_md()


def prompt(name: str) -> str:
    return f"""# go-workflow prompt for {name}

You are working in a repository that uses **go-workflow**.

## Mandatory startup

1. Read `AGENTS.md`.
2. Read `.go-workflow/config.yaml`.
3. Read `.go-workflow/goals.yaml` and `.go-workflow/tasks.yaml`.
4. Validate: `python3 scripts/next_task.py --validate`.
5. Show the next tasks first: `python3 scripts/next_task.py --list --limit 5`.
6. Claim exactly one task: `python3 scripts/next_task.py --claim --agent {name}`.
7. Read the generated handoff in `.go-workflow/runs/`.

## Workflow phases

SETUP → PLAN → ROUTE/CLAIM → BUILD → VERIFY → DOCS/LEDGER → DEVIL → ANTISLOP → SHIP

## Non-negotiables

- The repo is the source of truth.
- Stay inside `scope.modify` for the claimed task.
- Run the listed verification.
- Finish only with explicit evidence.
- Report changed files, verification, evidence, and git state.
"""


def agents_md(project: str, name: str | None) -> str:
    return f"""# Agent Instructions — go-workflow

This repo uses **go-workflow**: a repo-contained, model-agnostic workflow for Hermes, Codex, Claude, Gemini, and other agents.

## Startup

1. Read `.go-workflow/config.yaml`.
2. Read `.go-workflow/goals.yaml`.
3. Read `.go-workflow/tasks.yaml`.
4. Read `.go-workflow/gates.yaml`.
5. Run `python3 scripts/next_task.py --validate`.
6. Show the next tasks first with `python3 scripts/next_task.py --list --limit 5`; include that short list in the first checkpoint/report for every workflow trigger.
7. Claim one task before editing: `python3 scripts/next_task.py --claim --agent <agent-name>`.
8. Read the generated handoff in `.go-workflow/runs/`.

## Workflow

SETUP → PLAN → ROUTE/CLAIM → BUILD → VERIFY → DOCS/LEDGER → DEVIL → ANTISLOP → SHIP

## Rules

- Repository files are the source of truth, not chat memory.
- Do not edit before claiming a `ready` task unless the user explicitly asks for repo setup/hygiene.
- Stay within the claimed task's `scope.modify`.
- Run task verification and repo gates before finishing.
- For phase-specific work, use `.go-workflow/skills/go-workflow-<phase>/SKILL.md` for inputs, outputs, allowed mutations, required evidence, stop conditions, and handoff contract.
- For cross-cutting routing, use support skills under `.go-workflow/skills/go-workflow-<support>/SKILL.md`: audit, git, release, interview, cancel, and better.
- Use `python3 scripts/gate.py --task-id <TASK-ID> --phase <phase> --evidence key=value` when a phase needs hard evidence before advancing.
- Finish with `python3 scripts/finish_task.py <TASK-ID> --evidence "<evidence>" --agent <agent-name>`.
- Commit one task at a time; include evidence in the final report.

## Project

- id: `{project}`
- name: `{name or project}`
"""


def model_md(model: str) -> str:
    return f"""# {model} Instructions — go-workflow

Follow `AGENTS.md`. This file exists so {model}-style agents can discover the same repo-local workflow without Hermes-specific context.

Start with:

```bash
python3 scripts/next_task.py --validate
python3 scripts/next_task.py --list --limit 5
python3 scripts/next_task.py --claim --agent {model.lower()}
```

Show the max-5 task preview before claiming or doing work on every workflow-triggered run.

Then read the handoff, work only inside scope, verify, and finish with evidence.
"""


def readme(project: str, name: str | None) -> str:
    return f"""# go-workflow

One name, one workflow, one repo-contained skillbundle for agentic repo work.

`go-workflow` is deliberately independent from Hermes. Hermes can use it, but so can Codex, Claude, Gemini, or a plain terminal agent. The repository remains the source of truth.

## Workflow

```text
SETUP → PLAN → ROUTE/CLAIM → BUILD → VERIFY → DOCS/LEDGER → DEVIL → ANTISLOP → SHIP
```

## Repo contract

```text
.go-workflow/
  config.yaml          # workflow config and policy
  goals.yaml           # durable goals/outcomes
  tasks.yaml           # machine-readable task queue
  gates.yaml           # phase gates and done rules
  runs/                # generated handoffs and ledgers
  prompts/             # Hermes/Codex/Claude/Gemini prompts
  skills/go-workflow/              # root portable skill bundle
  skills/go-workflow-setup/        # phase skills with inputs/outputs/gates
  skills/go-workflow-plan/
  skills/go-workflow-route-claim/
  skills/go-workflow-build/
  skills/go-workflow-verify/
  skills/go-workflow-docs-ledger/
  skills/go-workflow-devil/
  skills/go-workflow-antislop/
  skills/go-workflow-ship/
  skills/go-workflow-audit/        # support routers / quality gates
  skills/go-workflow-git/
  skills/go-workflow-release/
  skills/go-workflow-interview/
  skills/go-workflow-cancel/
  skills/go-workflow-better/
  runtime/go_workflow/             # embedded runtime for self-contained scripts
.hermes/
  skills/go-workflow*/             # compatibility copies for Hermes-style skill discovery
scripts/
  next_task.py         # validate/list/claim
  finish_task.py       # finish with evidence
  gate.py              # self-contained phase evidence gate
AGENTS.md              # universal agent instructions
CLAUDE.md              # Claude-specific discovery file
GEMINI.md              # Gemini-specific discovery file
```

## Install/use in a repo

From a checkout of this package:

```bash
python3 -m pip install -e .
go-workflow init --project my-repo --name "My Repo"
```

Without installing:

```bash
PYTHONPATH=/path/to/go-workflow python3 -m go_workflow init --project my-repo --name "My Repo"
```

## Runtime dependency policy

Generated target repositories are self-contained for normal agent operations. The
`scripts/next_task.py`, `scripts/finish_task.py`, and `scripts/gate.py` wrappers
prefer the embedded `.go-workflow/runtime/go_workflow/` copy and can run in a
plain Python 3.10+ environment without `PYTHONPATH`, without an installed
`go-workflow` package, and without PyYAML. The embedded runtime includes a small
YAML subset reader/writer for the state files generated by go-workflow itself.

PyYAML remains the preferred parser when the package is installed in a developer
environment, especially for arbitrary hand-written YAML. Do not rely on the
fallback parser for general YAML syntax outside go-workflow's own generated
`config.yaml`, `goals.yaml`, `tasks.yaml`, `gates.yaml`, and roadmap files.

## Agent loop

```bash
python3 scripts/next_task.py --validate
python3 scripts/next_task.py --list --limit 5
python3 scripts/next_task.py --claim --agent codex
# read .go-workflow/runs/*-handoff.md
python3 scripts/gate.py --task-id <TASK-ID> --phase verify \
  --evidence "task_verification_run=pytest passed" \
  --evidence "repo_validation_passes=go-workflow validate passed"
python3 scripts/gate.py --task-id <TASK-ID> --phase docs-ledger \
  --evidence "evidence_recorded=task evidence captured" \
  --evidence "ledger_written=pending finish" \
  --evidence "human_cockpit_updated_if_present=tasks.md updated or n/a"
python3 scripts/gate.py --task-id <TASK-ID> --phase antislop \
  --evidence "diff_check_clean=git diff --check passed"
python3 scripts/gate.py --task-id <TASK-ID> --phase ship \
  --evidence "git_status_reported=clean or explained" \
  --evidence "commit_or_pr_linked=pending finish commit"
python3 scripts/finish_task.py <TASK-ID> --evidence "tests passed" --agent codex
```

Always show the max-5 upcoming task preview before claiming or executing work. The claim response also includes `up_next` so agents can report what remains next. `finish` refuses to mark a task done until required final phase evidence for `verify`, `docs_ledger`, `antislop`, and `ship` is recorded. Manual bypasses must use `--force --force-reason "..."`; the reason is written to the task and run ledger.

## Add goals and tasks

```bash
go-workflow add-goal --id G01 --title "Make workflow portable" --outcome "Any agent can run it"
go-workflow add-task --id T001 --goal G01 --title "Add gates" \
  --acceptance "gates.yaml exists" \
  --verification "python3 scripts/next_task.py --validate" \
  --modify .go-workflow/gates.yaml
```

## Phase gates

Each workflow phase has a generated skill under `.go-workflow/skills/go-workflow-<phase>/SKILL.md` and a Hermes compatibility copy under `.hermes/skills/`. A phase skill defines:

- inputs;
- outputs;
- allowed mutations;
- required evidence;
- failure/stop conditions;
- handoff contract.

`gates.yaml` also contains `phase_evidence_required`. Use the self-contained gate wrapper to reject missing evidence before moving on:

```bash
python3 scripts/gate.py \
  --task-id T001 \
  --phase verify \
  --evidence "task_verification_run=pytest passed" \
  --evidence "repo_validation_passes=go-workflow validate passed"
```

## Support routers

Generated repos also include non-phase support skills: `go-workflow-audit`, `go-workflow-git`, `go-workflow-release`, `go-workflow-interview`, `go-workflow-cancel`, and `go-workflow-better`. Audit covers selectable code/test/arch/api/perf/a11y/ux/security-style gates. Git covers status hygiene, selective staging, ship, fix-ci, PR/check inspection, and clean return to main. Release keeps version/tag/release evidence separate from generic docs-ledger work.

## Current project

- id: `{project}`
- name: `{name or project}`
"""


def ensure_gitignore() -> None:
    path = Path(".gitignore")
    add = [".go-workflow/runs/*", "!.go-workflow/runs/.gitkeep", ".go-workflow/state/*", "!.go-workflow/state/.gitkeep", "__pycache__/", "*.py[cod]", ".pytest_cache/", "*.egg-info/"]
    existing = path.read_text().splitlines() if path.exists() else []
    changed = False
    for line in add:
        if line not in existing:
            existing.append(line)
            changed = True
    if changed or not path.exists():
        path.write_text("\n".join(existing).rstrip() + "\n")


def cmd_init(args: argparse.Namespace) -> int:
    FLOW_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / ".gitkeep").touch()
    (FLOW_DIR / "state").mkdir(exist_ok=True)
    (FLOW_DIR / "state" / ".gitkeep").touch()
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    Path("scripts").mkdir(exist_ok=True)

    if not CONFIG_PATH.exists() or args.force:
        yaml_save(CONFIG_PATH, default_config(args.project, args.name))
    if not PROJECT_DOC_PATH.exists() or args.force:
        PROJECT_DOC_PATH.write_text(project_doc(args.project, args.name))
    if not GOALS_PATH.exists() or args.force:
        yaml_save(GOALS_PATH, {"version": 1, "project": args.project, "goals": []})
    if not TASKS_PATH.exists() or args.force:
        yaml_save(TASKS_PATH, {"version": 1, "project": args.project, "tasks": []})
    if not GATES_PATH.exists() or args.force:
        yaml_save(GATES_PATH, default_gates())
    (SKILLS_DIR / "SKILL.md").write_text(skill_md())
    for phase_id, phase_name, summary in PHASES:
        phase_dir = FLOW_DIR / "skills" / f"go-workflow-{phase_id}"
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / "SKILL.md").write_text(phase_skill_md(phase_id, phase_name, summary))
    for skill_id, contract in SUPPORT_SKILLS.items():
        support_dir = FLOW_DIR / "skills" / f"go-workflow-{skill_id}"
        support_dir.mkdir(parents=True, exist_ok=True)
        (support_dir / "SKILL.md").write_text(support_skill_md(skill_id, contract))
    source_runtime = Path(__file__).resolve().parent
    target_runtime = RUNTIME_DIR / "go_workflow"
    if target_runtime.exists():
        shutil.rmtree(target_runtime)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(source_runtime, target_runtime, ignore=ignore)
    for agent in ["hermes", "codex", "claude", "gemini"]:
        (PROMPTS_DIR / f"{agent}.md").write_text(prompt(agent))

    wrapper_template = """#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# Prefer the embedded runtime installed by `go-workflow init`; fall back to the
# repo root for this source checkout and to site-packages when installed.
sys.path.insert(0, str(ROOT / ".go-workflow" / "runtime"))
sys.path.insert(0, str(ROOT))

from go_workflow.__main__ import main

raise SystemExit(main([{subcommand!r}, *sys.argv[1:]]))
"""
    script_subcommands = {
        Path("scripts/next_task.py"): "next",
        Path("scripts/finish_task.py"): "finish",
        Path("scripts/gate.py"): "gate",
    }
    for script, subcommand in script_subcommands.items():
        if not script.exists() or args.force:
            script.write_text(wrapper_template.format(subcommand=subcommand))
            script.chmod(0o755)
    if not Path("AGENTS.md").exists() or args.force:
        Path("AGENTS.md").write_text(agents_md(args.project, args.name))
    Path("CLAUDE.md").write_text(model_md("Claude"))
    Path("GEMINI.md").write_text(model_md("Gemini"))
    Path("CODEX.md").write_text(model_md("Codex"))
    # Hermes-compatible copy, but .go-workflow remains canonical and no Hermes runtime is required.
    hermes_skill = HERMES_COMPAT_DIR / "skills" / "go-workflow"
    hermes_skill.mkdir(parents=True, exist_ok=True)
    (hermes_skill / "SKILL.md").write_text(skill_md())
    for phase_id, phase_name, summary in PHASES:
        compat_phase = HERMES_COMPAT_DIR / "skills" / f"go-workflow-{phase_id}"
        compat_phase.mkdir(parents=True, exist_ok=True)
        (compat_phase / "SKILL.md").write_text(phase_skill_md(phase_id, phase_name, summary))
    for skill_id, contract in SUPPORT_SKILLS.items():
        compat_support = HERMES_COMPAT_DIR / "skills" / f"go-workflow-{skill_id}"
        compat_support.mkdir(parents=True, exist_ok=True)
        (compat_support / "SKILL.md").write_text(support_skill_md(skill_id, contract))
    (HERMES_COMPAT_DIR / "README.md").write_text("# .hermes compatibility\n\nThis folder exposes go-workflow root and phase skills to Hermes-style agents. Canonical state lives in `.go-workflow/`.\n")
    if not Path("README.md").exists() or args.force:
        Path("README.md").write_text(readme(args.project, args.name))
    ensure_gitignore()
    print(f"initialized go-workflow for {args.project}")
    return 0


def validation_errors() -> list[str]:
    errors: list[str] = []
    tasks = load_tasks()
    goals = load_goals()
    gates = yaml_load(GATES_PATH, {})
    ids: set[str] = set()
    gids = goal_ids(goals)
    for i, goal in enumerate(goals.get("goals", []), start=1):
        if not goal.get("id"):
            errors.append(f"goal[{i}]: missing id")
        if not goal.get("title"):
            errors.append(f"goal[{i}]: missing title")
        if not goal.get("outcome"):
            errors.append(f"goal[{i}]: missing outcome")
    all_task_ids = {t.get("id") for t in tasks.get("tasks", []) if t.get("id")}
    for index, task in enumerate(tasks.get("tasks", []), start=1):
        prefix = f"task[{index}]"
        task_id = task.get("id")
        if not task_id:
            errors.append(f"{prefix}: missing id")
        elif task_id in ids:
            errors.append(f"{prefix}: duplicate id {task_id}")
        else:
            ids.add(task_id)
        if not task.get("title"):
            errors.append(f"{prefix}: missing title")
        if task.get("status") not in VALID_STATUSES:
            errors.append(f"{prefix}: invalid status {task.get('status')!r}")
        if not task.get("acceptance"):
            errors.append(f"{prefix}: missing acceptance")
        if not task.get("verification"):
            errors.append(f"{prefix}: missing verification")
        if task.get("goal") and gids and task.get("goal") not in gids:
            errors.append(f"{prefix}: unknown goal {task.get('goal')}")
        for dep in task.get("depends_on") or []:
            if dep not in all_task_ids:
                errors.append(f"{prefix}: unknown dependency {dep}")
    if not gates.get("gates"):
        errors.append("gates.yaml: missing gates")
    return errors


def cmd_validate(args: argparse.Namespace) -> int:
    errors = validation_errors()
    if errors:
        for error in errors:
            print(error, file=os.sys.stderr)
        return 1
    print("valid")
    return 0


def cmd_add_goal(args: argparse.Namespace) -> int:
    data = load_goals()
    if args.id in goal_ids(data):
        print(f"goal exists: {args.id}", file=os.sys.stderr)
        return 1
    goal = {"id": args.id, "title": args.title, "outcome": args.outcome, "status": args.status}
    if args.deliverable:
        goal["deliverables"] = normalize_list(args.deliverable)
    data["goals"].append(goal)
    save_goals(data)
    print(args.id)
    return 0


def cmd_add_task(args: argparse.Namespace) -> int:
    data = load_tasks()
    if task_by_id(data, args.id):
        print(f"task exists: {args.id}", file=os.sys.stderr)
        return 1
    task: dict[str, Any] = {
        "id": args.id,
        "title": args.title,
        "status": args.status,
        "acceptance": normalize_list(args.acceptance),
        "verification": normalize_list(args.verification),
    }
    if args.goal:
        task["goal"] = args.goal
    deps = normalize_list(args.depends_on)
    if deps:
        task["depends_on"] = deps
    scope: dict[str, Any] = {}
    for key, value in [("read", args.read), ("modify", args.modify)]:
        items = normalize_list(value)
        if items:
            scope[key] = items
    if scope:
        task["scope"] = scope
    docs = normalize_list(args.docs)
    if docs:
        task["docs"] = {"update": docs}
    if args.commit_message:
        task["commit"] = {"mode": "task", "message": args.commit_message}
    data["tasks"].append(task)
    save_tasks(data)
    print(args.id)
    return 0


def dependencies_done(data: dict[str, Any], task: dict[str, Any]) -> bool:
    for dep_id in task.get("depends_on") or []:
        dep = task_by_id(data, dep_id)
        if not dep or dep.get("status") != "done":
            return False
    return True


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def reclaim_expired_claims(data: dict[str, Any]) -> bool:
    changed = False
    current = now_utc()
    for task in data.get("tasks", []):
        if task.get("status") != "claimed":
            continue
        claim = task.get("claim") or {}
        lease_until = parse_iso(claim.get("lease_until"))
        if lease_until and lease_until <= current:
            history_entry = dict(claim)
            history_entry["reclaimed_at"] = iso(current)
            history_entry["reclaimed_reason"] = "lease_expired"
            task.setdefault("claim_history", []).append(history_entry)
            task.pop("claim", None)
            task["status"] = "ready"
            changed = True
    return changed


def ready_tasks(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [task for task in data.get("tasks", []) if task.get("status") == "ready" and dependencies_done(data, task)]


def task_preview(task: dict[str, Any]) -> dict[str, Any]:
    preview = {"id": task.get("id"), "title": task.get("title", ""), "status": task.get("status")}
    if task.get("goal"):
        preview["goal"] = task.get("goal")
    return preview


def upcoming_payload(tasks: list[dict[str, Any]], limit: int = 5, key: str = "ready") -> dict[str, Any]:
    bounded = max(0, limit)
    return {key: [task_preview(task) for task in tasks[:bounded]], "count": len(tasks), "limit": bounded}


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()[:50] or "task"


def write_handoff(task: dict[str, Any], agent: str) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = now_utc().strftime("%Y%m%d-%H%M%S")
    path = RUNS_DIR / f"{timestamp}-{task['id']}-{slug(task.get('title', 'task'))}-handoff.md"
    scope = task.get("scope") or {}
    content = [
        f"# Task Handoff — {task['id']}: {task.get('title', '')}",
        "", f"- Agent: {agent}", f"- Claimed: {iso(now_utc())}",
        "", "## Workflow", "SETUP → PLAN → ROUTE/CLAIM → BUILD → VERIFY → DOCS/LEDGER → DEVIL → ANTISLOP → SHIP",
        "", "## Scope", "", "### Read", *(f"- `{p}`" for p in scope.get("read", [])),
        "", "### Modify", *(f"- `{p}`" for p in scope.get("modify", [])),
        "", "## Acceptance", *(f"- {item}" for item in task.get("acceptance", [])),
        "", "## Verification", *(f"- `{item}`" for item in task.get("verification", [])),
        "", "## Finish", f"Run: `python3 scripts/finish_task.py {task['id']} --evidence <path-or-command> --agent {agent}`", "",
    ]
    path.write_text("\n".join(content))
    return path


def cmd_next(args: argparse.Namespace) -> int:
    if args.validate:
        return cmd_validate(args)
    data = load_tasks()
    reclaimed = reclaim_expired_claims(data)
    ready = ready_tasks(data)
    if args.list:
        if reclaimed:
            save_tasks(data)
        print(json.dumps(upcoming_payload(ready, args.limit), indent=2))
        return 0
    if not args.claim:
        if reclaimed:
            save_tasks(data)
        payload = {"next": task_preview(ready[0]) if ready else None, **upcoming_payload(ready, args.limit, key="up_next")}
        print(json.dumps(payload, indent=2))
        return 0
    if not ready:
        if reclaimed:
            save_tasks(data)
        print(json.dumps({"task_id": None, "message": "no ready tasks", **upcoming_payload([], args.limit, key="up_next")}))
        return 0
    task = ready[0]
    task["status"] = "claimed"
    task["claim"] = {"by": args.agent, "at": iso(now_utc()), "lease_until": iso(now_utc() + timedelta(minutes=args.lease_minutes))}
    handoff = write_handoff(task, args.agent)
    task["handoff"] = str(handoff)
    save_tasks(data)
    remaining_ready = [candidate for candidate in ready[1:] if candidate.get("id") != task.get("id")]
    print(json.dumps({"task_id": task["id"], "handoff": str(handoff), **upcoming_payload(remaining_ready, args.limit, key="up_next")}, indent=2))
    return 0


FINAL_FINISH_PHASES = ("verify", "docs_ledger", "antislop", "ship")
REQUIRED_FINISH_PHASE_EVIDENCE = {
    "verify": ["task_verification_run", "repo_validation_passes"],
    "docs_ledger": ["evidence_recorded", "ledger_written"],
    "antislop": ["diff_check_clean"],
    "ship": ["git_status_reported", "commit_or_pr_linked"],
}


def phase_required_evidence(evidence_required: dict[str, Any], phase: str) -> list[str] | None:
    if phase not in evidence_required and phase not in REQUIRED_FINISH_PHASE_EVIDENCE:
        return None
    configured_required = evidence_required.get(phase) or []
    fallback_required = REQUIRED_FINISH_PHASE_EVIDENCE.get(phase, [])
    return list(dict.fromkeys([*configured_required, *fallback_required]))


def missing_finish_phase_evidence(task: dict[str, Any]) -> dict[str, list[str]]:
    gates = yaml_load(GATES_PATH, {})
    evidence_required = gates.get("phase_evidence_required") or {}
    phase_data = task.get("phase_evidence") or {}
    missing: dict[str, list[str]] = {}
    for phase in FINAL_FINISH_PHASES:
        required = phase_required_evidence(evidence_required, phase) or []
        provided = phase_data.get(phase, [])
        phase_missing = [item for item in required if item not in evidence_names(provided)]
        if phase_missing:
            missing[phase] = phase_missing
    return missing


def write_finish_ledger(task: dict[str, Any], agent: str, evidence: list[str], force_reason: str | None = None) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = now_utc().strftime("%Y%m%d-%H%M%S")
    path = RUNS_DIR / f"{timestamp}-{task['id']}-{slug(task.get('title', 'task'))}-ledger.md"
    content = [
        f"# Task Ledger — {task['id']}: {task.get('title', '')}",
        "", f"- Agent: {agent}", f"- Completed: {iso(now_utc())}",
        "", "## Acceptance", *(f"- {item}" for item in task.get("acceptance", [])),
        "", "## Verification", *(f"- `{item}`" for item in task.get("verification", [])),
        "", "## Evidence", *(f"- {item}" for item in evidence),
    ]
    if force_reason:
        content.extend(["", "## Force override", f"- Reason: {force_reason}"])
    content.append("")
    path.write_text("\n".join(content))
    return path


def cmd_finish(args: argparse.Namespace) -> int:
    if not args.evidence:
        print("finish requires at least one --evidence", file=os.sys.stderr)
        return 1
    if args.force and not args.force_reason:
        print("force finish requires --force-reason", file=os.sys.stderr)
        return 1
    if args.force_reason and not args.force:
        print("--force-reason can only be used with --force", file=os.sys.stderr)
        return 1
    data = load_tasks()
    task = task_by_id(data, args.task_id)
    if not task:
        print(f"unknown task: {args.task_id}", file=os.sys.stderr)
        return 1
    if not args.force and task.get("status") not in {"claimed", "active", "review"}:
        print("finish requires a claimed/active/review task; pass --force for manual close", file=os.sys.stderr)
        return 1
    claim = task.get("claim") or {}
    if not args.force and claim.get("by") and claim.get("by") != args.agent:
        print(f"task is claimed by {claim.get('by')}; pass --force to override", file=os.sys.stderr)
        return 1
    missing = missing_finish_phase_evidence(task)
    if missing and not args.force:
        details = "; ".join(f"{phase}: {', '.join(items)}" for phase, items in missing.items())
        print(f"missing required phase evidence for {args.task_id}: {details}; pass --force --force-reason to override", file=os.sys.stderr)
        return 1
    evidence = normalize_list(args.evidence)
    task["status"] = "done"
    task["completed_at"] = iso(now_utc())
    task["completed_by"] = args.agent
    task["evidence"] = evidence
    if args.force_reason:
        task["force_reason"] = args.force_reason
    task.pop("claim", None)
    ledger = write_finish_ledger(task, args.agent, evidence, args.force_reason)
    task["ledger"] = str(ledger)
    save_tasks(data)
    print(json.dumps({"task_id": args.task_id, "status": "done", "ledger": str(ledger)}, indent=2))
    return 0


def cmd_render_md(args: argparse.Namespace) -> int:
    tasks = load_tasks()
    goals = load_goals()
    goal_titles = {g.get("id"): g.get("title") for g in goals.get("goals", [])}
    lines = ["# Tasks", ""]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for task in tasks.get("tasks", []):
        grouped.setdefault(task.get("goal") or "Ungrouped", []).append(task)
    for goal, items in grouped.items():
        title = goal_titles.get(goal)
        lines.extend([f"## {goal}" + (f": {title}" if title else ""), ""])
        for task in items:
            check = "x" if task.get("status") == "done" else " "
            line = f"- [{check}] {task['id']}: {task.get('title', '')} | status: {task.get('status')}"
            if task.get("evidence"):
                line += f" | evidence: `{'; '.join(task['evidence'])}`"
            lines.append(line)
        lines.append("")
    output = "\n".join(lines)
    if args.output:
        Path(args.output).write_text(output)
    else:
        print(output)
    return 0


def task_phase_evidence(task: dict[str, Any]) -> dict[str, Any]:
    return task.setdefault("phase_evidence", {})


def normalize_phase_name(value: str) -> str:
    return value.strip().replace("-", "_")


def evidence_names(values: list[str]) -> set[str]:
    names: set[str] = set()
    for value in values:
        key = str(value).split("=", 1)[0].strip()
        if key:
            names.add(key)
    return names


def evidence_value(item: str) -> tuple[str, str] | None:
    if not isinstance(item, str) or "=" not in item:
        return None
    key, value = item.split("=", 1)
    return key.strip(), value.strip()


def validate_phase_specific_evidence(phase: str, evidence: list[str]) -> str | None:
    if phase != "audit":
        return None
    allowed = set(SUPPORT_SKILLS["audit"]["commands"])
    selected_values = []
    for item in evidence:
        parsed = evidence_value(item)
        if parsed and parsed[0] == "audit_profile_selected":
            selected_values.append(parsed[1])
    for value in selected_values:
        profiles = [part for part in re.split(r"[+,;/\s]+", value.strip()) if part]
        if not profiles:
            return "audit_profile_selected must name at least one audit profile"
        unknown = [profile for profile in profiles if profile not in allowed]
        if unknown:
            return f"unknown audit profile(s): {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}"
    return None


def cmd_gate(args: argparse.Namespace) -> int:
    rc = cmd_validate(args)
    if rc:
        return rc
    commands = args.command or []
    for command in commands:
        result = subprocess.run(command, shell=True, text=True)
        if result.returncode != 0:
            return result.returncode
    if args.task_id or args.phase:
        if not args.task_id or not args.phase:
            print("gate requires both --task-id and --phase when checking phase evidence", file=os.sys.stderr)
            return 1
        phase = normalize_phase_name(args.phase)
        data = load_tasks()
        task = task_by_id(data, args.task_id)
        if not task:
            print(f"unknown task: {args.task_id}", file=os.sys.stderr)
            return 1
        gates = yaml_load(GATES_PATH, {})
        evidence_required = gates.get("phase_evidence_required") or {}
        required = phase_required_evidence(evidence_required, phase)
        if required is None:
            print(f"unknown phase: {args.phase}", file=os.sys.stderr)
            return 1
        provided = task_phase_evidence(task).get(phase, []) + normalize_list(args.evidence)
        missing = [item for item in required if item not in evidence_names(provided)]
        if missing:
            print(f"missing phase evidence for {args.task_id}/{phase}: {', '.join(missing)}", file=os.sys.stderr)
            return 1
        phase_error = validate_phase_specific_evidence(phase, provided)
        if phase_error:
            print(phase_error, file=os.sys.stderr)
            return 1
        if args.evidence:
            phase_data = task_phase_evidence(task)
            existing = phase_data.setdefault(phase, [])
            for item in normalize_list(args.evidence):
                if item not in existing:
                    existing.append(item)
            save_tasks(data)
    print("gates passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="go-workflow", description="Repo-contained GO workflow")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init"); p.add_argument("--project", required=True); p.add_argument("--name"); p.add_argument("--force", action="store_true"); p.set_defaults(func=cmd_init)
    p = sub.add_parser("validate"); p.set_defaults(func=cmd_validate)
    p = sub.add_parser("add-goal"); p.add_argument("--id", required=True); p.add_argument("--title", required=True); p.add_argument("--outcome", required=True); p.add_argument("--status", default="active"); p.add_argument("--deliverable", action="append"); p.set_defaults(func=cmd_add_goal)
    p = sub.add_parser("add-task"); p.add_argument("--id", required=True); p.add_argument("--title", required=True); p.add_argument("--status", default="ready", choices=sorted(VALID_STATUSES)); p.add_argument("--goal"); p.add_argument("--depends-on", action="append"); p.add_argument("--acceptance", action="append", required=True); p.add_argument("--verification", action="append", required=True); p.add_argument("--read", action="append"); p.add_argument("--modify", action="append"); p.add_argument("--docs", action="append"); p.add_argument("--commit-message"); p.set_defaults(func=cmd_add_task)
    p = sub.add_parser("next"); p.add_argument("--list", action="store_true"); p.add_argument("--claim", action="store_true"); p.add_argument("--agent", default=os.environ.get("USER", "agent")); p.add_argument("--lease-minutes", type=int, default=60); p.add_argument("--limit", type=int, default=5, help="maximum upcoming tasks to preview (default: 5)"); p.add_argument("--validate", action="store_true"); p.set_defaults(func=cmd_next)
    p = sub.add_parser("finish"); p.add_argument("task_id"); p.add_argument("--evidence", action="append"); p.add_argument("--agent", default=os.environ.get("USER", "agent")); p.add_argument("--force", action="store_true"); p.add_argument("--force-reason"); p.set_defaults(func=cmd_finish)
    p = sub.add_parser("render-md"); p.add_argument("--output"); p.set_defaults(func=cmd_render_md)
    p = sub.add_parser("gate"); p.add_argument("--command", action="append"); p.add_argument("--task-id"); p.add_argument("--phase"); p.add_argument("--evidence", action="append"); p.set_defaults(func=cmd_gate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
