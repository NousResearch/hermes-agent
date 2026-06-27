#!/usr/bin/env python3
"""Create a loop-engineering Hermes Kanban task chain.

Default dry-run prints exact hermes kanban commands.
Use --execute to create the chain with idempotency keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_ASSIGNEES = {
    "analyst": "analyst",
    "spec_reviewer": "spec-reviewer",
    "coder": "coder",
    "reviewer": "reviewer",
    "qa": "reviewer",  # QA skills currently run through reviewer profile.
    "observe": "default",
}


@dataclass
class TaskSpec:
    key: str
    title: str
    assignee: str
    skills: list[str]
    body: str
    parents: list[str] = field(default_factory=list)
    initial_status: str | None = None
    max_runtime: str | None = None


def shell_cmd(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def run_json(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {shell_cmd(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Expected JSON from: {shell_cmd(cmd)}\nOutput:\n{proc.stdout}") from exc


def verify_assignees(required: set[str]) -> None:
    proc = subprocess.run(["hermes", "kanban", "assignees"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"Could not list assignees:\n{proc.stderr or proc.stdout}")
    available = set()
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            available.add(parts[0])
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"Missing kanban assignees: {', '.join(missing)}\nAvailable: {', '.join(sorted(available))}")


def request_text(args: argparse.Namespace) -> str:
    chunks: list[str] = []
    if args.request:
        chunks.append(args.request.strip())
    if args.request_file:
        chunks.append(Path(args.request_file).read_text(encoding="utf-8").strip())
    if not chunks:
        return "No raw request body supplied. Orchestrator must fill in the request before dispatch."
    return "\n\n".join(c for c in chunks if c)


def body_header(args: argparse.Namespace, raw_request: str) -> str:
    return f"""Loop-engineering development pipeline task.

Original feature/task: {args.title}
Workspace: {args.workspace}
Artifact root: {args.artifact_root}

Raw request:
{raw_request}

Global hard gates:
- No SPEC_APPROVED → no coding.
- No evidence → not done.
- Merge is not product done; release must move to observation/validation.
- If the scope is unclear, block with the smallest concrete question instead of guessing.
""".strip()


def build_specs(args: argparse.Namespace) -> list[TaskSpec]:
    raw = request_text(args)
    header = body_header(args, raw)
    artifact = args.artifact_root.rstrip("/")
    title = args.title

    specs: list[TaskSpec] = []

    specs.append(TaskSpec(
        key="analyst",
        title=f"analyze: {title}",
        assignee=args.analyst,
        skills=["business-analyst-bundle", "loop-engineering-development-pipeline"],
        max_runtime=args.max_runtime_analysis,
        body=f"""{header}

ROLE: analyst.

Create/update the requirement contract before any implementation:
- find/update or draft the corresponding user story;
- create user story map;
- write concrete positive and negative examples;
- write domain rules and explicit out-of-scope;
- create Cucumber/Gherkin `.feature` file(s) under `{artifact}/<activity>/<task>/` with `@activity-* @task-*` tags;
- produce acceptance criteria and implementation handoff;
- record open domain questions.

Required output:
Outcome: SPEC_DRAFTED or BLOCKED_ON_DOMAIN_QUESTION
Artifacts: story-map, feature files, examples, domain-rules, open-questions, implementation-handoff
Next gate: spec-reviewer
""".strip(),
    ))

    specs.append(TaskSpec(
        key="spec",
        title=f"spec-review: {title}",
        assignee=args.spec_reviewer,
        skills=["spec-reviewer-bundle", "loop-engineering-development-pipeline"],
        parents=["analyst"],
        max_runtime=args.max_runtime_review,
        body=f"""{header}

ROLE: spec-reviewer.

Review analyst artifacts as a build/QA contract. Gate:
- user story readiness;
- story map;
- domain questions;
- BDD/Gherkin quality;
- positive/negative/boundary examples;
- traceability;
- acceptance criteria;
- explicit out-of-scope;
- coder handoff.

Return exactly one gate status:
SPEC_APPROVED / REQUEST_SPEC_CHANGES / BLOCKED_ON_DOMAIN_QUESTION / REJECTED_NOT_WORTH_BUILDING

Hard rule: only SPEC_APPROVED unlocks coder/QA implementation tasks.
""".strip(),
    ))

    if args.test_first_qa:
        specs.append(TaskSpec(
            key="qa_pre",
            title=f"qa-plan: {title}",
            assignee=args.qa,
            skills=["qa-bundle", "qa-bdd-test-writing", "loop-engineering-development-pipeline"],
            parents=["spec"],
            max_runtime=args.max_runtime_qa,
            body=f"""{header}

ROLE: QA, test-first acceptance coverage.

From SPEC_APPROVED `.feature` files:
- build a scenario-to-test coverage matrix;
- write/plan executable BDD/regression tests;
- cover positive, negative, boundary, permission, and integration examples where relevant;
- do not invent missing outcomes; block with domain questions.

Return:
Outcome: QA_PLAN_READY / PARTIAL / BLOCKED_FOR_SPEC / BLOCKED_FOR_ENV
Coverage matrix:
Commands run / evidence where possible:
Next gate: coder
""".strip(),
        ))

    coder_parents = ["qa_pre"] if args.test_first_qa else ["spec"]
    specs.append(TaskSpec(
        key="coder",
        title=f"implement: {title}",
        assignee=args.coder,
        skills=["coder-bundle", "loop-engineering-development-pipeline"],
        parents=coder_parents,
        max_runtime=args.max_runtime_dev,
        body=f"""{header}

ROLE: coder.

Implement only the SPEC_APPROVED contract. Work slice-by-slice:
1. write/confirm failing test;
2. implement minimal code;
3. run specific test;
4. run relevant suite;
5. report diff/commit and risks.

Use the approved user story, implementation scenarios, and linked `.feature` files as the primary contract. Do not invent business rules. If spec conflicts with repo reality, return BLOCKED_ON_SPEC_CONFLICT.

Return:
Outcome: IMPLEMENTED / PARTIAL / BLOCKED_ON_SPEC_CONFLICT / BLOCKED_ON_ENV
Files changed:
Tests added:
Commands run:
Actual output:
Diff/commit:
Risks:
Next gate: reviewer
""".strip(),
    ))

    specs.append(TaskSpec(
        key="reviewer",
        title=f"review: {title}",
        assignee=args.reviewer,
        skills=["reviewer-bundle", "loop-engineering-development-pipeline"],
        parents=["coder"],
        max_runtime=args.max_runtime_review,
        body=f"""{header}

ROLE: reviewer.

Do two-stage review:

Stage 1 — Spec compliance:
- approved spec/BDD/acceptance criteria matched;
- negative/boundary cases handled;
- no out-of-scope creep.

Stage 2 — Code quality:
- architecture;
- readability;
- error handling;
- security;
- performance;
- maintainability;
- test quality.

Return both statuses:
Spec compliance: SPEC_COMPLIANT / REQUEST_CHANGES / BLOCKED_ON_SPEC_AMBIGUITY
Code quality: APPROVED / REQUEST_CHANGES / BLOCKED_ON_TECH_RISK

If REQUEST_CHANGES, list exact required fixes.
""".strip(),
    ))

    specs.append(TaskSpec(
        key="qa",
        title=f"qa: {title}",
        assignee=args.qa,
        skills=["qa-bundle", "qa-bdd-test-writing", "qa-bdd-test-review", "loop-engineering-development-pipeline"],
        parents=["reviewer"],
        max_runtime=args.max_runtime_qa,
        body=f"""{header}

ROLE: QA.

Prove approved behaviour from `.feature` files. For user-visible flows, real browser/user-visible acceptance is required; API tests may support but do not replace acceptance.

Required coverage:
- positive scenarios;
- negative scenarios;
- boundary scenarios;
- permission/integration cases where relevant;
- regression for any bug fixed.

Return:
Outcome: QA_APPROVED / FAILED / PARTIAL / BLOCKED_FOR_ENV / BLOCKED_FOR_SPEC
Feature files covered:
Coverage matrix:
Commands run:
Evidence/report path:
Failures/blockers:
Next gate: release/observe
""".strip(),
    ))

    if not args.no_observe:
        specs.append(TaskSpec(
            key="observe",
            title=f"observe: {title}",
            assignee=args.observe,
            skills=["loop-engineering-development-pipeline"],
            parents=["qa"],
            initial_status="blocked" if args.block_observe else None,
            max_runtime=args.max_runtime_observe,
            body=f"""{header}

ROLE: release/observe loop owner.

After QA_APPROVED and release, validate product effect:
- release/version/rollback recorded;
- observation period defined;
- baseline/target/actual metric captured;
- support/log/user feedback checked;
- learning loop updated.

Return:
Outcome: RELEASED / OBSERVING / VALIDATED / NEEDS_ITERATION / FAILED_HYPOTHESIS
Metric evidence:
Support/log signals:
New regression/domain/process updates:
""".strip(),
        ))

    return specs


def build_create_cmd(args: argparse.Namespace, spec: TaskSpec, created: dict[str, str], idem_prefix: str) -> list[str]:
    cmd = ["hermes", "kanban"]
    if args.board:
        cmd += ["--board", args.board]
    cmd += [
        "create",
        spec.title,
        "--assignee", spec.assignee,
        "--workspace", args.workspace,
        "--body", spec.body,
        "--idempotency-key", f"{idem_prefix}:{spec.key}",
        "--created-by", args.created_by,
        "--json",
    ]
    for parent_key in spec.parents:
        parent_id = created.get(parent_key, f"${{{parent_key.upper()}}}")
        cmd += ["--parent", parent_id]
    for skill in spec.skills:
        cmd += ["--skill", skill]
    if spec.initial_status:
        cmd += ["--initial-status", spec.initial_status]
    if spec.max_runtime:
        cmd += ["--max-runtime", spec.max_runtime]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a loop-engineering Hermes Kanban chain.")
    parser.add_argument("--title", required=True, help="Feature/task title")
    parser.add_argument("--workspace", required=True, help="Hermes workspace, e.g. dir:/path/to/repo")
    parser.add_argument("--request", help="Raw request text")
    parser.add_argument("--request-file", help="File containing raw request/context")
    parser.add_argument("--artifact-root", default="features", help="Where analyst should place feature artifacts")
    parser.add_argument("--board", help="Kanban board slug")
    parser.add_argument("--idempotency-prefix", help="Stable dedup prefix; default derived from title/workspace")
    parser.add_argument("--created-by", default="hermes-loop-generator")
    parser.add_argument("--execute", action="store_true", help="Actually create tasks; default prints commands only")
    parser.add_argument("--dispatch", action="store_true", help="Run hermes kanban dispatch after creation; implies --execute")
    parser.add_argument("--test-first-qa", action="store_true", help="Insert QA planning task between spec-reviewer and coder")
    parser.add_argument("--no-observe", action="store_true", help="Do not create observe task")
    parser.add_argument("--block-observe", action=argparse.BooleanOptionalAction, default=True, help="Create observe task blocked by default")
    parser.add_argument("--analyst", default=DEFAULT_ASSIGNEES["analyst"])
    parser.add_argument("--spec-reviewer", default=DEFAULT_ASSIGNEES["spec_reviewer"])
    parser.add_argument("--coder", default=DEFAULT_ASSIGNEES["coder"])
    parser.add_argument("--reviewer", default=DEFAULT_ASSIGNEES["reviewer"])
    parser.add_argument("--qa", default=DEFAULT_ASSIGNEES["qa"])
    parser.add_argument("--observe", default=DEFAULT_ASSIGNEES["observe"])
    parser.add_argument("--max-runtime-analysis", default="2h")
    parser.add_argument("--max-runtime-dev", default="4h")
    parser.add_argument("--max-runtime-review", default="90m")
    parser.add_argument("--max-runtime-qa", default="2h")
    parser.add_argument("--max-runtime-observe", default="30m")
    args = parser.parse_args()

    if args.dispatch:
        args.execute = True

    specs = build_specs(args)
    required = {s.assignee for s in specs}
    verify_assignees(required)

    idem_prefix = args.idempotency_prefix or hashlib.sha1(f"{args.board or 'default'}\0{args.workspace}\0{args.title}".encode()).hexdigest()[:12]
    created: dict[str, str] = {}

    if not args.execute:
        print("# DRY RUN — commands not executed")
        print(f"# Idempotency prefix: {idem_prefix}")
        for spec in specs:
            print()
            print(f"# {spec.key}: {spec.title}")
            print(shell_cmd(build_create_cmd(args, spec, created, idem_prefix)))
            created[spec.key] = f"${{{spec.key.upper()}}}"
        if args.dispatch:
            dispatch_cmd = ["hermes", "kanban"] + (["--board", args.board] if args.board else []) + ["dispatch", "--max", str(len(specs))]
            print(shell_cmd(dispatch_cmd))
        return 0

    for spec in specs:
        cmd = build_create_cmd(args, spec, created, idem_prefix)
        data = run_json(cmd)
        task_id = data.get("id") or data.get("task", {}).get("id") or data.get("task_id")
        if not task_id:
            raise RuntimeError(f"Could not find task id in response for {spec.key}: {json.dumps(data, indent=2)}")
        created[spec.key] = task_id
        print(json.dumps({"key": spec.key, "id": task_id, "title": spec.title, "assignee": spec.assignee}, ensure_ascii=False))

    if args.dispatch:
        dispatch_cmd = ["hermes", "kanban"] + (["--board", args.board] if args.board else []) + ["dispatch", "--max", str(len(specs)), "--json"]
        print(json.dumps({"dispatch": run_json(dispatch_cmd)}, ensure_ascii=False))

    print(json.dumps({"created": created, "idempotency_prefix": idem_prefix}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
