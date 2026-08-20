"""Harness / Agenting Engineering Hermes plugin.

The plugin delegates intake-form operations to the bundled skill helper when the
repo ships it, while keeping an active-profile ``bin/hermes-harness`` helper as
a compatibility fallback for existing profiles.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

from hermes_constants import display_hermes_home, get_hermes_home

HELP_TEXT = """Harness / Agenting Engineering intake

CLI:
  hermes harness template
  hermes harness classify --text "Fix this WebUI bug and add tests"
  hermes harness new --title "My task" --workspace /path/to/repo --mode "Implement changes"
  hermes harness check /path/to/intake.md
  hermes harness prompt /path/to/intake.md
  hermes harness kanban create /path/to/intake.md --triage
  hermes harness evidence <task-id> --output evidence.md
  hermes harness gc-template --output weekly-harness-gc.md
  hermes harness migration-pack --output-dir /path/to/repo

Helper resolution:
  bundled skill script first, then the active profile's hermes-harness helper

Purpose:
  Move non-trivial AI-assisted coding tasks from vibe coding to sustainable
  Harness / Agenting Engineering by requiring task scope, acceptance criteria,
  risk surface, and verification evidence before implementation.
""".strip()


class TaskClassification:
    """Advisory task routing result for Harness preflight and CLI use."""

    def __init__(
        self,
        *,
        task_type: str,
        harness_required: bool,
        risk_level: str,
        route: str,
        signals: list[str] | None = None,
        recommended_next_steps: list[str] | None = None,
    ) -> None:
        self.task_type = task_type
        self.harness_required = harness_required
        self.risk_level = risk_level
        self.route = route
        self.signals = signals or []
        self.recommended_next_steps = recommended_next_steps or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "harness_required": self.harness_required,
            "risk_level": self.risk_level,
            "route": self.route,
            "signals": self.signals,
            "recommended_next_steps": self.recommended_next_steps,
        }


def _bundled_helper_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "skills"
        / "software-development"
        / "harness-agenting-engineering"
        / "scripts"
        / "harness_intake.py"
    )


def _user_helper_path() -> Path:
    return get_hermes_home() / "bin" / "hermes-harness"


def _helper_command() -> list[str] | None:
    bundled = _bundled_helper_path()
    if bundled.exists():
        return [os.environ.get("PYTHON", "python3"), str(bundled)]
    user_helper = _user_helper_path()
    if user_helper.exists():
        return [str(user_helper)]
    return None


def _run_helper(argv: Sequence[str]) -> int:
    command = _helper_command()
    if command is None:
        print("Missing Harness intake helper.")
        print(f"Expected bundled script: {_bundled_helper_path()}")
        print(f"Or profile-local helper: {display_hermes_home()}/bin/hermes-harness")
        return 2
    proc = subprocess.run([*command, *argv], check=False)
    return int(proc.returncode)


def _setup_harness_cli(parser) -> None:
    # Keep the public flag surface exact: argparse's default abbreviation would
    # otherwise continue accepting the removed `--out` spelling for `--output`.
    parser.allow_abbrev = False
    sub = parser.add_subparsers(dest="harness_action")

    template_p = sub.add_parser("template", help="Print or copy the Harness task intake template", allow_abbrev=False)
    template_p.add_argument("--output", "-o", default="", help="Copy template to this path instead of stdout")

    classify_p = sub.add_parser("classify", help="Classify a task and print advisory Harness routing", allow_abbrev=False)
    classify_p.add_argument("--text", "-t", default="", help="Task text to classify")
    classify_p.add_argument("--format", choices=("markdown", "json"), default="markdown", help="Output format")

    new_p = sub.add_parser("new", help="Create a new Harness task intake form", allow_abbrev=False)
    new_p.add_argument("--title", default="", help="Task title")
    new_p.add_argument("--workspace", default="", help="Workspace / repo path")
    new_p.add_argument("--mode", default="", help="Desired mode / permission level")
    new_p.add_argument("--output", "-o", default="", help="Output path or directory")
    new_p.add_argument("--print-path", action="store_true", help="Print only the created file path")

    check_p = sub.add_parser("check", help="Validate a filled Harness intake form", allow_abbrev=False)
    check_p.add_argument("file", help="Intake markdown file")

    prompt_p = sub.add_parser("prompt", help="Render a compact prompt from a filled intake form", allow_abbrev=False)
    prompt_p.add_argument("file", help="Intake markdown file")
    prompt_p.add_argument("--allow-incomplete", action="store_true", help="Render even if required fields are missing")
    prompt_p.add_argument("--output", "-o", default="", help="Write prompt to this file instead of stdout")
    prompt_p.add_argument("--force", action="store_true", help="Overwrite output file if it exists")

    kanban_p = sub.add_parser("kanban", help="Bridge Harness intake into Kanban lifecycle", allow_abbrev=False)
    kanban_sub = kanban_p.add_subparsers(dest="harness_kanban_action")

    kanban_create = kanban_sub.add_parser("create", help="Create a Kanban triage card from a filled intake form", allow_abbrev=False)
    kanban_create.add_argument("file", help="Filled intake markdown file")
    kanban_create.add_argument("--assignee", default="architect", help="Kanban assignee/profile (default: architect)")
    kanban_create.add_argument("--workspace", default="", help="Override workspace; defaults to the intake form workspace")
    kanban_create.add_argument("--triage", action="store_true", default=True, help="Park card in triage (default)")
    kanban_create.add_argument("--no-triage", dest="triage", action="store_false", help="Create directly in todo/runnable state")
    kanban_create.add_argument("--dry-run", action="store_true", help="Print the hermes kanban command instead of running it")
    kanban_create.add_argument("--json", action="store_true", help="Emit JSON from hermes kanban create when running")

    kanban_decompose = kanban_sub.add_parser("decompose", help="Create or print implementation/review child-card commands", allow_abbrev=False)
    kanban_decompose.add_argument("task_id", help="Parent Kanban task id")
    kanban_decompose.add_argument("--worker", default="loop-worker", help="Worker profile for implementation card")
    kanban_decompose.add_argument("--reviewer", default="loop-reviewer", help="Reviewer profile for review card")
    kanban_decompose.add_argument("--workspace", default="worktree", help="Workspace hint for implementation card")
    kanban_decompose.add_argument("--branch", default="", help="Branch name for implementation worktree tasks")
    kanban_decompose.add_argument("--execute", action="store_true", help="Actually create child cards; default is dry-run")
    kanban_decompose.add_argument("--json", action="store_true", help="Emit JSON from hermes kanban create when executing")

    evidence_p = sub.add_parser("evidence", help="Generate a Harness Evidence markdown report for a Kanban task or workspace", allow_abbrev=False)
    evidence_p.add_argument("task_id", nargs="?", default="", help="Optional Kanban task id to include via hermes kanban show")
    evidence_p.add_argument("--workspace", default="", help="Repo/workspace to inspect; defaults to cwd")
    evidence_p.add_argument("--output", "-o", default="", help="Write evidence markdown to this file")

    gc_p = sub.add_parser("gc-template", help="Write a weekly Harness GC / drift-review checklist template", allow_abbrev=False)
    gc_p.add_argument("--output", "-o", default="", help="Write template to this file instead of stdout")
    gc_p.add_argument("--board", default="hermes-engineering-loop", help="Target board slug to mention in the template")

    migration_p = sub.add_parser("migration-pack", help="Generate cross-agent Harness migration files", allow_abbrev=False)
    migration_p.add_argument("--output-dir", "-o", default=".", help="Repository/directory where pack files should be written")
    migration_p.add_argument("--force", action="store_true", help="Overwrite existing migration-pack files")
    migration_p.add_argument("--json", action="store_true", help="Print written file paths as JSON")

    parser.set_defaults(func=_handle_harness_cli)


def _handle_harness_cli(args) -> None:
    action = getattr(args, "harness_action", None)
    if not action:
        print(HELP_TEXT)
        raise SystemExit(0)

    if action == "kanban":
        raise SystemExit(_handle_harness_kanban_cli(args))
    if action == "evidence":
        raise SystemExit(_handle_harness_evidence_cli(args))
    if action == "gc-template":
        raise SystemExit(_handle_harness_gc_template_cli(args))
    if action == "migration-pack":
        raise SystemExit(_handle_harness_migration_pack_cli(args))

    argv: list[str] = [action]
    if action == "classify":
        text = getattr(args, "text", "")
        if not text:
            print("Missing --text for harness classify.")
            raise SystemExit(2)
        classification = classify_task(text)
        if getattr(args, "format", "markdown") == "json":
            print(json.dumps(classification.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(_render_classification_markdown(classification))
        raise SystemExit(0)
    if action == "new":
        for flag in ("title", "workspace", "mode"):
            value = getattr(args, flag, "")
            if value:
                argv.extend([f"--{flag}", value])
        out_value = getattr(args, "output", "")
        if out_value:
            argv.extend(["--output", out_value])
        # The helper already prints the created path for `new`; keep
        # --print-path as a plugin-side compatibility flag without passing it
        # to the helper.
    elif action == "check":
        argv.append(getattr(args, "file"))
    elif action == "prompt":
        argv.append(getattr(args, "file"))
        if getattr(args, "allow_incomplete", False):
            argv.append("--allow-incomplete")
        output_value = getattr(args, "output", "")
        if output_value:
            argv.extend(["--output", output_value])
        if getattr(args, "force", False):
            argv.append("--force")
    elif action == "template":
        output_value = getattr(args, "output", "")
        if output_value:
            argv.extend(["--output", output_value])
    else:
        print(HELP_TEXT)
        raise SystemExit(2)
    raise SystemExit(_run_helper(argv))



INTAKE_TITLE = re.compile(r"^- Task title:[ \t]*(.*)$", re.MULTILINE)
INTAKE_WORKSPACE = re.compile(r"^- Workspace / repo:[ \t]*(.*)$", re.MULTILINE)
INTAKE_PROBLEM = re.compile(r"^- Problem / request:[ \t]*(.*)$", re.MULTILINE)
INTAKE_ACCEPTANCE = re.compile(r"^\s*\d+\.[ \t]+(.+)$", re.MULTILINE)
INTAKE_RISK = re.compile(r"^\s*- \[x\][ \t]+(.+)$", re.MULTILINE | re.IGNORECASE)


def _read_text(path: str | Path) -> str:
    return Path(path).expanduser().resolve().read_text(encoding="utf-8")


def _first(pattern: re.Pattern[str], content: str, default: str = "") -> str:
    match = pattern.search(content)
    return match.group(1).strip() if match else default


def _parse_intake_summary(path: str | Path) -> dict[str, Any]:
    intake_path = Path(path).expanduser().resolve()
    content = _read_text(intake_path)
    title = _first(INTAKE_TITLE, content) or intake_path.stem
    workspace = _first(INTAKE_WORKSPACE, content)
    problem = _first(INTAKE_PROBLEM, content)
    acceptance = [m.group(1).strip() for m in INTAKE_ACCEPTANCE.finditer(content) if m.group(1).strip()]
    risks = [m.group(1).strip() for m in INTAKE_RISK.finditer(content) if m.group(1).strip()]
    return {
        "path": str(intake_path),
        "title": title,
        "workspace": workspace,
        "problem": problem,
        "acceptance": acceptance,
        "risks": risks,
    }


def _run_capture(command: list[str], *, cwd: str | None = None) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    return int(proc.returncode), proc.stdout.strip(), proc.stderr.strip()


def _kanban_body_from_intake(summary: dict[str, Any]) -> str:
    lines = [
        "## Harness Intake",
        "",
        f"Source intake: `{summary['path']}`",
        f"Workspace: `{summary.get('workspace') or '<unspecified>'}`",
        "",
        "### Problem",
        summary.get("problem") or "<unspecified>",
        "",
        "### Acceptance criteria",
    ]
    acceptance = summary.get("acceptance") or []
    lines.extend(f"- {item}" for item in acceptance) if acceptance else lines.append("- <unspecified>")
    lines.extend(["", "### Risk surface"])
    risks = summary.get("risks") or []
    lines.extend(f"- {item}" for item in risks) if risks else lines.append("- <unspecified>")
    lines.extend([
        "",
        "### Lifecycle policy",
        "- Keep this card in triage until scope, affected paths, validation commands, rollback/recovery, and evidence are explicit.",
        "- Do not dispatch implementation workers from this bridge unless a human explicitly promotes/decomposes the card.",
        "- Attach Harness Evidence before marking done.",
    ])
    return "\n".join(lines)


def _handle_harness_kanban_cli(args) -> int:
    subaction = getattr(args, "harness_kanban_action", None)
    if subaction == "create":
        summary = _parse_intake_summary(getattr(args, "file"))
        workspace = getattr(args, "workspace", "") or summary.get("workspace") or "scratch"
        command = [
            "hermes", "kanban", "create", summary["title"],
            "--body", _kanban_body_from_intake(summary),
            "--assignee", getattr(args, "assignee", "architect") or "architect",
            "--workspace", workspace,
            "--skill", "harness-agenting-engineering",
            "--skill", "hermes-engineering-loop",
            "--idempotency-key", f"harness-intake:{summary['path']}",
        ]
        if getattr(args, "triage", True):
            command.append("--triage")
        if getattr(args, "json", False):
            command.append("--json")
        if getattr(args, "dry_run", False):
            print(json.dumps({"command": command, "intake": summary}, ensure_ascii=False, indent=2))
            return 0
        code, out, err = _run_capture(command)
        if out:
            print(out)
        if err:
            print(err)
        return code
    if subaction == "decompose":
        parent = getattr(args, "task_id")
        workspace = getattr(args, "workspace", "worktree") or "worktree"
        branch = getattr(args, "branch", "")
        worker_cmd = [
            "hermes", "kanban", "create", f"Implement Harness task {parent}",
            "--parent", parent,
            "--assignee", getattr(args, "worker", "loop-worker") or "loop-worker",
            "--workspace", workspace,
            "--body", "Implement the parent Harness intake in an isolated worktree. Preserve scope and attach verification evidence before completion.",
            "--skill", "harness-agenting-engineering",
            "--skill", "hermes-engineering-loop",
        ]
        if branch:
            worker_cmd.extend(["--branch", branch])
        reviewer_cmd = [
            "hermes", "kanban", "create", f"Review Harness task {parent}",
            "--parent", parent,
            "--assignee", getattr(args, "reviewer", "loop-reviewer") or "loop-reviewer",
            "--workspace", "scratch",
            "--body", "Review only. Check spec compliance, scope control, tests/evidence, Hermes invariants, and safety boundaries. Do not implement changes in this review card.",
            "--skill", "harness-agenting-engineering",
            "--skill", "hermes-engineering-loop",
            "--initial-status", "blocked",
        ]
        if getattr(args, "json", False):
            worker_cmd.append("--json")
            reviewer_cmd.append("--json")
        if not getattr(args, "execute", False):
            print(json.dumps({"dry_run": True, "commands": [worker_cmd, reviewer_cmd]}, ensure_ascii=False, indent=2))
            return 0
        results = []
        for command in (worker_cmd, reviewer_cmd):
            code, out, err = _run_capture(command)
            results.append({"command": command, "exit_code": code, "stdout": out, "stderr": err})
            if out:
                print(out)
            if err:
                print(err)
            if code != 0:
                return code
        return 0
    print("Usage: hermes harness kanban {create,decompose} ...")
    return 2


def _handle_harness_evidence_cli(args) -> int:
    workspace = Path(getattr(args, "workspace", "") or os.getcwd()).expanduser().resolve()
    task_id = getattr(args, "task_id", "") or ""
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "## Harness Evidence",
        "",
        f"Generated: {stamp}",
        f"Workspace: `{workspace}`",
    ]
    if task_id:
        code, out, err = _run_capture(["hermes", "kanban", "show", task_id, "--json"])
        lines.extend(["", "### Kanban task", ""])
        if code == 0 and out:
            lines.append("```json")
            lines.append(out)
            lines.append("```")
        else:
            lines.append(f"Unable to read Kanban task `{task_id}` with `hermes kanban show --json`.")
            if err:
                lines.append(f"Error: `{err}`")
    for heading, cmd in [
        ("Git HEAD", ["git", "rev-parse", "--short", "HEAD"]),
        ("Git status", ["git", "status", "--short"]),
        ("Diff stat", ["git", "diff", "--stat"]),
    ]:
        code, out, err = _run_capture(cmd, cwd=str(workspace))
        lines.extend(["", f"### {heading}", "", "```text"])
        lines.append(out if out else ("<empty>" if code == 0 else err or f"exit {code}"))
        lines.append("```")
    lines.extend([
        "",
        "### Required completion notes",
        "- Spec compliance:",
        "- Automated verification:",
        "- Manual/negative checks:",
        "- Rollback or recovery:",
        "- Retention decision:",
    ])
    content = "\n".join(lines) + "\n"
    output = getattr(args, "output", "")
    if output:
        target = Path(output).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        print(target)
    else:
        print(content)
    return 0


def _handle_harness_gc_template_cli(args) -> int:
    board = getattr(args, "board", "hermes-engineering-loop") or "hermes-engineering-loop"
    stamp = _dt.datetime.now().strftime("%Y-%m-%d")
    content = f"""## Weekly Harness GC / Drift Review

Date: {stamp}
Board: `{board}`
Assignee: reviewer / architect

### Scope

Create/update Kanban triage cards for drift. Do not auto-repair, auto-restart services, delete state, modify credentials, or dispatch implementation workers from this GC pass.

### Checks

- [ ] Blocked Kanban tasks older than policy window have current evidence and owner.
- [ ] Done Harness tasks include evidence, validation commands, and rollback/recovery notes.
- [ ] Profile-local plugins or scripts that are relied on operationally are solidified into repo/skill tap or explicitly marked local-only.
- [ ] `docs/CONTRACTS.md`, `AGENTS.md`, and Harness docs still point to valid commands and current contracts.
- [ ] Skills patched repeatedly are consolidated and still load successfully.
- [ ] Temporary scripts/reports are either archived, promoted, or deleted with human approval.
- [ ] WebUI Harness gate was run for WebUI-facing changes.
- [ ] PR/body handoffs include Contract Routing and Verification when publication is used.

### Evidence commands

```bash
hermes kanban boards switch {board}
hermes kanban list --status blocked
hermes kanban list --archived
hermes harness evidence --workspace /path/to/repo --output /tmp/harness-evidence.md
```

### Output

For each drift finding, create a triage card with evidence, affected paths, risk level, and human-gated recommendation. Leave destructive actions blocked until explicitly approved.
"""
    output = getattr(args, "output", "")
    if output:
        target = Path(output).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        print(target)
    else:
        print(content)
    return 0


MIGRATION_PACK_FILES = {
    "CODEX.md": """# Codex Harness Rules

Use the repository `AGENTS.md` plus this Harness migration pack before non-trivial coding tasks.

## Required Loop

1. Classify the task with `hermes harness classify --text \"<request>\" --format json` when Hermes CLI is available.
2. For high-risk, multi-agent, scheduled, filesystem, auth, secret, approval, or cross-module work, create/fill an intake with `hermes harness new` and validate it with `hermes harness check`.
3. Read applicable contracts before editing: `AGENTS.md`, `CONTRIBUTING.md`, `docs/CONTRACTS.md`, subsystem RFC/docs, and tests for the touched area.
4. Keep implementation and review separate. Use `hermes harness kanban decompose <task-id>` for worker/reviewer child-card commands when Kanban is available.
5. Attach evidence with commands run, failures, manual checks, risk/rollback notes, and retention decisions before claiming done.

## Safety Defaults

- Do not bypass user approval for destructive shell, filesystem, credential, cron, gateway, release, or force-push operations.
- Do not treat upstream PR/CI state as the local completion gate when a local Harness manifest/gate exists.
- Prefer focused tests and local deterministic gates over broad claims.
""",
    "CLAUDE.md": """# Claude Harness Rules

This repository uses Hermes Harness Engineering constraints. Follow them as project memory.

## Context

- `AGENTS.md` and `docs/CONTRACTS.md` are authoritative local entry points.
- `hermes harness classify` routes tasks into direct answer, research, bounded engineering, advisory, or intake-required modes.
- `hermes harness evidence` records completion proof.

## Work Policy

Before code changes, identify scope, acceptance criteria, non-goals, risk surface, rollback/recovery, and verification commands. For multi-agent work, keep implementer and reviewer responsibilities separate.

Do not mutate secrets, cron jobs, credentials, memory, profile config, or destructive filesystem state without explicit human approval.
""",
    "OPENCODE.md": """# OpenCode Harness Rules

Use this file as the OpenCode-compatible entry rule for Harness Engineering.

- Load `AGENTS.md`, `CONTRIBUTING.md`, and `docs/CONTRACTS.md` before repository edits.
- Run `hermes harness classify --text \"<task>\" --format json` for non-trivial work when available.
- Use `hermes harness new/check/prompt` for high-risk or cross-module tasks.
- Produce Harness Evidence before done: spec compliance, automated checks, manual/negative checks, rollback, retention.
- Keep publication PRs separate from local completion gates.
""",
    ".cursor/rules/harness.mdc": """---
description: Harness Engineering local constraints
globs:
  - \"**/*\"
alwaysApply: true
---

Use AGENTS.md, docs/CONTRACTS.md, and Hermes Harness commands before non-trivial edits.

For risky/cross-module work, require intake, explicit risk surface, verification evidence, and rollback notes. Do not perform destructive state changes without human approval.
""",
    ".windsurfrules": """# Harness Engineering Rules

Read AGENTS.md and docs/CONTRACTS.md before editing. Use `hermes harness classify` for non-trivial work and `hermes harness evidence` before claiming done.

High-risk changes require explicit scope, acceptance criteria, risk surface, negative checks, rollback/recovery, and human approval for destructive state.
""",
    "prompts/task-intake.md": """# Harness Task Intake Prompt

Fill this before delegating non-trivial AI engineering work.

- Problem:
- Acceptance criteria:
- Non-goals:
- Workspace / affected paths:
- Applicable contracts/docs:
- Risk surface:
- Rollback or recovery:
- Required automated checks:
- Required manual/negative checks:
- Retention decision:

After filling, run:

```bash
hermes harness check /path/to/intake.md
hermes harness prompt /path/to/intake.md
```
""",
}


def _handle_harness_migration_pack_cli(args) -> int:
    output_dir = Path(getattr(args, "output_dir", ".") or ".").expanduser().resolve()
    force = bool(getattr(args, "force", False))
    written: list[str] = []
    skipped: list[str] = []
    for rel, content in MIGRATION_PACK_FILES.items():
        target = output_dir / rel
        if target.exists() and not force:
            skipped.append(str(target))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(str(target))
    payload = {"output_dir": str(output_dir), "written": written, "skipped_existing": skipped}
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        for path in written:
            print(path)
        for path in skipped:
            print(f"SKIP existing: {path}")
        if skipped:
            print("Use --force to overwrite existing migration-pack files.")
    return 0

ENGINEERING_KEYWORDS = re.compile(
    r"("
    r"修复|修 bug|bug|报错|异常|失败|不生效|调试|排查|"
    r"实现|开发|改代码|重构|优化|接入|集成|迁移|发布|部署|"
    r"测试|单测|pytest|CI|PR|代码审查|review|"
    r"feature|fix|debug|refactor|implement|integrat|migrat|deploy|release|test|coverage"
    r")",
    re.IGNORECASE,
)
RESEARCH_PATTERNS = re.compile(
    r"(调研|研究|查资料|搜索|对比|总结资料|文献|论文|market|research|search|compare|survey|literature)",
    re.IGNORECASE,
)
SMALL_CODE_PATTERNS = re.compile(
    r"(小修|简单修|typo|文案|样式|one[- ]?line|small fix|minor|quick fix|加测试|单测)",
    re.IGNORECASE,
)
LARGE_CODE_PATTERNS = re.compile(
    r"(重构|架构|迁移|端到端|全流程|跨模块|系统性|大规模|多文件|refactor|architecture|migration|end[- ]?to[- ]?end|cross[- ]?module|systematic)",
    re.IGNORECASE,
)
HIGH_RISK_PATTERNS = re.compile(
    r"(认证|授权|权限|密钥|凭证|token|secret|password|cookie|支付|license|删除|清空|覆盖|force[- ]?push|deploy|release|auth|oauth|credential|filesystem|webhook|database|migration)",
    re.IGNORECASE,
)
MULTI_AGENT_PATTERNS = re.compile(
    r"(多代理|子代理|并行|分解|派工|kanban|codex|claude|opencode|multi[- ]?agent|subagent|parallel|delegate|orchestrate)",
    re.IGNORECASE,
)
OPS_SCHEDULED_PATTERNS = re.compile(
    r"(cron|定时|周期|监控|告警|任务队列|scheduler|scheduled|monitor|daemon|gateway)",
    re.IGNORECASE,
)
LOW_RISK_PATTERNS = re.compile(
    r"^(怎么看|是什么|解释|总结|翻译|润色|写一段|生成文案|画|查一下|搜索|what is|explain|summarize|translate)\b",
    re.IGNORECASE,
)
HARNESS_ALREADY_PRESENT = re.compile(
    r"(Harness\s*/\s*Agenting|harness-agenting|hermes\s+harness|/intake|intake\s+form|验收标准|验证证据|risk surface)",
    re.IGNORECASE,
)

PREFLIGHT_NOTICE = """[Harness / Agenting Engineering preflight]\nThis appears to be a non-trivial engineering task. Before implementing, use the harness discipline:\n- define scope, acceptance criteria, risk surface, and rollback plan;\n- inspect the codebase before editing;\n- preserve tests / quality gates as evidence;\n- prefer reusable Skill/Rules/Plugin updates for repeated workflow.\nIf the task is underspecified, ask only for missing information that changes the implementation.\n\nOriginal user request:\n"""

STRICT_NOTICE = """[Harness / Agenting Engineering preflight: intake required]\nThis appears to be a high-risk engineering task. Ask the user to create or fill a Harness intake before implementation:\n  hermes harness new --title \"<task>\" --workspace \"<repo>\" --mode \"Implement changes\" --output /tmp/task-intake.md\n  hermes harness check /tmp/task-intake.md\nProceed only after scope, acceptance criteria, risk surface, and verification evidence are explicit.\n\nOriginal user request:\n"""


def _has(pattern: re.Pattern[str], text: str) -> bool:
    return bool(pattern.search(text))


def classify_task(text: str) -> TaskClassification:
    """Classify task text into advisory Harness routing buckets."""
    compact = (text or "").strip()
    if not compact:
        return TaskClassification(
            task_type="simple_chat",
            harness_required=False,
            risk_level="low",
            route="answer_directly",
            signals=["empty_text"],
            recommended_next_steps=["Ask for the task text before routing."],
        )
    if compact.startswith("/"):
        return TaskClassification(
            task_type="simple_chat",
            harness_required=False,
            risk_level="low",
            route="slash_command",
            signals=["slash_command"],
            recommended_next_steps=["Handle as an in-session command."],
        )

    signals: list[str] = []
    if _has(HARNESS_ALREADY_PRESENT, compact):
        signals.append("harness_context_present")
    if _has(ENGINEERING_KEYWORDS, compact):
        signals.append("engineering_keywords")
    if _has(RESEARCH_PATTERNS, compact):
        signals.append("research_keywords")
    if _has(SMALL_CODE_PATTERNS, compact):
        signals.append("small_code_keywords")
    if _has(LARGE_CODE_PATTERNS, compact):
        signals.append("large_code_keywords")
    if _has(HIGH_RISK_PATTERNS, compact):
        signals.append("high_risk_keywords")
    if _has(MULTI_AGENT_PATTERNS, compact):
        signals.append("multi_agent_keywords")
    if _has(OPS_SCHEDULED_PATTERNS, compact):
        signals.append("ops_scheduled_keywords")

    if _has(LOW_RISK_PATTERNS, compact) and len(compact) < 220 and "engineering_keywords" not in signals:
        return TaskClassification(
            task_type="simple_chat",
            harness_required=False,
            risk_level="low",
            route="answer_directly",
            signals=[*signals, "low_risk_prompt_shape"],
            recommended_next_steps=["Answer directly; no Harness intake needed."],
        )

    if "multi_agent_keywords" in signals:
        task_type = "multi_agent_project"
    elif "high_risk_keywords" in signals:
        task_type = "high_risk_change"
    elif "ops_scheduled_keywords" in signals:
        task_type = "ops_scheduled"
    elif "large_code_keywords" in signals:
        task_type = "large_code_change"
    elif "engineering_keywords" in signals:
        task_type = "small_code_change" if "small_code_keywords" in signals else "large_code_change"
    elif "research_keywords" in signals:
        task_type = "research"
    else:
        task_type = "simple_chat"

    if task_type in {"high_risk_change", "multi_agent_project", "ops_scheduled"}:
        return TaskClassification(
            task_type=task_type,
            harness_required=True,
            risk_level="high",
            route="intake_required",
            signals=signals or ["no_special_signal"],
            recommended_next_steps=[
                "Create or fill a Harness intake before implementation.",
                "Name risk surface, rollback plan, and verification evidence.",
            ],
        )
    if task_type == "large_code_change":
        return TaskClassification(
            task_type=task_type,
            harness_required=True,
            risk_level="medium",
            route="harness_advisory",
            signals=signals or ["no_special_signal"],
            recommended_next_steps=[
                "Define scope, acceptance criteria, risk surface, and tests before editing.",
                "Inspect project rules and touched subsystem contracts first.",
            ],
        )
    if task_type == "small_code_change":
        return TaskClassification(
            task_type=task_type,
            harness_required=False,
            risk_level="medium",
            route="bounded_engineering",
            signals=signals or ["no_special_signal"],
            recommended_next_steps=["Keep the change scoped and run focused tests before reporting done."],
        )
    if task_type == "research":
        return TaskClassification(
            task_type=task_type,
            harness_required=False,
            risk_level="low",
            route="research_then_report",
            signals=signals or ["no_special_signal"],
            recommended_next_steps=["Gather source evidence and report assumptions or gaps explicitly."],
        )
    return TaskClassification(
        task_type="simple_chat",
        harness_required=False,
        risk_level="low",
        route="answer_directly",
        signals=signals or ["no_special_signal"],
        recommended_next_steps=["Answer directly; no Harness intake needed."],
    )


def _render_classification_markdown(classification: TaskClassification) -> str:
    lines = [
        "## Harness Task Classification",
        "",
        f"Task type: `{classification.task_type}`",
        f"Risk level: `{classification.risk_level}`",
        f"Route: `{classification.route}`",
        f"Harness intake required: `{str(classification.harness_required).lower()}`",
        "",
        "Signals:",
    ]
    lines.extend(f"- `{signal}`" for signal in classification.signals)
    lines.extend(["", "Recommended next steps:"])
    lines.extend(f"- {step}" for step in classification.recommended_next_steps)
    return "\n".join(lines)


def _configured_preflight_mode() -> str:
    try:
        from hermes_cli.config import cfg_get, load_config

        value = cfg_get(load_config(), "harness_engineering", "preflight_mode", default="advisory")
    except Exception:
        value = "advisory"
    return str(value or "advisory").strip().lower()


def _preflight_mode() -> str:
    return _configured_preflight_mode()


def _looks_like_engineering_task(text: str) -> bool:
    classification = classify_task(text)
    return classification.task_type in {
        "small_code_change",
        "large_code_change",
        "high_risk_change",
        "multi_agent_project",
        "ops_scheduled",
    }


def _handle_pre_gateway_dispatch(event: Any = None, **_: Any) -> dict[str, str] | None:
    """Soft Level-4 preflight for gateway messages.

    Modes via config.yaml `harness_engineering.preflight_mode`:
      off/0/false/no  -> disabled
      advisory/warn/rewrite (default) -> prepend Harness discipline reminder
      strict -> prepend an intake-required instruction

    The hook returns only `allow` or `rewrite`; it never skips messages.
    """
    mode = _preflight_mode()
    if mode in {"", "off", "0", "false", "no", "disabled"}:
        return {"action": "allow"}
    text = getattr(event, "text", "") if event is not None else ""
    if not isinstance(text, str):
        return {"action": "allow"}
    classification = classify_task(text)
    if classification.task_type not in {
        "small_code_change",
        "large_code_change",
        "high_risk_change",
        "multi_agent_project",
        "ops_scheduled",
    }:
        return {"action": "allow"}
    if mode in {"strict", "require", "required"}:
        return {"action": "rewrite", "text": STRICT_NOTICE + text}
    if classification.harness_required:
        return {"action": "rewrite", "text": STRICT_NOTICE + text}
    return {"action": "rewrite", "text": PREFLIGHT_NOTICE + text}


def _handle_intake_slash(raw_args: str = "") -> str:
    raw = (raw_args or "").strip()
    if raw:
        return (
            f"/intake currently provides entry instructions only. Received: {raw}\n\n"
            f"{HELP_TEXT}"
        )
    return HELP_TEXT


def register(ctx) -> None:
    ctx.register_cli_command(
        "harness",
        help="Harness / Agenting Engineering task intake helper",
        description=HELP_TEXT,
        setup_fn=_setup_harness_cli,
        handler_fn=_handle_harness_cli,
    )
    ctx.register_command(
        "intake",
        handler=_handle_intake_slash,
        description="Show Harness / Agenting Engineering task intake instructions.",
        args_hint="[optional note]",
    )
    ctx.register_hook("pre_gateway_dispatch", _handle_pre_gateway_dispatch)
