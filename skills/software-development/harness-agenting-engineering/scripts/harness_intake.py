#!/usr/bin/env python3
"""Harness / Agenting Engineering task intake helper.

Local command helper for the user-local `harness-agenting-engineering` skill.
It intentionally avoids importing Hermes internals so it can run from CLI, WebUI
terminals, WSL, cron scripts, or other agents.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import shutil
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
TEMPLATE = SKILL_DIR / "templates" / "task-intake-form.md"
DEFAULT_OUT_DIR = Path.home() / ".hermes" / "harness" / "intake"

FIELD_PATTERNS = {
    "task_title": re.compile(r"^- Task title:[ \t]*(.*)$", re.MULTILINE),
    "workspace": re.compile(r"^- Workspace / repo:[ \t]*(.*)$", re.MULTILINE),
    "problem": re.compile(r"^- Problem / request:[ \t]*(.*)$", re.MULTILINE),
    "business_outcome": re.compile(r"^- Business or user outcome:[ \t]*(.*)$", re.MULTILINE),
    "chosen_extension": re.compile(r"^Chosen extension surface:[ \t]*(.*)$", re.MULTILINE),
    "reason": re.compile(r"^Reason:[ \t]*(.*)$", re.MULTILINE),
}

CHECKED_LINE = re.compile(r"^\s*- \[x\][ \t]+(.+)$", re.MULTILINE | re.IGNORECASE)
ACCEPTANCE_ITEM = re.compile(r"^\s*\d+\.[ \t]+(.+)$", re.MULTILINE)
REQUIRED_NEGATIVE = re.compile(r"^Required negative checks:\n(?P<body>(?:^-.*\n?)*)", re.MULTILINE)
VERIFICATION_SECTION = re.compile(
    r"## 6\. Verification evidence required before done\n(?P<body>.*?)(?:\n## 7\.|\Z)",
    re.DOTALL,
)
RETENTION_SECTION = re.compile(r"## 7\. Retention decision\n(?P<body>.*?)(?:\n## 8\.|\Z)", re.DOTALL)


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:80] or "task"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, content: str, force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise SystemExit(f"Refusing to overwrite existing file: {path}\nUse --force to overwrite.")
    path.write_text(content, encoding="utf-8")


def command_new(args: argparse.Namespace) -> int:
    if not TEMPLATE.exists():
        raise SystemExit(f"Template not found: {TEMPLATE}")
    content = _read(TEMPLATE)
    title = args.title.strip() if args.title else ""
    workspace = args.workspace.strip() if args.workspace else os.getcwd()
    if title:
        content = content.replace("- Task title:", f"- Task title: {title}", 1)
    if workspace:
        content = content.replace("- Workspace / repo:", f"- Workspace / repo: {workspace}", 1)
    if args.mode:
        # Mark a matching Desired mode checkbox when the label is present.
        mode = args.mode.strip().lower()
        lines = []
        for line in content.splitlines():
            if line.strip().startswith("- [ ]") and line.strip()[6:].strip().lower() == mode:
                line = line.replace("[ ]", "[x]", 1)
            lines.append(line)
        content = "\n".join(lines) + "\n"

    out = Path(args.output) if args.output else None
    if out is None:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = _slugify(title) if title else "task"
        out = DEFAULT_OUT_DIR / f"{stamp}-{slug}.md"
    _write(out.expanduser().resolve(), content, force=args.force)
    print(out.expanduser().resolve())
    return 0


def _field(content: str, key: str) -> str:
    m = FIELD_PATTERNS[key].search(content)
    return (m.group(1).strip() if m else "")


def _checked_in_section(content: str, start_heading: str, next_heading: str | None = None) -> list[str]:
    start = content.find(start_heading)
    if start == -1:
        return []
    end = len(content)
    if next_heading:
        n = content.find(next_heading, start + len(start_heading))
        if n != -1:
            end = n
    section = content[start:end]
    return [m.group(1).strip() for m in CHECKED_LINE.finditer(section)]


def parse_form(content: str) -> dict[str, object]:
    acceptance = [m.group(1).strip() for m in ACCEPTANCE_ITEM.finditer(content) if m.group(1).strip()]
    modes = _checked_in_section(content, "## 0. Submission mode", "## 1.")
    risks = _checked_in_section(content, "## 3. Risk surface", "## 4.")
    extension_choices = _checked_in_section(content, "## 4. Extension choice", "## 5.")

    neg = ""
    m = REQUIRED_NEGATIVE.search(content)
    if m:
        neg = m.group("body").strip()
    verification = ""
    m = VERIFICATION_SECTION.search(content)
    if m:
        verification = m.group("body").strip()
    retention = ""
    m = RETENTION_SECTION.search(content)
    if m:
        retention = m.group("body").strip()

    return {
        "task_title": _field(content, "task_title"),
        "workspace": _field(content, "workspace"),
        "problem": _field(content, "problem"),
        "business_outcome": _field(content, "business_outcome"),
        "acceptance": acceptance,
        "modes": modes,
        "risks": risks,
        "chosen_extension": _field(content, "chosen_extension"),
        "extension_reason": _field(content, "reason"),
        "extension_choices": extension_choices,
        "negative_checks": neg,
        "verification": verification,
        "retention": retention,
    }


def validate_form(data: dict[str, object]) -> list[str]:
    missing: list[str] = []
    for key, label in [
        ("task_title", "Task title"),
        ("workspace", "Workspace / repo"),
        ("problem", "Problem / request"),
    ]:
        if not str(data.get(key) or "").strip():
            missing.append(label)
    if not data.get("modes"):
        missing.append("At least one desired mode / permission checkbox")
    if not data.get("acceptance"):
        missing.append("Acceptance criteria")
    if not data.get("risks"):
        missing.append("Risk surface checkbox")
    if not str(data.get("verification") or "").strip():
        missing.append("Verification evidence required")
    return missing


def command_check(args: argparse.Namespace) -> int:
    path = Path(args.form).expanduser().resolve()
    content = _read(path)
    data = parse_form(content)
    missing = validate_form(data)
    if missing:
        print("Missing or incomplete fields:")
        for item in missing:
            print(f"- {item}")
        return 2
    print(f"OK: {path}")
    return 0


def command_prompt(args: argparse.Namespace) -> int:
    path = Path(args.form).expanduser().resolve()
    content = _read(path)
    data = parse_form(content)
    missing = validate_form(data)
    if missing and not args.allow_incomplete:
        print("Form is incomplete; use `check` to see details or pass --allow-incomplete.", file=sys.stderr)
        for item in missing:
            print(f"- {item}", file=sys.stderr)
        return 2

    prompt = f"""Please execute this task under Harness / Agenting Engineering discipline.

Task: {data['task_title'] or '<title>'}
Workspace/repo: {data['workspace'] or '<workspace>'}
Mode and permissions: {', '.join(data['modes']) if data['modes'] else '<mode + permission level>'}
Spec: Problem: {data['problem'] or '<problem>'}; Outcome: {data['business_outcome'] or '<outcome>'}; Acceptance criteria: {'; '.join(data['acceptance']) if data['acceptance'] else '<acceptance criteria>'}
Context routing: inspect the form for docs/files/skills/MCP/tools before editing; do not skip AGENTS/CONTRIBUTING/contracts when present.
Risk surface: {', '.join(data['risks']) if data['risks'] else '<checked risks>'}; Required negative checks: {data['negative_checks'] or '<negative checks>'}
Extension choice: {data['chosen_extension'] or ', '.join(data['extension_choices']) or '<Plugin/MCP/Command/Skill/Rule/Contract/Built-in>'}; Reason: {data['extension_reason'] or '<reason>'}
Implementation expectation: smallest useful change; keep scope narrow; update docs/tests alongside behavior changes.
Verification required: {data['verification'] or '<tests, lint, manual evidence, security checks>'}
Retention: {data['retention'] or '<skill/rule/contract/memory decision>'}

Do not claim completion without evidence. If required context is missing and cannot be retrieved, ask a targeted clarification before implementation.
"""
    if args.output:
        _write(Path(args.output).expanduser().resolve(), prompt, force=args.force)
        print(Path(args.output).expanduser().resolve())
    else:
        print(prompt)
    return 0


def command_template(args: argparse.Namespace) -> int:
    target = Path(args.output).expanduser().resolve() if args.output else None
    if target:
        shutil.copyfile(TEMPLATE, target)
        print(target)
    else:
        print(_read(TEMPLATE))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes-harness",
        description="Harness / Agenting Engineering task intake helper",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new", help="create a new task intake form")
    p_new.add_argument("--title", "-t", default="", help="task title to prefill")
    p_new.add_argument("--workspace", "-w", default="", help="workspace/repo to prefill; defaults to cwd")
    p_new.add_argument("--mode", "-m", default="", help="desired mode label to check, e.g. 'Implement changes'")
    p_new.add_argument("--output", "-o", default="", help="output markdown path")
    p_new.add_argument("--force", action="store_true", help="overwrite output if it exists")
    p_new.set_defaults(func=command_new)

    p_check = sub.add_parser("check", help="validate required intake fields")
    p_check.add_argument("form", help="filled intake form markdown")
    p_check.set_defaults(func=command_check)

    p_prompt = sub.add_parser("prompt", help="render condensed prompt from a filled form")
    p_prompt.add_argument("form", help="filled intake form markdown")
    p_prompt.add_argument("--allow-incomplete", action="store_true", help="render even if required fields are missing")
    p_prompt.add_argument("--output", "-o", default="", help="write prompt to file instead of stdout")
    p_prompt.add_argument("--force", action="store_true", help="overwrite output if it exists")
    p_prompt.set_defaults(func=command_prompt)

    p_template = sub.add_parser("template", help="print or copy the blank template")
    p_template.add_argument("--output", "-o", default="", help="copy template to path instead of stdout")
    p_template.set_defaults(func=command_template)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
