#!/usr/bin/env python3
"""Read-only readiness check for the multi-AI project workflow.

The checker verifies the files that let multiple AI tools coordinate in a
project. It intentionally avoids secret-bearing and runtime-heavy paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_ISSUE_FIELDS = (
    "issue_id:",
    "phase:",
    "owner_role:",
    "assigned_ai:",
    "worktree_path:",
    "branch:",
    "goal:",
    "scope:",
    "out_of_scope:",
    "done_when:",
    "verify_commands:",
    "localhost_check:",
    "vps_check:",
    "status:",
    "done_percent:",
    "remaining_percent:",
    "evidence:",
)

REQUIRED_HANDOFF_FIELDS = (
    "task:",
    "latest_state:",
    "next_agent:",
    "next_step:",
    "verification_run:",
    "remaining_risk:",
)

REQUIRED_AI_PAIR_CODER_BRIEF_FIELDS = (
    "review_focus:",
    "commands_run:",
)

REQUIRED_AI_PAIR_REVIEW_RESULT_FIELDS = (
    "decision:",
    "required_changes:",
)


def _read_text(path: Path, max_bytes: int = 80_000) -> str:
    try:
        with path.open("rb") as fh:
            return fh.read(max_bytes).decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    except OSError as exc:
        return f"__READ_ERROR__ {type(exc).__name__}: {exc}"


def _check(code: str, ok: bool, message: str, evidence: str = "") -> dict[str, Any]:
    return {
        "code": code,
        "ok": bool(ok),
        "message": message,
        "evidence": evidence,
    }


def _has_any_file(path: Path, patterns: tuple[str, ...]) -> bool:
    return any(candidate.is_file() for pattern in patterns for candidate in path.glob(pattern))


def _missing_fields(text: str, fields: tuple[str, ...]) -> list[str]:
    return [field for field in fields if field not in text]


def _has_closeout_protocol(text: str) -> bool:
    has_closeout = "Closeout Protocol" in text or "Closeout" in text
    has_next_step = "recommended next step" in text or "ขั้นตอนถัดไป" in text
    has_verification = "verification" in text or "ตรวจสอบ" in text or "ตรวจจริง" in text
    return has_closeout and has_next_step and has_verification


def inspect_project(project: str | Path) -> dict[str, Any]:
    root = Path(project).expanduser().resolve()
    checks: list[dict[str, Any]] = []

    agents = root / "AGENTS.md"
    claude = root / "CLAUDE.md"
    qwen = root / "QWEN.md"
    gemini = root / "GEMINI.md"
    cursor_rules = root / ".cursor" / "rules"
    hermes = root / ".hermes"
    issues = hermes / "issues"
    ai_pair = hermes / "ai-pair"
    issue_template = root / "docs" / "multi-ai-workflow" / "templates" / "issue.md"
    handoff_template = root / "docs" / "multi-ai-workflow" / "templates" / "handoff.md"
    ai_pair_templates = root / "docs" / "multi-ai-workflow" / "templates" / "ai-pair"
    coder_brief_template = ai_pair_templates / "coder-brief.md"
    review_result_template = ai_pair_templates / "review-result.md"

    checks.append(
        _check(
            "project-path-present",
            root.exists() and root.is_dir(),
            "Project directory exists.",
            str(root),
        )
    )

    agents_text = _read_text(agents)
    checks.append(
        _check(
            "agents-adapter-present",
            agents.exists(),
            "AGENTS.md exists as the shared instruction anchor.",
            str(agents),
        )
    )
    checks.append(
        _check(
            "agents-shortcut-registry",
            "Prompt Shortcut Registry" in agents_text or "prompt-shortcut-registry" in agents_text,
            "AGENTS.md points agents to the prompt shortcut registry.",
            "Prompt Shortcut Registry",
        )
    )
    checks.append(
        _check(
            "agents-closeout-protocol",
            _has_closeout_protocol(agents_text),
            "AGENTS.md requires a closeout with verification and a recommended next step.",
            "Closeout Protocol",
        )
    )
    checks.append(
        _check(
            "claude-adapter-present",
            claude.exists(),
            "CLAUDE.md exists for Claude Code.",
            str(claude),
        )
    )
    checks.append(
        _check(
            "qwen-adapter-present",
            qwen.exists(),
            "QWEN.md exists for Qwen Code.",
            str(qwen),
        )
    )
    checks.append(
        _check(
            "gemini-adapter-present",
            gemini.exists(),
            "GEMINI.md exists for Gemini CLI.",
            str(gemini),
        )
    )
    checks.append(
        _check(
            "cursor-adapter-present",
            cursor_rules.exists() and _has_any_file(cursor_rules, ("*.mdc", "*.md")),
            "Cursor project rules exist.",
            str(cursor_rules),
        )
    )

    for name in ("context.md", "active.md", "decisions.md", "handoff.md"):
        path = hermes / name
        checks.append(
            _check(
                f"hermes-{name}-present",
                path.exists(),
                f".hermes/{name} exists for repo-local state.",
                str(path),
            )
        )

    checks.append(
        _check(
            "issue-registry-present",
            issues.exists() and (issues / "README.md").exists(),
            ".hermes/issues/ exists with README.md.",
            str(issues),
        )
    )
    checks.append(
        _check(
            "ai-pair-registry-present",
            ai_pair.exists() and (ai_pair / "README.md").exists(),
            ".hermes/ai-pair/ exists with README.md.",
            str(ai_pair),
        )
    )

    issue_text = _read_text(issue_template)
    issue_missing = _missing_fields(issue_text, REQUIRED_ISSUE_FIELDS)
    checks.append(
        _check(
            "issue-template-fields",
            issue_template.exists() and not issue_missing,
            "Issue template includes assignment, scope, verification, percent, and evidence fields.",
            ", ".join(issue_missing) if issue_missing else str(issue_template),
        )
    )

    handoff_text = _read_text(handoff_template)
    handoff_missing = _missing_fields(handoff_text, REQUIRED_HANDOFF_FIELDS)
    checks.append(
        _check(
            "handoff-template-fields",
            handoff_template.exists() and not handoff_missing,
            "Handoff template includes continuation and verification fields.",
            ", ".join(handoff_missing) if handoff_missing else str(handoff_template),
        )
    )

    coder_brief_text = _read_text(coder_brief_template)
    coder_brief_missing = _missing_fields(coder_brief_text, REQUIRED_AI_PAIR_CODER_BRIEF_FIELDS)
    checks.append(
        _check(
            "ai-pair-coder-template-fields",
            coder_brief_template.exists() and not coder_brief_missing,
            "AI Pair coder brief template includes review focus and verification fields.",
            ", ".join(coder_brief_missing) if coder_brief_missing else str(coder_brief_template),
        )
    )

    review_result_text = _read_text(review_result_template)
    review_result_missing = _missing_fields(review_result_text, REQUIRED_AI_PAIR_REVIEW_RESULT_FIELDS)
    checks.append(
        _check(
            "ai-pair-review-template-fields",
            review_result_template.exists() and not review_result_missing,
            "AI Pair review result template includes decision and required changes fields.",
            ", ".join(review_result_missing) if review_result_missing else str(review_result_template),
        )
    )

    passed = sum(1 for check in checks if check["ok"])
    total = len(checks)
    failed = total - passed

    return {
        "project": str(root),
        "ok": failed == 0,
        "summary": {
            "passed": passed,
            "failed": failed,
            "total": total,
            "done_percent": round((passed / total) * 100, 2) if total else 0,
            "remaining_percent": round((failed / total) * 100, 2) if total else 0,
        },
        "checks": checks,
    }


def render_report(report: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, ensure_ascii=False, indent=2) + "\n"

    lines = [
        f"Multi-AI workflow readiness: {'OK' if report['ok'] else 'FAIL'}",
        f"Project: {report['project']}",
        (
            "Summary: "
            f"{report['summary']['passed']}/{report['summary']['total']} passed, "
            f"{report['summary']['done_percent']}% done, "
            f"{report['summary']['remaining_percent']}% remaining"
        ),
        "",
    ]
    for check in report["checks"]:
        marker = "PASS" if check["ok"] else "FAIL"
        evidence = f" ({check['evidence']})" if check.get("evidence") else ""
        lines.append(f"[{marker}] {check['code']}: {check['message']}{evidence}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".", help="Project directory to inspect.")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    args = parser.parse_args(argv)

    report = inspect_project(args.project)
    print(render_report(report, args.format), end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
