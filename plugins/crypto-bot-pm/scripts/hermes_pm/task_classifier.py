#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

TASK_CLASSIFIER_SCHEMA_VERSION = "hermes.pm.task_classification.v1"

TASK_CLASSES = {
    "read",
    "plan",
    "propose",
    "branch_write",
    "forge_write",
    "ci_trial",
    "deploy",
    "runtime_admin",
    "secret",
    "financial",
    "unknown",
}

NON_ACTION_BOOLEANS = {
    "executes_task": False,
    "writes_files": False,
    "calls_gitea_write_api": False,
    "starts_runner": False,
    "runs_workflows": False,
    "deploys": False,
    "runtime_actions": False,
    "financial_actions": False,
    "secret_access": False,
    "branch_writer_invoked": False,
}

FINANCIAL_PATTERNS = (
    r"\bbroker\b",
    r"\bexchange\b",
    r"\brobinhood\b",
    r"\bwallet\b",
    r"\border\b",
    r"\bposition\b",
    r"\btrade\b",
    r"\btrading\b",
    r"\blive[-_ ]?market\b",
    r"\baccount\b",
    r"\bbuy\b",
    r"\bsell\b",
)
SECRET_PATTERNS = (
    r"\.env\b",
    r"\bapi[-_ ]?key\b",
    r"\bauthorization\b",
    r"\bbearer\b",
    r"\bcookie\b",
    r"\bcredential\b",
    r"\bkeychain\b",
    r"\bpassword\b",
    r"\bprivate[-_ ]?key\b",
    r"\bsecret\b",
    r"\btoken\b",
)
RUNTIME_PATTERNS = (
    r"\bdaemon\b",
    r"\bworker\b",
    r"\bscheduler\b",
    r"\bservice[-_ ]?(start|stop|restart)\b",
    r"\bstart\b.*\brunner\b",
    r"\brunner\b.*\bdaemon\b",
    r"\blaunchctl\b",
    r"\blaunchd\b",
    r"\bruntime\b",
    r"\bqmd\b",
)
DEPLOY_PATTERNS = (
    r"\bdeploy\b",
    r"\bkubernetes\b",
    r"\bkubectl\b",
    r"\bflux\b",
    r"\bharbor\b",
    r"\bhelm\b",
    r"\bdocker build\b",
    r"\bpublish image\b",
)
CI_TRIAL_PATTERNS = (
    r"\bci trial\b",
    r"\bsmoke trial\b",
    r"\brun workflow\b",
    r"\brun workflows\b",
    r"\bgitea actions\b",
    r"\bworkflow[_ -]?dispatch\b",
)
FORGE_WRITE_PATTERNS = (
    r"\bcreate\b.*\bissue\b",
    r"\bupdate\b.*\bissue\b",
    r"\bclose\b.*\bissue\b",
    r"\bcomment\b.*\b(pr|pull request|issue)\b",
    r"\bopen\b.*\b(pr|pull request)\b",
    r"\blabel\b",
    r"\bproject board\b",
    r"\bkanban\b.*\bwrite\b",
)
BRANCH_WRITE_PATTERNS = (
    r"\bedit\b",
    r"\bwrite\b",
    r"\bpatch\b",
    r"\bmodify\b",
    r"\bupdate\b.*\b(file|doc|docs|code|test|script)\b",
    r"\bcreate\b.*\b(file|doc|docs|test|script)\b",
    r"\bcommit\b",
)
READ_PATTERNS = (
    r"\bread\b",
    r"\binspect\b",
    r"\blist\b",
    r"\bsummarize\b",
    r"\bstatus\b",
    r"\breport\b",
)
PLAN_PATTERNS = (
    r"\bplan\b",
    r"\barchitecture\b",
    r"\bdesign\b",
    r"\brunbook\b",
    r"\bcadence\b",
)
PROPOSE_PATTERNS = (
    r"\bpropose\b",
    r"\bdraft\b",
    r"\brecommend\b",
    r"\bclassify\b",
    r"\btriage\b",
)


def _payload_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        values: list[str] = []
        for key in ("title", "summary", "description", "request", "target", "path"):
            value = payload.get(key)
            if value is not None:
                values.append(str(value))
        paths = payload.get("paths")
        if isinstance(paths, list):
            values.extend(str(item) for item in paths)
        return " ".join(values)
    return str(payload)


def _matches(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _base_result(
    *,
    text: str,
    task_class: str,
    decision: str,
    forbidden: bool,
    approval_required: bool,
    approval_requirements: list[str],
    suggested_subagent_role: str,
    next_safe_action: str,
) -> dict[str, Any]:
    if task_class not in TASK_CLASSES:
        task_class = "unknown"
    return {
        "schema_version": TASK_CLASSIFIER_SCHEMA_VERSION,
        "tool": "hermes_pm_task_classifier",
        "task_class": task_class,
        "decision": decision,
        "forbidden": forbidden,
        "approval_required": approval_required,
        "approval_requirements": approval_requirements,
        "suggested_subagent_role": suggested_subagent_role,
        "next_safe_action": next_safe_action,
        "matched_text_excerpt": text[:180],
        "non_action_booleans": dict(NON_ACTION_BOOLEANS),
    }


def classify_task(payload: Any) -> dict[str, Any]:
    text = _payload_text(payload).strip()
    lowered = text.lower()
    if not text:
        return _base_result(
            text="",
            task_class="unknown",
            decision="blocked",
            forbidden=False,
            approval_required=True,
            approval_requirements=["Operator must clarify the requested task."],
            suggested_subagent_role="operator_triage",
            next_safe_action=(
                "Ask the Operator for a concrete read/propose/write scope."
            ),
        )
    if _matches(SECRET_PATTERNS, lowered):
        return _base_result(
            text=text,
            task_class="secret",
            decision="blocked",
            forbidden=True,
            approval_required=True,
            approval_requirements=[
                "Secret, credential, token, Keychain, cookie, or .env access "
                "is forbidden."
            ],
            suggested_subagent_role="security_reviewer_blocked",
            next_safe_action=(
                "Refuse secret access and propose a redacted evidence or "
                "policy review instead."
            ),
        )
    if _matches(FINANCIAL_PATTERNS, lowered):
        return _base_result(
            text=text,
            task_class="financial",
            decision="blocked",
            forbidden=True,
            approval_required=True,
            approval_requirements=[
                "Trading, broker, exchange, account, order, wallet, or "
                "live-market access is forbidden."
            ],
            suggested_subagent_role="financial_safety_reviewer_blocked",
            next_safe_action=(
                "Refuse financial action and offer a read-only architecture "
                "or safety-plan review."
            ),
        )
    if _matches(RUNTIME_PATTERNS, lowered):
        return _base_result(
            text=text,
            task_class="runtime_admin",
            decision="blocked",
            forbidden=True,
            approval_required=True,
            approval_requirements=[
                "Runtime service, runner daemon, worker, scheduler, launchd, "
                "or qmd control is forbidden."
            ],
            suggested_subagent_role="runtime_admin_blocked",
            next_safe_action=(
                "Prepare a proposal-only runbook or approval packet; do not "
                "start anything."
            ),
        )
    if _matches(DEPLOY_PATTERNS, lowered):
        return _base_result(
            text=text,
            task_class="deploy",
            decision="approval_required",
            forbidden=False,
            approval_required=True,
            approval_requirements=[
                "Deployment requires an explicit operator approval and a "
                "dedicated deploy-gated checkpoint."
            ],
            suggested_subagent_role="gitops_or_release_specialist",
            next_safe_action=(
                "Create a deployment proposal and evidence checklist only."
            ),
        )
    if _matches(CI_TRIAL_PATTERNS, lowered):
        return _base_result(
            text=text,
            task_class="ci_trial",
            decision="approval_required",
            forbidden=False,
            approval_required=True,
            approval_requirements=[
                "CI trials require explicit approval before starting runners "
                "or running workflows."
            ],
            suggested_subagent_role="ci_evidence_specialist",
            next_safe_action=(
                "Summarize CI evidence and propose a future no-op trial plan."
            ),
        )
    if _matches(FORGE_WRITE_PATTERNS, lowered):
        return _base_result(
            text=text,
            task_class="forge_write",
            decision="approval_required",
            forbidden=False,
            approval_required=True,
            approval_requirements=[
                "Gitea issue, PR, label, comment, or project-board mutation "
                "requires explicit approval."
            ],
            suggested_subagent_role="forge_coordinator",
            next_safe_action=(
                "Emit a proposal for the forge mutation without calling write "
                "APIs."
            ),
        )
    if _matches(BRANCH_WRITE_PATTERNS, lowered):
        return _base_result(
            text=text,
            task_class="branch_write",
            decision="approval_required",
            forbidden=False,
            approval_required=True,
            approval_requirements=[
                "Branch file changes require explicit scoped approval and "
                "branch-write evidence gates."
            ],
            suggested_subagent_role="branch_coding_specialist",
            next_safe_action=(
                "Prepare a write-plan proposal; do not invoke a writer or "
                "apply changes."
            ),
        )
    if _matches(PROPOSE_PATTERNS, lowered):
        return _base_result(
            text=text,
            task_class="propose",
            decision="proposal_only",
            forbidden=False,
            approval_required=False,
            approval_requirements=[],
            suggested_subagent_role="pm_architect",
            next_safe_action=(
                "Draft a proposal, task card, or approval request for operator "
                "review."
            ),
        )
    if _matches(PLAN_PATTERNS, lowered):
        return _base_result(
            text=text,
            task_class="plan",
            decision="proposal_only",
            forbidden=False,
            approval_required=False,
            approval_requirements=[],
            suggested_subagent_role="architect_planner",
            next_safe_action=(
                "Create a plan or runbook without mutating repo, forge, "
                "runtime, or CI."
            ),
        )
    if _matches(READ_PATTERNS, lowered):
        return _base_result(
            text=text,
            task_class="read",
            decision="allowed_read_only",
            forbidden=False,
            approval_required=False,
            approval_requirements=[],
            suggested_subagent_role="status_analyst",
            next_safe_action="Read safe evidence and report compact status.",
        )
    return _base_result(
        text=text,
        task_class="unknown",
        decision="blocked",
        forbidden=False,
        approval_required=True,
        approval_requirements=[
            "Unknown tasks default to blocked until Hermes can classify "
            "authority and risk."
        ],
        suggested_subagent_role="operator_triage",
        next_safe_action=(
            "Ask for clarification or reduce the task to read/propose scope."
        ),
    )


def load_task_payload(path: Path | None, text: str | None) -> Any:
    if text is not None:
        return text
    if path is None or str(path) == "-":
        raw = sys.stdin.read()
    else:
        raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def format_classification_text(classification: dict[str, Any]) -> str:
    lines = [
        "Hermes PM task classification",
        f"Class: {classification.get('task_class')}",
        f"Decision: {classification.get('decision')}",
        f"Forbidden: {'yes' if classification.get('forbidden') else 'no'}",
        (
            "Approval required: "
            + ("yes" if classification.get("approval_required") else "no")
        ),
        f"Role: {classification.get('suggested_subagent_role')}",
        f"Next: {classification.get('next_safe_action')}",
    ]
    return "\n".join(lines)


OPERATOR_AUTHORITY_METADATA = {
    "tool": "hermes_pm_task_classifier",
    "authority_class": "propose",
    "schema_version": TASK_CLASSIFIER_SCHEMA_VERSION,
    "read_only": True,
    "mutation_capability": False,
    **NON_ACTION_BOOLEANS,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify a Hermes PM task into read/plan/propose/write/CI/deploy/"
            "runtime/secret/financial authority without executing it."
        )
    )
    parser.add_argument("--task-file", type=Path, help="Task JSON/text path, or '-'")
    parser.add_argument("--text", help="Task text to classify")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument(
        "--describe-authority",
        action="store_true",
        help="Print this tool's Hermes PM authority metadata as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    indent = 2 if args.pretty else None
    if args.describe_authority:
        print(json.dumps(OPERATOR_AUTHORITY_METADATA, indent=indent, sort_keys=True))
        return 0
    payload = load_task_payload(args.task_file, args.text)
    result = classify_task(payload)
    if args.format == "text":
        print(format_classification_text(result))
    else:
        print(json.dumps(result, indent=indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
