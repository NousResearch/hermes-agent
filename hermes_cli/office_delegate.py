"""First-class Office delegation helpers.

This module backs the /delegate slash command used by the CLI and gateway.
It intentionally bypasses the LLM: when Akhil says to delegate a task, we
create Kanban tasks directly so Office workers can pick them up via the
existing dispatcher.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from hermes_cli import kanban_db as kb
from hermes_cli import agent_office

OFFICE_ROLES: tuple[str, ...] = (
    "triage",
    "chief",
    "supervisor",
    "pm",
    "architect",
    "research",
    "coder",
    "tooling",
    "memory",
    "reviewer",
    "qa",
    "security",
    "approval",
    "observability",
    "devops",
    "docs",
    "demo",
)

ROLE_ALIASES: dict[str, str] = {
    "office": "triage",
    "intake": "triage",
    "frontdesk": "triage",
    "front-desk": "triage",
    "product": "pm",
    "product-manager": "pm",
    "product_manager": "pm",
    "manager": "pm",
    "builder": "coder",
    "code": "coder",
    "developer": "coder",
    "engineer": "coder",
    "qa-engineer": "qa",
    "quality": "qa",
    "tester": "qa",
    "test": "qa",
    "review": "reviewer",
    "code-review": "reviewer",
    "sec": "security",
    "ops": "devops",
    "operations": "devops",
    "documentation": "docs",
    "writer": "docs",
    "showcase": "demo",
}

_ROLE_PATTERN = "|".join(
    sorted(
        {re.escape(r) for r in OFFICE_ROLES} | {re.escape(a) for a in ROLE_ALIASES},
        key=len,
        reverse=True,
    )
)

_CHAIN_SEP_RE = re.compile(r"\s*(?:then|->|→|,|&|and)\s*", re.IGNORECASE)


@dataclass(frozen=True)
class DelegatedTask:
    id: str
    assignee: str
    status: str
    title: str


@dataclass(frozen=True)
class DelegationResult:
    tasks: tuple[DelegatedTask, ...]
    workflow: tuple[str, ...]
    original_request: str


def _canonical_role(raw: str) -> str | None:
    token = raw.strip().lower().replace(" ", "-")
    if not token:
        return None
    if token in OFFICE_ROLES:
        return token
    return ROLE_ALIASES.get(token)


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _parse_role_chain(text: str) -> tuple[list[str], str]:
    """Return (roles, task_text) from flexible /delegate arguments.

    Supported examples:
    - "build office page" -> [triage], "build office page"
    - "to coder: fix tests" -> [coder], "fix tests"
    - "coder fix tests" -> [coder], "fix tests"
    - "pm then coder then qa: build feature" -> [pm, coder, qa], "build feature"
    - "send to office: build feature" -> [triage], "build feature"
    """
    s = text.strip()
    if not s:
        return ["triage"], ""

    # Strip common natural-language wrappers when users paste the same phrase
    # they would say to the agent.
    s = re.sub(r"^(?:please\s+)?(?:delegate|deligate|send|give|assign)\s+(?:this\s+)?(?:task\s+)?", "", s, flags=re.I).strip()

    # "pm then coder then qa: ..." / "to coder: ..." is the most
    # reliable form, so parse an explicit colon before the looser prefix
    # matcher below. The loose matcher intentionally allows no colon for
    # single-role commands like "/delegate coder fix tests".
    if ":" in s:
        before, after = s.split(":", 1)
        role_part = re.sub(r"^(?:to|for)\s+", "", before.strip(), flags=re.I)
        raw_roles = [p for p in _CHAIN_SEP_RE.split(role_part) if p.strip()]
        roles = _dedupe_keep_order(r for r in (_canonical_role(p) for p in raw_roles) if r)
        if roles and after.strip():
            return roles, after.strip()

    # "to coder ...", "to pm then coder then qa ...", "office ..."
    prefix_match = re.match(
        rf"^(?:(?:to|for)\s+)?(?P<roles>{_ROLE_PATTERN}(?:\s*(?:then|->|→|,|&|and)\s*{_ROLE_PATTERN})*)\s*(?::|-|—)?\s*(?P<rest>.*)$",
        s,
        flags=re.I,
    )
    if prefix_match:
        raw_roles = [p for p in _CHAIN_SEP_RE.split(prefix_match.group("roles")) if p.strip()]
        roles = _dedupe_keep_order(r for r in (_canonical_role(p) for p in raw_roles) if r)
        rest = prefix_match.group("rest").strip()
        # Avoid treating a normal task that starts with a role-ish verb as an
        # assignment unless there is either explicit punctuation, "to/for", a
        # multi-role chain, or remaining text that clearly follows the role.
        if roles and rest:
            return roles, rest
        if roles and not rest and (s.lower().startswith(("to ", "for ")) or ":" in s):
            return roles, ""

    # "... to coder", "... to pm then coder then qa" at the end.
    suffix_match = re.search(
        rf"\s+(?:to|for|assigned\s+to)\s+(?P<roles>{_ROLE_PATTERN}(?:\s*(?:then|->|→|,|&|and)\s*{_ROLE_PATTERN})*)\s*$",
        s,
        flags=re.I,
    )
    if suffix_match:
        raw_roles = [p for p in _CHAIN_SEP_RE.split(suffix_match.group("roles")) if p.strip()]
        roles = _dedupe_keep_order(r for r in (_canonical_role(p) for p in raw_roles) if r)
        rest = s[: suffix_match.start()].strip(" :-—")
        if roles and rest:
            return roles, rest

    return ["triage"], s


def _title_for(task_text: str, role: str, index: int, total: int) -> str:
    base = " ".join(task_text.split()) or "Office task"
    if len(base) > 90:
        base = base[:87].rstrip() + "..."
    if total == 1:
        return base
    return f"{index}. {role}: {base}"


def _requires_approval_mode(text: str) -> bool:
    cfg = agent_office.office_config()
    needles = cfg.get("approval_mode_keywords") or [
        "keep me in the loop", "ask me", "take my permission",
        "approval required", "do not yolo", "not yolo",
    ]
    lowered = text.casefold()
    return any(str(n).casefold() in lowered for n in needles)


def _workflow_with_approval_gate(roles: list[str], original_text: str) -> tuple[list[str], str]:
    mode = "approval" if _requires_approval_mode(original_text) else "yolo"
    if mode == "approval" and (not roles or roles[0] != "approval"):
        roles = ["approval", *roles]
    return roles, mode


def create_office_delegation(
    text: str,
    *,
    created_by: str = "office-delegate",
    tenant: str | None = None,
    priority: int = 0,
) -> DelegationResult:
    roles, task_text = _parse_role_chain(text)
    roles, operating_mode = _workflow_with_approval_gate(roles, text)
    if not task_text.strip():
        raise ValueError(
            "usage: /delegate [to <role>|<role> then <role>:] <task>\n"
            "examples:\n"
            "  /delegate build the Office status card\n"
            "  /delegate to coder: fix dashboard tests\n"
            "  /delegate pm then coder then qa: build Telegram status command"
        )

    created: list[DelegatedTask] = []
    parent_ids: list[str] = []
    total = len(roles)
    with kb.connect() as conn:
        for idx, role in enumerate(roles, start=1):
            assignee = agent_office.resolve_profile_for_role(conn, role)
            manager = agent_office.ROLE_MANAGERS.get(role, "chief")
            title = _title_for(task_text, role, idx, total)
            body_lines = [
                "Created by /delegate for Agent Office.",
                f"Original request: {task_text}",
                f"Workflow: {' -> '.join(roles)}",
                f"This step role: {role}",
                f"This step concrete assignee: {assignee}",
                f"Reports to: {manager}",
                f"Operating mode: {operating_mode}. Office default is yolo/hands-free; approval mode is active only because the request explicitly asked for permission/loop.",
                "If this is an approval step, notify Akhil via Telegram/home channel using `.hermes/scripts/office_approval_notify.py` and wait for approval/deny before downstream work.",
                "Workspace boundary: /Users/akhilkinnera/Documents/My Workspace. Workers may create/edit/download inside that tree; installation/security scrutiny routes through security/devops hands-free.",
                "Office quality gate for EVERY task: do not trust agent 'done' prose. End with an evidence-backed gate scorecard listing requested/implicit gates, commands/checks run, exit codes, artifact paths, and pass/fail/partial/blocked/not_applicable verdicts.",
                "No silent scope reduction: if any spec requirement cannot be satisfied after a serious attempt, stop and emit a parseable SCOPE_CHANGE_REQUEST block that starts with SCOPE_CHANGE_REQUEST, includes requirement_ref, requested_change, reason, attempted_evidence, impact, and options, and ends with END_SCOPE_CHANGE_REQUEST before proceeding.",
                "Benchmarks/performance claims require real produced artifacts (for example BENCHMARKS.md with numbers, Criterion output, k6 JSON, live-server pytest report, helm kind/minikube install log), not merely scripts or claims that they could be produced later.",
                "Substantive implementation work must be followed by reviewer/QA evidence review; reviewers must inspect diffs plus programmatic evidence before approving completion.",
            ]
            if parent_ids:
                body_lines.append(f"Depends on: {', '.join(parent_ids)}")
            task_id = kb.create_task(
                conn,
                title=title,
                body="\n".join(body_lines),
                assignee=assignee,
                created_by=created_by,
                tenant=tenant,
                priority=priority,
                parents=tuple(parent_ids[-1:]),
                workspace_kind="scratch",
            )
            task = kb.get_task(conn, task_id)
            created.append(
                DelegatedTask(
                    id=task.id,
                    assignee=task.assignee or role,
                    status=task.status,
                    title=task.title,
                )
            )
            parent_ids = [task_id]
    return DelegationResult(tasks=tuple(created), workflow=tuple(roles), original_request=task_text)


def format_delegation_result(result: DelegationResult, *, dispatcher_note: bool = True) -> str:
    if len(result.tasks) == 1:
        t = result.tasks[0]
        lines = [
            f"Delegated to Office: {t.id}",
            f"Assignee: {t.assignee}",
            f"Status: {t.status}",
            f"Task: {t.title}",
        ]
    else:
        lines = [
            "Delegated to Office workflow:",
            f"Flow: {' -> '.join(result.workflow)}",
        ]
        for t in result.tasks:
            lines.append(f"- {t.id}  {t.status:8s}  {t.assignee:14s}  {t.title}")
    if dispatcher_note:
        lines.extend([
            "",
            "Monitor it in the Office dashboard or with `/kanban show <task_id>`.",
            "Workers will start when the gateway/dispatcher is running.",
        ])
    return "\n".join(lines)


def run_delegate_slash(text: str, *, created_by: str = "office-delegate") -> str:
    result = create_office_delegation(text, created_by=created_by)
    return format_delegation_result(result)
