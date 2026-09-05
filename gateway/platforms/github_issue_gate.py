"""Local GitHub issue/PR webhook gate for agent-triggering routes.

The generic webhook adapter already validates HMAC signatures and can wake an
agent for accepted events.  GitHub issue-worker routes need one more guardrail:
cheaply decide whether an event is worth an LLM run before the prompt reaches
the agent.  This module is deliberately pure Python / payload-only so it can be
unit tested without a gateway, network, GitHub API, or model provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_RELEVANT_ISSUE_ACTIONS = frozenset(
    {"opened", "edited", "closed", "reopened", "labeled", "unlabeled"}
)
_RELEVANT_COMMENT_ACTIONS = frozenset({"created", "edited"})
_RELEVANT_PR_ACTIONS = frozenset(
    {"opened", "edited", "closed", "reopened", "synchronize", "labeled", "unlabeled"}
)


@dataclass(frozen=True)
class GitHubIssueGateDecision:
    keep: bool
    reason: str
    repo: str
    number: int | None
    action: str
    event_type: str
    llm_reason: str = ""


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    return [value]


def _string_set(value: Any) -> set[str]:
    return {str(item).strip() for item in _as_list(value) if str(item).strip()}


def _labels_from(item: Any) -> set[str]:
    labels: set[str] = set()
    if not isinstance(item, dict):
        return labels
    for raw in _as_list(item.get("labels")):
        if isinstance(raw, dict):
            name = raw.get("name")
        else:
            name = raw
        if name is not None and str(name).strip():
            labels.add(str(name).strip())
    label = item.get("label")
    if isinstance(label, dict) and label.get("name"):
        labels.add(str(label["name"]).strip())
    return labels


def _repo_name(payload: dict[str, Any]) -> str:
    repo = payload.get("repository")
    return str(repo.get("full_name") or "") if isinstance(repo, dict) else ""


def _sender_login(payload: dict[str, Any]) -> str:
    sender = payload.get("sender")
    return str(sender.get("login") or "") if isinstance(sender, dict) else ""


def _body_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("comment", "issue", "pull_request"):
        item = payload.get(key)
        if isinstance(item, dict):
            body = item.get("body")
            if body:
                parts.append(str(body))
    return "\n".join(parts)


def _subject(payload: dict[str, Any], event_type: str) -> tuple[dict[str, Any] | None, bool]:
    if event_type == "issue_comment":
        issue = payload.get("issue")
        return (issue, bool(isinstance(issue, dict) and issue.get("pull_request")))
    if event_type == "issues":
        issue = payload.get("issue")
        return (issue, False)
    if event_type == "pull_request":
        return (payload.get("pull_request"), True)
    return (None, False)


def evaluate_github_issue_gate(
    config: dict[str, Any] | None,
    payload: dict[str, Any],
    event_type: str,
) -> GitHubIssueGateDecision:
    """Return whether a GitHub webhook event should wake an agent.

    Supported route config under ``github_issue_gate``:
      - repositories: optional list of ``owner/repo`` names
      - labels_all / labels_any: labels required on the issue/PR
      - self_users: GitHub logins whose events are ignored
      - self_markers: body markers whose events are ignored
      - include_pull_requests / include_pushes: opt into PR/push events
    """

    cfg = config or {}
    repo = _repo_name(payload)
    action = str(payload.get("action") or "")
    subject, is_pr_subject = _subject(payload, event_type)
    number = None
    if isinstance(subject, dict):
        try:
            number = int(subject.get("number") or payload.get("number"))
        except (TypeError, ValueError):
            number = None

    repositories = _string_set(cfg.get("repositories"))
    if repositories and repo not in repositories:
        return GitHubIssueGateDecision(False, "repository_not_configured", repo, number, action, event_type)

    if _sender_login(payload) in _string_set(cfg.get("self_users")):
        return GitHubIssueGateDecision(False, "self_sender", repo, number, action, event_type)

    body = _body_text(payload)
    for marker in _string_set(cfg.get("self_markers")):
        if marker and marker in body:
            return GitHubIssueGateDecision(False, "self_marker", repo, number, action, event_type)

    if event_type == "issues":
        if action not in _RELEVANT_ISSUE_ACTIONS:
            return GitHubIssueGateDecision(False, "irrelevant_issue_action", repo, number, action, event_type)
    elif event_type == "issue_comment":
        if action not in _RELEVANT_COMMENT_ACTIONS:
            return GitHubIssueGateDecision(False, "irrelevant_comment_action", repo, number, action, event_type)
        if is_pr_subject and not bool(cfg.get("include_pull_requests", False)):
            return GitHubIssueGateDecision(False, "pull_request_comment_disabled", repo, number, action, event_type)
    elif event_type == "pull_request":
        if not bool(cfg.get("include_pull_requests", False)):
            return GitHubIssueGateDecision(False, "pull_request_disabled", repo, number, action, event_type)
        if action not in _RELEVANT_PR_ACTIONS:
            return GitHubIssueGateDecision(False, "irrelevant_pull_request_action", repo, number, action, event_type)
    elif event_type == "push":
        if not bool(cfg.get("include_pushes", False)):
            return GitHubIssueGateDecision(False, "push_disabled", repo, number, action, event_type)
    else:
        return GitHubIssueGateDecision(False, "unsupported_event", repo, number, action, event_type)

    if subject is None and event_type != "push":
        return GitHubIssueGateDecision(False, "missing_subject", repo, number, action, event_type)

    labels = _labels_from(subject or {}) | _labels_from(payload)
    labels_all = _string_set(cfg.get("labels_all"))
    if labels_all and not labels_all.issubset(labels):
        return GitHubIssueGateDecision(False, "missing_required_labels", repo, number, action, event_type)

    labels_any = _string_set(cfg.get("labels_any"))
    if labels_any and not (labels & labels_any):
        return GitHubIssueGateDecision(False, "missing_any_label", repo, number, action, event_type)

    if event_type == "push":
        llm_reason = "configured push event may change issue-worker code or status"
    elif event_type == "pull_request":
        llm_reason = f"pull request {action} matched configured GitHub issue gate"
    elif event_type == "issue_comment":
        llm_reason = f"issue comment {action} matched configured labels"
    else:
        llm_reason = f"issue {action} matched configured labels"
    return GitHubIssueGateDecision(True, "matched", repo, number, action, event_type, llm_reason)
