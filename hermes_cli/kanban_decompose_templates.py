"""Kanban decompose templates for an implementer/reviewer split.

Templates are configured under ``kanban.decompose_templates``.  They match a
triage title, require two installed and distinct profiles, and harden the LLM
plan so implementation and independent review cannot collapse onto one owner.

Example::

    decompose_templates:
      - name: sl_analysis_editorial
        match_title_requires_any: [suomen liittokunta, sl, pohjolan ihme]
        match_title_contains: [analysis, editorial, rothbard]
        enforce_review_child: true
        implementer: suomen-liittokunta
        reviewer: academic
        require_mention: true
        credential_isolation: true

``match_title_requires_any`` is the project/path anchor.  At least one anchor
and one ``match_title_contains`` signal must match when both are configured.
Single-word matching is boundary-aware, so the ``sl`` anchor does not match a
word such as ``slack``.
"""

from __future__ import annotations

import re
from typing import Optional


_CREDENTIAL_POLICY = (
    "Credential isolation: never read, reuse, request, or pass another "
    "profile's secrets, tokens, auth files, or credentials. Each profile owns "
    "its own credentials."
)


def load_templates(cfg: dict) -> list[dict]:
    """Return the ``kanban.decompose_templates`` list, or [] if malformed."""
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    templates = kanban_cfg.get("decompose_templates")
    if not isinstance(templates, list):
        return []
    return [template for template in templates if isinstance(template, dict)]


def _normalize_match_text(value: object) -> str:
    text = str(value or "").casefold().replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def _matches_token(title: str, token: object) -> bool:
    needle = _normalize_match_text(token)
    if not needle:
        return False
    if " " in needle:
        return needle in title
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", title) is not None


def select_template(title: str, templates: list[dict]) -> Optional[dict]:
    """Return the most-specific title template.

    If ``match_title_requires_any`` is present, at least one of those anchors
    must match.  If ``match_title_contains`` is present, at least one of those
    signals must also match.  Legacy templates without anchors retain their
    old any-token behavior.
    """
    normalized_title = _normalize_match_text(title)
    if not normalized_title:
        return None

    best: Optional[dict] = None
    best_score = -1
    for template in templates:
        anchors = template.get("match_title_requires_any") or []
        signals = template.get("match_title_contains") or []
        if not isinstance(anchors, list) or not isinstance(signals, list):
            continue
        if not anchors and not signals:
            continue

        anchor_matches = sum(
            1 for token in anchors if _matches_token(normalized_title, token)
        )
        signal_matches = sum(
            1 for token in signals if _matches_token(normalized_title, token)
        )
        if anchors and anchor_matches == 0:
            continue
        if signals and signal_matches == 0:
            continue

        # Anchors dominate ties because they encode the project/path boundary.
        score = anchor_matches * 100 + signal_matches
        if score > best_score:
            best_score = score
            best = template
    return best


def _append_credential_policy(body: object) -> str:
    text = body.strip() if isinstance(body, str) else ""
    if _CREDENTIAL_POLICY in text:
        return text
    if text:
        return f"{text}\n\n{_CREDENTIAL_POLICY}"
    return _CREDENTIAL_POLICY


def _review_body_for(template: dict, task, original_body: object = "") -> str:
    """Build a reviewer body containing the hard policy boundaries."""
    lines = [
        "**Independent review required.** You are the review counterpart to the "
        "implementation work on this task. Do NOT implement or modify the "
        "artifact. Critique it, verify it against the acceptance criteria, and "
        "gate completion.",
        "require_mention is enforced: this assignment is the explicit request "
        "for this specific review; do not act on other council/mention-gated "
        "surfaces without a separate explicit request.",
        _CREDENTIAL_POLICY,
    ]

    supplied = original_body.strip() if isinstance(original_body, str) else ""
    if supplied:
        lines.append("Review scope from the decomposer:\n" + supplied)

    root_body = getattr(task, "body", "")
    root_body = root_body.strip() if isinstance(root_body, str) else ""
    if root_body:
        lines.append("Original task acceptance context:\n" + _truncate(root_body, 4000))
    return "\n\n".join(lines)


def _is_reviewer_task(entry: dict) -> bool:
    title = str(entry.get("title", "")).strip().casefold()
    return title.startswith("review")


def _apply_to_plan(
    task,
    parsed: dict,
    template: dict,
    valid_names: set[str],
) -> tuple[dict, bool]:
    """Force one installed implementer and an independent review gate.

    Policy flags fail closed: a template is ignored unless
    ``require_mention`` and ``credential_isolation`` are literally true.  A
    fanout plan keeps supporting non-review children and routes every such
    child to the configured implementer.  Every reviewer is routed
    to the configured reviewer, receives the policy body, and depends on all
    non-review children.
    """
    implementer = template.get("implementer")
    reviewer = template.get("reviewer")
    policy_enabled = (
        template.get("require_mention") is True
        and template.get("credential_isolation") is True
    )
    if (
        template.get("enforce_review_child") is not True
        or not policy_enabled
        or not isinstance(implementer, str)
        or not isinstance(reviewer, str)
    ):
        return parsed, False

    implementer = implementer.strip()
    reviewer = reviewer.strip()
    if (
        not implementer
        or not reviewer
        or implementer == reviewer
        or implementer not in valid_names
        or reviewer not in valid_names
    ):
        return parsed, False

    if not bool(parsed.get("fanout")):
        title = parsed.get("title") or getattr(task, "title", "") or "Implement work"
        title = title.strip() if isinstance(title, str) else "Implement work"
        implementation = {
            "title": title,
            "body": _append_credential_policy(parsed.get("body")),
            "assignee": implementer,
            "parents": [],
        }
        review = {
            "title": "Review: " + _truncate(title, 60),
            "body": _review_body_for(template, task),
            "assignee": reviewer,
            "parents": [0],
        }
        return {
            "fanout": True,
            "rationale": parsed.get("rationale")
            or "template-enforced separate implementer and reviewer",
            "tasks": [implementation, review],
        }, True

    raw_tasks = parsed.get("tasks")
    tasks = [dict(entry) if isinstance(entry, dict) else entry for entry in (
        raw_tasks if isinstance(raw_tasks, list) else []
    )]

    reviewer_indices: list[int] = []
    implementation_indices: list[int] = []
    for index, entry in enumerate(tasks):
        if not isinstance(entry, dict):
            continue
        if _is_reviewer_task(entry):
            entry["assignee"] = reviewer
            entry["body"] = _review_body_for(template, task, entry.get("body"))
            reviewer_indices.append(index)
        else:
            entry["body"] = _append_credential_policy(entry.get("body"))
            implementation_indices.append(index)

    if not implementation_indices:
        implementation_index = len(tasks)
        root_title = getattr(task, "title", "") or "work"
        root_body = getattr(task, "body", "") or ""
        tasks.append({
            "title": "Implement: " + _truncate(str(root_title), 60),
            "body": _append_credential_policy(root_body),
            "assignee": implementer,
            "parents": [],
        })
        implementation_indices.append(implementation_index)
    else:
        # All non-review children are part of the template's implementation
        # lane. Routing every one prevents an artifact-producing child from
        # retaining a generic or otherwise unintended owner.
        for index in implementation_indices:
            tasks[index]["assignee"] = implementer

    # Review must remain a terminal gate. If the LLM made implementation
    # depend on a reviewer, replacing only the reviewer's parents below would
    # leave an implementer <-> reviewer cycle. Preserve unrelated work edges
    # while removing every dependency on a review child.
    reviewer_index_set = set(reviewer_indices)
    if reviewer_index_set:
        for index in implementation_indices:
            parents = tasks[index].get("parents")
            if isinstance(parents, list):
                tasks[index]["parents"] = [
                    parent for parent in parents
                    if parent not in reviewer_index_set
                ]

    if reviewer_indices:
        for index in reviewer_indices:
            # Review is a terminal gate. Replacing LLM-provided review edges
            # avoids self/cross-review cycles while covering every work child.
            tasks[index]["parents"] = list(implementation_indices)
    else:
        tasks.append({
            "title": "Review: " + _truncate(getattr(task, "title", "") or "work", 60),
            "body": _review_body_for(template, task),
            "assignee": reviewer,
            "parents": list(implementation_indices),
        })

    result = dict(parsed)
    result["fanout"] = True
    result["tasks"] = tasks
    return result, True


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
