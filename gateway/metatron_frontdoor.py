"""Metatron front-door actions for messaging transports.

This module keeps Telegram/Discord adapters thin: detect deterministic hub
commands, call Paperclip's Metatron endpoints, and format a short reply.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:3100/api/metatron"
DEFAULT_TIMEOUT_SECONDS = 10.0

MetatronAction = dict[str, str]


def _metatron_base_url() -> str:
    return os.getenv("PAPERCLIP_METATRON_URL", DEFAULT_BASE_URL).rstrip("/")


def detect_metatron_action(text: str) -> MetatronAction | None:
    """Return a deterministic Metatron hub action for supported commands."""
    raw = (text or "").strip()
    normalized = raw.lower()

    if (
        re.search(r"\b(bootstrap|initialize|init|seed|setup)\b", normalized)
        and re.search(r"\b(the|metatron|hub|command)\b", normalized)
    ):
        return {"type": "bootstrap_hub"}

    task_match = re.search(
        r"(?:metatron[,:\s]+)?(?:create|make|open|add)\s+"
        r"(?:a\s+)?(?:coding\s+)?(?:task|issue)\s+"
        r"(?:to|for|that)\s+(.+)",
        raw,
        re.IGNORECASE,
    )
    if task_match and task_match.group(1).strip():
        return {
            "type": "create_coding_task",
            "message": task_match.group(1).strip(),
        }

    if re.search(r"\b(blocked|blockers|stuck|status|what'?s\s+blocked)\b", normalized):
        return {
            "type": "ask_metatron",
            "message": raw,
        }

    return None


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return json.loads(body) if body else {}


async def perform_metatron_action(
    action: MetatronAction,
    *,
    base_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Execute a deterministic Paperclip Metatron action."""
    root = (base_url or _metatron_base_url()).rstrip("/")

    try:
        if action["type"] == "bootstrap_hub":
            return await asyncio.to_thread(
                _post_json,
                f"{root}/bootstrap-the",
                {},
                timeout,
            )
        if action["type"] == "create_coding_task":
            return await asyncio.to_thread(
                _post_json,
                f"{root}/coding-tasks",
                {
                    "message": action["message"],
                    "requestedBy": "telegram-metatron-frontdoor",
                },
                timeout,
            )
        if action["type"] == "ask_metatron":
            return await asyncio.to_thread(
                _post_json,
                f"{root}/orchestrate",
                {
                    "message": action["message"],
                    "requestedBy": "telegram-metatron-frontdoor",
                },
                timeout,
            )
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return {
            "blocked": True,
            "message": (
                "Paperclip Metatron hub is unreachable. "
                f"Start Paperclip on 127.0.0.1:3100, then retry. ({error})"
            ),
        }

    return {
        "blocked": True,
        "message": f"Unsupported Metatron action: {action.get('type', 'unknown')}",
    }


def format_metatron_telegram_response(result: dict[str, Any]) -> str:
    """Format Paperclip Metatron result for Telegram."""
    if result.get("blocked"):
        return str(result.get("message", "Metatron action is blocked."))

    response = result.get("response")
    if isinstance(response, str) and response.strip():
        return response.strip()

    issue = result.get("issue")
    project = result.get("project")
    assignee = result.get("assignee")
    route = result.get("route")
    if isinstance(issue, dict) and isinstance(project, dict) and isinstance(assignee, dict):
        identifier = issue.get("identifier") or issue.get("id") or "new issue"
        project_name = project.get("name") or "selected project"
        assignee_name = assignee.get("name") or "selected agent"
        reason = route.get("reason") if isinstance(route, dict) else None
        suffix = f"\nRoute: {reason}" if reason else ""
        return f"Created {identifier} in {project_name}, assigned to {assignee_name}.{suffix}"

    company = result.get("company")
    projects = result.get("projects")
    agents = result.get("agents")
    if isinstance(company, dict) and isinstance(projects, list) and isinstance(agents, list):
        created_projects = sum(1 for project in projects if isinstance(project, dict) and project.get("created"))
        created_agents = sum(1 for agent in agents if isinstance(agent, dict) and agent.get("created"))
        prefix = company.get("issuePrefix") or "THE"
        return (
            f"{prefix} hub is ready. "
            f"Created {created_projects} projects and {created_agents} agents."
        )

    return "Metatron action completed."


async def handle_metatron_frontdoor_text(text: str) -> str | None:
    """Return a Telegram-ready response if text is a Metatron hub command."""
    action = detect_metatron_action(text)
    if not action:
        return None
    result = await perform_metatron_action(action)
    return format_metatron_telegram_response(result)
