"""Opt-in, fail-open bridge from Hermes Kanban evidence to narrative vault reports."""

from __future__ import annotations

import datetime as dt
import json
import logging
import subprocess
from typing import Any, Iterable

logger = logging.getLogger("gateway.run")
ELIGIBLE_EVENTS = frozenset({
    "completed",
    "blocked",
    "gave_up",
    "crashed",
    "timed_out",
    "review_requested",
    "block_loop_detected",
})


def _settings(config: object) -> dict[str, Any] | None:
    if not isinstance(config, dict):
        return None
    value = (config.get("kanban") or {}).get("vault_reports")
    if not isinstance(value, dict) or value.get("enabled") is not True:
        return None
    command = value.get("command")
    projects = value.get("projects")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(x, str) and x for x in command)
    ):
        return None
    if not isinstance(value.get("vault_path"), str) or not isinstance(projects, dict):
        return None
    return value


def _run(settings: dict[str, Any], tail: list[str], *, stdin: str | None = None):
    timeout = max(1, min(int(settings.get("timeout_seconds", 5)), 15))
    return subprocess.run(
        [*settings["command"], *tail, "--vault", settings["vault_path"]],
        input=stdin,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        shell=False,
    )


def write_terminal_report(
    config: object,
    *,
    board: str,
    root_task_id: str,
    event_kind: str,
    title: str,
    outcome: str,
    branch: str | None = None,
    commit: str | None = None,
) -> bool:
    settings = _settings(config)
    project = settings.get("projects", {}).get(board) if settings else None
    if not settings or not project or event_kind not in ELIGIBLE_EVENTS:
        return False
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    payload = {
        "schema": "exampleapp_agent_report_v1",
        "agent": "hermes",
        "project": project,
        "workstream_id": f"{board}--{root_task_id}",
        "board": board,
        "root_task_id": root_task_id,
        "verified_at": now.isoformat().replace("+00:00", "Z"),
        "review_date": (now.date() + dt.timedelta(days=3)).isoformat(),
        "title": title[:200],
        "outcome": outcome[:2000],
        "changes": [f"Kanban terminal event: {event_kind}."],
        "verification": [
            "Recorded from the source-authorized Hermes Kanban event stream."
        ],
        "blockers": [outcome[:500]]
        if event_kind
        in {"blocked", "gave_up", "crashed", "timed_out", "block_loop_detected"}
        else [],
        "next_action": "Verify current Git, Kanban, and governed artifacts before continuing.",
        "branch": branch,
        "commit": commit,
        "source_paths": [],
        "artifact_paths": [],
    }
    try:
        result = _run(settings, ["report", "write"], stdin=json.dumps(payload))
        return result.returncode == 0
    except Exception as exc:
        logger.debug("vault report write unavailable: %s", exc)
        return False


def append_vault_context(
    config: object, response: str, *, board: str, root_task_ids: Iterable[str]
) -> str:
    settings = _settings(config)
    project = settings.get("projects", {}).get(board) if settings else None
    if not settings or not project:
        return response
    remaining = max(0, min(int(settings.get("max_context_chars", 600)), 1200))
    snippets: list[str] = []
    for root_id in list(dict.fromkeys(root_task_ids))[:3]:
        try:
            result = _run(
                settings,
                [
                    "report",
                    "read",
                    "--project",
                    project,
                    "--workstream-id",
                    f"{board}--{root_id}",
                    "--board",
                    board,
                    "--root-task-id",
                    root_id,
                    "--max-chars",
                    str(remaining),
                    "--stale-after-hours",
                    str(max(1, int(settings.get("stale_after_hours", 72)))),
                ],
            )
            if result.returncode != 0 or len(result.stdout) > 16_384:
                continue
            payload = json.loads(result.stdout)
            context = payload.get("context") if isinstance(payload, dict) else None
            if isinstance(context, str) and context.strip():
                snippets.append(context.strip())
                remaining -= len(context)
                if remaining <= 0:
                    break
        except Exception as exc:
            logger.debug("vault context unavailable: %s", exc)
    if not snippets:
        return response
    return (response + "\n\n" + "\n".join(snippets))[:2200]


def load_live_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        value = load_config()
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}
