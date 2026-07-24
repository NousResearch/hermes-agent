"""Deployment preview monitors backed by Kanban cards.

This module is deliberately non-LLM. A monitor card stores a URL, expected
markers, and a deadline. ``tick`` checks the URL once and either completes the
card, blocks it after the deadline, or leaves it ready for the next scheduler
tick.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import urlparse

from hermes_cli import kanban_db as kb

TEMPLATE_ID = "deployment-monitor"
STEP_KEY = "watching"
_CONFIG_FENCE_RE = re.compile(
    r"```json deployment-monitor\s*(\{.*?\})\s*```",
    re.DOTALL,
)
_MAX_BODY_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class MonitorConfig:
    url: str
    markers: list[str]
    created_at: int
    deadline_at: int
    parent: Optional[str] = None


@dataclass(frozen=True)
class FetchResult:
    status_code: Optional[int]
    body: str = ""
    final_url: Optional[str] = None
    error: Optional[str] = None


@dataclass
class TickResult:
    checked: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "checked": self.checked,
            "completed": self.completed,
            "blocked": self.blocked,
            "pending": self.pending,
            "skipped": self.skipped,
            "errors": self.errors,
        }


def _validate_url(url: str) -> str:
    text = str(url or "").strip()
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an absolute http(s) URL")
    return text


def _clean_markers(markers: list[str]) -> list[str]:
    cleaned: list[str] = []
    for marker in markers:
        text = str(marker or "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
    if not cleaned:
        raise ValueError("at least one --marker is required")
    return cleaned


def _parse_duration_seconds(value: str) -> int:
    text = str(value or "").strip().lower()
    if not text:
        raise ValueError("duration is required")
    match = re.fullmatch(r"(\d+)([smhd]?)", text)
    if not match:
        raise ValueError("duration must look like 90s, 15m, 2h, 1d, or seconds")
    amount = int(match.group(1))
    unit = match.group(2) or "s"
    scale = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    seconds = amount * scale
    if seconds <= 0:
        raise ValueError("duration must be positive")
    return seconds


def _default_title(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    label = f"{parsed.netloc}{path}"
    if len(label) > 80:
        label = label[:77].rstrip() + "..."
    return f"Monitor deployment: {label}"


def _build_monitor_body(config: MonitorConfig) -> str:
    payload = {
        "kind": TEMPLATE_ID,
        "url": config.url,
        "markers": config.markers,
        "created_at": config.created_at,
        "deadline_at": config.deadline_at,
        "parent": config.parent,
    }
    return (
        "Hermes deployment monitor. This card is checked by "
        "`hermes deployment-monitor tick`.\n\n"
        "```json deployment-monitor\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)}\n"
        "```\n"
    )


def parse_monitor_config(body: str) -> MonitorConfig:
    match = _CONFIG_FENCE_RE.search(body or "")
    if not match:
        raise ValueError("deployment monitor config block not found")
    try:
        raw = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError(f"deployment monitor config is invalid JSON: {exc}") from exc
    if raw.get("kind") != TEMPLATE_ID:
        raise ValueError("deployment monitor config has wrong kind")
    url = _validate_url(str(raw.get("url") or ""))
    markers_raw = raw.get("markers")
    if not isinstance(markers_raw, list):
        raise ValueError("deployment monitor markers must be a list")
    markers = _clean_markers([str(m) for m in markers_raw])
    try:
        created_at = int(raw.get("created_at"))
        deadline_at = int(raw.get("deadline_at"))
    except (TypeError, ValueError) as exc:
        raise ValueError("deployment monitor timestamps must be integers") from exc
    if deadline_at <= created_at:
        raise ValueError("deployment monitor deadline must be after created_at")
    parent = raw.get("parent")
    if parent is not None:
        parent = str(parent).strip() or None
    return MonitorConfig(
        url=url,
        markers=markers,
        created_at=created_at,
        deadline_at=deadline_at,
        parent=parent,
    )


def create_monitor(
    conn,
    *,
    url: str,
    markers: list[str],
    parent: Optional[str] = None,
    deadline_seconds: int = 15 * 60,
    now: Optional[int] = None,
    title: Optional[str] = None,
    created_by: str = "deployment-monitor",
    idempotency_key: Optional[str] = None,
) -> str:
    url = _validate_url(url)
    markers = _clean_markers(markers)
    if deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be positive")
    created_at = int(time.time() if now is None else now)
    config = MonitorConfig(
        url=url,
        markers=markers,
        created_at=created_at,
        deadline_at=created_at + int(deadline_seconds),
        parent=parent,
    )
    task_id = kb.create_task(
        conn,
        title=title or _default_title(url),
        body=_build_monitor_body(config),
        assignee=None,
        created_by=created_by,
        parents=(parent,) if parent else (),
        priority=1,
        idempotency_key=idempotency_key,
    )
    with kb.write_txn(conn):
        conn.execute(
            """
            UPDATE tasks
               SET workflow_template_id = ?,
                   current_step_key = ?
             WHERE id = ?
            """,
            (TEMPLATE_ID, STEP_KEY, task_id),
        )
    return task_id


def _read_text_response(resp) -> str:
    raw = resp.read(_MAX_BODY_BYTES)
    charset = None
    try:
        charset = resp.headers.get_content_charset()
    except Exception:
        charset = None
    return raw.decode(charset or "utf-8", errors="replace")


def fetch_url(url: str, *, timeout: float = 20.0) -> FetchResult:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "HermesDeploymentMonitor/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return FetchResult(
                status_code=getattr(resp, "status", None),
                body=_read_text_response(resp),
                final_url=getattr(resp, "url", url),
            )
    except urllib.error.HTTPError as exc:
        return FetchResult(
            status_code=exc.code,
            body=_read_text_response(exc),
            final_url=getattr(exc, "url", url),
            error=str(exc),
        )
    except urllib.error.URLError as exc:
        return FetchResult(
            status_code=None,
            body="",
            final_url=url,
            error=str(exc.reason),
        )
    except Exception as exc:  # pragma: no cover - network edge
        return FetchResult(
            status_code=None,
            body="",
            final_url=url,
            error=str(exc),
        )


def _missing_markers(body: str, markers: list[str]) -> list[str]:
    return [marker for marker in markers if marker not in body]


def _is_ready(fetch: FetchResult, config: MonitorConfig) -> bool:
    if fetch.status_code is None or not (200 <= int(fetch.status_code) < 400):
        return False
    return not _missing_markers(fetch.body or "", config.markers)


def _status_label(fetch: FetchResult) -> str:
    if fetch.status_code is None:
        return "status=unavailable"
    return f"status={fetch.status_code}"


def _block_reason(config: MonitorConfig, fetch: FetchResult) -> str:
    missing = _missing_markers(fetch.body or "", config.markers)
    details = [
        "deployment monitor timed out",
        f"url={config.url}",
        _status_label(fetch),
    ]
    if missing:
        details.append("missing_markers=" + json.dumps(missing, ensure_ascii=False))
    if fetch.error:
        details.append(f"error={fetch.error}")
    return "; ".join(details)


def tick(
    conn,
    *,
    now: Optional[int] = None,
    fetcher: Callable[[str], FetchResult] = fetch_url,
    limit: Optional[int] = None,
) -> TickResult:
    result = TickResult()
    now_ts = int(time.time() if now is None else now)
    tasks = kb.list_tasks(
        conn,
        workflow_template_id=TEMPLATE_ID,
        include_archived=False,
        order_by="created",
    )
    if limit is not None:
        tasks = tasks[: max(0, int(limit))]
    for task in tasks:
        if task.status not in {"ready", "running"}:
            result.skipped.append(task.id)
            continue
        result.checked.append(task.id)
        try:
            config = parse_monitor_config(task.body or "")
        except ValueError as exc:
            reason = f"deployment monitor invalid config: {exc}"
            if kb.block_task(conn, task.id, reason=reason, kind="capability"):
                result.blocked.append(task.id)
            else:
                result.errors[task.id] = reason
            continue
        fetch = fetcher(config.url)
        if _is_ready(fetch, config):
            summary = (
                "Deployment monitor passed: "
                f"{fetch.final_url or config.url} ({_status_label(fetch)})"
            )
            metadata = {
                "url": config.url,
                "final_url": fetch.final_url or config.url,
                "status_code": fetch.status_code,
                "markers": config.markers,
                "checked_at": now_ts,
            }
            if kb.complete_task(conn, task.id, summary=summary, metadata=metadata):
                result.completed.append(task.id)
            else:
                result.errors[task.id] = "failed to complete monitor task"
            continue
        if now_ts >= config.deadline_at:
            reason = _block_reason(config, fetch)
            if kb.block_task(conn, task.id, reason=reason, kind="transient"):
                result.blocked.append(task.id)
            else:
                result.errors[task.id] = reason
            continue
        result.pending.append(task.id)
    return result


def build_parser(parent_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = parent_subparsers.add_parser(
        "deployment-monitor",
        help="Create and tick URL/marker deployment monitors",
        description=(
            "Create lightweight Kanban-backed monitors for preview deployments. "
            "The tick command performs one URL check; schedule it from cron or "
            "call it from a gateway loop."
        ),
    )
    parser.set_defaults(func=deployment_monitor_command)
    parser.add_argument("--board", default=None, help="Kanban board slug")
    sub = parser.add_subparsers(dest="deployment_monitor_action")
    sub.required = True

    p_create = sub.add_parser("create", help="Create a deployment monitor card")
    p_create.add_argument("--url", required=True, help="Preview URL to check")
    p_create.add_argument(
        "--marker",
        action="append",
        required=True,
        dest="markers",
        help="Required text marker in the response body (repeatable)",
    )
    p_create.add_argument("--parent", default=None, help="Parent Kanban task id")
    p_create.add_argument(
        "--deadline",
        default="15m",
        help="Deadline from now before blocking (default: 15m)",
    )
    p_create.add_argument("--title", default=None, help="Monitor card title")
    p_create.add_argument("--created-by", default="deployment-monitor")
    p_create.add_argument("--idempotency-key", default=None)
    p_create.add_argument("--json", action="store_true")

    p_tick = sub.add_parser("tick", help="Check ready deployment monitor cards once")
    p_tick.add_argument("--limit", type=int, default=None)
    p_tick.add_argument("--json", action="store_true")
    return parser


def deployment_monitor_command(args: argparse.Namespace) -> int:
    action = getattr(args, "deployment_monitor_action", None)
    if action == "create":
        try:
            deadline_seconds = _parse_duration_seconds(args.deadline)
            with kb.connect_closing(board=getattr(args, "board", None)) as conn:
                task_id = create_monitor(
                    conn,
                    url=args.url,
                    markers=args.markers,
                    parent=args.parent,
                    deadline_seconds=deadline_seconds,
                    title=args.title,
                    created_by=args.created_by,
                    idempotency_key=args.idempotency_key,
                )
                task = kb.get_task(conn, task_id)
        except ValueError as exc:
            print(f"deployment-monitor create: {exc}", file=sys.stderr)
            return 2
        if getattr(args, "json", False):
            print(json.dumps({"id": task_id, "status": task.status}, indent=2))
        else:
            print(f"Created deployment monitor {task_id} ({task.status})")
        return 0
    if action == "tick":
        with kb.connect_closing(board=getattr(args, "board", None)) as conn:
            result = tick(conn, limit=getattr(args, "limit", None))
        if getattr(args, "json", False):
            print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
        else:
            print(
                "deployment-monitor tick: "
                f"checked={len(result.checked)} "
                f"completed={len(result.completed)} "
                f"blocked={len(result.blocked)} "
                f"pending={len(result.pending)} "
                f"skipped={len(result.skipped)}"
            )
        return 0
    build_parser(argparse.ArgumentParser(prog="deployment-monitor").add_subparsers())
    return 2
