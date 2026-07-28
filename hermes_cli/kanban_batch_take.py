"""Plan and promote a selected Kanban batch without overlapping workers.

The planner asks a low-cost auxiliary model only for ordering constraints
between the chosen cards.  It never rewrites existing dependencies and uses
``link_tasks`` for every proposed edge, so the database remains the source of
truth for DAG validation.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from hermes_cli import kanban_db as kb

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You plan a selected batch of Kanban tasks. Serialize only
CLEAR problems, not every touch. Add an ordering edge when two tasks would
heavily rewrite the same code, configuration, or versioned knowledge base in
different directions — that is the expensive merge-conflict case to prevent.
Light overlap (each task adds a little in different places of the same file)
merges trivially: keep those parallel. Judge by the volume and nature of the
edits in the shared area: additive changes are peaceful, rewrites conflict.
Do not invent dependencies for merely related tasks; preserve parallelism
whenever possible.

Return JSON only:
{"edges":[{"before":"task-id","after":"task-id","reason":"short reason"}]}

Rules: ids must come from the supplied tasks; before and after differ; use the
fewest edges necessary; an empty edges list is valid."""
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class BatchTakeOutcome:
    ok: bool
    reason: str = ""
    edges: list[dict[str, str]] | None = None
    promoted: list[str] | None = None
    waiting: list[str] | None = None
    skipped: list[dict[str, str]] | None = None


def _json(raw: str) -> Optional[dict[str, Any]]:
    text = _FENCE_RE.sub("", (raw or "").strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _prompt(tasks: list[Any]) -> str:
    rows = []
    for task in tasks:
        rows.append({
            "id": task.id,
            "title": (task.title or "")[:400],
            "body": (task.body or "")[:3000],
            "status": task.status,
        })
    return "Selected tasks:\n" + json.dumps(rows, ensure_ascii=False)


def _incomplete_parents(conn: Any, task_id: str) -> bool:
    row = conn.execute(
        """SELECT 1 FROM task_links l JOIN tasks p ON p.id=l.parent_id
           WHERE l.child_id=? AND p.status != 'done' LIMIT 1""",
        (task_id,),
    ).fetchone()
    return row is not None


def plan_and_take(task_ids: list[str], *, timeout: int = 90) -> BatchTakeOutcome:
    """Add safe ordering edges, then promote immediately runnable tasks.

    The operation is deliberately conservative: malformed/failed model output
    changes nothing, and a graph edge rejected by the DB is reported rather
    than bypassing cycle checks.
    """
    ids = list(dict.fromkeys(str(task_id) for task_id in task_ids if task_id))
    if not ids:
        return BatchTakeOutcome(False, "ids is required")
    with kb.connect_closing() as conn:
        tasks = []
        skipped: list[dict[str, str]] = []
        for task_id in ids:
            task = kb.get_task(conn, task_id)
            if task is None:
                skipped.append({"id": task_id, "reason": "not found"})
            elif task.status not in {"todo", "ready"}:
                skipped.append({"id": task_id, "reason": f"status {task.status!r} cannot be batch-taken"})
            else:
                tasks.append(task)
        if not tasks:
            return BatchTakeOutcome(False, "no eligible tasks", skipped=skipped)

        try:
            from agent.auxiliary_client import call_llm
            response = call_llm(
                task="kanban_batch_planner",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _prompt(tasks)},
                ],
                temperature=0.1,
                max_tokens=1500,
                timeout=timeout,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            logger.info("batch-take planner failed: %s", exc)
            return BatchTakeOutcome(False, f"planner unavailable: {type(exc).__name__}", skipped=skipped)
        parsed = _json(raw)
        if (
            parsed is None
            or "edges" not in parsed
            or not isinstance(parsed.get("edges"), list)
        ):
            return BatchTakeOutcome(False, "planner returned malformed JSON", skipped=skipped)

        eligible = {task.id for task in tasks}
        edges: list[dict[str, str]] = []
        for item in parsed.get("edges", []):
            if not isinstance(item, dict):
                continue
            before, after = item.get("before"), item.get("after")
            if not isinstance(before, str) or not isinstance(after, str):
                continue
            if before == after or before not in eligible or after not in eligible:
                continue
            edge = {"before": before, "after": after, "reason": str(item.get("reason") or "overlapping work")[:300]}
            if edge not in edges:
                edges.append(edge)

        # Apply planner edges one-by-one through the canonical DAG guard.
        applied: list[dict[str, str]] = []
        for edge in edges:
            try:
                kb.link_tasks(conn, edge["before"], edge["after"])
                applied.append(edge)
            except ValueError as exc:
                skipped.append({"id": edge["after"], "reason": f"dependency rejected: {exc}"})

        promoted: list[str] = []
        waiting: list[str] = []
        # Direct status updates mirror the dashboard bulk endpoint. A task
        # with unfinished parents must remain todo; dispatcher's normal DAG
        # promotion will move it to ready after those parents finish.
        for task in tasks:
            if _incomplete_parents(conn, task.id):
                if task.status != "todo":
                    with kb.write_txn(conn):
                        conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (task.id,))
                waiting.append(task.id)
            else:
                if task.status != "ready":
                    ok, reason = kb.promote_task(
                        conn,
                        task.id,
                        actor="batch-planner",
                        reason="batch take: no unfinished dependencies",
                    )
                    if not ok:
                        skipped.append({"id": task.id, "reason": reason or "promotion refused"})
                        continue
                promoted.append(task.id)
        return BatchTakeOutcome(True, edges=applied, promoted=promoted, waiting=waiting, skipped=skipped)
