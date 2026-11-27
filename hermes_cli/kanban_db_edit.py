"""Text/priority/result edits, separate from operator execution-control edits."""
from __future__ import annotations

import json


def edit_task_fields(conn, task_id, *, title=None, body=None, priority=None,
                     result=None, summary=None, metadata=None, board=None) -> bool:
    from hermes_cli import kanban_db as kb

    changed_fields = [
        field for field, value in (("title", title), ("body", body), ("priority", priority))
        if value is not None
    ]
    with kb.write_txn(conn):
        status = kb._task_status(conn, task_id)
        if status is None or (result is not None and status != "done"):
            return False
        assignments = []
        params = []
        for field, value in (("title", title), ("body", body), ("priority", priority)):
            if value is not None:
                assignments.append(f"{field} = ?")
                params.append(value)
        if result is not None:
            assignments.append("result = ?")
            params.append(result)
            changed_fields.append("result")
        if not assignments:
            return False
        conn.execute(
            f"UPDATE tasks SET {', '.join(assignments)} WHERE id = ?",
            (*params, task_id),
        )
        if priority is not None:
            kb._append_event(conn, task_id, "reprioritized", {"priority": priority})
        if result is None:
            non_priority_fields = [field for field in changed_fields if field != "priority"]
            if non_priority_fields:
                kb._append_event(conn, task_id, "edited", {"fields": non_priority_fields})
        else:
            handoff_summary = summary if summary is not None else result
            changed_fields.append("summary")
            if metadata is not None:
                changed_fields.append("metadata")
            run = conn.execute(
                """
                SELECT id FROM task_runs
                 WHERE task_id = ?
                   AND outcome = 'completed'
                 ORDER BY COALESCE(ended_at, started_at, 0) DESC, id DESC
                 LIMIT 1
                """,
                (task_id,),
            ).fetchone()
            if run is None:
                run_id = kb._synthesize_ended_run(
                    conn, task_id, outcome="completed", summary=handoff_summary, metadata=metadata,
                )
            else:
                run_id = int(run["id"])
                conn.execute("UPDATE task_runs SET summary = ? WHERE id = ?", (handoff_summary, run_id))
                if metadata is not None:
                    conn.execute(
                        "UPDATE task_runs SET metadata = ? WHERE id = ?",
                        (json.dumps(metadata, ensure_ascii=False), run_id),
                    )
            kb._append_event(
                conn, task_id, "edited",
                {
                    "fields": ["result", "summary"] + (["metadata"] if metadata is not None else []),
                    "result_len": len(result) if result else 0,
                    "summary": kb._first_line(handoff_summary, 400) or None,
                },
                run_id=run_id,
            )
    kb.notify_task_updated(conn, task_id, changed_fields, board=board)
    return True
