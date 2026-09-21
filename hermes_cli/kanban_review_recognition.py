"""Private encouragement from an independently reviewed task, scoped to its implementer."""
from hermes_cli import kanban_db as kb


def recipient_before_completion(conn, task_id, prior_status):
    source = kb._retry_status_for_run(conn, task_id) if prior_status == "running" else prior_status
    if source != "review":
        return None
    event = kb._latest_event(conn, task_id, "review_requested")
    payload = kb._json_dict(event["payload"]) if event else {}
    return kb._nonblank_str(payload.get("implementer"))


def record_approval(conn, task_id, recipient, run_id):
    if recipient:
        kb._append_event(conn, task_id, "private_recognition", {
            "basis": "independent_review_approved",
            "message": "Strong work: your work passed independent review. "
                       "Carry forward the same evidence-first, tightly scoped approach.",
            "recipient_profile": recipient,
        }, run_id=run_id)


def append_private_context(lines, conn, assignee):
    if not assignee:
        return
    rows = conn.execute("SELECT payload FROM task_events WHERE kind = 'private_recognition' "
                        "ORDER BY id DESC LIMIT 500").fetchall()
    for row in rows:
        payload = kb._json_dict(row["payload"])
        if payload.get("recipient_profile") != assignee:
            continue
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            lines.extend(["## Private recognition", kb._ctx_cap(message),
                          "_This is encouragement only. Re-verify the current task and keep all "
                          "safety, scope, and acceptance gates unchanged._", ""])
        break
