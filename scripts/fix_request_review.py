#!/usr/bin/env python3
"""Fix request_review to include artifact staging logic from upstream."""
import re

with open('/Users/mikedemott/hermes-fork-work/fork-repo/hermes_cli/kanban_db.py', 'r') as f:
    content = f.read()

# Find the request_review function and add artifact staging logic
old_block = '''    initial_state = (task.status, task.current_run_id)
    summary = redact_review_value(summary)
    metadata = redact_review_value(metadata)
    try:
        with write_txn(conn):'''

new_block = '''    initial_state = (task.status, task.current_run_id)
    summary = redact_review_value(summary)
    metadata = redact_review_value(metadata)
    # Extract artifact paths for staging before we open the txn (staging is inside the txn
    # so files are cleaned up if the txn fails, but the list is built here for clarity).
    artifacts_to_stage = []
    if isinstance(metadata, dict):
        for p in (metadata.get("artifacts") or []):
            if p:
                artifacts_to_stage.append(str(p))

    staged_artifacts: list[Path] = []
    try:
        with write_txn(conn):'''

content = content.replace(old_block, new_block)

# Also need to add the artifact staging logic inside the write_txn block
# Find the run_id = _end_or_synthesize_run call and add artifact staging before it
old_artifact_block = '''        if cur.rowcount != 1:
            return _ret(
                False, "task is not in running/ready (or expected_run_id did not match the current run)",
            )
        run_id = _end_or_synthesize_run(
            conn, task_id, outcome="review_requested", status="review",
            summary=summary, metadata=metadata, synthesize=bool(summary or metadata),
        )
        _append_event(
            conn,
            task_id,
            "review_requested",
            {
                "summary": _first_line(summary, 400) or None,
                "implementer": implementer,
                "reviewer": reviewer,
            },
            run_id=run_id,
        )'''

new_artifact_block = '''        if cur.rowcount != 1:
            return _ret(
                False, "task is not in running/ready (or expected_run_id did not match the current run)",
            )
        # Stage artifact files to the task's attachment directory so they survive
        # workspace cleanup after the reviewer's completion (#review-artifacts).
        if artifacts_to_stage:
            import shutil as _shutil
            dest_dir = task_attachments_dir(task_id)
            dest_dir.mkdir(parents=True, exist_ok=True)
            for path_str in artifacts_to_stage:
                src = Path(path_str)
                if src.is_file():
                    safe_name = _safe_attachment_name(src.name)
                    dest = _collision_free_path(dest_dir, safe_name)
                    _shutil.copy2(src, dest)
                    staged_artifacts.append(dest)
        run_id = _end_or_synthesize_run(
            conn, task_id, outcome="review_requested", status="review",
            summary=summary, metadata=metadata, synthesize=bool(summary or metadata),
        )
        # Register staged files as attachments (raw INSERT — already inside a txn).
        _now_ts = int(time.time())
        for staged_path in staged_artifacts:
            _size = staged_path.stat().st_size if staged_path.exists() else 0
            conn.execute(
                "INSERT INTO task_attachments "
                "(task_id, filename, stored_path, content_type, size, created_at) "
                "VALUES (?, ?, ?, NULL, ?, ?)",
                (task_id, staged_path.name, str(staged_path.resolve()), _size, _now_ts),
            )
        staged_paths = [str(p) for p in staged_artifacts]
        _append_event(
            conn,
            task_id,
            "review_requested",
            {
                "summary": _first_line(summary, 400) or None,
                "implementer": implementer,
                "reviewer": reviewer,
                **({"artifacts": staged_paths} if staged_paths else {}),
            },
            run_id=run_id,
        )
    except Exception:
        for f in staged_artifacts:
            with contextlib.suppress(OSError):
                f.unlink(missing_ok=True)
        raise'''

content = content.replace(old_artifact_block, new_artifact_block)

with open('/Users/mikedemott/hermes-fork-work/fork-repo/hermes_cli/kanban_db.py', 'w') as f:
    f.write(content)

print("Fixed!")