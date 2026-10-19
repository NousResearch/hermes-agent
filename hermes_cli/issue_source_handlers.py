"""Select explicit per-Issue policies on the existing Gateway control socket."""
from functools import partial
from pathlib import Path

from github_bridge.config import load_config
from hermes_cli.issue22_gateway_verb import build_issue22_admit_handler, build_issue22_poll_handler
from hermes_cli.issue_ingress import issue22_task_is_active, publish_pending_task_result, validate_ingress_config


def build_source_handlers(connect_owner_db, config_path, *, enabled=False):
    if enabled is not True:
        disabled = lambda _params: {"verdict": "REJECT", "reason": "INGRESS_DISABLED"}
        return disabled, disabled
    cfg = load_config(config_path)
    sources = cfg.issue_ingress_sources or {str(cfg.issue_ingress_issue_number): str(config_path)}
    admits, polls = {}, []

    def owner_call(operation, **kwargs):
        conn = connect_owner_db()
        try:
            return operation(conn, **kwargs)
        finally:
            conn.close()

    def processed(conn, comment_id):
        for table in ("issue22_control_receipts", "issue22_pending_intakes"):
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
                continue
            row = conn.execute(f"SELECT * FROM {table} WHERE comment_id=?", (comment_id,)).fetchone()
            if row and row["result_comment_id"] is not None:
                return table == "issue22_control_receipts" or ("dispatch_state" in row.keys() and row["dispatch_state"] != "awaiting_binding")
        return False

    for key, path in sorted(sources.items()):
        number, path = int(key), Path(path)
        if validate_ingress_config(path).issue_number != number:
            raise ValueError("source selector differs from authenticated issue pin")
        admit = build_issue22_admit_handler(connect_owner_db=connect_owner_db,
            config_path=path, enabled=enabled, issue_number=number)
        admits[number] = admit
        polls.append(build_issue22_poll_handler(config_path=path, enabled=enabled,
            issue_number=number, admit_handler=admit,
            publish_task_result=partial(owner_call, publish_pending_task_result, config_path=path),
            task_active=partial(owner_call, issue22_task_is_active),
            is_processed=lambda comment_id: owner_call(processed, comment_id=comment_id)))

    def admit(params):
        if enabled is not True:
            return {"verdict": "REJECT", "reason": "INGRESS_DISABLED"}
        if (type(params) is not dict or type(params.get("issue_number")) is not int
                or params["issue_number"] not in admits):
            return {"verdict": "REJECT", "reason": "SOURCE_NOT_ALLOWED"}
        return admits[params["issue_number"]](params)

    def poll(params):
        if params not in ({}, None):
            return {"verdict": "REJECT", "reason": "REQUEST_SCHEMA"}
        results = [handler(params) for handler in polls]
        accepted = next((row for row in results if row.get("verdict") == "ACK"), None)
        if accepted is not None:
            return accepted
        return next((row for row in results if row.get("verdict") not in {"NOOP", "BUSY"}), results[-1])

    return admit, poll
