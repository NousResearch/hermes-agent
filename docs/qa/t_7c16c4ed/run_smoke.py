from __future__ import annotations

import json
import os

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

kb._INITIALIZED_PATHS.discard(str(kb.kanban_db_path(board="default").resolve()))
kb.init_db()
os.environ["HERMES_COMPLETION_GATE"] = "1"


def payload(artifact=None):
    result = {
        "task": {"status": "done", "summary": "smoke", "evidence": [{"command": "smoke", "exit_code": 0, "output": "completion gate smoke output"}]},
        "repo": {"dirty": False, "commit": "smoke-commit", "pushed": True, "pr": {"number": 1, "state": "merged"}, "ci": {"status": "success"}, "review": {"approved": True}},
        "preflight": {"requested_skills": [], "available_skills": [], "review_checks": [], "supported_flags": [], "dependency_copies": 0, "max_dependency_copies": 0},
        "scope": {"changed_paths": [], "allowed_roots": [], "secret_findings": []},
    }
    if artifact is not None:
        result["artifact"] = artifact
    return result


with kbc.connect() as conn:
    missing = kb.create_task(conn, title="smoke missing artifact", assignee="qa", completion_contract="deployed")
    kb.claim_task(conn, missing, claimer=kb._claimer_id())
    try:
        kb.complete_task(conn, missing, summary="should remain running", metadata={"verification": payload()})
    except kb.CompletionVerificationError as exc:
        task = kb.get_task(conn, missing)
        assert task is not None
        print(json.dumps({"missing_artifact": {"status": task.status, "ok": exc.report["ok"], "failure_codes": [x["code"] for x in exc.report["failures"]]}}))

    complete = kb.create_task(conn, title="smoke with artifact", assignee="qa", completion_contract="deployed")
    kb.claim_task(conn, complete, claimer=kb._claimer_id())
    kb.complete_task(conn, complete, summary="completed", metadata={"verification": payload({"type": "deployed", "url": "https://dev.example.test", "smoke_test": {"command": "curl /health", "exit_code": 0}})})
    task = kb.get_task(conn, complete)
    assert task is not None
    print(json.dumps({"required_artifact": {"status": task.status}}))
