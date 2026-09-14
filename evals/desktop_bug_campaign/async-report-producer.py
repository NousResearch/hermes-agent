"""Real offline completion producers -> durable SQLite rows; never run a model."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import time

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))
output = Path(sys.argv[1])
with tempfile.TemporaryDirectory(prefix="async-report-producer-") as temporary:
    home = Path(temporary)
    for key in list(os.environ):
        if key.startswith("HERMES_") or key.endswith(("_API_KEY", "_TOKEN")):
            os.environ.pop(key, None)
    os.environ.update(HOME=str(home), HERMES_HOME=str(home / ".hermes"))
    from cron.jobs import create_job, save_job_output
    from gateway.wake import persist_delegation_delivery
    from hermes_state import SessionDB
    from tools.cronjob_tools import _manual_run_completion
    from tools.process_registry_notifications import format_process_notification

    report = "# Generated inspection\n\n**Conclusion:** attention needed.\n\n| Strategy | Status |\n|---|---|\n| Canary | running |\n\nFirst line  \nSecond line\n\n```python\nvalue = 1  \nvalue = 2 \n```"
    job = create_job(prompt="Offline fixture; never run", schedule="0 0 * * *", name="Markdown probe", deliver="local")
    save_job_output(job["id"], report)
    result = _manual_run_completion({"success": True}, job["id"], job["name"], "local", time.time())
    common = dict(type="async_delegation", delegation_id="report-probe", goal="PRIVATE GOAL", context="PRIVATE CONTEXT")
    events = {
        "cron": dict(common, **result),
        "process": dict(common, status="completed", summary=report),
        "batch": dict(common, is_batch=True, goals=["PRIVATE FIRST", "PRIVATE SECOND\nPRIVATE CONTINUATION"], results=[
            dict(task_index=0, status="completed", summary=report, live_transcript="/private/transcript"),
            dict(task_index=1, status="completed", summary="Plain batch result"),
        ]),
        "legacy": dict(common, status="completed", summary=report),
        "plain": dict(common, status="completed", summary="Plain result"),
        "malformed": dict(common, status="completed"),
    }
    db = SessionDB(db_path=home / "delivery.db")

    class DeliveryStore:
        def _ensure_session_db(self):
            return db

    cases = {}
    for name, event in events.items():
        envelope = format_process_notification(event)
        assert envelope is not None
        if name == "legacy":
            envelope = report
        elif name == "malformed":
            envelope = "[ASYNC DELEGATION COMPLETE — malformed]\nPRIVATE INSTRUCTIONS"
        db.create_session(session_id=name, source="desktop")
        asyncio.run(persist_delegation_delivery(DeliveryStore(), text=envelope, session_id=name, evt=event))
        rows = db.get_messages_as_conversation(name, include_row_ids=True)
        cases[name] = dict(envelope=envelope, rows=rows, stored_exactly=rows[0]["content"] == envelope)
    output.write_text(json.dumps({"report": report, "cases": cases}, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "stored_exactly": all(c["stored_exactly"] for c in cases.values()), "producer": "cron save/formatter -> process formatter -> durable delivery -> SQLite readback", "model_requests": 0}))
    db.close()
