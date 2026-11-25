"""Collect real cron completion producer + durable delivery, without inference."""
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
with tempfile.TemporaryDirectory(prefix="markdown-producer-") as temporary:
    home = Path(temporary)
    for key in list(os.environ):
        if key.startswith("HERMES_") or key.endswith(("_API_KEY", "_TOKEN")):
            os.environ.pop(key, None)
    os.environ.update(HOME=str(home), HERMES_HOME=str(home / ".hermes"))
    from cron.jobs import create_job, save_job_output
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.wake import persist_delegation_delivery
    from persistence.conversation.messages import ConversationRequest
    from persistence.conversation.sessions import NewSession
    from persistence_assembly import open_persistence, profile_scope
    from tools.cronjob_tools import _manual_run_completion
    from tools.process_registry_notifications import format_process_notification

    report = "# Generated inspection\n\n**Conclusion:** attention needed.\n\n| Strategy | Status |\n|---|---|\n| Canary | running |\n\nFirst line  \nSecond line\n\n```python\nvalue = 1  \nvalue = 2 \n```"
    job = create_job(prompt="Offline fixture; never run", schedule="0 0 * * *", name="Markdown probe", deliver="local")
    save_job_output(job["id"], report)
    result = _manual_run_completion({"success": True}, job["id"], job["name"], "local", time.time())
    event = dict(result, type="async_delegation", delegation_id="markdown-probe", goal="Inspect a generated report")
    envelope = format_process_notification(event)
    assert envelope is not None
    store = open_persistence(profile_scope(), eager=True)
    store.sessions.create(NewSession(session_id="markdown-probe", source="desktop"))
    # The adapter opens this home's store itself.
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    asyncio.run(persist_delegation_delivery(adapter, text=envelope, session_id="markdown-probe", evt=event))
    rows = [m.as_openai_message() for m in store.messages.read_conversation("markdown-probe", ConversationRequest(include_row_ids=True))]
    output.write_text(json.dumps({"report": report, "result": result, "envelope": envelope, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"producer": "save_job_output -> _manual_run_completion -> format_process_notification -> persist_delegation_delivery -> store readback", "rows": len(rows), "report_preserved": report in rows[0]["content"], "display_kind": rows[0].get("display_kind"), "model_requests": 0}))
    adapter._close_cached_session_dbs()
    store.close()
