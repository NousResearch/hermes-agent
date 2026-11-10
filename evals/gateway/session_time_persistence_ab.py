"""Real SessionStore/SQLite A/B, isolated fresh process; no provider calls."""
import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
home = Path(tempfile.mkdtemp(prefix="idle-time-ab-"))
for key in list(os.environ):
    if key.startswith("HERMES_") or key.endswith(("_API_KEY", "_TOKEN")):
        os.environ.pop(key, None)
os.environ["HOME"] = str(home)
os.environ["HERMES_HOME"] = str(home / ".hermes")
sys.path.insert(0, sys.argv[1])
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionStore, SessionSource
results = []
for mode in ("idle", "daily", "both", "none"):
    cfg = GatewayConfig.from_dict({"default_reset_policy": {"mode": mode, "idle_minutes": 1}})
    store = SessionStore(home / mode, cfg)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=mode, user_id="probe")
    old = store.get_or_create_session(source)
    for role in ("user", "assistant"):
        store.append_to_transcript(old.session_id, {"role": role, "content": "durable probe"})
    old.updated_at = datetime.now() - timedelta(days=3)
    old.resume_pending = True
    old.last_resume_marked_at = old.updated_at
    store._save()
    new = store.get_or_create_session(source)
    result = {"mode": mode, "same_session": new.session_id == old.session_id,
              "routed_messages": len(store.load_transcript(new.session_id))}
    explicit = store.reset_session(new.session_key)
    result["explicit_rotated"] = explicit.session_id != new.session_id
    result["explicit_end_reason"] = store._db.get_session(new.session_id)["end_reason"]
    results.append(result)
    store._db.close()
print(json.dumps(results, indent=2), flush=True)
assert all(r["same_session"] and r["routed_messages"] == 2 and r["explicit_rotated"] and r["explicit_end_reason"] == "session_reset" for r in results), "elapsed-time rotation discarded routed conversation"
