"""Isolated production-library controls; no transport/provider calls."""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

repo = Path(sys.argv[1]).resolve()
home = Path(tempfile.mkdtemp(prefix="idle-controls-"))
for key in list(os.environ):
    if key.startswith("HERMES_") or key.endswith(("_API_KEY", "_TOKEN")):
        os.environ.pop(key, None)
os.environ.update(HOME=str(home), HERMES_HOME=str(home / ".hermes"), SESSION_IDLE_MINUTES="1", SESSION_RESET_HOUR="0")
sys.path.insert(0, str(repo))
import yaml
from gateway.config import load_gateway_config, GatewayConfig, Platform
from gateway.session import SessionStore, SessionSource
from gateway.run import GatewayRunner
from gateway.agent_cache_pressure import AgentCacheBounds

Path(os.environ["HERMES_HOME"]).mkdir(exist_ok=True)
Path(os.environ["HERMES_HOME"], "config.yaml").write_text(yaml.safe_dump({"session_reset": {"mode": "both", "idle_minutes": 1}, "gateway": {"session_reset": {"mode": "daily", "at_hour": 0}}}), encoding="utf-8")
config = load_gateway_config()
assert "default_reset_policy" not in config.to_dict()
source = SessionSource(platform=Platform.TELEGRAM, chat_id="cache", user_id="control")
store = SessionStore(home / "sessions", config)
entry = store.get_or_create_session(source)
messages = [{"role": "user", "content": "persist across soft release"}]
store.append_to_transcript(entry.session_id, messages[0])
results = {"legacy_yaml_env_ignored": True, "cache": []}

for strategy in ("ttl", "lru", "pressure"):
    runner = object.__new__(GatewayRunner)
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache_bounds_cache = AgentCacheBounds(max_size=1, idle_ttl_secs=1, memory_high_mb=1, protect_recent=0)
    runner.session_store = store
    trace = []
    released = threading.Event()
    def commit(msgs):
        assert msgs == messages
        Path(home, strategy + "-memory.json").write_text(json.dumps(msgs), encoding="utf-8")
        trace.append("memory")
    def release():
        assert json.loads(Path(home, strategy + "-memory.json").read_text(encoding="utf-8")) == messages
        trace.append("release")
        released.set()
    agent = SimpleNamespace(_memory_manager=object(), _session_messages=list(messages),
                            _last_activity_ts=time.time() - 86400, _last_flushed_db_idx=len(messages),
                            commit_memory_session=commit, release_clients=release)
    active = SimpleNamespace(_last_activity_ts=time.time() - 86400)
    runner._running_agents = {"active": active}
    runner._agent_cache = OrderedDict([(entry.session_key, (agent, "sig")), ("active", (active, "sig"))])
    if strategy == "ttl":
        assert runner._sweep_idle_cached_agents() == 1
    elif strategy == "pressure":
        unflushed = SimpleNamespace(_session_messages=[{"role": "user", "content": "not on disk"}], _last_flushed_db_idx=0)
        runner._agent_cache["unflushed"] = (unflushed, "sig")
        assert runner._sweep_agent_cache_under_pressure() == 1
        assert "unflushed" in runner._agent_cache
    else:
        with runner._agent_cache_lock:
            runner._enforce_agent_cache_cap()
    assert released.wait(10), strategy
    assert trace == ["memory", "release"]
    assert "active" in runner._agent_cache
    resumed = store.get_or_create_session(source)
    assert resumed.session_id == entry.session_id
    assert store.load_transcript(resumed.session_id)[0]["content"] == messages[0]["content"]
    assert store._db.get_session(resumed.session_id)["end_reason"] is None
    results["cache"].append({"strategy": strategy, "order": trace, "same_transcript": True, "active_preserved": True})

store._db.end_session(entry.session_id, "ws_orphan_reap")
recovered = store.get_or_create_session(source)
assert recovered.session_id == entry.session_id
assert store._db.get_session(entry.session_id)["end_reason"] is None
results["ws_orphan_end_reopens_same_transcript"] = True

store.suspend_session(entry.session_key)
suspended = store.get_or_create_session(source)
assert suspended.session_id != entry.session_id
assert store._db.get_session(entry.session_id)["end_reason"] == "suspended"
results["explicit_suspension_rotates"] = True
store.append_to_transcript(suspended.session_id, messages[0])
# Historical finalization wrote both the flag and durable reset boundary.
store._db.set_expiry_finalized(suspended.session_id)
store._db.promote_to_session_reset(suspended.session_id)
store._db.replace_gateway_routing_entries({}, scope=store._routing_scope())
store._entries.clear()
fenced = store.get_or_create_session(source)
assert fenced.session_id != suspended.session_id
results["historical_expiry_fence_preserved"] = True
store._db.close()

spec = importlib.util.spec_from_file_location("migration_probe", repo / "optional-skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
source_dir = home / "migration-source"
target = home / "migration-target"
source_dir.mkdir()
target.mkdir()
(target / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
before = (target / "config.yaml").read_bytes()
migrator = module.Migrator(source_dir, target, True, None, False, False, home / "archive-output")
migrator.migrate_session_config({"session": {"reset": {"mode": "daily", "atHour": 3}, "identityLinks": {"test": "identity"}}})
assert (target / "config.yaml").read_bytes() == before
assert json.loads((migrator.archive_dir / "session-config.json").read_text(encoding="utf-8")) == {"identityLinks": {"test": "identity"}}
results["migration_timers_ignored_advanced_archived"] = True
from contextlib import redirect_stdout
from io import StringIO
from hermes_cli.cli_info_mixin import CLIInfoMixin
status_output = StringIO()
with redirect_stdout(status_output):
    CLIInfoMixin._show_gateway_status(object.__new__(CLIInfoMixin))
assert "Error loading gateway config" not in status_output.getvalue()
assert "Conversations persist" in status_output.getvalue()
results["gateway_status_without_policy"] = True
print(json.dumps(results, indent=2), flush=True)
