"""Fresh-process exit/recovery and CLI parser-to-cleanup integration, no LLM."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("blocked", ["interrupt", "ledger", "sqlite", "effect"])
def test_shutdown_exits_with_blocked_effect_and_recovers(tmp_path, blocked):
    code = r''' 
import json, threading, time, sqlite3, sys
from tools import async_delegation as ad
from tools.process_registry import process_registry
mode = sys.argv[1]
r = dict(delegation_id="exit-test", goal="offline", dispatched_at=time.time(), status="running", interrupt_fn=threading.Event().wait if mode == "interrupt" else lambda: None)
ad._persist_dispatch(r)
ad._records[r["delegation_id"]] = r
if mode == "ledger": ad._DB_LOCK.acquire()
if mode == "sqlite":
    conn = sqlite3.connect(ad._db_path())
    conn.execute("BEGIN EXCLUSIVE")
if mode == "effect": ad._persist_completion = lambda *a: threading.Event().wait()
start = time.monotonic()
result = ad.finalize_for_oneshot_shutdown(grace_seconds=.02)
assert time.monotonic() - start < 2
again = ad.finalize_for_oneshot_shutdown(grace_seconds=0)
assert len([t for t in threading.enumerate() if t.name.startswith("oneshot-")]) <= 2
assert result["unknown"] + result["pending"] == 1
if mode != "interrupt":
    assert result["unknown"] == 0 and result["pending"] == 1
    assert r["status"] == "finalizing"
    ad._finalize("exit-test", {"summary": "late"}, "completed")
    assert process_registry.completion_queue.empty()
print(json.dumps(result), flush=True)
'''
    env = {**os.environ, "HERMES_HOME": str(tmp_path)}
    child = subprocess.run([sys.executable, "-c", code, blocked], cwd=ROOT, env=env,
                           text=True, capture_output=True, timeout=8)
    assert child.returncode == 0, child.stdout + child.stderr
    recovery = subprocess.run([sys.executable, "-c", "from tools import async_delegation as a; a.recover_abandoned_delegations(); assert a.get_durable_delegation('exit-test')['state'] == 'unknown'"], cwd=ROOT, env=env, text=True, capture_output=True, timeout=8)
    assert recovery.returncode == 0, recovery.stderr


@pytest.mark.parametrize("flag", ["-q", "-z"])
def test_real_cli_entrypoint_terminalizes_offline_delegation(tmp_path, flag):
    code = r'''
import sys, time
from types import SimpleNamespace
from tools import async_delegation as ad
from hermes_cli import main as entry
import cli
import hermes_cli.oneshot as oneshot
# Only the conversation/startup seams are offline; parser, route and teardown are real.
def conversation(*a, **k):
    r = dict(delegation_id="cli-exit", goal="offline", dispatched_at=time.time(), status="running", interrupt_fn=lambda: None)
    ad._persist_dispatch(r)
    ad._records[r["delegation_id"]] = r
    return "offline-result"
class FakeCLI:
    def __init__(self, **k):
        self.console = SimpleNamespace(print=lambda *a, **k: None)
        self.session_id = "offline"
        self.agent = None
    def _claim_active_session(self, *a, **k): return True
    def _release_active_session(self): pass
    def _show_security_advisories(self): pass
    def _print_exit_summary(self, **k): pass
    chat = staticmethod(conversation)
cli.HermesCLI = FakeCLI
oneshot._run_agent = lambda *a, **k: (conversation(), {})
entry._has_any_provider_configured = lambda: True
entry._start_chat_background_prefetch = lambda: None
entry._prepare_agent_startup = lambda *a: None
entry.main()
'''
    child = subprocess.run([sys.executable, "-c", code, *(["chat"] if flag == "-q" else []), flag, "offline"], cwd=ROOT,
                           env={**os.environ, "HERMES_HOME": str(tmp_path)},
                           text=True, capture_output=True, timeout=15)
    assert child.returncode == 0, child.stdout + child.stderr
    with sqlite3.connect(tmp_path / "state.db") as conn:
        assert conn.execute("SELECT state FROM async_delegations WHERE delegation_id='cli-exit'").fetchone() == ("unknown",)
