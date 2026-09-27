"""Exercise real plugin discovery, adapter queueing and top-level persistence together."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
CORE = str(ROOT)


@pytest.mark.parametrize("mode", ["approve", "silent", "steer", "failed_flush", "nested", "no_database", "spill_failure"])
def test_busy_parent_receives_checkpoint_before_ending_turn(tmp_path, mode):
    home = tmp_path / "home"
    (home / "plugins").mkdir(parents=True)
    (home / "plugins/subagent-progress").symlink_to(ROOT / "plugins/subagent_progress")
    (home / "config.yaml").write_text("plugins:\n  enabled: [subagent-progress]\n")
    code = r'''
import asyncio
from concurrent.futures import Future, TimeoutError
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
import weakref

from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from run_agent import AIAgent
from agent.subagent_lifecycle import bind_subagent_parent
from agent.tool_executor import _flush_session_db_after_tool_progress
from hermes_cli.lifecycle import invoke_hook
from tools.registry import registry
from tools.delegate_tool_registry import _active_subagents
from tools.delegate_tool_deadline import ReviewedDeadline, wait_with_reviewed_deadline
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.config import Platform

mode = sys.argv[1]
home = get_hermes_home()
parent = AIAgent(model="test-model", provider="openai-compat", api_key="test",
    base_url="http://127.0.0.1:1/v1", quiet_mode=True, max_iterations=8,
    skip_context_files=True, skip_memory=True, session_id="owning-parent",
    platform="subagent" if mode == "nested" else "telegram",
    session_db=SessionDB(db_path=home / "session.db"))
clock = SimpleNamespace(now=100.0)
class Child:
    pass
child = Child()
child.session_id = "worker"
child._subagent_id = "busy-worker"
child._delegate_depth = 2 if mode == "nested" else 1
child._delegate_parent_ref = weakref.ref(parent)
child._parent_session_id = parent.session_id
child._interrupt_requested = False
child.steer = lambda text: True
child._delegate_reviewed_deadline = ReviewedDeadline(600, clock=lambda: clock.now)
lease = child._delegate_reviewed_deadline
_active_subagents[child._subagent_id] = {"agent": child,
    "owner_agent_session_id": parent.session_id, "accepting_steer": True}
invoke_hook("subagent_start", parent_session_id=parent.session_id,
    child_session_id=child.session_id, child_subagent_id=child._subagent_id,
    child_goal="Inspect the evidence file and continue the child work")
with bind_subagent_parent(parent):
    # The turn has already begun: its one pre_llm_call cannot see the later report.
    assert not invoke_hook("pre_llm_call", session_id=parent.session_id,
                           platform=parent.platform, is_first_turn=False)
evidence = home / "evidence.json"
evidence.write_text('{"verified_input": true}')
clock.now = 450.0
with bind_subagent_parent(child):
    report = json.loads(registry.get_entry("report_progress").handler({
        "completed": "INPUT_VERIFIED", "evidence": [str(evidence)],
        "next_step": "Continue bounded computation"}))
assert report["success"], report
checkpoint = report["checkpoint_id"]
assert lease.deadline == 700 and lease.renewals == 0
# This actual busy adapter path queues the wake; it does not run another parent turn.
source = SessionSource(platform=Platform.TELEGRAM, chat_id="test", chat_type="dm", user_id="test")
event = MessageEvent(text=f"[SUBAGENT CHECKPOINT ID: {checkpoint}]", message_type=MessageType.TEXT,
    source=source, internal=True, allow_gateway_control=False,
    metadata={"hermes_plugin_injection": True, "hermes_plugin_id": "subagent-progress"})
adapter = SimpleNamespace(name="test", _busy_session_handler=None, _pending_messages={},
    _is_queue_text_debounce_candidate=lambda event: False, _canonicalize=lambda source: None)
asyncio.run(BasePlatformAdapter._handle_message_while_active(adapter, event, "parent-route"))
assert adapter._pending_messages["parent-route"] is event
assert event._gateway_accepted
messages = [{"role": "user", "content": "Original work is still running"},
    {"role": "assistant", "tool_calls": [{"id": "outer-code", "type": "function",
      "function": {"name": "execute_code", "arguments": "{}"}}]},
    {"role": "tool", "name": "execute_code", "tool_call_id": "outer-code", "content": "parent work result"}]
real_flush = parent._flush_messages_to_session_db
session_db = parent._session_db
if mode == "failed_flush":
    parent._flush_messages_to_session_db = lambda *a, **k: False
elif mode == "no_database":
    parent._session_db = None
elif mode == "spill_failure":
    import tools.hook_output_spill as spill
    blocked = home / "blocked-spill"
    blocked.write_text("not a directory")
    spill_config = {"enabled": True, "max_chars": 100, "preview_head": 5,
                    "preview_tail": 5, "directory": str(blocked)}
    spill.get_spill_config = lambda: spill_config
with bind_subagent_parent(parent):
    flushed = _flush_session_db_after_tool_progress(parent, messages, stage="busy parent tool")
if mode != "spill_failure":
    assert "INPUT_VERIFIED" in messages[-1]["content"], "checkpoint remained queued until next turn"
else:
    assert "INPUT_VERIFIED" not in messages[-1]["content"]
assert lease.deadline == 700 and lease.renewals == 0, "delivery itself renewed the deadline"
failed_delivery = mode in {"failed_flush", "no_database", "spill_failure"}
with sqlite3.connect(home / "state/subagent-progress.sqlite3") as db:
    consumed = db.execute("SELECT consumed FROM reports WHERE id=?", (checkpoint,)).fetchone()[0]
    receipt_count = db.execute("SELECT COUNT(*) FROM context_deliveries").fetchone()[0]
assert consumed == (0 if failed_delivery else 1)
if failed_delivery:
    assert not receipt_count
    assert flushed == (mode != "failed_flush")
    parent._session_db = session_db
    parent._flush_messages_to_session_db = real_flush
    if mode == "spill_failure":
        spill_config["directory"] = str(home / "recovered-spill")
    messages.extend([
        {"role": "assistant", "tool_calls": [{"id": "retry-tool", "type": "function",
          "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "name": "read_file", "tool_call_id": "retry-tool", "content": "retry result"}])
    with bind_subagent_parent(parent):
        assert _flush_session_db_after_tool_progress(parent, messages, stage="recovered delivery")
    with sqlite3.connect(home / "state/subagent-progress.sqlite3") as db:
        assert db.execute("SELECT consumed FROM reports WHERE id=?", (checkpoint,)).fetchone()[0] == 1
        receipt = json.loads(db.execute("SELECT receipt FROM context_deliveries WHERE report=?", (checkpoint,)).fetchone()[0])
        assert receipt["persisted"]
    assert messages[-1]["content"] == session_db.get_messages(parent.session_id)[-1]["content"]
    if mode == "spill_failure":
        assert any("INPUT_VERIFIED" in p.read_text() for p in (home / "recovered-spill").glob("**/*.txt"))
    else:
        assert "INPUT_VERIFIED" in messages[-1]["content"]
    assert lease.deadline == 700 and lease.renewals == 0
else:
    assert flushed
    assert "INPUT_VERIFIED" in str(parent._session_db.get_messages(parent.session_id)[-1]["content"])
    assert invoke_hook("pre_gateway_dispatch", event=event)[0]["action"] == "skip"
    clock.now = 650.0
    if mode != "silent":
        # The owning parent's next step reads real evidence, then explicitly reviews.
        assert json.loads(evidence.read_text())["verified_input"]
        with bind_subagent_parent(parent):
            review = json.loads(registry.get_entry("review_subagent_progress").handler({
                "checkpoint_id": checkpoint, "decision": "steer" if mode == "steer" else "approve",
                "reason": "Evidence file was read and agrees with the commissioned check",
                "evidence_checked": [str(evidence)]}))
        assert review["success"], review
    clock.now = 701.0
    future = Future()
    future.set_result("child work completed after original deadline")
    if mode in {"approve", "nested"}:
        assert wait_with_reviewed_deadline(future, lease).startswith("child work completed")
        assert lease.renewals == 1 and lease.deadline == 1250
    else:
        try:
            wait_with_reviewed_deadline(future, lease)
        except TimeoutError:
            pass
        else:
            raise AssertionError("absence of approval must expire the child")
        assert lease.renewals == 0
assert not parent._interrupt_requested and not child._interrupt_requested
print("BUSY_PARENT_CHECK_PASSED", mode)
parent._flush_messages_to_session_db = real_flush
parent.close()
'''
    result = subprocess.run([sys.executable, "-c", code, mode],
        env={**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": CORE},
        text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BUSY_PARENT_CHECK_PASSED" in result.stdout
