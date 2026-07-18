"""Behavior tests for crash-safe active gateway run recovery."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import replace

import pytest

from gateway.recovery import (
    PHASE_EXECUTING,
    PHASE_RESPONSE_READY,
    RECOVERY_AUTO_RESUME,
    RECOVERY_PAUSE_DELIVERY,
    RECOVERY_PAUSE_INPUT,
    RECOVERY_PAUSE_RETRY_LIMIT,
    RECOVERY_PAUSE_SIDE_EFFECT,
    RECOVERY_WAIT_FOR_PROCESS,
    ActiveRunRecord,
    ActiveRunStore,
    classify_active_run,
)


def _record(**overrides) -> ActiveRunRecord:
    values = {
        "session_key": "agent:main:discord:dm:chat-1",
        "run_id": "run-1",
        "started_at": 100.0,
        "trigger_message_id": "msg-1",
        "phase": PHASE_EXECUTING,
        "recovery_attempts": 1,
    }
    values.update(overrides)
    return ActiveRunRecord(**values)


def _user(timestamp: float = 101.0) -> dict:
    return {
        "role": "user",
        "content": "do the work",
        "message_id": "msg-1",
        "timestamp": timestamp,
    }


def _tool_call(name: str, call_id: str = "call-1") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


class TestActiveRunStore:
    def test_atomic_roundtrip_and_no_message_content(self, tmp_path):
        store = ActiveRunStore(tmp_path)
        record = store.begin(
            "agent:main:discord:dm:chat-1", trigger_message_id="msg-1", started_at=12.5
        )

        restored = ActiveRunStore(tmp_path).get(record.session_key)
        assert restored == record
        payload = json.loads(store.path.read_text(encoding="utf-8"))
        assert payload["version"] == 1
        assert "do the work" not in store.path.read_text(encoding="utf-8")
        assert not list(tmp_path.glob(".*.tmp"))

    def test_corrupt_journal_recovers_on_next_write(self, tmp_path, caplog):
        path = tmp_path / ".active_runs.json"
        path.write_text("{not-json", encoding="utf-8")
        store = ActiveRunStore(tmp_path)

        assert store.snapshot() == []
        assert "corrupt active-run journal" in caplog.text
        record = store.begin("session-a", trigger_message_id="m1")
        assert ActiveRunStore(tmp_path).get("session-a") == record

    def test_run_id_cas_and_phase(self, tmp_path):
        store = ActiveRunStore(tmp_path)
        first = store.begin("session-a", trigger_message_id="m1")
        second = store.begin("session-a", trigger_message_id="m2")

        assert first.run_id != second.run_id
        assert store.mark_response_ready("session-a", first.run_id) is False
        assert store.finish("session-a", first.run_id) is False
        assert store.mark_response_ready("session-a", second.run_id) is True
        assert store.get("session-a").phase == PHASE_RESPONSE_READY
        assert store.finish("session-a", second.run_id) is True
        assert store.get("session-a") is None

    def test_recovery_reclaims_same_run_without_resetting_state(self, tmp_path):
        store = ActiveRunStore(tmp_path)
        original = store.begin("session-a", trigger_message_id="m1", started_at=10)
        attempted = store.record_recovery_attempt(
            "session-a", original.run_id, "boot-a"
        )

        reclaimed = store.begin(
            "session-a",
            trigger_message_id=None,
            recovery_run_id=original.run_id,
            started_at=999,
        )
        assert reclaimed == attempted
        assert reclaimed.started_at == 10
        assert reclaimed.recovery_attempts == 1

    def test_recovery_attempt_is_deduplicated_per_boot(self, tmp_path):
        store = ActiveRunStore(tmp_path)
        record = store.begin("session-a")

        once = store.record_recovery_attempt("session-a", record.run_id, "boot-a")
        duplicate = store.record_recovery_attempt("session-a", record.run_id, "boot-a")
        twice = store.record_recovery_attempt("session-a", record.run_id, "boot-b")

        assert once.recovery_attempts == 1
        assert duplicate == once
        assert twice.recovery_attempts == 2
        assert twice.last_recovery_boot_id == "boot-b"

    @pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="requires SIGKILL")
    def test_sigkill_fault_injection_leaves_one_recoverable_run(self, tmp_path):
        script = """
import os
import signal
import sys
from gateway.recovery import ActiveRunStore
store = ActiveRunStore(sys.argv[1])
store.begin('session-killed', trigger_message_id='message-killed', started_at=42)
os.kill(os.getpid(), signal.SIGKILL)
"""
        child = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            cwd=os.getcwd(),
            check=False,
        )
        assert child.returncode == -signal.SIGKILL

        store = ActiveRunStore(tmp_path)
        records = store.snapshot()
        assert len(records) == 1
        assert records[0].session_key == "session-killed"
        first = store.record_recovery_attempt(
            "session-killed", records[0].run_id, "new-boot"
        )
        duplicate = store.record_recovery_attempt(
            "session-killed", records[0].run_id, "new-boot"
        )
        assert first.recovery_attempts == 1
        assert duplicate == first


class TestActiveRunClassification:
    def test_durable_user_tail_auto_resumes(self):
        decision = classify_active_run(_record(), [_user()])
        assert decision.disposition == RECOVERY_AUTO_RESUME

    def test_complete_tool_result_auto_resumes(self):
        transcript = [
            _user(),
            _tool_call("web_search"),
            {"role": "tool", "tool_call_id": "call-1", "content": "done"},
        ]
        assert (
            classify_active_run(_record(), transcript).disposition
            == RECOVERY_AUTO_RESUME
        )

    def test_dangling_read_only_tool_auto_resumes(self):
        transcript = [_user(), _tool_call("web_search")]
        assert (
            classify_active_run(_record(), transcript).disposition
            == RECOVERY_AUTO_RESUME
        )

    def test_dangling_side_effecting_tool_pauses(self):
        transcript = [_user(), _tool_call("terminal")]
        assert (
            classify_active_run(_record(), transcript).disposition
            == RECOVERY_PAUSE_SIDE_EFFECT
        )

    def test_interrupted_side_effecting_tool_pauses(self):
        transcript = [
            _user(),
            _tool_call("terminal"),
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "[Command interrupted] exit_code: 130",
            },
        ]
        assert (
            classify_active_run(_record(), transcript).disposition
            == RECOVERY_PAUSE_SIDE_EFFECT
        )

    @pytest.mark.parametrize(
        "record,transcript",
        [
            (_record(phase=PHASE_RESPONSE_READY), [_user()]),
            (_record(), [_user(), {"role": "assistant", "content": "finished"}]),
        ],
    )
    def test_delivery_unknown_pauses(self, record, transcript):
        assert (
            classify_active_run(record, transcript).disposition
            == RECOVERY_PAUSE_DELIVERY
        )

    def test_missing_trigger_input_pauses(self):
        transcript = [{"role": "user", "content": "older", "message_id": "other"}]
        assert (
            classify_active_run(_record(), transcript).disposition
            == RECOVERY_PAUSE_INPUT
        )

    def test_triggerless_run_requires_a_timestamped_current_input(self):
        record = _record(trigger_message_id=None)
        transcript = [{"role": "user", "content": "unverifiable legacy input"}]
        assert (
            classify_active_run(record, transcript).disposition == RECOVERY_PAUSE_INPUT
        )

    def test_active_process_waits_for_watcher(self):
        decision = classify_active_run(_record(), [_user()], has_active_process=True)
        assert decision.disposition == RECOVERY_WAIT_FOR_PROCESS

    def test_third_recovery_interruption_pauses(self):
        decision = classify_active_run(
            replace(_record(), recovery_attempts=3), [_user()]
        )
        assert decision.disposition == RECOVERY_PAUSE_RETRY_LIMIT
