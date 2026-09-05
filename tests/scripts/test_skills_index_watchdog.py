"""Fault-injection tests for the skills-index recovery guard."""

import json
import subprocess

import scripts.skills_index_watchdog as watchdog


def test_active_run_suppresses_duplicate_recovery():
    assert watchdog.recovery_action(["completed", "in_progress"]) == "already-running"
    assert watchdog.recovery_action(["queued"]) == "already-running"


def test_only_terminal_runs_allow_recovery_dispatch():
    assert watchdog.recovery_action(["completed", "cancelled", "failure"]) == "dispatch"


def test_run_list_failure_fails_closed(monkeypatch):
    calls = []

    def fake_run(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, "", "rate limited")

    monkeypatch.setattr(watchdog, "_run", fake_run)

    assert watchdog.trigger_recovery("NousResearch/hermes-agent", "skills-index.yml") == "check-failed"
    assert len(calls) == 1


def test_active_run_does_not_dispatch(monkeypatch):
    calls = []

    def fake_run(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps([{"status": "in_progress"}]), "")

    monkeypatch.setattr(watchdog, "_run", fake_run)

    assert watchdog.trigger_recovery("NousResearch/hermes-agent", "skills-index.yml") == "already-running"
    assert len(calls) == 1


def test_dispatch_failure_is_reported_without_failing_watchdog(monkeypatch):
    calls = []

    def fake_run(command):
        calls.append(command)
        if command[1:3] == ["run", "list"]:
            return subprocess.CompletedProcess(command, 0, "[]", "")
        return subprocess.CompletedProcess(command, 1, "", "forbidden")

    monkeypatch.setattr(watchdog, "_run", fake_run)

    assert watchdog.trigger_recovery("NousResearch/hermes-agent", "skills-index.yml") == "dispatch-failed"
    assert len(calls) == 2
