"""Cron terminal ownership follows the worker lifetime, including detached workers."""
import json
import time
from concurrent.futures import Future
from contextvars import copy_context
from pathlib import Path
from threading import Event, Thread

import pytest

from cron import scheduler
from cron.scheduler_run_scope import _CronRunScope
from tools import terminal_tool
from tools.terminal_tool_lifecycle import cleanup_vm
from tui_gateway import server as gateway


def _homes(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default = tmp_path / ".hermes"
    work = default / "profiles" / "work"
    for home in (default, work):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(json.dumps({
            "model": "test/model", "cron": {"preflight": False},
            "terminal": {"backend": "local"}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(default))
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    return default, work


def _controlled_agent(monkeypatch, observed, future, pending, release):
    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def close(self):
            observed["close_key"] = terminal_tool._resolve_container_task_id(observed["task_id"])
            observed["close_entered"].set()
            assert release.wait(timeout=30)
            cleanup_vm(observed["key"])

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "api_key": "test-key", "base_url": "https://example.invalid/v1",
        "provider": "openrouter", "api_mode": "chat_completions"})

    def controlled_worker(agent, prompt, job, job_id, job_name, task_id, cancel_event, *, worker_state):
        observed["task_id"] = task_id
        observed["context"] = copy_context()
        observed["key"] = terminal_tool._resolve_container_task_id(task_id)
        seeded = json.loads(terminal_tool.terminal_tool("export CRON_LIFETIME_PROBE=own-run", task_id=task_id))
        assert seeded["exit_code"] == 0
        worker_state["future"] = future
        if not pending:
            future.set_result({"final_response": "done"})
        return {"final_response": "done"}

    monkeypatch.setattr(scheduler, "_run_agent_with_watchdog", controlled_worker)


def _wait_for_release(context, task_id, key):
    deadline = time.monotonic() + 10
    while context.run(terminal_tool._resolve_container_task_id, task_id) == key and time.monotonic() < deadline:
        Event().wait(0.01)
    assert context.run(terminal_tool._resolve_container_task_id, task_id) != key


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("profile", ["default", "work"])
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("slow_cleanup", [False, True])
def test_run_job_retains_owner_until_worker_and_cleanup_finish(tmp_path, monkeypatch, pending, profile, deferred, slow_cleanup):
    default, work = _homes(tmp_path, monkeypatch)
    home = default if profile == "default" else work
    observed = {"close_entered": Event()}
    future = Future()
    future.set_running_or_notify_cancel()
    release = Event()
    if not slow_cleanup:
        release.set()
    _controlled_agent(monkeypatch, observed, future, pending, release)
    monkeypatch.setattr(scheduler, "_cron_cleanup_timeout_seconds", lambda: 0.02 if slow_cleanup else 10)
    deferred_agents = [] if deferred else None
    try:
        with gateway._session_profile_runtime_scope({"profile_home": str(home)}, hydrate_secrets=False):
            assert scheduler.run_job({"id": "lifetime-job", "prompt": "hello"}, execution_id="same-run",
                                     defer_agent_teardown=deferred_agents)[0]
            context = observed["context"]
            if pending or deferred or slow_cleanup:
                assert context.run(terminal_tool._resolve_container_task_id, observed["task_id"]) == observed["key"]
                late = json.loads(context.run(terminal_tool.terminal_tool,
                    'printf %s "$CRON_LIFETIME_PROBE"', task_id=observed["task_id"]))
                assert late["output"] == "own-run"
        if pending:
            completion = Thread(target=future.set_result, args=({"final_response": "late"},))
            completion.start()
            completion.join(timeout=10)
            assert not completion.is_alive()
        elif deferred:
            assert len(deferred_agents) == 1
            # Called after leaving the owning profile; the deferred action must carry its context.
            deferred_agents.pop()()
        assert observed["close_entered"].wait(timeout=10)
        assert observed["close_key"] == observed["key"]
        if slow_cleanup:
            assert context.run(terminal_tool._resolve_container_task_id, observed["task_id"]) == observed["key"]
        release.set()
        _wait_for_release(context, observed["task_id"], observed["key"])
    finally:
        release.set()
        if not future.done():
            future.set_result({"final_response": "late"})
        if "key" in observed:
            cleanup_vm(observed["key"])
            cleanup_vm("default")


@pytest.mark.parametrize("profile", ["default", "work"])
@pytest.mark.parametrize("delivery_error", [False, True])
def test_delivery_pipeline_retains_owner_through_deferred_teardown(tmp_path, monkeypatch, profile, delivery_error):
    from cron.jobs import create_job

    default, work = _homes(tmp_path, monkeypatch)
    home = default if profile == "default" else work
    observed = {"close_entered": Event()}
    future = Future()
    future.set_running_or_notify_cancel()
    release = Event()
    release.set()
    _controlled_agent(monkeypatch, observed, future, False, release)
    real_save_deliver = scheduler._save_compose_deliver

    def save_deliver(*args, **kwargs):
        observed["delivery_key"] = terminal_tool._resolve_container_task_id(observed["task_id"])
        assert "close_key" not in observed
        late = json.loads(terminal_tool.terminal_tool(
            'printf %s "$CRON_LIFETIME_PROBE"', task_id=observed["task_id"]))
        observed["delivery_output"] = late["output"]
        if delivery_error:
            raise RuntimeError("delivery interrupted")
        return real_save_deliver(*args, **kwargs)

    monkeypatch.setattr(scheduler, "_save_compose_deliver", save_deliver)
    try:
        with gateway._session_profile_runtime_scope({"profile_home": str(home)}, hydrate_secrets=False):
            job = create_job(prompt="hello", schedule="every 1h", deliver="local")
            scheduler._run_one_job_body(job)
        assert observed["delivery_key"] == observed["key"]
        assert observed["delivery_output"] == "own-run"
        assert observed["close_key"] == observed["key"]
        _wait_for_release(observed["context"], observed["task_id"], observed["key"])
    finally:
        release.set()
        if "key" in observed:
            cleanup_vm(observed["key"])
            cleanup_vm("default")


def test_same_run_id_in_two_profiles_has_independent_ownership(tmp_path, monkeypatch):
    default, work = _homes(tmp_path, monkeypatch)
    runs = []
    for home in (default, work):
        with gateway._session_profile_runtime_scope({"profile_home": str(home)}, hydrate_secrets=False):
            context = copy_context()
            scope = context.run(_CronRunScope, {"id": "same-job"}, "same-job", "same-run")
            runs.append((context, scope))
    try:
        keys = [context.run(terminal_tool._resolve_container_task_id, scope.task_id) for context, scope in runs]
        assert keys[0] != keys[1]
        first_context, first = runs[0]
        first_context.run(first.release)
        second_context, second = runs[1]
        assert second_context.run(terminal_tool._resolve_container_task_id, second.task_id) == keys[1]
    finally:
        for context, scope in runs:
            context.run(scope.exit)
            context.run(scope.release)
