"""Real worker children must receive the assignee's grants, not launch residue."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def worker_profiles(tmp_path, monkeypatch):
    from agent import secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.env_passthrough import clear_env_passthrough

    root = tmp_path / ".hermes"
    source = root / "profiles" / "source"
    target = root / "profiles" / "target"
    for home in (root, source, target):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (target / "home").mkdir()
    # A scheduled job already has a target cron store before handoff.
    (target / "cron").mkdir()
    (source / ".env").write_text(
        "SOURCE_ONLY=alpha\nSHARED_LOGIN=alpha-shared\n", encoding="utf-8"
    )
    (target / ".env").write_text(
        "TARGET_ONLY=beta\nSHARED_LOGIN=beta-shared\nUNPERMITTED=private\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.setenv("SOURCE_ONLY", "alpha")
    monkeypatch.setenv("SHARED_LOGIN", "alpha-shared")
    monkeypatch.setenv("TARGET_ONLY", "stale-target")
    monkeypatch.setenv("UNPERMITTED", "stale-private")
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "launch-profile-image")
    carriers = (
        "APPTAINERENV_SOURCE_ONLY",
        "SINGULARITYENV_APPTAINERENV_SOURCE_ONLY",
        "_HERMES_FORCE_SOURCE_ONLY",
    )
    for name in carriers:
        monkeypatch.setenv(name, "alpha")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    scope_token = secret_scope.set_secret_scope(None)
    home_token = set_hermes_home_override(source)
    clear_env_passthrough()
    try:
        yield source, target, carriers
    finally:
        clear_env_passthrough()
        secret_scope.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)


def _set_forwarding(target, enabled):
    (target / "config.yaml").write_text(
        "terminal:\n  env_passthrough: "
        + json.dumps(["TARGET_ONLY"] if enabled else []) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("caller_is_target", [False, True])
def test_cron_handoff_reprojects_target_grants_on_each_launch(
    worker_profiles, tmp_path, monkeypatch, caller_is_target
):
    import cron.scheduler as scheduler
    from hermes_constants import (
        get_hermes_home_override, reset_hermes_home_override, set_hermes_home_override,
    )
    from tools.process_registry import GatewayChildDispatch

    source, target, carriers = worker_profiles
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: target)
    monkeypatch.setattr(scheduler, "mark_execution_handoff_pending", lambda _: {"id": "exec"})
    output = tmp_path / "cron-child.json"
    names = ["TARGET_ONLY", "SHARED_LOGIN", "SOURCE_ONLY", "UNPERMITTED",
             "TERMINAL_DOCKER_IMAGE", "HERMES_HOME", *carriers]

    def dispatch(command, **_kwargs):
        # Substitute only the systemd/agent payload, not environment construction.
        ack = command[command.index("--ack-file") + 1]
        payload = command[command.index("--external-worker-file") + 1]
        probe = (
            "import json,os,pathlib;"
            f"payload=json.loads(pathlib.Path({payload!r}).read_text(encoding='utf-8'));"
            f"pathlib.Path({str(output)!r}).write_text(json.dumps({{k:os.getenv(k) for k in {names!r}}}),encoding='utf-8');"
            f"pathlib.Path({ack!r}).write_text(json.dumps({{'pid':os.getpid(),'execution_id':payload['job']['execution_id']}}),encoding='utf-8')"
        )
        return GatewayChildDispatch("degraded", [sys.executable, "-c", probe])

    monkeypatch.setattr("tools.process_registry.restart_safe_gateway_child_argv", dispatch)

    def wait(process, *, job_id, handoff_files, **_kwargs):
        try:
            assert process.wait(timeout=10) == 0
            return True
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            for path in handoff_files:
                path.unlink(missing_ok=True)
            with scheduler._running_lock:
                scheduler._restart_safe_waiter_job_ids.discard(job_id)

    monkeypatch.setattr(scheduler, "_wait_for_external_cron_worker", wait)
    caller = target if caller_is_target else source
    token = set_hermes_home_override(caller)
    try:
        for enabled in (True, False):
            _set_forwarding(target, enabled)
            job = {"id": "projection", "execution_id": f"exec-{enabled}", "prompt": "unused"}
            assert scheduler._launch_external_cron_worker(job) is True
            expected = dict.fromkeys(names)
            expected.update(TARGET_ONLY="beta" if enabled else None,
                            SHARED_LOGIN="beta-shared", HERMES_HOME=str(target))
            assert json.loads(output.read_text(encoding="utf-8")) == expected
            assert get_hermes_home_override() == str(caller)
            assert os.environ["SOURCE_ONLY"] == "alpha"
    finally:
        reset_hermes_home_override(token)


@pytest.mark.linux_only
def test_kanban_cross_profile_child_observes_only_current_assignee_grant(
    worker_profiles, tmp_path, monkeypatch
):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as dispatch
    from hermes_cli.kanban_db_connect import connect
    from hermes_constants import get_hermes_home_override

    source, target, carriers = worker_profiles
    monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "prior-task")
    db = tmp_path / "board.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    output = tmp_path / "kanban-child.json"
    names = ["TARGET_ONLY", "SHARED_LOGIN", "SOURCE_ONLY", "UNPERMITTED",
             "TERMINAL_DOCKER_IMAGE", "HERMES_HOME", "HOME", "HERMES_KANBAN_TASK", *carriers]
    probe = (
        "import json,os,pathlib;"
        f"pathlib.Path({str(output)!r}).write_text(json.dumps({{k:os.getenv(k) for k in {names!r}}}),encoding='utf-8')"
    )
    monkeypatch.setattr(dispatch, "_resolve_hermes_argv", lambda: [sys.executable, "-c", probe])
    real_popen = subprocess.Popen
    children = []

    def capture(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(dispatch.subprocess, "Popen", capture)
    conn = connect(db)
    try:
        for enabled in (True, False):
            _set_forwarding(target, enabled)
            task_id = kb.create_task(conn, title="assigned child", assignee="target")
            kb.claim_task(conn, task_id)
            task = kb.get_task(conn, task_id)
            assert task is not None
            pid = dispatch._default_spawn(task, str(tmp_path), board="default")
            assert pid == children[-1].pid
            assert children[-1].wait(timeout=10) == 0
            expected = dict.fromkeys(names)
            expected.update(TARGET_ONLY="beta" if enabled else None,
                            SHARED_LOGIN="beta-shared", HERMES_HOME=str(target),
                            HOME=str(target / "home"), HERMES_KANBAN_TASK=task_id)
            assert json.loads(output.read_text(encoding="utf-8")) == expected
            assert get_hermes_home_override() == str(source)
            assert os.environ["HERMES_KANBAN_TASK"] == "prior-task"
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
        conn.close()
