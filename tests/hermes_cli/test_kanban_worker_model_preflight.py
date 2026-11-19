"""A card whose profile names no model is refused before dispatch spawns a worker; a profile the
CLI and gateway serve through the provider's catalog default still spawns (#144027).

Fixture and refusal assertions adapted from hehuang139's #144088."""

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def dispatch_profile(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch

    root = tmp_path / ".hermes"
    profile = root / "profiles" / "worker"
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text("model:\n  default: launch-model\n", encoding="utf-8")
    (profile / ".env").write_text("", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(root))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(dispatch, "_memory_pressure_level", lambda: "normal")
    monkeypatch.setattr(dispatch, "_restart_safe_worker_argv", lambda task, command: command)
    spawned = []
    monkeypatch.setattr("subprocess.Popen", lambda command, **kw: spawned.append(command) or SimpleNamespace(pid=424242))
    kb.create_board("work")
    return profile, spawned


def _dispatch(profile, config_yaml, **task):
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch
    from hermes_cli.kanban_db_connect import connect_closing

    (profile / "config.yaml").write_text(config_yaml, encoding="utf-8")
    with connect_closing(board="work") as conn:
        task_id = kb.create_task(conn, title="needs a model", assignee="worker", board="work", **task)
        result = dispatch.dispatch_once(conn, board="work")
        return task_id, result, kb.get_task(conn, task_id), kb.list_events(conn, task_id), kb.list_runs(conn, task_id)


def test_unresolved_model_refused_before_spawn_with_task_log(dispatch_profile):
    from hermes_cli import kanban as cli

    profile, spawned = dispatch_profile
    task_id, result, task, events, runs = _dispatch(
        profile, "model:\n  provider: custom\n  base_url: https://llm.example.test/v1\n")
    assert not spawned and not result.spawned
    assert task.status == "ready" and task.worker_pid is None
    assert "No model configured for profile worker" in task.last_failure_error
    assert any(e.kind == "spawn_failed" and "hermes -p worker model" in e.payload["error"] for e in events)
    assert len(runs) == 1 and runs[0].outcome == "spawn_failed"
    assert "hermes -p worker model" in cli.run_slash(f"--board work log {task_id}")


@pytest.mark.parametrize("config_yaml, task", [
    ("model:\n  provider: openrouter\n", {}),
    ("model:\n  provider: custom\n  base_url: https://llm.example.test/v1\n", {"model_override": "task-model"}),
], ids=["catalog-default", "task-override"])
def test_resolvable_model_still_spawns(dispatch_profile, config_yaml, task):
    profile, spawned = dispatch_profile
    task_id, result, row, _, _ = _dispatch(profile, config_yaml, **task)
    assert len(spawned) == 1 and result.spawned[0][0] == task_id
    assert row.status == "running" and row.worker_pid == 424242


def _card(**kw):
    return SimpleNamespace(**{"id": "t_x", "assignee": "worker", "model_override": None, "provider_override": None, **kw})


_LAB = {"model": {"provider": "openrouter", "default": "profile-model"},
        "custom_providers": [{"name": "lab", "base_url": "https://lab.example.test/v1", "model": "entry-model"}]}


def test_named_provider_pick_brings_its_entry_model_and_task_model_still_wins():
    from hermes_cli.kanban_worker_preflight import task_model

    assert task_model(_card(provider_override="custom:lab"), _LAB) == ("entry-model", False)
    assert task_model(_card(provider_override="custom:lab", model_override="task-model"), _LAB) == ("task-model", False)
    assert task_model(_card(), _LAB) == ("profile-model", False)


def test_owner_asks_a_single_model_loopback_server_the_dispatcher_never_does():
    import http.server
    import json
    import threading

    from hermes_cli.kanban_worker_preflight import task_model

    hits = []

    class Models(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            body = json.dumps({"data": [{"id": "served-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Models)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        config = {"model": {"provider": "custom", "base_url": f"http://127.0.0.1:{server.server_port}/v1"}}
        assert task_model(_card(), config) == ("", False) and hits == []
        assert task_model(_card(), config, discover=True) == ("served-model", False)
        assert hits == ["/v1/models"]
    finally:
        server.shutdown()


def test_preflight_never_imports_the_gateway_runner(tmp_path):
    import os
    import subprocess
    import sys

    root = tmp_path / ".hermes"
    profile = root / "profiles" / "worker"
    profile.mkdir(parents=True)
    (profile / ".env").write_text("", encoding="utf-8")
    (profile / "config.yaml").write_text("model:\n  provider: custom\n  base_url: https://llm.example.test/v1\n", encoding="utf-8")
    probe = (
        "import os, sys\n"
        "from types import SimpleNamespace\n"
        "from hermes_cli.kanban_worker_preflight import refuse_unresolved_model\n"
        "card = SimpleNamespace(id='t_fresh', assignee='worker', model_override=None, provider_override=None)\n"
        "try:\n"
        "    refuse_unresolved_model(card, '.', None)\n"
        "except RuntimeError as exc:\n"
        "    print('refused', exc)\n"
        "print('gateway.run' in sys.modules, os.environ.get('_HERMES_GATEWAY'))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "_HERMES_GATEWAY"}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(root), HERMES_KANBAN_HOME=str(root))
    out = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[0].startswith("refused No model configured for profile worker")
    assert f"HERMES_HOME={profile.parent.parent}" in lines[0]
    assert lines[-1] == "False None"


def test_unavailable_config_backend_is_an_infrastructure_spawn_failure(dispatch_profile, monkeypatch):
    from hermes_cli.config_backend import ConfigBackendUnavailable

    from hermes_cli import config_effective

    profile, spawned = dispatch_profile
    real = config_effective.load_user_config_effective

    def unavailable(config_path=None, **kwargs):
        if config_path is not None and Path(config_path).parent == profile:
            raise ConfigBackendUnavailable("config plane unreachable")
        return real(config_path, **kwargs)

    monkeypatch.setattr(config_effective, "load_user_config_effective", unavailable)
    _, result, task, _, runs = _dispatch(profile, "model:\n  provider: openrouter\n")
    assert not spawned and not result.spawned
    assert task.status == "ready" and task.worker_pid is None and task.consecutive_failures == 0
    assert "config plane unreachable" in task.last_failure_error
    assert len(runs) == 1 and runs[0].outcome == "spawn_failed" and runs[0].metadata.get("infrastructure") is True
