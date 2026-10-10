"""Runtime reclamation for background threads and probe executors.

Regression for #128969: long-running gateway/serve processes accumulated
threads with no runtime reclamation — only a restart cleared them. Covered:

- ``CLICommandsMixin._reclaim_background_tasks`` drops finished ``/bg``
  thread references while keeping live threads and never-started stand-ins.
- The status-bar snapshot reaps before counting, so ``▶ N`` stays truthful.
- New ``/bg`` submissions reap before insert and never leak on ``start()``
  failure.
- Bounded probes (model-info, gateway-health, Z.AI) run on daemon workers
  with non-blocking shutdown, so an abandoned probe never pins a non-daemon
  thread (and its atexit join) until GC.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

import pytest

from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "anthropic/claude-opus-4.6"
    cli_obj.agent = None
    cli_obj._background_tasks = {}
    cli_obj.session_start = datetime.now()
    cli_obj._agent_running = False
    cli_obj._spinner_text = ""
    cli_obj._app = None
    return cli_obj


def _finished_thread() -> threading.Thread:
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert t.ident is not None
    return t


def _live_thread(stop: threading.Event) -> threading.Thread:
    t = threading.Thread(target=stop.wait, daemon=True)
    t.start()
    assert t.is_alive()
    return t


def _stub_thread() -> threading.Thread:
    return threading.Thread(target=lambda: None)


def test_reclaim_drops_only_finished_threads_and_reports_count():
    cli_obj = _make_cli()
    stop = threading.Event()
    try:
        live = _live_thread(stop)
        finished = _finished_thread()
        stub = _stub_thread()
        cli_obj._background_tasks = {"live": live, "dead": finished, "stub": stub}
        reclaimed = cli_obj._reclaim_background_tasks()
        assert reclaimed == 1
        assert "dead" not in cli_obj._background_tasks
        assert cli_obj._background_tasks["live"] is live
        assert cli_obj._background_tasks["stub"] is stub
    finally:
        stop.set()


def test_reclaim_is_best_effort_and_never_raises():
    cli_obj = _make_cli()
    # Missing / wrong-typed state.
    del cli_obj._background_tasks
    assert cli_obj._reclaim_background_tasks() == 0
    cli_obj._background_tasks = None  # type: ignore[assignment]
    assert cli_obj._reclaim_background_tasks() == 0
    cli_obj._background_tasks = "not-a-dict"  # type: ignore[assignment]
    assert cli_obj._reclaim_background_tasks() == 0

    class _Boom:
        ident = 1234

        def is_alive(self):
            raise RuntimeError("boom")

    cli_obj._background_tasks = {"boom": _Boom()}
    assert cli_obj._reclaim_background_tasks() == 0
    assert "boom" in cli_obj._background_tasks


def test_snapshot_reaps_dead_threads_before_counting():
    cli_obj = _make_cli()
    stop = threading.Event()
    try:
        live_stub = _stub_thread()
        dead = _finished_thread()
        cli_obj._background_tasks = {"bg_live": live_stub, "bg_dead": dead}
        snap = cli_obj._get_status_bar_snapshot()
        assert "bg_dead" not in cli_obj._background_tasks
        assert "bg_live" in cli_obj._background_tasks
        assert snap["active_background_tasks"] == len(cli_obj._background_tasks) == 1
    finally:
        stop.set()


def test_handle_background_command_reaps_before_insert(monkeypatch):
    import hermes_cli.cli_commands_mixin as cmds

    cli_obj = _make_cli()
    dead = _finished_thread()
    cli_obj._background_tasks = {"old_dead": dead}
    cli_obj._background_task_counter = 0
    cli_obj._ensure_runtime_credentials = lambda: True  # type: ignore[method-assign]
    cli_obj._resolve_turn_agent_config = lambda prompt: (  # type: ignore[method-assign]
        {"runtime": {}, "model": "m", "request_overrides": None}
    )
    monkeypatch.setattr(cmds, "_command_arg", lambda cmd: "hello")
    monkeypatch.setattr(cmds, "_ellipsize", lambda s, n: s[:n])
    monkeypatch.setattr(cmds, "_t", lambda _k, **_kw: "x")
    monkeypatch.setattr(cmds, "_cp", lambda *a, **k: None)

    started = threading.Event()

    class _FakeThread(threading.Thread):
        def __init__(self):
            super().__init__(target=lambda: None, daemon=True)

        def start(self):  # noqa: D102 - test double
            started.set()

    fake = _FakeThread()
    monkeypatch.setattr(cli_obj, "_side_worker", lambda *a, **k: fake)

    cli_obj._handle_background_command("/bg hello")
    assert started.is_set()
    assert "old_dead" not in cli_obj._background_tasks
    assert len(cli_obj._background_tasks) == 1
    assert next(iter(cli_obj._background_tasks.values())) is fake


def test_handle_background_command_start_failure_leaves_no_reference(monkeypatch):
    import hermes_cli.cli_commands_mixin as cmds

    cli_obj = _make_cli()
    cli_obj._background_task_counter = 0
    cli_obj._ensure_runtime_credentials = lambda: True  # type: ignore[method-assign]
    cli_obj._resolve_turn_agent_config = lambda prompt: (  # type: ignore[method-assign]
        {"runtime": {}, "model": "m", "request_overrides": None}
    )
    monkeypatch.setattr(cmds, "_command_arg", lambda cmd: "hello")
    monkeypatch.setattr(cmds, "_ellipsize", lambda s, n: s[:n])
    monkeypatch.setattr(cmds, "_t", lambda _k, **_kw: "x")
    monkeypatch.setattr(cmds, "_cp", lambda *a, **k: None)

    class _FailThread(threading.Thread):
        def __init__(self):
            super().__init__(target=lambda: None, daemon=True)

        def start(self):  # noqa: D102 - test double
            raise RuntimeError("thread limit")

    monkeypatch.setattr(cli_obj, "_side_worker", lambda *a, **k: _FailThread())
    with pytest.raises(RuntimeError, match="thread limit"):
        cli_obj._handle_background_command("/bg hello")
    assert cli_obj._background_tasks == {}


def test_model_info_probe_timeout_returns_quickly_on_daemon_worker(monkeypatch):
    import hermes_cli.web_routers.models as models_mod

    def _hang(**_kw):
        time.sleep(30.0)
        return 128000

    monkeypatch.setattr(models_mod, "_MODEL_INFO_PROBE_BUDGET_S", 0.2)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", _hang)
    began = time.monotonic()
    assert models_mod._bounded_context_length_probe("m", "http://x", "p") == 0
    assert time.monotonic() - began < 2.0
    workers = [t for t in threading.enumerate() if "model-info-probe" in (t.name or "")]
    assert workers, "expected an abandoned probe worker to exist"
    assert all(t.daemon for t in workers), "abandoned probe must not block interpreter exit"


def test_gateway_health_probe_timeout_returns_fallback_on_daemon_worker(monkeypatch):
    import hermes_cli.web_routers.status as status_mod

    def _hang():
        time.sleep(30.0)
        return True, None

    monkeypatch.setattr(status_mod, "_probe_gateway_health", _hang)
    began = time.monotonic()
    assert status_mod._bounded_health_probe() == (False, None)
    assert time.monotonic() - began < 2.0
    workers = [t for t in threading.enumerate() if "gateway-health-probe" in (t.name or "")]
    assert workers, "expected an abandoned health-probe worker to exist"
    assert all(t.daemon for t in workers), "abandoned probe must not block interpreter exit"


def test_zai_probe_abandoned_workers_are_daemon(monkeypatch):
    import hermes_cli.auth_zai_kimi as zai_mod

    def _fake_probe(api_key, endpoint, timeout):
        if endpoint[0] == "global":
            return {"id": "global", "base_url": endpoint[1], "model": "glm-5", "label": "Global"}
        time.sleep(30.0)
        return None

    monkeypatch.setattr(zai_mod, "_probe_single_zai_endpoint", _fake_probe)
    began = time.monotonic()
    result = zai_mod.detect_zai_endpoint("key", timeout=5.0)
    elapsed = time.monotonic() - began
    assert result is not None and result["id"] == "global"
    assert elapsed < 5.0, f"early return defeated: {elapsed:.2f}s"
    workers = [t for t in threading.enumerate() if "zai-probe" in (t.name or "")]
    assert workers, "expected probe workers to exist"
    assert all(t.daemon for t in workers), "abandoned Z.AI probes must not block interpreter exit"
