"""Side-agent AIAgent instances must be closed on every exit path (#50197).

Four long-lived surfaces build a one-shot ``AIAgent`` and, before this fix,
never called ``close()`` — the owner boundary that releases memory-provider
sessions, tool subprocesses and httpx clients:

* ``batch_runner._process_single_prompt`` — one agent per prompt, N prompts
  per batch process: an unclosed agent leaked terminals/VMs/clients for the
  batch's whole run.
* ``plugins.platforms/feishu/feishu_comment._run_comment_agent`` — one agent
  per comment run in a long-lived gateway process.
* ``tui_gateway/methods_prompt`` ``prompt.background`` — one side agent per
  background turn in the gateway process.
* ``hermes_cli/cli_commands_mixin._handle_background_command`` — one agent
  per ``/bg`` task in a long-lived CLI process.

The tests drive each real call path with a recording fake ``AIAgent`` and
assert ``close()`` ran on both the success and the failure path. They are
lifecycle invariants, not change-detectors: the fake fails the assertions on
any code path that forgets the boundary again.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import hermes_bootstrap  # noqa: F401  (process boot before tui_gateway.server)


class RecordingAgent:
    """Minimal AIAgent stand-in: records close() and optional run failure."""

    instances: list["RecordingAgent"] = []

    def __init__(self, *args, **kwargs):
        self.closed = False
        self.close_calls = 0
        self.fail = kwargs.pop("_fail", False)
        type(self).instances.append(self)

    def run_conversation(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("agent run failed")
        return {"final_response": "ok", "messages": [], "completed": True, "api_calls": 0}

    def _convert_to_trajectory_format(self, *args, **kwargs):
        return []

    def close(self):
        self.closed = True
        self.close_calls += 1

    @classmethod
    def reset(cls):
        cls.instances = []

    @classmethod
    def use(cls, monkeypatch, *, where="run_agent.AIAgent", fail=False):
        cls.reset()
        monkeypatch.setattr(where, cls)
        return cls


# ── batch_runner: one agent per prompt ─────────────────────────────────────


class TestBatchRunnerClosesAgent:
    CONFIG = {"model": "m", "max_iterations": 1, "distribution": "d", "verbose": False}

    def test_agent_closed_after_successful_prompt(self, monkeypatch):
        import batch_runner
        RecordingAgent.use(monkeypatch, where="batch_runner.AIAgent")
        monkeypatch.setattr(batch_runner, "sample_toolsets_from_distribution", lambda name: [])

        result = batch_runner._process_single_prompt(0, {"prompt": "hi"}, 0, self.CONFIG)

        assert result["success"] is True
        assert len(RecordingAgent.instances) == 1
        assert RecordingAgent.instances[0].closed

    def test_agent_closed_when_run_fails(self, monkeypatch):
        import batch_runner

        class FailingAgent(RecordingAgent):
            def __init__(self, *args, **kwargs):
                kwargs["_fail"] = True
                super().__init__(*args, **kwargs)

        FailingAgent.reset()
        FailingAgent.instances = []
        monkeypatch.setattr("batch_runner.AIAgent", FailingAgent)
        monkeypatch.setattr(batch_runner, "sample_toolsets_from_distribution", lambda name: [])

        result = batch_runner._process_single_prompt(0, {"prompt": "hi"}, 0, self.CONFIG)

        assert result["success"] is False
        assert FailingAgent.instances[0].closed


# ── feishu comment agent: one per comment run in the gateway ───────────────


class TestFeishuCommentClosesAgent:
    def _run(self, monkeypatch):
        from plugins.platforms.feishu import feishu_comment

        RecordingAgent.reset()
        monkeypatch.setattr("run_agent.AIAgent", RecordingAgent)
        monkeypatch.setattr(feishu_comment, "_resolve_model_and_runtime", lambda: ("m", {}))
        # session_key="" keeps the cross-card history cache out of the test.
        response = feishu_comment._run_comment_agent("hi", client=None, session_key="")
        return response

    def test_agent_closed_after_successful_comment(self, monkeypatch):
        response = self._run(monkeypatch)
        assert response == "ok"
        assert len(RecordingAgent.instances) == 1
        assert RecordingAgent.instances[0].closed

    def test_agent_closed_when_run_fails(self, monkeypatch):
        from plugins.platforms.feishu import feishu_comment

        class FailingAgent(RecordingAgent):
            def __init__(self, *args, **kwargs):
                kwargs["_fail"] = True
                super().__init__(*args, **kwargs)

        FailingAgent.instances = []
        monkeypatch.setattr("run_agent.AIAgent", FailingAgent)
        monkeypatch.setattr(feishu_comment, "_resolve_model_and_runtime", lambda: ("m", {}))
        response = feishu_comment._run_comment_agent("hi", client=None, session_key="")
        assert response == ""
        assert FailingAgent.instances[0].closed


# ── tui_gateway prompt.background: one side agent per bg turn ──────────────


def _bg_session(server, sid: str) -> dict:
    session = {
        "agent": SimpleNamespace(model="m"),
        "agent_ready": threading.Event(),
        "agent_error": None,
        "attached_images": [],
        "cwd": "/tmp",
        "history": [],
        "history_lock": threading.RLock(),
        "history_version": 0,
        "image_counter": 0,
        "profile_home": "/tmp",
        "running": False,
        "session_key": sid,
        "transport": None,
    }
    session["agent_ready"].set()
    server._sessions[sid] = session
    return session


class TestPromptBackgroundClosesAgent:
    def _call(self, monkeypatch, sid, agent_cls):
        from tui_gateway import server
        import contextlib

        monkeypatch.setattr(server, "_start_agent_build", lambda sid_, session_: None)
        monkeypatch.setattr(
            server, "_background_agent_kwargs", lambda agent, task_id: {"model": "m"})
        monkeypatch.setattr(
            server, "_side_agent_session_db",
            lambda parent_db: contextlib.nullcontext(parent_db))
        bodies = []

        def fake_spawn(rid, session, task_id, parent, event, body, **kwargs):
            bodies.append(body)
            return {"result": {"task_id": task_id}}

        monkeypatch.setattr(server, "_spawn_side_agent", fake_spawn)
        agent_cls.reset()
        monkeypatch.setattr("run_agent.AIAgent", agent_cls)

        response = server._methods["prompt.background"](
            1, {"session_id": sid, "text": "do work"})

        assert "error" not in response, response
        # The side-agent body runs synchronously through the fake spawner.
        assert len(bodies) == 1
        bodies[0]()
        return response

    def test_agent_closed_after_background_turn(self, monkeypatch):
        from tui_gateway import server
        sid = "close-lifecycle-bg"
        _bg_session(server, sid)
        try:
            self._call(monkeypatch, sid, RecordingAgent)
            assert len(RecordingAgent.instances) == 1
            assert RecordingAgent.instances[0].closed
        finally:
            server._sessions.pop(sid, None)

    def test_agent_closed_when_background_turn_fails(self, monkeypatch):
        from tui_gateway import server

        class FailingAgent(RecordingAgent):
            def __init__(self, *args, **kwargs):
                kwargs["_fail"] = True
                super().__init__(*args, **kwargs)

        FailingAgent.instances = []
        sid = "close-lifecycle-bg-fail"
        _bg_session(server, sid)
        try:
            with pytest.raises(RuntimeError):
                self._call(monkeypatch, sid, FailingAgent)
            assert FailingAgent.instances[0].closed
        finally:
            server._sessions.pop(sid, None)


# ── classic CLI /bg: one agent per background task ─────────────────────────


def _make_cli():
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._background_task_counter = 0
    cli._background_tasks = {}
    cli._agent_running = False
    cli._app = None
    cli.max_turns = 5
    cli.enabled_toolsets = []
    cli._session_db = None
    cli.reasoning_config = None
    cli.service_tier = None
    for attr in ("_providers_only", "_providers_ignore", "_providers_order", "_provider_sort",
                 "_provider_require_params", "_provider_data_collection",
                 "_openrouter_min_coding_score", "_fallback_model",
                 "_sudo_password_callback", "_approval_callback", "_vault_unlock_callback",
                 "_vault_save_login_callback", "_secret_capture_callback"):
        setattr(cli, attr, None)
    cli._ensure_runtime_credentials = lambda: True
    cli._resolve_turn_agent_config = lambda prompt: {
        "model": "m", "runtime": {}, "request_overrides": None}
    return cli


class TestCliBackgroundCommandClosesAgent:
    def _run_bg(self, monkeypatch, agent_cls):
        cli = _make_cli()
        agent_cls.reset()
        monkeypatch.setattr("run_agent.AIAgent", agent_cls)
        for target in ("cli.set_sudo_password_callback", "cli.set_approval_callback",
                       "cli.set_secret_capture_callback",
                       "agent.vault_backends.unlock.set_code_prompt_callback",
                       "agent.vault_backends.unlock.set_save_login_prompt_callback",
                       "agent.vault_backends.unlock.set_unlock_prompt_callback"):
            monkeypatch.setattr(target, lambda *a, **k: None)
        produced = {}

        def fake_side_worker(produce, **kwargs):
            produced["produce"] = produce
            return SimpleNamespace(start=lambda: None)

        cli._side_worker = fake_side_worker
        cli._handle_background_command("/bg do work")
        assert "produce" in produced, "no background worker was spawned"
        try:
            produced["produce"]()
        except RuntimeError:
            pass  # the failing-agent variant: the worker surfaces the error
        return produced

    def test_agent_closed_after_bg_task(self, monkeypatch):
        self._run_bg(monkeypatch, RecordingAgent)
        assert len(RecordingAgent.instances) == 1
        assert RecordingAgent.instances[0].closed

    def test_agent_closed_when_bg_task_fails(self, monkeypatch):
        class FailingAgent(RecordingAgent):
            def __init__(self, *args, **kwargs):
                kwargs["_fail"] = True
                super().__init__(*args, **kwargs)

        FailingAgent.instances = []
        self._run_bg(monkeypatch, FailingAgent)
        assert FailingAgent.instances[0].closed
