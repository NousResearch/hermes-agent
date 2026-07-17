from __future__ import annotations

import json
from types import SimpleNamespace

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from tools import threadwire_telegram_worker_tool as worker


def test_gateway_bound_source_reaches_tool_dispatch(monkeypatch, tmp_path):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100123",
        chat_type="group",
        user_id="44",
        thread_id="7",
    )
    context = SessionContext(
        source=source,
        connected_platforms=[Platform.TELEGRAM],
        home_channels={},
        session_key="safe-test-key",
    )
    captured = []

    class FakeProcess:
        pid = 123
        stdin = None

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def fake_popen(argv, **kwargs):
        captured.append((argv, kwargs))
        return FakeProcess()

    monkeypatch.setattr(worker, "check_threadwire_requirements", lambda: True)
    monkeypatch.setattr(worker.subprocess, "Popen", fake_popen)
    tokens = runner._set_session_env(context)
    try:
        result = worker._handle_threadwire({
            "provider": "codex", "prompt": "work", "cwd": str(tmp_path)
        })
    finally:
        runner._clear_session_env(tokens)

    assert json.loads(result) == {"status": "completed", "exit_code": 0}
    argv = captured[0][0]
    assert argv[argv.index("--target") + 1] == "telegram:-100123:7"


def test_bound_context_wins_over_conflicting_process_environment(monkeypatch, tmp_path):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    context = SessionContext(
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="333",
            chat_type="dm",
            user_id="333",
        ),
        connected_platforms=[Platform.TELEGRAM],
        home_channels={},
    )
    captured = []

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "999")
    monkeypatch.setattr(worker, "check_threadwire_requirements", lambda: True)
    monkeypatch.setattr(
        worker.subprocess,
        "Popen",
        lambda argv, **kwargs: captured.append(argv) or SimpleNamespace(
            pid=123, stdin=None, poll=lambda: 0, wait=lambda timeout=None: 0
        ),
    )

    tokens = runner._set_session_env(context)
    try:
        result = worker.launch_telegram_coding_worker({
            "provider": "claude", "prompt": "work", "cwd": str(tmp_path)
        })
    finally:
        runner._clear_session_env(tokens)

    assert json.loads(result)["status"] == "completed"
    assert captured[0][captured[0].index("--target") + 1] == "telegram:333"
