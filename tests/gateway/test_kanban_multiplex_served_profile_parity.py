"""Standalone-vs-served parity for the kanban dispatcher and notifier under
``gateway.multiplex_profiles``: a worker spawned for served profile X gets the env a standalone
``hermes -p X`` dispatcher would build, and X's notifications are rendered/filtered under X's
config.
"""

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.secret_scope import set_multiplex_active
from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_notify as kbn
from hermes_constants import get_hermes_home


@pytest.fixture
def served(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    alpha = root / "profiles" / "alpha"
    alpha.mkdir(parents=True)
    (root / ".env").write_text("HERMES_MODEL=default-model\nTERMINAL_ENV=docker\n")
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\nterminal:\n  backend: docker\n")
    (alpha / ".env").write_text("")
    (alpha / "config.yaml").write_text("display:\n  language: zh\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_MODEL", "default-model")
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    set_multiplex_active(True)
    try:
        yield SimpleNamespace(root=root, alpha=alpha)
    finally:
        set_multiplex_active(False)


def test_worker_for_served_profile_gets_its_own_env_and_toolset_pin(served, monkeypatch):
    """The dispatcher (root context) spawns alpha's worker: no launch-profile settings leak into the
    child and the ``--toolsets`` pin (whose probes read credentials) is resolved under alpha's scope."""
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="alpha")
        task = kb.get_task(conn, tid)
    finally:
        conn.close()

    spawned = {}

    def fake_popen(argv, **kwargs):
        spawned["argv"], spawned["env"] = argv, kwargs["env"]
        return SimpleNamespace(pid=4242)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(kbd, "_open_worker_log", lambda task, board: open("/dev/null", "w"))
    monkeypatch.setattr(kbd, "_hermes_argv", lambda: ["hermes"], raising=False)
    kbd._default_spawn(task, str(served.alpha), board=None)

    env = spawned["env"]
    assert env["HERMES_HOME"] == str(served.alpha)
    assert "HERMES_MODEL" not in env and "TERMINAL_ENV" not in env
    assert "--toolsets" in spawned["argv"]


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        from gateway.media_policy import media_delivery_strict
        self.sent.append({"text": text, "home": str(get_hermes_home()), "strict": media_delivery_strict()})
        return SimpleNamespace(success=True, error=None)

    async def handle_message(self, event):
        event._gateway_accepted = True


def test_notifier_pings_run_under_the_subscribers_profile(served, monkeypatch):
    """A subscription owned by served alpha is pinged with alpha's home active, so its media policy
    and display language apply — alpha's own adapter was already selected before this fix."""
    (served.alpha / "config.yaml").write_text("display:\n  language: zh\ngateway:\n  strict: true\n")
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="notify parity", assignee="alpha")
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="1001", notifier_profile="alpha")
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.config = SimpleNamespace(multiplex_profiles=True, profile_routes=[])
    alpha_adapter = RecordingAdapter()
    runner.adapters = {Platform.TELEGRAM: RecordingAdapter()}
    runner._profile_adapters = {"alpha": {Platform.TELEGRAM: alpha_adapter}}
    runner._primary_profile_name = "default"
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    runner._profile_failed_platforms = {}

    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    asyncio.run(runner._kanban_notifier_watcher(interval=1))

    assert len(alpha_adapter.sent) == 1
    assert alpha_adapter.sent[0]["home"] == str(served.alpha)
    assert alpha_adapter.sent[0]["strict"] is True
