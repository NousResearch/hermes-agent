"""/agents links registered processes to their structured conversation origin."""

from contextlib import contextmanager
import json
import re
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session import SessionSource, build_session_key
from tests.waiting import wait_until
from tools.process_registry import ProcessRegistry


@pytest.fixture
def registry(monkeypatch):
    registry = ProcessRegistry()
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    return registry


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._background_tasks = set()
    runner._session_key_for_source = lambda source: build_session_key(source, profile=source.profile)
    return runner


@contextmanager
def _registered_process(registry, key, cwd, *, owner="foreground-task"):
    # stdin gates the real child; the timer bounds it even if the test aborts.
    proc = subprocess.Popen(
        [sys.executable, "-c", "import os, sys, threading; "
         "threading.Timer(30, lambda: os._exit(0)).start(); sys.stdin.read(1)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        session = registry.adopt_local(
            proc, command="test command", cwd=str(cwd), task_id=owner,
            owner_task_id=owner, session_key=key, notify_on_complete=False,
        )
        yield session
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        proc.stdin.close()
        wait_until(lambda: not registry.list_sessions() or all(
            row["status"] == "exited" for row in registry.list_sessions()
        ), what="registered process exited")


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [None, "research"])
@pytest.mark.parametrize("thread,prospective,destination", [
    ("222222222222222222", "333333333333333333", "222222222222222222"),
    (None, "333333333333333333", "333333333333333333"),
    (None, None, "111111111111111111"),
])
async def test_registered_process_links_inactive_conversation(registry, tmp_path, profile, thread, prospective, destination):
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="111111111111111111", thread_id=thread,
        prospective_thread_id=prospective, parent_chat_id="444444444444444444", profile=profile,
    )
    runner = _runner()
    key = runner._session_key_for_source(source)
    runner._cache_session_source(key, source)
    with _registered_process(registry, key, tmp_path) as session:
        rows = registry.list_sessions()
        assert rows[0].get("session_key") == key
        out = await runner._handle_agents_command(SimpleNamespace(source=source))
        row = next(line for line in out.splitlines() if session.id in line)
        prefix, uptime, command, label = row.split(" · ")
        assert prefix == f"- `{session.id}`"
        assert re.fullmatch(r"(?:\d+m )?\d+s", uptime)
        assert command == "`test command`"
        assert label == f"<#{destination}>"
    assert not runner._running_agents
    assert key not in out
    assert out.count("<#") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["missing-cache", "missing-profile", "empty-key", "empty-destination", "local-source", "telegram-request"])
async def test_process_origin_fallback(registry, tmp_path, case):
    source = SessionSource(
        platform=Platform.LOCAL if case == "local-source" else Platform.DISCORD,
        chat_id="111111111111111111", thread_id="222222222222222222", profile="research",
    )
    runner = _runner()
    key = "" if case == "empty-key" else runner._session_key_for_source(source)
    if case == "empty-destination":
        runner._cache_session_source(key, SessionSource(platform=Platform.DISCORD, chat_id=""))
    elif case == "missing-profile":
        runner._cache_session_source(build_session_key(source), source)
    elif case != "missing-cache":
        runner._cache_session_source(key, source)
    request = SimpleNamespace(source=SessionSource(
        platform=Platform.TELEGRAM if case == "telegram-request" else Platform.DISCORD, chat_id="555555555555555555",
    ))
    with _registered_process(registry, key, tmp_path) as session:
        out = await runner._handle_agents_command(request)
        row = next(line for line in out.splitlines() if session.id in line)
    suffix = f" · `{key}`" if key and case != "telegram-request" else ""
    prefix, uptime, rest = row.split(" · ", 2)
    assert prefix == f"- `{session.id}`"
    assert re.fullmatch(r"(?:\d+m )?\d+s", uptime)
    assert rest == f"`test command`{suffix}"
    assert "<#" not in out


@pytest.mark.asyncio
@pytest.mark.platforms("posix")
async def test_child_terminal_preserves_conversation_across_profile_switches(registry, tmp_path, monkeypatch):
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.delegate_tool_child_run import _ChildRun
    from tools.terminal_tool import cleanup_all_environments, terminal_tool

    # This test exercises origin propagation, not host cgroup admission.
    monkeypatch.setattr("tools.process_registry._build_systemd_scope_argv", lambda argv, **kwargs: argv)
    command = shlex.join([sys.executable, "-c", "import threading; threading.Event().wait(30)"])
    runner = _runner()
    sessions = []

    def child_conversation(**kwargs):
        return json.loads(terminal_tool(
            command=command, background=True, force=True, task_id=kwargs["task_id"], workdir=str(tmp_path),
        ))

    try:
        for profile, thread in [("research", "222222222222222222"), ("other", "333333333333333333"), ("research", "222222222222222222")]:
            source = SessionSource(platform=Platform.DISCORD, chat_id="111111111111111111", thread_id=thread, profile=profile)
            key = runner._session_key_for_source(source)
            runner._cache_session_source(key, source)
            tokens = set_session_vars(platform="discord", session_key=key, profile=profile)
            child_id = f"child-{len(sessions)}"
            child = SimpleNamespace(session_id=child_id, run_conversation=child_conversation)
            try:
                result, error, deferred = _ChildRun(
                    child=child, parent_agent=None, task_index=0, goal="spawn process",
                    subagent_id=None, child_progress_cb=None, child_task_id=child_id,
                ).await_child()
            finally:
                clear_session_vars(tokens)
            assert error is None and not deferred, error
            assert result.get("error") is None, result
            sessions.append(registry.get(result["session_id"]))
            assert sessions[-1].session_key == key
            assert sessions[-1].owner_task_id == child_id
            out = await runner._handle_agents_command(SimpleNamespace(source=source))
            row = next(line for line in out.splitlines() if sessions[-1].id in line)
            assert row.endswith(f" · <#{thread}>")
            assert row.startswith(f"- `{sessions[-1].id}` · ")
    finally:
        for session in sessions:
            registry.kill_process(session.id)
        cleanup_all_environments()
