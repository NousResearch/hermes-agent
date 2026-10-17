"""`/agents` must list the `/bg` background sessions of this CLI process.

A `/bg` task runs as a thread inside the CLI (recorded in `_background_tasks` /
`_background_task_meta`), so it is in neither the process registry nor the async
delegation registry that `/agents` already reports. Before this, a user who started a
`/bg` task saw "zero" from `/agents` while the status bar still counted it.
"""

import threading
import time

from cli import HermesCLI  # imported at collection time like the other CLI tests
from hermes_cli.cli_commands_session_tools import CLICommandsSessionToolsMixin, _background_session_lines


class _Cli(CLICommandsSessionToolsMixin):
    def __init__(self):
        self._background_tasks = {}
        self._background_task_meta = {}
        self._agent_running = False


class _Agent:
    def __init__(self, snap):
        self._snap = snap

    def get_activity_summary(self):
        return dict(self._snap)


def _no_processes(monkeypatch):
    """No local processes, no gateway view, no async delegations: only the /bg rows remain."""
    import tools.process_registry as pr
    from hermes_cli import gateway_tools
    monkeypatch.setattr(pr.process_registry, "list_sessions", list)
    monkeypatch.setattr(gateway_tools, "_view", None)
    monkeypatch.setattr("hermes_cli.cli_commands_session_tools._probe", lambda *a, **k: [])


def _live_thread():
    stop = threading.Event()
    t = threading.Thread(target=stop.wait, daemon=True)
    t.start()
    return t, stop


def test_no_background_sessions_prints_nothing():
    assert _background_session_lines(_Cli()) == []


def test_background_session_listed_with_activity():
    cli = _Cli()
    thread, stop = _live_thread()
    try:
        now = time.time()
        cli._background_tasks["bg_154924_897b31"] = thread
        cli._background_task_meta["bg_154924_897b31"] = {
            "number": 1, "prompt": "Surface granular viewer data in the console\n\nadd device dims",
            "started_at": now - 754,
            "agent": _Agent({"api_call_count": 58, "current_tool": "terminal", "seconds_since_activity": 3.2}),
        }
        lines = _background_session_lines(cli, now=now)
    finally:
        stop.set()
    text = "\n".join(lines)
    assert "Background sessions (/bg): 1" in text
    assert "#1 bg_154924_897b31 · running 12m" in text
    assert "58 api calls" in text and "in terminal" in text and "last activity 3s ago" in text
    # the prompt is shown on one line, newlines collapsed
    assert '"Surface granular viewer data in the console add device dims"' in text
    assert "approval prompts appear in this window" in text


def test_background_session_before_agent_exists_and_after_thread_ends():
    cli = _Cli()
    finished = threading.Thread(target=lambda: None)
    finished.start()
    finished.join()
    cli._background_tasks["bg_a"] = finished  # done() has not popped it yet
    cli._background_task_meta["bg_a"] = {"number": 2, "prompt": "p", "started_at": time.time(), "agent": None}
    lines = _background_session_lines(cli)
    assert "#2 bg_a · finishing" in lines[1]
    assert "starting" in lines[1]  # no activity snapshot yet


def test_agents_command_includes_background_sessions(monkeypatch):
    printed = []
    monkeypatch.setattr("hermes_cli.cli_commands_session_tools._cp", lambda *ls: printed.extend(ls))
    _no_processes(monkeypatch)
    cli = _Cli()
    thread, stop = _live_thread()
    try:
        cli._background_tasks["bg_x"] = thread
        cli._background_task_meta["bg_x"] = {"number": 3, "prompt": "hello", "started_at": time.time(),
                                             "agent": _Agent({"api_call_count": 1, "current_tool": None,
                                                              "seconds_since_activity": 0.0})}
        cli._handle_agents_command()
    finally:
        stop.set()
    text = "\n".join(printed)
    assert "Background sessions (/bg): 1" in text
    assert "#3 bg_x · running" in text and "between turns" in text
    assert "Agent: idle" in text


def test_bg_handler_registers_task_for_agents_and_clears_it_when_done(monkeypatch):
    """Through the real /bg handler: while the side agent runs, /agents lists the task with its
    live activity; once the task finishes, both maps are empty again."""
    import sys
    import types

    release = threading.Event()
    started = threading.Event()

    class FakeAgent:
        def __init__(self, *a, **kw):
            self._print_fn = None
            self.thinking_callback = None

        def get_activity_summary(self):
            return {"api_call_count": 7, "current_tool": "terminal", "seconds_since_activity": 1.0}

        def run_conversation(self, user_message, task_id=None, **kw):
            started.set()
            release.wait(10)
            return {"final_response": "done: " + user_message}

        def close(self):
            pass

    # the handler imports AIAgent lazily; importing the real run_agent bootstraps the real home
    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=FakeAgent))
    cli = HermesCLI.__new__(HermesCLI)
    cli._background_tasks = {}
    cli._background_task_meta = {}
    cli._background_task_counter = 0
    cli._agent_running = False
    cli._spinner_text = ""
    cli._app = None
    cli.max_turns = 5
    cli.enabled_toolsets = None
    cli.reasoning_config = None
    cli.service_tier = None
    cli._session_db = None
    for attr in ("_sudo_password_callback", "_approval_callback", "_vault_unlock_callback",
                 "_vault_save_login_callback", "_vault_code_callback", "_secret_capture_callback"):
        setattr(cli, attr, None)
    from hermes_cli import cli_commands_background as bg, cli_commands_mixin as m
    for kw, attr in bg._BG_PROVIDER_KWARGS.items():
        setattr(cli, attr, None)
    cli._ensure_runtime_credentials = lambda: True
    cli._resolve_turn_agent_config = lambda prompt: {"model": "m", "runtime": {}, "request_overrides": None}
    cli._ring_bell = lambda **kw: None
    monkeypatch.setattr(bg, "_cp", lambda *ls: None)
    monkeypatch.setattr(m, "_print_side_result_panel", lambda *a, **kw: None)  # _side_worker reads it there
    printed = []
    monkeypatch.setattr("hermes_cli.cli_commands_session_tools._cp", lambda *ls: printed.extend(ls))
    _no_processes(monkeypatch)
    from hermes_cli import gateway_side
    monkeypatch.setattr(gateway_side, "connection_for", lambda cli_obj: None)

    cli._handle_background_command("/bg count the viewers")
    assert started.wait(5), "side agent did not start"
    (task_id,) = cli._background_tasks
    assert cli._background_task_meta[task_id]["number"] == 1
    assert cli._background_task_meta[task_id]["prompt"] == "count the viewers"
    assert isinstance(cli._background_task_meta[task_id]["agent"], FakeAgent)

    cli._handle_agents_command()
    text = "\n".join(printed)
    assert "Background sessions (/bg): 1" in text
    assert f"#1 {task_id} · running" in text and "7 api calls" in text and "in terminal" in text
    assert '"count the viewers"' in text

    release.set()
    cli._background_tasks[task_id].join(5)
    deadline = time.time() + 5
    while cli._background_tasks and time.time() < deadline:
        time.sleep(0.05)
    assert cli._background_tasks == {} and cli._background_task_meta == {}
    printed.clear()
    cli._handle_agents_command()
    assert "Background sessions" not in "\n".join(printed)
