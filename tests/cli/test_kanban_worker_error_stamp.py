"""Integration: kanban workers stamp fatal errors before exit.

Regression coverage for #46593 / #46985 — both dispatcher launch modes:

* ordinary workers: ``hermes chat -q …`` → ``HermesCLI.chat`` (no ``-Q``)
* goal-mode workers: ``chat -q … -Q`` → ``main``'s full-quiet branch

Both call ``_stamp_kanban_worker_error_if_needed``.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli


@pytest.fixture
def kanban_worker_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="worker task", assignee="test-worker")
        kb.claim_task(conn, tid)
        run_id = kb._current_run_id(conn, tid)
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    if run_id is not None:
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid


def _fake_cli_for_quiet(result):
    def run_conversation(*, user_message, conversation_history):
        return result

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.provider = "test-provider"
            self.model = "test-model"
            self.session_id = "kanban-worker-session"
            self.conversation_history = []
            self._active_agent_route_signature = "same-route"
            self.tool_progress_mode = "off"
            self.console = SimpleNamespace(print=lambda *a, **k: None)
            self.agent = SimpleNamespace(
                session_id="kanban-worker-session",
                platform="cli",
                quiet_mode=False,
                suppress_status_output=False,
                stream_delta_callback=object(),
                tool_gen_callback=object(),
                run_conversation=run_conversation,
            )

        def _claim_active_session(self, surface, *, stderr=False):
            return True

        def _ensure_runtime_credentials(self):
            return True

        def _resolve_turn_agent_config(self, effective_query):
            return {
                "signature": "same-route",
                "model": None,
                "runtime": None,
                "request_overrides": None,
            }

        def _init_agent(self, **kwargs):
            return True

        def _show_security_advisories(self):
            return None

        def _print_exit_summary(self, clear_screen=True):
            return None

        def chat(self, message, images=None):
            raise AssertionError(
                "full-quiet (-Q) path must not fall through to HermesCLI.chat"
            )

    return FakeCLI


def _drive_quiet_worker(monkeypatch, result):
    """Run ``cli.main(quiet=True)`` — the goal_mode / ``-Q`` launch path."""
    monkeypatch.setattr(cli, "HermesCLI", _fake_cli_for_quiet(result))
    monkeypatch.setattr(cli.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_finalize_single_query", lambda _cli: None)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(query="work task", quiet=True, toolsets="terminal")
    return exc_info.value.code


def _drive_default_q_worker(monkeypatch, result):
    """Run ``cli.main(query=..., quiet=False)`` — ordinary dispatcher launch.

    Dispatcher's ``_default_spawn`` uses ``chat -q`` without ``-Q`` unless
    ``task.goal_mode``. Fire maps that to ``query=…, quiet=False``, which
    calls ``HermesCLI.chat``.
    """
    called = {"chat": False}

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.provider = "test-provider"
            self.model = "test-model"
            self.session_id = "kanban-worker-session"
            self.conversation_history = []
            self.console = SimpleNamespace(print=lambda *a, **k: None)

        def _claim_active_session(self, surface, *, stderr=False):
            return True

        def _show_security_advisories(self):
            return None

        def _print_exit_summary(self, clear_screen=True):
            return None

        def chat(self, message, images=None):
            called["chat"] = True
            # Same call site HermesCLI.chat uses after run_conversation.
            cli._stamp_kanban_worker_error_if_needed(result)
            err = result.get("error") if isinstance(result, dict) else None
            if err and (
                result.get("failed") or result.get("partial")
            ):
                return f"Error: {err}"
            return result.get("final_response", "") if isinstance(result, dict) else ""

    monkeypatch.setattr(cli, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_finalize_single_query", lambda _cli: None)

    # Ordinary single-query path returns normally (rc=0) — that is what
    # produces the protocol-violation reap; the stamp must land first.
    cli.main(query="work kanban task", quiet=False, toolsets="terminal")
    assert called["chat"], (
        "ordinary non-goal-mode workers must reach HermesCLI.chat "
        "(quiet=False single-query path), not the full-quiet (-Q) branch"
    )
    return called


def test_chat_calls_shared_kanban_worker_error_stamp():
    """``HermesCLI.chat`` (default ``-q`` path) must invoke the shared stamp."""
    src = inspect.getsource(cli.HermesCLI.chat)
    assert "_stamp_kanban_worker_error_if_needed" in src, (
        "ordinary dispatcher workers use HermesCLI.chat; the shared stamp "
        "helper must be called there, not only in the full-quiet (-Q) branch"
    )


def test_quiet_branch_calls_shared_kanban_worker_error_stamp():
    """Goal-mode ``-Q`` path must also use the shared stamp helper."""
    src = inspect.getsource(cli.main)
    assert "_stamp_kanban_worker_error_if_needed" in src


def test_default_q_kanban_worker_stamps_error_before_exit(
    kanban_worker_env, monkeypatch
):
    """Ordinary ``chat -q`` (quiet=False) must persist worker errors on the open run."""
    from hermes_cli import kanban_db as kb

    worker_error = "litellm.NotFoundError: model 'gpt-bogus' does not exist"
    _drive_default_q_worker(
        monkeypatch,
        {"final_response": "", "error": worker_error, "failed": True, "partial": True},
    )

    conn = kb.connect()
    try:
        assert kb._open_run_error(conn, kanban_worker_env) == worker_error
    finally:
        conn.close()


def test_default_q_kanban_worker_skips_rate_limit_error(
    kanban_worker_env, monkeypatch
):
    """Rate-limit/billing failures must not stamp on the ordinary ``-q`` path."""
    from hermes_cli import kanban_db as kb

    _drive_default_q_worker(
        monkeypatch,
        {
            "final_response": "",
            "error": "litellm.RateLimitError: quota exhausted",
            "failed": True,
            "failure_reason": "rate_limit",
        },
    )

    conn = kb.connect()
    try:
        assert kb._open_run_error(conn, kanban_worker_env) is None
    finally:
        conn.close()


def test_quiet_kanban_worker_stamps_error_before_exit(kanban_worker_env, monkeypatch):
    """``cli.main(..., quiet=True)`` / ``-Q`` must still stamp worker errors."""
    from hermes_cli import kanban_db as kb

    worker_error = "litellm.NotFoundError: model 'gpt-bogus' does not exist"
    code = _drive_quiet_worker(
        monkeypatch,
        {"final_response": "", "error": worker_error, "failed": True},
    )
    assert code == 1

    conn = kb.connect()
    try:
        assert kb._open_run_error(conn, kanban_worker_env) == worker_error
    finally:
        conn.close()


def test_quiet_kanban_worker_skips_rate_limit_error(kanban_worker_env, monkeypatch):
    """A rate-limit/billing failure on ``-Q`` exits with the quota sentinel and must NOT stamp."""
    from hermes_cli import kanban_db as kb

    code = _drive_quiet_worker(
        monkeypatch,
        {
            "final_response": "",
            "error": "litellm.RateLimitError: quota exhausted",
            "failed": True,
            "failure_reason": "rate_limit",
        },
    )
    assert code == kb.KANBAN_RATE_LIMIT_EXIT_CODE

    conn = kb.connect()
    try:
        assert kb._open_run_error(conn, kanban_worker_env) is None
    finally:
        conn.close()


def test_default_spawn_omits_Q_unless_goal_mode(kanban_worker_env, monkeypatch):
    """Ordinary dispatcher workers must launch ``chat -q`` without ``-Q``.

    ``-Q`` is only appended for ``goal_mode``; if ordinary workers required
    ``-Q`` to stamp errors, #46593 would still reproduce (sweeper review).
    """
    from hermes_cli import kanban_db as kb

    captured = {}

    class FakeProc:
        pid = 424242

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    conn = kb.connect()
    try:
        task = kb.get_task(conn, kanban_worker_env)
        workspace = kb.resolve_workspace(task)
        assert task.goal_mode is False or task.goal_mode is None
        pid = kb._default_spawn(task, str(workspace))
        assert pid == 424242
    finally:
        conn.close()

    cmd = captured["cmd"]
    assert "chat" in cmd
    assert "-q" in cmd
    assert "-Q" not in cmd, f"ordinary workers must not get -Q: {cmd}"
