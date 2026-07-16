from types import SimpleNamespace

import pytest

import cli


@pytest.fixture(autouse=True)
def reset_single_query_finalize_state(monkeypatch):
    monkeypatch.setattr(cli, "_single_query_finalize_attempted_session_ids", set())
    monkeypatch.setattr(cli, "_cleanup_done", False)


def test_finalize_single_query_runs_cleanup_without_reemitting_finalize_before_release(monkeypatch):
    calls = []
    fake_cli = SimpleNamespace(_release_active_session=lambda: calls.append(("release", {})))

    def cleanup(**kwargs):
        calls.append(("cleanup", kwargs))

    monkeypatch.setattr(
        cli,
        "_notify_single_query_session_finalize",
        lambda _cli: calls.append(("finalize", {})),
    )
    monkeypatch.setattr(cli, "_run_cleanup", cleanup)

    cli._finalize_single_query(fake_cli)

    assert calls == [
        ("finalize", {}),
        ("cleanup", {"notify_session_finalize": False}),
        ("release", {}),
    ]


def test_finalize_single_query_releases_session_when_cleanup_fails(monkeypatch):
    calls = []
    fake_cli = SimpleNamespace(_release_active_session=lambda: calls.append("release"))

    def cleanup(**kwargs):
        calls.append("cleanup")
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(
        cli,
        "_notify_single_query_session_finalize",
        lambda _cli: calls.append("finalize"),
    )
    monkeypatch.setattr(cli, "_run_cleanup", cleanup)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        cli._finalize_single_query(fake_cli)

    assert calls == ["finalize", "cleanup", "release"]


def test_finalize_single_query_runs_cleanup_when_finalize_hook_fails(monkeypatch):
    calls = []
    fake_agent = SimpleNamespace(session_id="agent-session", platform="cli")
    fake_cli = SimpleNamespace(
        agent=fake_agent,
        session_id="cli-session",
        _release_active_session=lambda: calls.append("release"),
    )

    def invoke_hook(name, **kwargs):
        calls.append("finalize")
        raise RuntimeError("hook failed")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    monkeypatch.setattr(cli, "_run_cleanup", lambda **kwargs: calls.append("cleanup"))

    cli._finalize_single_query(fake_cli)

    assert calls == ["finalize", "cleanup", "release"]


def test_finalize_single_query_signal_window_does_not_reemit_during_atexit(monkeypatch):
    calls = []
    fake_agent = SimpleNamespace(session_id="agent-session", platform="cli")
    fake_cli = SimpleNamespace(
        agent=fake_agent,
        session_id="cli-session",
        _release_active_session=lambda: calls.append(("release", {})),
    )

    def invoke_hook(name, **kwargs):
        calls.append((name, kwargs))

    def interrupted_cleanup(**_kwargs):
        raise KeyboardInterrupt()

    expected_finalize = (
        "on_session_finalize",
        {
            "session_id": "agent-session",
            "platform": "cli",
            "reason": "shutdown",
        },
    )

    original_run_cleanup = cli._run_cleanup
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    monkeypatch.setattr(cli, "_run_cleanup", interrupted_cleanup)

    with pytest.raises(KeyboardInterrupt):
        cli._finalize_single_query(fake_cli)

    assert calls == [expected_finalize, ("release", {})]

    # Simulate later atexit cleanup after the interrupted one-shot path. The
    # active agent may already be unavailable by then.
    monkeypatch.setattr(cli, "_run_cleanup", original_run_cleanup)
    monkeypatch.setattr(cli, "_active_agent_ref", None)
    monkeypatch.setattr(cli, "_reset_terminal_input_modes_on_exit", lambda: None)
    monkeypatch.setattr(cli, "_cleanup_all_terminals", lambda: None)
    monkeypatch.setattr(cli, "_cleanup_all_browsers", lambda: None)
    monkeypatch.setattr("tools.mcp_tool.shutdown_mcp_servers", lambda: None)
    monkeypatch.setattr("agent.auxiliary_client.shutdown_cached_clients", lambda: None)

    cli._run_cleanup()

    assert calls == [expected_finalize, ("release", {})]


def test_notify_single_query_session_finalize_uses_agent_session(monkeypatch):
    calls = []
    fake_agent = SimpleNamespace(session_id="agent-session", platform="cli")
    fake_cli = SimpleNamespace(agent=fake_agent, session_id="cli-session")

    def invoke_hook(name, **kwargs):
        calls.append((name, kwargs))

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)

    cli._notify_single_query_session_finalize(fake_cli)

    assert calls == [
        (
            "on_session_finalize",
            {
                "session_id": "agent-session",
                "platform": "cli",
                "reason": "shutdown",
            },
        )
    ]


def test_human_single_query_main_finalizes_after_query(monkeypatch):
    calls = []

    import cli as cli_mod

    class _Console:
        def print(self, *_args, **_kwargs):
            calls.append("query-label")

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.console = _Console()
            self.session_id = "single-query-session"
            self.agent = SimpleNamespace(
                session_id="single-query-session",
                platform="cli",
            )

        def _claim_active_session(self, surface, *, stderr=False):
            calls.append(("claim", surface, stderr))
            return True

        def _show_security_advisories(self):
            calls.append("advisories")

        def chat(self, query, images=None):
            calls.append(("chat", query, images))
            return "done"

        def _print_exit_summary(self, clear_screen=True):
            calls.append("summary")

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli_mod,
        "_finalize_single_query",
        lambda fake_cli: calls.append(("finalize", fake_cli.session_id)),
    )

    cli_mod.main(query="hello", quiet=False, toolsets="terminal")

    assert calls == [
        ("claim", "cli", False),
        "query-label",
        "advisories",
        ("chat", "hello", None),
        "summary",
        ("finalize", "single-query-session"),
    ]


def test_kanban_billing_failure_preserves_error_and_uses_sentinel(monkeypatch):
    import cli as cli_mod
    from hermes_cli import kanban_db as kb

    calls = []
    provider_error = "HTTP 429: Insufficient balance. Please recharge."
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_billing")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "host:worker")
    monkeypatch.setattr(
        kb,
        "record_kanban_worker_failure_error",
        lambda task_id, claim_lock, error: calls.append(
            (task_id, claim_lock, error)
        ) or True,
    )

    exit_code = cli_mod._single_query_failure_exit_code(
        {
            "failed": True,
            "failure_reason": "billing",
            "error": provider_error,
        }
    )

    assert exit_code == kb.KANBAN_BILLING_BLOCKER_EXIT_CODE
    assert calls == [("t_billing", "host:worker", provider_error)]


def test_kanban_rate_limit_failure_uses_sentinel_without_persisting(monkeypatch):
    import cli as cli_mod
    from hermes_cli import kanban_db as kb

    calls = []
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_rate_limit")
    monkeypatch.setattr(
        kb,
        "record_kanban_worker_failure_error",
        lambda *args: calls.append(args) or True,
    )

    exit_code = cli_mod._single_query_failure_exit_code(
        {"failed": True, "failure_reason": "rate_limit", "error": "HTTP 429"}
    )

    assert exit_code == kb.KANBAN_RATE_LIMIT_EXIT_CODE
    assert calls == []


def test_goal_mode_midloop_billing_failure_reaches_sentinel(tmp_path, monkeypatch):
    import cli as cli_mod
    from hermes_cli import goals
    from hermes_cli import kanban_db as kb

    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    claim_lock = "host:goal-worker"
    provider_error = "HTTP 429: Insufficient balance. Please recharge."

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="finish the goal", assignee="a")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        conn.commit()
        assert kb.claim_task(conn, task_id, claimer=claim_lock) is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", claim_lock)
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *_args, **_kwargs: ("continue", "not done", False, False),
    )

    failure = {
        "failed": True,
        "failure_reason": "billing",
        "error": provider_error,
        "final_response": "Billing or credits exhausted",
    }
    calls = []
    fake_cli = SimpleNamespace(
        session_id="goal-session",
        conversation_history=[],
        agent=SimpleNamespace(
            session_id="goal-session",
            run_conversation=lambda **kwargs: calls.append(kwargs) or failure,
        ),
    )

    returned = cli_mod._run_kanban_goal_loop_q(fake_cli, "starting work")
    assert returned is failure
    assert len(calls) == 1
    assert cli_mod._single_query_failure_exit_code(returned) == (
        kb.KANBAN_BILLING_BLOCKER_EXIT_CODE
    )

    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "running"
        assert task.last_failure_error == provider_error


def test_quiet_single_query_main_finalizes_while_preserving_exit_code(monkeypatch):
    calls = []

    import cli as cli_mod

    def run_conversation(*, user_message, conversation_history):
        calls.append(("run", user_message, conversation_history))
        return {
            "final_response": "",
            "error": "provider failed",
            "failed": True,
        }

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.provider = "test-provider"
            self.model = "test-model"
            self.session_id = "quiet-session"
            self.conversation_history = []
            self._active_agent_route_signature = "same-route"
            self.agent = SimpleNamespace(
                session_id="quiet-session",
                platform="cli",
                quiet_mode=False,
                suppress_status_output=False,
                stream_delta_callback=object(),
                tool_gen_callback=object(),
                run_conversation=run_conversation,
            )

        def _claim_active_session(self, surface, *, stderr=False):
            calls.append(("claim", surface, stderr))
            return True

        def _ensure_runtime_credentials(self):
            calls.append("credentials")
            return True

        def _resolve_turn_agent_config(self, effective_query):
            calls.append(("resolve", effective_query))
            return {
                "signature": "same-route",
                "model": None,
                "runtime": None,
                "request_overrides": None,
            }

        def _init_agent(self, **kwargs):
            calls.append(("init", kwargs))
            return True

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli_mod,
        "_finalize_single_query",
        lambda fake_cli: calls.append(("finalize", fake_cli.session_id)),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 1
    assert ("claim", "cli", True) in calls
    assert ("run", "hello", []) in calls
    assert calls[-1] == ("finalize", "quiet-session")
