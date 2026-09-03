from types import SimpleNamespace

import pytest

import cli


@pytest.fixture(autouse=True)
def reset_single_query_finalize_state(monkeypatch):
    monkeypatch.setattr(cli, "_single_query_finalize_attempted_session_ids", set())
    monkeypatch.setattr(cli, "_cleanup_done", False)




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
            # Mirrors the real chat(): the turn's raw result is published
            # for the single-query exit-code path.
            self._last_turn_result = {
                "final_response": "done",
                "completed": True,
            }
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


def _run_human_single_query(monkeypatch, calls, turn_result):
    """Drive main()'s human-facing ``chat -q`` branch with a stub CLI.

    Returns the ``SystemExit`` code, or ``None`` when main() returned
    normally (the success path falls through instead of exiting).
    """
    import cli as cli_mod

    class _Console:
        def print(self, *_args, **_kwargs):
            pass

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.console = _Console()
            self.session_id = "human-session"
            self.agent = SimpleNamespace(
                session_id="human-session",
                platform="cli",
            )

        def _claim_active_session(self, surface, *, stderr=False):
            return True

        def _show_security_advisories(self):
            pass

        def chat(self, query, images=None):
            calls.append(("chat", query))
            self._last_turn_result = turn_result
            return (turn_result or {}).get("final_response", "")

        def _print_exit_summary(self, clear_screen=True):
            calls.append("summary")

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        cli_mod,
        "_finalize_single_query",
        lambda fake_cli: calls.append(("finalize", fake_cli.session_id)),
    )

    try:
        cli_mod.main(query="hello", quiet=False, toolsets="terminal")
    except SystemExit as exc:
        return exc.code
    return None


def test_human_single_query_failed_turn_exits_nonzero(monkeypatch):
    """A failed turn on the ``-q`` path must not look like success.

    This branch is what the kanban dispatcher spawns; before the fix it
    returned 0 for every provider failure, so the dispatcher recorded a
    protocol violation ("worker exited cleanly without kanban_complete")
    and burned one of the card's failure-counter lives per attempt.
    """
    calls = []
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    code = _run_human_single_query(
        monkeypatch,
        calls,
        {
            "final_response": "",
            "failed": True,
            "error": "provider failed",
            "failure_reason": "api_error",
        },
    )

    assert code == 1
    # The exit code must not cost us session finalization.
    assert calls[-1] == ("finalize", "human-session")


def test_human_single_query_quota_wall_exits_tempfail_for_kanban(monkeypatch):
    """A quota wall in a kanban worker exits 75, not 0 and not 1."""
    from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE

    calls = []
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_d16778b1")

    code = _run_human_single_query(
        monkeypatch,
        calls,
        {
            "final_response": "",
            "failed": True,
            "error": "You've hit your session limit",
            "failure_reason": "rate_limit",
        },
    )

    assert code == KANBAN_RATE_LIMIT_EXIT_CODE
    assert calls[-1] == ("finalize", "human-session")


def test_human_single_query_successful_turn_does_not_exit(monkeypatch):
    """Success still falls through to a normal return (no SystemExit)."""
    calls = []
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    code = _run_human_single_query(
        monkeypatch,
        calls,
        {"final_response": "done", "completed": True},
    )

    assert code is None
    assert calls[-1] == ("finalize", "human-session")
