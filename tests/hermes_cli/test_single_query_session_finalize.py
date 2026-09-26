from types import SimpleNamespace
from typing import Any

import pytest

import cli


@pytest.fixture(autouse=True)
def reset_single_query_finalize_state(monkeypatch):
    monkeypatch.setattr(cli, "_single_query_finalize_attempted_session_ids", set())
    monkeypatch.setattr(cli, "_cleanup_done", False)




def test_finalize_single_query_releases_lease_before_cleanup(monkeypatch):
    """Settlement (finalize hook) runs before the release, the release before
    cleanup; a cleanup failure can never skip or precede the release."""
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

    assert calls == ["finalize", "release", "cleanup"]


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

    assert calls == ["finalize", "release", "cleanup"]


def test_finalize_single_query_frees_the_lease_ahead_of_the_linger(tmp_path, monkeypatch):
    """The #118826 contract with a REAL lease: once the one-shot's turns are done,
    the session entry must be gone from the active-session registry BEFORE the
    bounded exit linger runs — a alive-but-idle process must not keep refusing
    deliveries ("Refused active session") for minutes after its turn ended."""
    from hermes_cli import active_sessions

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    lease, message = active_sessions.try_acquire_active_session(
        session_id="s1", surface="cli", config={}, metadata={"live_session_id": "s1"}
    )
    assert lease is not None, message

    def wait_only_if_released(_cli):
        entries = active_sessions._read_entries(active_sessions._state_path(home))
        # The linger step swallows exceptions by design, so RECORD instead of
        # asserting here — the assertion below is what must fail on unfixed code.
        seen["held_during_linger"] = any(
            str(e.get("session_id") or "") == "s1" for e in entries
        )

    seen: dict[str, bool] = {}
    fake_cli = SimpleNamespace(
        session_id="s1", agent=None, _release_active_session=lease.release
    )
    monkeypatch.setattr(cli, "_wait_for_oneshot_background_completions", wait_only_if_released)
    monkeypatch.setattr(cli, "_flush_one_shot_session_store", lambda _cli: None)
    monkeypatch.setattr(cli, "_notify_single_query_session_finalize", lambda _cli: None)
    monkeypatch.setattr(cli, "_run_cleanup", lambda **_k: None)

    cli._finalize_single_query(fake_cli)

    assert seen.get("held_during_linger") is False, (
        "session lease still held when the exit linger began"
    )
    entries = active_sessions._read_entries(active_sessions._state_path(home))
    assert not any(str(e.get("session_id") or "") == "s1" for e in entries)


def test_finalize_settles_session_before_a_successor_can_take_over(tmp_path, monkeypatch):
    """P1 handoff race: a successor that acquires the session DURING the
    predecessor's exit linger must never receive the predecessor's stale
    ``cli_close`` end-stamp — settlement (flush incl. end_session) must complete
    before the lease is released.

    Exercises the real seam with a real SessionDB and a real lease: the linger is
    patched to play the successor (acquire the freed lease, reopen the row), and
    the final assertion is that the successor's open interval survives the
    predecessor's whole finalize. On the unsafe order (release → linger → flush)
    the flush's unconditional ``end_session`` stamps the successor's interval and
    this test fails."""
    from hermes_cli import active_sessions
    from hermes_state import SessionDB

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "s1"
    db.create_session(sid, "cli")
    lease_a, message = active_sessions.try_acquire_active_session(
        session_id=sid, surface="cli", config={}, metadata={"live_session_id": sid}
    )
    assert lease_a is not None, message

    agent_a = SimpleNamespace(
        session_id=sid, _session_db=db, _session_messages=[],
        _persist_disabled=False, _persist_session=lambda *a, **k: None,
    )
    cli_a = SimpleNamespace(
        agent=agent_a, session_id=sid, conversation_history=[], _session_db=db,
        _release_active_session=lease_a.release,
    )

    taken_over: dict[str, Any] = {}

    def linger_with_takeover(_cli):
        # The successor arrives mid-linger. Early admission must hold (the point of
        # the PR)…
        lease_b, msg_b = active_sessions.try_acquire_active_session(
            session_id=sid, surface="cli", config={}, metadata={"live_session_id": sid}
        )
        assert lease_b is not None, f"successor could not acquire during the linger: {msg_b}"
        taken_over["lease_b"] = lease_b
        # …and it resumes the row (the predecessor's settlement already ended it).
        db.reopen_session(sid)

    monkeypatch.setattr(cli, "_wait_for_oneshot_background_completions", linger_with_takeover)
    monkeypatch.setattr(cli, "_notify_single_query_session_finalize", lambda _c: None)
    monkeypatch.setattr(cli, "_run_cleanup", lambda **_k: None)

    cli._finalize_single_query(cli_a)

    row = db.get_session(sid)
    assert row is not None and row["ended_at"] is None, (
        "successor's open interval was end-stamped by the predecessor's finalize"
    )
    taken_over["lease_b"].release()




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
            self._last_turn_result = {"final_response": "done", "completed": True}
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

    # The non-quiet one-shot path exits with the turn's outcome (0 here), like ``-Q``.
    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, toolsets="terminal")

    assert exc_info.value.code == 0
    assert ("chat", "hello", None) in calls
    assert calls[-1] == ("finalize", "single-query-session")


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
