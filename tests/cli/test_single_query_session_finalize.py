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
