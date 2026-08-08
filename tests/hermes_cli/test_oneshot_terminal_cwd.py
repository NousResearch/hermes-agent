from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _restore_process_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep production chdir calls from leaking into later tests."""
    monkeypatch.chdir(Path.cwd())


def _session_db(workspace: Path, *, events: list[object]):
    class FakeSessionDB:
        def get_session(self, session_id):
            if session_id == "session-1":
                return {"id": session_id, "cwd": str(workspace)}
            return None

        def resolve_session_by_title(self, _title):
            return None

        def resolve_resume_session_id(self, session_id):
            return session_id

        def get_resume_conversations(self, _session_id):
            return ([{"role": "user", "content": "prior context"}], [])

        def reopen_session(self, session_id):
            events.append(("reopen", session_id, os.environ.get("TERMINAL_CWD")))

        def close(self):
            events.append("close")

    return FakeSessionDB()


def test_resume_publishes_terminal_cwd_before_session_reopen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import hermes_cli.oneshot as oneshot

    workspace = tmp_path / "recorded-workspace"
    workspace.mkdir()
    events: list[object] = []
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "stale-workspace"))

    session_id, history = oneshot._load_oneshot_resume(
        _session_db(workspace, events=events),
        resume_session_id="session-1",
        continue_last=None,
        restore_resume_cwd=True,
    )

    assert session_id == "session-1"
    assert history == [{"role": "user", "content": "prior context"}]
    assert Path.cwd() == workspace.resolve()
    assert os.environ["TERMINAL_CWD"] == str(workspace)
    assert events == [("reopen", "session-1", str(workspace))]


def test_missing_recorded_workspace_does_not_mutate_terminal_context_or_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import hermes_cli.oneshot as oneshot

    events: list[object] = []
    previous = str(tmp_path / "previous-workspace")
    monkeypatch.setenv("TERMINAL_CWD", previous)

    with pytest.raises(FileNotFoundError, match="working directory is unavailable"):
        oneshot._load_oneshot_resume(
            _session_db(tmp_path / "missing", events=events),
            resume_session_id="session-1",
            continue_last=None,
            restore_resume_cwd=True,
        )

    assert os.environ["TERMINAL_CWD"] == previous
    assert events == []


def test_failed_chdir_does_not_publish_terminal_context_or_reopen_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import hermes_cli.oneshot as oneshot

    workspace = tmp_path / "recorded-workspace"
    workspace.mkdir()
    events: list[object] = []
    monkeypatch.setenv("TERMINAL_CWD", "/previous")

    def fail_chdir(_path: object) -> None:
        raise OSError("blocked")

    monkeypatch.setattr(os, "chdir", fail_chdir)

    with pytest.raises(RuntimeError, match="Failed to restore"):
        oneshot._load_oneshot_resume(
            _session_db(workspace, events=events),
            resume_session_id="session-1",
            continue_last=None,
            restore_resume_cwd=True,
        )

    assert os.environ["TERMINAL_CWD"] == "/previous"
    assert events == []


def test_no_restore_cwd_preserves_explicit_caller_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import hermes_cli.oneshot as oneshot

    workspace = tmp_path / "recorded-workspace"
    workspace.mkdir()
    events: list[object] = []
    previous = str(tmp_path / "intentional-caller-workspace")
    monkeypatch.setenv("TERMINAL_CWD", previous)
    original_cwd = Path.cwd()

    oneshot._load_oneshot_resume(
        _session_db(workspace, events=events),
        resume_session_id="session-1",
        continue_last=None,
        restore_resume_cwd=False,
    )

    assert Path.cwd() == original_cwd
    assert os.environ["TERMINAL_CWD"] == previous
    assert events == [("reopen", "session-1", previous)]


def test_agent_construction_observes_restored_terminal_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import hermes_cli.oneshot as oneshot

    workspace = tmp_path / "recorded-workspace"
    workspace.mkdir()
    events: list[object] = []
    observed: list[str | None] = []
    db = _session_db(workspace, events=events)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "stale"))

    class FakeAgent:
        def __init__(self, **_kwargs):
            observed.append(os.environ.get("TERMINAL_CWD"))
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()
            self._session_messages = []

        def run_conversation(self, _prompt, **_kwargs):
            return {"final_response": "ok"}

        def shutdown_memory_provider(self, _messages=None):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=FakeAgent))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "test-model", "provider": "openai"}},
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "test",
            "base_url": "https://example.invalid",
            "provider": "openai",
            "requested_provider": "openai",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: db)
    monkeypatch.setattr(oneshot, "get_fallback_chain", lambda _cfg: [])

    result = oneshot._run_agent(
        "continue",
        model="test-model",
        provider="openai",
        use_config_toolsets=False,
        resume_session_id="session-1",
        restore_resume_cwd=True,
    )

    assert result == ("ok", {"final_response": "ok"})
    assert observed == [str(workspace)]