from __future__ import annotations

from types import SimpleNamespace

import pytest


class _FakeSessionDB:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def end_session(self, session_id: str, end_reason: str) -> None:
        self.calls.append((session_id, end_reason))


def test_finalize_single_query_uses_current_agent_session_id() -> None:
    """Single-query cleanup must close the live continuation session.

    Auto-compression can rotate ``agent.session_id`` during the turn.  The
    finalizer should therefore prefer the agent's current session id over the
    CLI object's initially-created session id.
    """
    import cli as cli_mod

    db = _FakeSessionDB()
    shell = SimpleNamespace(
        session_id="parent-session",
        agent=SimpleNamespace(session_id="continuation-session"),
        _session_db=db,
        _last_turn_interrupted=False,
    )

    cli_mod._finalize_single_query_session(shell, {"completed": True})

    assert db.calls == [("continuation-session", "single_query_complete")]


def test_finalize_single_query_records_failed_result() -> None:
    import cli as cli_mod

    db = _FakeSessionDB()
    shell = SimpleNamespace(
        session_id="single-query-session",
        agent=SimpleNamespace(session_id="single-query-session"),
        _session_db=db,
        _last_turn_interrupted=False,
    )

    cli_mod._finalize_single_query_session(shell, {"failed": True})

    assert db.calls == [("single-query-session", "single_query_failed")]


def test_finalize_single_query_records_partial_result_as_failed() -> None:
    import cli as cli_mod

    db = _FakeSessionDB()
    shell = SimpleNamespace(
        session_id="single-query-session",
        agent=SimpleNamespace(session_id="single-query-session"),
        _session_db=db,
        _last_turn_interrupted=False,
    )

    cli_mod._finalize_single_query_session(shell, {"partial": True})

    assert db.calls == [("single-query-session", "single_query_failed")]


def test_finalize_single_query_records_interrupted_result() -> None:
    import cli as cli_mod

    db = _FakeSessionDB()
    shell = SimpleNamespace(
        session_id="single-query-session",
        agent=SimpleNamespace(session_id="single-query-session"),
        _session_db=db,
        _last_turn_interrupted=True,
    )

    cli_mod._finalize_single_query_session(shell, {"failed": True})

    assert db.calls == [("single-query-session", "single_query_interrupted")]


def test_chat_preserves_failed_turn_result_for_single_query_finalizer() -> None:
    """Human ``chat -q`` needs the full turn result, not only rendered text."""
    import queue
    import cli as cli_mod

    result = {
        "final_response": "Error: provider failed",
        "messages": [],
        "api_calls": 1,
        "completed": False,
        "failed": True,
        "error": "provider failed",
    }

    class FakeAgent:
        session_id = "single-query-session"
        max_iterations = 90

        def run_conversation(self, **_kwargs):
            return result

        def interrupt(self, _message=None) -> None:
            pass

    shell = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
    shell.session_id = "single-query-session"
    shell.agent = FakeAgent()
    shell._session_db = None
    shell.conversation_history = []
    shell._active_agent_route_signature = "same"
    shell._last_turn_interrupted = False
    shell._last_chat_result = {"failed": False}
    shell._ensure_runtime_credentials = lambda: True
    shell._resolve_turn_agent_config = lambda _message: {
        "signature": "same",
        "model": None,
        "runtime": None,
        "request_overrides": None,
    }
    shell._init_agent = lambda **_kwargs: True
    shell._preprocess_images_with_vision = lambda message, _images: message
    shell._reset_stream_state = lambda: None
    shell._flush_stream = lambda: None
    shell._scrollback_box_width = lambda *_, **__: 80
    shell._voice_speak_response_async = lambda _response: None
    shell._sudo_password_callback = lambda *_, **__: None
    shell._approval_callback = lambda *_, **__: None
    shell._secret_capture_callback = lambda *_, **__: None
    shell._pending_model_switch_note = None
    shell._pending_skills_reload_note = None
    shell._interrupt_queue = queue.Queue()
    shell._pending_input = queue.Queue()
    shell._clarify_state = None
    shell._clarify_freetext = False
    shell._should_exit = False
    shell._voice_tts = False
    shell._voice_mode = False
    shell._voice_continuous = False
    shell.show_timestamps = False
    shell.show_reasoning = False
    shell._stream_started = False
    shell._stream_box_opened = False
    shell.final_response_markdown = False
    shell.bell_on_complete = False
    shell.provider = ""
    shell.model = ""
    shell.base_url = ""
    shell.api_key = ""
    shell.api_mode = ""

    response = cli_mod.HermesCLI.chat(shell, "hello")

    assert response == "Error: provider failed"
    assert shell._last_chat_result is result


def test_quiet_single_query_main_finalizes_session(monkeypatch) -> None:
    """The machine-readable ``hermes chat -q -Q`` path exits via sys.exit()."""
    import cli as cli_mod

    db = _FakeSessionDB()

    class FakeAgent:
        session_id = "quiet-agent-session"

        def run_conversation(self, **_kwargs):
            return {"final_response": "done", "completed": True, "failed": False}

    class FakeCLI:
        def __init__(self, **_kwargs) -> None:
            self.session_id = "initial-cli-session"
            self.agent = FakeAgent()
            self._session_db = db
            self.conversation_history = []
            self._active_agent_route_signature = "same"
            self.tool_progress_mode = "all"
            self._last_turn_interrupted = False

        def _ensure_runtime_credentials(self) -> bool:
            return True

        def _resolve_turn_agent_config(self, _message):
            return {
                "signature": "same",
                "model": None,
                "runtime": None,
                "request_overrides": None,
            }

        def _init_agent(self, **_kwargs) -> bool:
            return True

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_args, **_kwargs: None)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 0
    assert db.calls == [("quiet-agent-session", "single_query_complete")]


def test_human_single_query_main_finalizes_session(monkeypatch) -> None:
    """The normal human-facing ``hermes chat -q`` path should also close DB state."""
    import cli as cli_mod

    db = _FakeSessionDB()

    class _Console:
        def print(self, *_args, **_kwargs) -> None:
            pass

    class FakeAgent:
        session_id = "human-agent-session"

    class FakeCLI:
        def __init__(self, **_kwargs) -> None:
            self.session_id = "initial-cli-session"
            self.agent = FakeAgent()
            self._session_db = db
            self.conversation_history = []
            self.console = _Console()
            self._last_turn_interrupted = False

        def _show_security_advisories(self) -> None:
            pass

        def chat(self, _query, images=None):
            assert images is None
            return "done"

        def _print_exit_summary(self) -> None:
            pass

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_args, **_kwargs: None)

    cli_mod.main(query="hello", quiet=False, toolsets="terminal")

    assert db.calls == [("human-agent-session", "single_query_complete")]


def test_human_single_query_main_uses_failed_chat_result(monkeypatch) -> None:
    """A failed turn with a rendered error string should close as failed."""
    import cli as cli_mod

    db = _FakeSessionDB()

    class _Console:
        def print(self, *_args, **_kwargs) -> None:
            pass

    class FakeAgent:
        session_id = "human-agent-session"

    class FakeCLI:
        def __init__(self, **_kwargs) -> None:
            self.session_id = "initial-cli-session"
            self.agent = FakeAgent()
            self._session_db = db
            self.conversation_history = []
            self.console = _Console()
            self._last_turn_interrupted = False
            self._last_chat_result = None

        def _show_security_advisories(self) -> None:
            pass

        def chat(self, _query, images=None):
            assert images is None
            self._last_chat_result = {
                "final_response": "Error: provider failed",
                "completed": False,
                "failed": True,
            }
            return "Error: provider failed"

        def _print_exit_summary(self) -> None:
            pass

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_args, **_kwargs: None)

    cli_mod.main(query="hello", quiet=False, toolsets="terminal")

    assert db.calls == [("human-agent-session", "single_query_failed")]
