"""``hermes chat -q … --format stream-json`` emits a parseable JSONL event stream and nothing else on stdout."""

import json
import signal

import pytest

from hermes_cli.stream_json import StreamJsonEmitter


def _events(capsys):
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line]


def test_emitter_event_stream_is_valid_jsonl(capsys):
    emitter = StreamJsonEmitter(model="test-model", session_id="s-1")
    emitter.on_text_delta("hel")
    emitter.on_text_delta("\n  ")  # whitespace deltas are part of the answer and must be forwarded verbatim
    emitter.on_text_delta("lo")
    emitter.on_text_delta(None)  # the turn-end sentinel the agent sends
    emitter.on_text_delta("")
    emitter.on_tool_progress("tool.started", "read_file", "preview", {"path": "x"}, tool_call_id="call-a")
    emitter.on_tool_progress("tool.started", "read_file", "preview", {"path": "y"}, tool_call_id="call-b")
    emitter.on_tool_progress("reasoning.available", "_thinking", "hmm", None)  # not part of the protocol
    emitter.on_tool_progress("tool.completed", "read_file", None, None, tool_call_id="call-b", result="y")
    emitter.on_tool_progress("tool.completed", "read_file", None, None, tool_call_id="call-a", duration=0.5,
                             is_error=False, result="x" * 6000)
    code = emitter.emit_result({"final_response": "", "failed": True, "error": "boom", "input_tokens": 3}, exit_code=0)

    events = _events(capsys)
    assert [e["type"] for e in events] == ["system", "text", "text", "text", "tool_use", "tool_use", "tool_result",
                                           "tool_result", "result"]
    assert "".join(e["text"] for e in events if e["type"] == "text") == "hel\n  lo"
    assert events[0]["subtype"] == "init" and events[0]["model"] == "test-model"
    assert events[4]["input"] == {"path": "x"} and events[4]["tool_call_id"] == "call-a"
    # concurrent same-name calls: each result pairs with its own start, not the last-started one
    assert [e["tool_call_id"] for e in events if e["type"] == "tool_result"] == ["call-b", "call-a"]
    assert events[6]["duration_ms"] < 500
    assert events[7]["duration_ms"] == 500 and events[7]["output"].endswith("...") and len(events[7]["output"]) == 5003
    assert code == 1 and events[-1] == {**events[-1], "exit_code": 1, "error": "boom", "session_id": "s-1"}
    assert events[-1]["tokens"]["input"] == 3
    assert all("timestamp" in e for e in events)


def _run_stream_json_chat(monkeypatch, capsys, run_conversation, credentials_ok=True,
                          argv=None, chat_result=None, resolves_tui=False):
    """parser → cmd_chat → cli.main → single-query path with a deterministic fake agent/CLI."""
    import cli
    import hermes_cli.main as cli_entry
    from hermes_cli._parser import build_top_level_parser

    class _Console:
        def print(self, *_a, **_k):
            pass

    class FakeAgent:
        model = "test-model"
        session_id = "session-123"

        def run_conversation(self, **_kwargs):
            return run_conversation(self)

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.session_id = "session-123"
            self.conversation_history = []
            self.agent = None
            self._active_agent_route_signature = None
            self.tool_progress_mode = None
            self._last_turn_result = None
            self.console = _Console()

        def _claim_active_session(self, *_a, **_k):
            return True

        def _ensure_runtime_credentials(self):
            return credentials_ok

        def _resolve_turn_agent_config(self, _query):
            return {"signature": "r", "model": None, "runtime": None, "request_overrides": None}

        def _init_agent(self, **_kwargs):
            self.agent = FakeAgent()
            return True

        def _show_security_advisories(self):
            pass

        def _print_exit_summary(self, **_k):
            pass

        def chat(self, _query, images=None):
            print("human output")  # must never reach stdout under stream-json
            self._last_turn_result = chat_result
            return (chat_result or {}).get("final_response") or ""

    monkeypatch.setattr(cli, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli, "_finalize_single_query", lambda _cli: None)
    monkeypatch.setattr(cli, "_emit_interrupted_session_end", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_start_worktree_setup", lambda *_a, **_k: None)
    monkeypatch.setattr(cli.atexit, "register", lambda *_a, **_k: None)
    monkeypatch.setattr(signal, "signal", lambda *_a, **_k: None)
    monkeypatch.setattr(cli_entry, "_resolve_use_tui",
                        (lambda _args: False) if resolves_tui
                        else (lambda _args: pytest.fail("TUI resolution consulted")))
    monkeypatch.setattr(cli_entry, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(cli_entry, "_start_chat_background_prefetch", lambda: None)
    monkeypatch.setattr(cli_entry, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(cli_entry, "_confirm_startup_expensive_model_override", lambda _a: None)
    monkeypatch.setattr(cli_entry, "_warn_retired_xai_models", lambda: None)
    monkeypatch.setattr("hermes_cli.free_tier_bootstrap.run_bootstrap", lambda **_k: None)
    monkeypatch.setattr("hermes_cli.quiet_single_query.continue_quiet_notify_completions", lambda *_a, **_k: None)

    parser, _, _ = build_top_level_parser()
    args = parser.parse_args(argv or ["chat", "-q", "hello", "--format", "stream-json"])
    with pytest.raises(SystemExit) as exc_info:
        cli_entry.cmd_chat(args)
    if args.output_format == "stream-json":
        return exc_info.value.code, _events(capsys)
    return exc_info.value.code, capsys.readouterr()


def _ok_turn(agent):
    agent.stream_delta_callback("hello")
    agent.tool_progress_callback("tool.started", "read_file", "p", {"path": "f"})
    agent.tool_progress_callback("tool.completed", "read_file", None, None, duration=0.01, result="contents")
    return {"final_response": "hello", "failed": False}


def _error_only_turn(_agent):
    """A turn that carries an ``error`` without a ``failed`` flag: the shape runtimes like the
    codex app-server report (``result["error"]`` text) before flag normalization."""
    return {"final_response": "", "error": "provider unavailable"}


def test_emit_result_never_reports_success_for_an_errored_result(capsys):
    """The terminal envelope must not carry ``error`` while reporting ``exit_code: 0``;
    CI consumers key success off that field."""
    emitter = StreamJsonEmitter(model="m", session_id="s-err")
    code = emitter.emit_result({"final_response": "", "error": "provider unavailable"})
    events = _events(capsys)
    assert code != 0
    assert events[-1]["exit_code"] == code and events[-1]["exit_code"] != 0
    assert events[-1]["error"] == "provider unavailable"


def test_emit_result_explicit_exit_code_wins(capsys):
    """An explicit nonzero exit code is never folded down by the error heuristic."""
    emitter = StreamJsonEmitter(model="m", session_id="s-1")
    assert emitter.emit_result({"error": "boom"}, exit_code=5) == 5
    emitter2 = StreamJsonEmitter(model="m", session_id="s-2")
    assert emitter2.emit_result({"final_response": "ok"}) == 0
    results = [e for e in _events(capsys) if e["type"] == "result"]
    assert [e["exit_code"] for e in results] == [5, 0]


def test_single_query_exit_code_treats_a_bare_error_as_failure():
    """``_single_query_exit_code`` feeds the plain ``-q``/``-Q`` process exit and the turn
    report's exit_code field; an ``error``-carrying result is not a clean completion."""
    from cli import _single_query_exit_code
    assert _single_query_exit_code({"final_response": "", "error": "provider unavailable"}) == 1
    assert _single_query_exit_code({"final_response": "ok", "completed": True}) == 0


def _interrupted_turn(_agent):
    raise KeyboardInterrupt


@pytest.mark.parametrize("turn, credentials_ok, exit_code, types", [
    (_ok_turn, True, 0, ["system", "text", "tool_use", "tool_result", "result"]),
    (_interrupted_turn, True, 130, ["system", "result"]),
    (_error_only_turn, True, 1, ["system", "result"]),  # error without a failed flag still fails
    (_ok_turn, False, 1, ["system", "result"]),  # credentials fail before the agent exists
])
def test_chat_stream_json_implies_quiet_and_closes_with_result(monkeypatch, capsys, turn, credentials_ok, exit_code,
                                                                types):
    """No ``-Q`` needed; stdout is only JSONL; the stream always ends in a ``result`` carrying the exit code."""
    code, events = _run_stream_json_chat(monkeypatch, capsys, turn, credentials_ok=credentials_ok)
    assert code == exit_code
    assert [e["type"] for e in events] == types
    assert events[-1]["exit_code"] == exit_code and events[-1]["session_id"] == "session-123"


def test_chat_plain_single_query_exits_nonzero_on_an_errored_turn(monkeypatch, capsys):
    """The plain ``-q`` path maps ``_last_turn_result`` through ``_single_query_exit_code``;
    a turn carrying ``error`` must not exit 0 (scripts and the Kanban dispatcher read it)."""
    code, _ = _run_stream_json_chat(
        monkeypatch, capsys, _ok_turn, argv=["chat", "-q", "hello"],
        chat_result={"final_response": "", "error": "provider unavailable"}, resolves_tui=True,
    )
    assert code == 1


def test_chat_quiet_single_query_surfaces_an_errored_turn_on_stderr(monkeypatch, capsys):
    """``-Q`` (no emitter): an ``error``-carrying result with no reply exits nonzero AND prints the
    error on stderr; without the surfacing the failure would be invisible outside the exit code."""
    code, captured = _run_stream_json_chat(
        monkeypatch, capsys, _error_only_turn, argv=["chat", "-q", "hello", "-Q"],
        resolves_tui=True,
    )
    assert code == 1
    assert "provider unavailable" in captured.err


@pytest.mark.parametrize("argv, message", [
    (["chat", "--format", "stream-json"], "requires -q/--query"),
    (["chat", "-q", "hi", "--format", "stream-json", "--tui"], "cannot be used with --tui"),
    (["--tui", "chat", "-q", "hi", "--format", "stream-json"], "cannot be used with --tui"),
])
def test_chat_stream_json_rejects_interactive_combinations(monkeypatch, capsys, argv, message):
    import hermes_cli.main as cli_entry
    from hermes_cli._parser import build_top_level_parser

    monkeypatch.setattr(cli_entry, "_launch_tui", lambda *_a, **_k: pytest.fail("TUI launched"))
    parser, _, _ = build_top_level_parser()
    with pytest.raises(SystemExit) as exc_info:
        cli_entry.cmd_chat(parser.parse_args(argv))
    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err
