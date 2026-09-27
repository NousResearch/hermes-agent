"""One-shot output contracts through the real parser → chat → runner path.

The model and CLI lifecycle are fakes; parsing, routing, output and file I/O are real.
"""
import json
import signal
from types import SimpleNamespace

import pytest


@pytest.fixture
def chat(monkeypatch, capsys):
    import cli
    import hermes_cli.main as entry
    from hermes_cli._parser import build_top_level_parser

    state = SimpleNamespace(calls=[], started=0, results=[], credentials=True, correction_requests=[], corrections=[], after_turn=lambda agent: None)

    class Agent:
        model = "selected-model"
        provider = "test-provider"
        api_mode = "chat_completions"
        session_id = "structured-session"
        _cached_system_prompt = "Frozen system prompt"
        ephemeral_system_prompt = None
        prefill_messages = []
        _image_rejecting_models = set()
        _force_ascii_payload = False
        _current_turn_timestamp = 0
        _use_prompt_caching = False
        context_compressor = None
        tools = [{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object"}}}]

        def _should_sanitize_tool_calls(self):
            return False

        def _copy_reasoning_content_for_api(self, original, target):
            pass

        def _sanitize_api_messages(self, messages):
            return messages

        def _drop_thinking_only_and_merge_users(self, messages, **kwargs):
            return messages

        def _build_api_kwargs(self, messages, tools_for_api=None):
            return {"model": self.model, "messages": messages, "tools": tools_for_api if tools_for_api is not None else self.tools}

        def _get_transport(self):
            from agent.transports.chat_completions import ChatCompletionsTransport
            return ChatCompletionsTransport()

        def _build_assistant_message(self, response, finish_reason):
            return {"role": "assistant", "content": response.content, "finish_reason": finish_reason}

        def _persist_session(self, messages):
            self._session_messages = messages

        def _interruptible_api_call(self, request):
            state.correction_requests.append(request)
            result = state.corrections.pop(0)
            if isinstance(result, BaseException):
                raise result
            if not isinstance(result, str):
                return result
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=result, tool_calls=None), finish_reason="stop")], usage=None)

        def run_conversation(self, **kwargs):
            state.calls.append(kwargs)
            result = state.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            state.after_turn(self)
            return {"messages": [{"role": "user", "content": kwargs["user_message"]},
                                 {"role": "assistant", "content": result.get("final_response", "")}], **result}

    class CLI:
        def __init__(self, **kwargs):
            state.started += 1
            self.session_id = Agent.session_id
            self.model = Agent.model
            self.conversation_history = []
            self.agent = None
            self._active_agent_route_signature = None

        def _claim_active_session(self, *args, **kwargs):
            return True

        def _ensure_runtime_credentials(self):
            return state.credentials

        def _resolve_turn_agent_config(self, query):
            return {"signature": "r", "model": Agent.model, "runtime": None}

        def _init_agent(self, **kwargs):
            self.agent = Agent()
            state.agent = self.agent
            return True

        def run(self):
            pytest.fail("interactive UI launched")

    monkeypatch.setattr(cli, "HermesCLI", CLI)
    # Lifecycle fake has no session accounting; native-agent coverage below exercises it.
    monkeypatch.setattr("agent.turn_usage.record_response_usage", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_finalize_single_query", lambda cli: None)
    monkeypatch.setattr(cli, "_emit_interrupted_session_end", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_start_worktree_setup", lambda *a, **k: None)
    monkeypatch.setattr(cli.atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(entry, "_resolve_use_tui", lambda args: False)
    monkeypatch.setattr(entry, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(entry, "_start_chat_background_prefetch", lambda: None)
    monkeypatch.setattr(entry, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(entry, "_confirm_startup_expensive_model_override", lambda args: None)
    monkeypatch.setattr(entry, "_warn_retired_xai_models", lambda: None)
    monkeypatch.setattr("hermes_cli.free_tier_bootstrap.run_bootstrap", lambda **k: None)
    monkeypatch.setattr("hermes_cli.quiet_single_query.continue_quiet_notify_completions", lambda *a, **k: None)

    def run(*args, results=None):
        state.calls.clear()
        state.results = list(results or [{"final_response": "hello", "completed": True}])
        parser, _, _ = build_top_level_parser()
        with pytest.raises(SystemExit) as exit:
            try:
                entry.cmd_chat(parser.parse_args(["chat", *args]))
            except KeyboardInterrupt:
                pytest.fail("KeyboardInterrupt escaped without a terminal result")
        captured = capsys.readouterr()
        return exit.value.code, captured.out, captured.err

    state.run = run
    return state


def test_stream_init_keeps_resolved_model_and_session(chat):
    code, stdout, stderr = chat.run("-q", "hi", "--format", "stream-json")
    events = [json.loads(line) for line in stdout.splitlines()]
    assert code == 0
    assert events[0]["type"] == "system"
    assert events[0]["model"] == "selected-model"
    assert events[0]["session_id"] == "structured-session"
    assert len([event for event in events if event["type"] == "system"]) == 1


def test_json_format_emits_one_final_envelope(chat):
    code, stdout, stderr = chat.run("-q", "hello", "--format", "json")
    assert code == 0, stderr
    result = json.loads(stdout)
    assert result["type"] == "result"
    assert result["exit_code"] == code
    assert result["text"] == "hello"
    assert result["session_id"] == "structured-session"
    assert {"tokens", "duration_ms", "timestamp"} <= result.keys()
    assert "session_id: structured-session" in stderr
    assert len(chat.calls) == 1


@pytest.mark.parametrize("value", [{"ok": True}, False, None])
@pytest.mark.parametrize("output_format", ["text", "json", "stream-json"])
def test_schema_validates_whole_result_with_local_refs(chat, tmp_path, value, output_format):
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"$defs": {"answer": {"const": value}}, "$ref": "#/$defs/answer"}), encoding="utf-8")
    query = tmp_path / "query.txt"
    query.write_text("answer this", encoding="utf-8")
    raw = json.dumps(value)
    target = tmp_path / "answer.json"
    target.write_text("old artifact", encoding="utf-8")
    code, stdout, stderr = chat.run(
        "--query-file", str(query), "--format", output_format, "--output-schema", str(schema), "-o", str(target),
        results=[{"final_response": raw, "completed": True}],
    )
    assert code == 0, stderr
    assert target.read_text(encoding="utf-8-sig") == raw
    if output_format == "text":
        assert stdout == raw + "\n"
    else:
        envelope = json.loads(stdout.splitlines()[-1])
        assert envelope["structured_output"] == value
        assert envelope["text"] == raw
    prompt = chat.calls[0]["user_message"]
    assert "answer this" in prompt
    assert "#/$defs/answer" in prompt
    assert "JSON" in prompt


@pytest.mark.parametrize("schema_text, diagnostic", [
    (None, "output schema"),
    ("{", "output schema"),
    ('{"type": "not-a-type"}', "output schema"),
    ('{"$schema": "https://example.invalid/custom"}', "Unsupported"),
    ('{"$ref": "https://example.invalid/schema"}', "local"),
    ('{"$ref": "file:///private/schema.json"}', "local"),
    ('{"$ref": "#/$defs/missing"}', "reference"),
    ('{"type": "number", "type": "string"}', "Duplicate"),
    ('42', "object or boolean"), ('null', "object or boolean"), ('[]', "object or boolean"),
    ('{"$defs": {"x": {"$schema": "https://example.invalid/custom"}}}', "Unsupported"),
    ('{"$vocabulary": {"https://example.invalid/custom": true}}', "Unsupported"),
    ('{"payload": {"type": "invalid"}, "$ref": "#/payload"}', "output schema"),
    ('{"payload": 5, "$ref": "#/payload"}', "object or boolean"),
    ('{"$schema": []}', "output schema"),
])
def test_bad_schema_fails_before_agent_starts(chat, tmp_path, schema_text, diagnostic):
    schema = tmp_path / "schema.json"
    if schema_text is not None:
        schema.write_text(schema_text, encoding="utf-8")
    code, stdout, stderr = chat.run("-q", "hi", "--format", "json", "--output-schema", str(schema))
    assert code != 0
    assert chat.started == 0 and not chat.calls
    result = json.loads(stdout)
    assert result["exit_code"] == code
    assert diagnostic in result["error"]
    assert "structured_output" not in result


@pytest.mark.parametrize("raw", ['NaN', 'Infinity', '-Infinity', '1e400', '{"x": 1, "x": 2}',
                                  '```json\n{}\n```', 'Here: {}', '{} trailing', '{} {}', ''])
def test_invalid_answer_is_failed_with_raw_text(chat, tmp_path, raw):
    schema = tmp_path / "schema.json"
    schema.write_text('{}', encoding="utf-8")
    chat.corrections = [raw]
    code, stdout, stderr = chat.run("-q", "hi", "--format", "json", "--output-schema", str(schema),
                                   results=[{"final_response": raw, "completed": True}]*2)
    result = json.loads(stdout)
    assert code == result["exit_code"] == 1
    assert result["failed"] is True
    assert result["text"] == raw
    assert len(result["schema_errors"]) == 2
    assert len(chat.correction_requests) == 1
    assert "structured_output" not in result


@pytest.mark.parametrize("corrected, expected_code", [('42', 0), ('"wrong"', 1)])
def test_one_correction_preserves_transcript_model_and_tools(chat, tmp_path, corrected, expected_code):
    schema = tmp_path / "schema.json"
    schema.write_text('{"type": "integer"}', encoding="utf-8")
    transcript = [{"role": "user", "content": "perform task"},
                  {"role": "assistant", "tool_calls": [{"id": "c", "type": "function", "function": {
                      "name": "write_file", "arguments": "{}"}}], "content": ""},
                  {"role": "tool", "tool_call_id": "c", "content": "already written"},
                  {"role": "assistant", "content": '"wrong"'}]
    chat.corrections = [corrected]
    target = tmp_path / "answer.json"
    target.write_text("old artifact", encoding="utf-8")
    code, stdout, stderr = chat.run("-q", "perform task", "--format", "json", "--output-schema", str(schema),
                                   "-o", str(target),
                                   results=[{"final_response": '"wrong"', "completed": True, "messages": transcript}])
    assert code == expected_code, stdout + stderr
    assert len(chat.correction_requests) == 1, stdout
    assert len(chat.calls) == 1  # no second task run, hence no repeated tool effects
    request = chat.correction_requests[0]
    assert request["model"] == chat.agent.model
    assert request["tools"] == chat.agent.tools
    assert request["messages"][0] == {"role": "system", "content": chat.agent._cached_system_prompt}
    assert request["messages"][1:-1] == transcript
    assert request["messages"][-1]["role"] == "user"
    assert "Do not" in request["messages"][-1]["content"]
    result = json.loads(stdout)
    assert result["text"] == corrected
    if code == 0:
        assert result["structured_output"] == 42
        assert target.read_text(encoding="utf-8-sig") == corrected
    else:
        assert target.read_text(encoding="utf-8-sig") == "old artifact"
        assert result["failed"] is True and len(result["schema_errors"]) == 2
        assert "structured_output" not in result


@pytest.mark.parametrize("finish_reason, tools", [
    ("length", None), ("content_filter", None),
    ("tool_calls", [SimpleNamespace(id="c2", type="function", function=SimpleNamespace(name="write_file", arguments="{}"))]),
])
def test_correction_partial_or_tool_call_cannot_publish_success(chat, tmp_path, finish_reason, tools):
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"integer"}', encoding="utf-8")
    chat.corrections = [SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="42", tool_calls=tools), finish_reason=finish_reason)], usage=None)]
    code, stdout, stderr = chat.run("-q", "hi", "--format", "json", "--output-schema", str(schema),
                                   results=[{"final_response": '"bad"', "completed": True}])
    result = json.loads(stdout)
    assert code == 1
    assert result["text"] == "42"
    assert result["failed"]
    assert "structured_output" not in result
    assert len(chat.calls) == len(chat.correction_requests) == 1


@pytest.mark.parametrize("output_format", ["text", "json", "stream-json"])
def test_output_file_contains_only_final_response_anchored_before_chdir(chat, tmp_path, monkeypatch, output_format):
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / "answer.txt"
    destination.write_text("old artifact", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    import cli
    monkeypatch.setattr(cli, "_start_worktree_setup", lambda *a, **k: monkeypatch.chdir(elsewhere))
    code, stdout, stderr = chat.run("-q", "hi", "--format", output_format, "-o", "answer.txt")
    assert code == 0, stderr
    assert destination.read_text(encoding="utf-8-sig") == "hello"
    assert not (elsewhere / "answer.txt").exists()
    if output_format == "text":
        assert stdout == "hello\n"
    else:
        assert json.loads(stdout.splitlines()[-1])["text"] == "hello"


def test_failed_atomic_replacement_preserves_old_artifact(chat, tmp_path, monkeypatch):
    import os
    from pathlib import Path
    destination = tmp_path / "answer.txt"
    destination.write_text("old artifact", encoding="utf-8")
    attempts = []
    real_replace = os.replace

    def refuse_replace(source, target):
        if Path(target) == destination:
            attempts.append(Path(source))
            assert Path(source).parent == destination.parent
            assert Path(source).read_text(encoding="utf-8-sig") == "hello"
            assert destination.read_text(encoding="utf-8-sig") == "old artifact"
            raise OSError("replacement denied")
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", refuse_replace)
    code, stdout, stderr = chat.run("-q", "hi", "--format", "json", "-o", str(destination))
    assert code == 1
    result = json.loads(stdout)
    assert result["exit_code"] == code
    assert "replacement denied" in result["error"]
    assert destination.read_text(encoding="utf-8-sig") == "old artifact"
    assert len(attempts) == 1 and not attempts[0].exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("kind", ["directory", "symlink", "dangling", "fifo"])
@pytest.mark.parametrize("during_turn", [False, True])
def test_output_target_must_be_regular_before_and_after_turn(chat, tmp_path, kind, during_turn):
    import os
    target = tmp_path / "output"
    protected = tmp_path / "protected"
    protected.write_text("do not replace", encoding="utf-8")

    def make_unsafe(agent=None):
        if kind == "directory":
            target.mkdir()
        elif kind in {"symlink", "dangling"}:
            target.symlink_to(protected if kind == "symlink" else tmp_path / "absent")
        else:
            os.mkfifo(target)

    if during_turn:
        chat.after_turn = make_unsafe
    else:
        make_unsafe()
    code, stdout, stderr = chat.run("-q", "hi", "--format", "json", "-o", str(target))
    result = json.loads(stdout)
    assert code == result["exit_code"] == 1
    assert "regular" in result["error"]
    assert protected.read_text(encoding="utf-8-sig") == "do not replace"
    assert bool(chat.calls) is during_turn
    if kind in {"symlink", "dangling"}:
        assert target.is_symlink()


@pytest.mark.parametrize("failure, expected_code", [(KeyboardInterrupt(), 130), (InterruptedError("cancelled"), 130),
                                                    (RuntimeError("provider unavailable"), 1)])
def test_correction_failure_keeps_raw_answer_and_old_file(chat, tmp_path, failure, expected_code):
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"integer"}', encoding="utf-8")
    target = tmp_path / "output"
    target.write_text("old", encoding="utf-8")
    chat.corrections = [failure]
    code, stdout, stderr = chat.run("-q", "hi", "--format", "json", "--output-schema", str(schema), "-o", str(target),
                                   results=[{"final_response": '"bad"', "completed": True}])
    result = json.loads(stdout)
    assert code == result["exit_code"] == expected_code
    assert result["text"] == '"bad"'
    assert "structured_output" not in result
    assert target.read_text(encoding="utf-8-sig") == "old"
    assert len(chat.correction_requests) == 1


def test_missing_validator_fails_closed_before_agent(chat, tmp_path, monkeypatch):
    import builtins
    schema = tmp_path / "schema.json"
    schema.write_text('{}', encoding="utf-8")
    real_import = builtins.__import__

    def without_validator(name, *args, **kwargs):
        if name == "jsonschema" or name.startswith("jsonschema."):
            raise ImportError("validator unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_validator)
    code, stdout, stderr = chat.run("-q", "hi", "--format", "json", "--output-schema", str(schema))
    assert code == 1 and chat.started == 0
    assert "jsonschema" in json.loads(stdout)["error"]


@pytest.mark.parametrize("failure, expected_code", [
    ({"failed": True, "error": "provider down", "failure_reason": "server_error"}, 1),
    ({"partial": True, "error": "partial answer"}, 1),
    ({"completed": False, "error": "iteration budget exhausted"}, 1),
    ({"interrupted": True, "error": "interrupted"}, 130),
    (KeyboardInterrupt(), 130), (InterruptedError("cancelled"), 130), (RuntimeError("provider down"), 1),
])
@pytest.mark.parametrize("output_format", ["text", "json", "stream-json"])
def test_failed_turn_never_corrects_or_replaces_artifact(chat, tmp_path, failure, expected_code, output_format):
    schema = tmp_path / "schema.json"
    schema.write_text('{}', encoding="utf-8")
    target = tmp_path / "output"
    target.write_text("old", encoding="utf-8")
    turn = {"final_response": "42", **failure} if isinstance(failure, dict) else failure
    code, stdout, stderr = chat.run("-q", "hi", "--format", output_format, "--output-schema", str(schema),
                                   "-o", str(target), results=[turn])
    assert code == expected_code
    assert not chat.correction_requests
    assert target.read_text(encoding="utf-8-sig") == "old"
    if output_format != "text":
        result = json.loads(stdout.splitlines()[-1])
        assert result["exit_code"] == code
        assert "structured_output" not in result
    elif isinstance(failure, dict):
        assert failure["error"] in stderr


@pytest.mark.parametrize("budget", ["time", "iterations", "interrupt"])
def test_no_correction_after_run_budget_or_interrupt(chat, tmp_path, budget):
    import time
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"integer"}', encoding="utf-8")
    target = tmp_path / "output"
    target.write_text("old", encoding="utf-8")

    def exhausted(agent):
        if budget == "time":
            agent.run_budget_seconds = 1
            agent._run_budget_started_at = time.time() - 2
        elif budget == "iterations":
            agent.iteration_budget = SimpleNamespace(remaining=0)
        else:
            agent._interrupt_requested = True

    chat.after_turn = exhausted
    chat.corrections = ['42']
    code, stdout, stderr = chat.run("-q", "hi", "--format", "json", "--output-schema", str(schema),
                                   "-o", str(target), results=[{"final_response": '"bad"', "completed": True}])
    assert code != 0
    assert not chat.correction_requests
    assert target.read_text(encoding="utf-8-sig") == "old"
    assert "structured_output" not in json.loads(stdout)


@pytest.mark.parametrize("options, label", [(["--format", "json"], "--format json"),
                                           (["--output-schema", "schema.json"], "--output-schema"),
                                           (["-o", "answer.txt"], "--output-last-message")])
@pytest.mark.parametrize("query_args, message", [([], "requires -q/--query"), (["-q", "hi", "--tui"], "cannot be used with --tui")])
def test_output_options_reject_interactive_combinations(chat, options, label, query_args, message):
    code, stdout, stderr = chat.run(*query_args, *options)
    assert code == 2
    assert chat.started == 0
    assert label in stderr and message in stderr


@pytest.mark.parametrize("failure", ["exception", "interrupt", "worktree", "exit"])
def test_json_closes_early_startup_failures(chat, monkeypatch, failure):
    import cli
    if failure == "worktree":
        monkeypatch.setattr(cli, "_start_worktree_setup", lambda *a, **k: lambda: None)
    else:
        def fail_init(**kwargs):
            print("startup diagnostic")
            if failure == "interrupt":
                raise KeyboardInterrupt
            if failure == "exit":
                raise SystemExit(0)
            raise RuntimeError("startup failed")
        monkeypatch.setattr(cli, "HermesCLI", fail_init)
    code, stdout, stderr = chat.run("-q", "hi", "--format", "json")
    result = json.loads(stdout)
    assert result["exit_code"] == code != 0
    assert result["failed"] is True
    if failure == "exception":
        assert "startup failed" in result["error"]
        assert "startup diagnostic" in stderr


@pytest.mark.parametrize("reference", ["https://example.invalid/schema", "file:///private/schema.json"])
def test_no_reference_retrieval_even_through_nonkeyword_local_pointer(chat, tmp_path, monkeypatch, reference):
    import urllib.request
    retrieved = []

    def forbidden_retrieval(*args, **kwargs):
        retrieved.append(args)
        raise AssertionError("Schema reference retrieval is forbidden")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden_retrieval)
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"payload": {"$ref": reference}, "$ref": "#/payload"}), encoding="utf-8")
    code, stdout, stderr = chat.run("-q", "hi", "--format", "json", "--output-schema", str(schema))
    assert code == 1
    assert not retrieved
    assert chat.started == 0
    assert "local" in json.loads(stdout)["error"]


@pytest.mark.parametrize("corrected", ['42', '"still wrong"'])
def test_correction_usage_included_even_when_validation_fails(chat, tmp_path, corrected):
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"integer"}', encoding="utf-8")
    chat.corrections = [SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=corrected, tool_calls=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14))]
    code, stdout, stderr = chat.run("-q", "hi", "--format", "json", "--output-schema", str(schema),
                                   results=[{"final_response": '"bad"', "completed": True,
                                             "input_tokens": 20, "output_tokens": 6, "total_tokens": 26}])
    result = json.loads(stdout)
    assert result["tokens"]["input"] == 30
    assert result["tokens"]["output"] == 10
    assert result["tokens"]["total"] == 40


@pytest.mark.parametrize("option", ["json", "schema", "file"])
def test_output_options_force_one_shot_even_on_tty(chat, tmp_path, monkeypatch, option):
    import sys
    import hermes_cli.main as entry
    schema = tmp_path / "schema.json"
    schema.write_text('{}', encoding="utf-8")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(entry, "_resolve_use_tui", lambda args: pytest.fail("Interactive UI resolution attempted"))
    args = {"json": ["--format", "json"], "schema": ["--output-schema", str(schema)],
            "file": ["-o", str(tmp_path / "answer.txt")]}[option]
    code, stdout, stderr = chat.run("-q", "hi", *args, results=[{"final_response": "null", "completed": True}])
    assert code == 0, stdout + stderr
    assert len(chat.calls) == 1


def test_relative_schema_is_loaded_before_cwd_changes(chat, tmp_path, monkeypatch):
    import cli
    monkeypatch.chdir(tmp_path)
    (tmp_path / "schema.json").write_text('{"type":"integer"}', encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "schema.json").write_text('false', encoding="utf-8")
    monkeypatch.setattr(cli, "_start_worktree_setup", lambda *a, **k: monkeypatch.chdir(elsewhere))
    code, stdout, stderr = chat.run("-q", "hi", "--format", "json", "--output-schema", "schema.json",
                                   results=[{"final_response": "42", "completed": True}])
    assert code == 0, stdout + stderr
    assert json.loads(stdout)["structured_output"] == 42


@pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages"])
def test_native_correction_uses_selected_transport_and_accounts_usage(tmp_path, monkeypatch, api_mode):
    import copy
    from run_agent import AIAgent
    from agent.chat_completion_helpers import _iteration_summary_api_messages
    from hermes_cli.structured_output import OutputSchema

    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.setenv("HERMES_DISABLE_PLUGINS", "1")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    agent = AIAgent(api_key="test-key", base_url="http://127.0.0.1:9/v1", provider="custom", model="test-model",
                    api_mode=api_mode, quiet_mode=True, skip_context_files=True, skip_memory=True,
                    save_trajectories=False, enabled_toolsets=["file"])
    agent._cached_system_prompt = "Frozen system prompt"
    agent._current_turn_timestamp = 0  # this fixture bypasses the turn prologue
    schema_file = tmp_path / "schema.json"
    schema_file.write_text('{"type":"integer"}', encoding="utf-8")
    transcript = [{"role": "user", "content": "task"}, {"role": "assistant", "content": '"wrong"'}]
    original = copy.deepcopy(transcript)
    before = agent._build_api_kwargs(_iteration_summary_api_messages(agent, transcript))
    requests = []

    def answer(request):
        requests.append(request)
        if api_mode == "anthropic_messages":
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="42")], stop_reason="end_turn",
                                   usage=SimpleNamespace(input_tokens=10, output_tokens=4))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="42", tool_calls=None),
                                                        finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14))

    monkeypatch.setattr(agent, "_interruptible_api_call", answer)
    try:
        result = OutputSchema.load(schema_file).apply({"final_response": '"wrong"', "completed": True,
                                                      "messages": transcript}, agent)
        assert result.get("structured_output") == 42, result
        assert len(requests) == 1
        assert requests[0]["model"] == before["model"]
        assert requests[0]["tools"] == before["tools"]
        if api_mode == "anthropic_messages":
            assert requests[0]["system"] == before["system"]
        else:
            assert requests[0]["messages"][0] == before["messages"][0]
        assert transcript == original
        assert agent.session_api_calls == 1
        assert agent.session_total_tokens == 14
    finally:
        agent.close()


@pytest.fixture
def moa_correction_agent(tmp_path, monkeypatch):
    from run_agent import AIAgent

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.setenv("HERMES_DISABLE_PLUGINS", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-not-a-credential")
    (home / "config.yaml").write_text(json.dumps({
        "model": {"provider": "custom", "base_url": "http://127.0.0.1:9/v1"},
        "moa": {"default_preset": "fixture-preset", "presets": {
            "fixture-preset": {
                "reference_models": [{"provider": "custom", "model": "fixture-reference"}],
                "aggregator": {"provider": "custom", "model": "fixture-aggregator"},
                "fanout": "user_turn",
            },
        }},
    }), encoding="utf-8")
    agent = AIAgent(
        api_key="moa-virtual-provider", provider="moa", model="fixture-preset",
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False, enabled_toolsets=["file"],
    )
    agent._cached_system_prompt = "Frozen system prompt"
    agent._current_turn_timestamp = 0  # bypass the turn prologue, not native dispatch
    try:
        yield agent
    finally:
        agent.close()


def test_moa_schema_correction_runs_one_reference_fanout(moa_correction_agent, tmp_path, monkeypatch):
    import copy
    from hermes_cli.structured_output import OutputSchema

    calls = []

    def complete(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        text = "42" if kwargs["task"] == "moa_aggregator" else "reference advice"
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None), finish_reason="stop")],
            usage=None, model="fixture")

    # Only provider I/O is replaced: assembly, MoA preparation and native dispatch are real.
    monkeypatch.setattr("agent.moa_loop.call_llm", complete)
    schema_file = tmp_path / "schema.json"
    schema_file.write_text('{"type":"integer"}', encoding="utf-8")
    schema = OutputSchema.load(schema_file)
    result = schema.apply({"final_response": "not JSON", "completed": True, "messages": [
        {"role": "user", "content": "task"}, {"role": "assistant", "content": "not JSON"},
    ]}, moa_correction_agent)

    assert result.get("structured_output") == schema.validate(result["final_response"]) == 42
    assert result["completed"] and not result.get("failed")
    assert [call["task"] for call in calls] == ["moa_reference", "moa_aggregator"]
    assert sum(str(message.get("content")).count("[Mixture of Agents reference context]")
               for message in calls[-1]["messages"]) == 1


def test_moa_schema_correction_does_not_leak_prepared_request_to_replaced_client(
    moa_correction_agent, tmp_path, monkeypatch,
):
    import httpx
    from openai import OpenAI
    from hermes_cli.structured_output import OutputSchema

    wire_requests = []
    reference_calls = []

    def respond(request):
        wire_requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "fixture-correction", "object": "chat.completion", "created": 0,
            "model": "fixture-native", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "42"}}],
        })

    with OpenAI(api_key="fixture-not-a-credential", base_url="http://127.0.0.1:9/v1",
                http_client=httpx.Client(transport=httpx.MockTransport(respond))) as native:
        def reference(**kwargs):
            reference_calls.append(kwargs["task"])
            # A prepared request survives, but the live client is no longer the MoA facade.
            moa_correction_agent.client = native
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="reference advice", tool_calls=None),
                finish_reason="stop")], usage=None, model="fixture-reference")

        monkeypatch.setattr("agent.moa_loop.call_llm", reference)
        schema_file = tmp_path / "schema.json"
        schema_file.write_text('{"type":"integer"}', encoding="utf-8")
        result = OutputSchema.load(schema_file).apply({
            "final_response": "not JSON", "completed": True, "messages": [
                {"role": "user", "content": "task"}, {"role": "assistant", "content": "not JSON"},
            ],
        }, moa_correction_agent)

    assert result.get("structured_output") == 42, result
    assert reference_calls == ["moa_reference"]
    assert len(wire_requests) == 1  # the real OpenAI SDK rejects unknown kwargs before HTTP I/O
    assert "_moa_prepared_request" not in wire_requests[0]
    assert sum(str(message.get("content")).count("[Mixture of Agents reference context]")
               for message in wire_requests[0]["messages"]) == 1


def test_unserializable_result_fails_before_replacing_output_file(chat, tmp_path):
    target = tmp_path / "answer.json"
    target.write_text("old artifact", encoding="utf-8")
    code, stdout, stderr = chat.run("-q", "hi", "--format", "json", "-o", str(target),
                                   results=[{"final_response": "42", "completed": True,
                                             "structured_output": object()}])
    assert code == 1
    assert target.read_text(encoding="utf-8-sig") == "old artifact"
    result = json.loads(stdout)
    assert result["exit_code"] == 1 and result["failed"] is True
    assert "serializable" in result["error"]


@pytest.mark.parametrize("contract", ["schema", "file", "schema-and-file"])
@pytest.mark.parametrize("failure", ["worktree", "early-exit-zero"])
def test_text_output_contract_cannot_succeed_without_a_result(chat, tmp_path, monkeypatch, capsys, contract, failure):
    import cli
    import hermes_cli.main as entry
    from hermes_cli._parser import build_top_level_parser

    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    target = tmp_path / "answer.txt"
    target.write_text("old artifact", encoding="utf-8")
    if failure == "worktree":
        monkeypatch.setattr(cli, "_start_worktree_setup", lambda *a, **k: lambda: None)
    else:
        def exit_early(**kwargs):
            raise SystemExit(0)
        monkeypatch.setattr(cli, "HermesCLI", exit_early)
    options = ["--output-schema", str(schema)] if contract != "file" else []
    if contract != "schema":
        options += ["-o", str(target)]
    parser, _, _ = build_top_level_parser()
    try:
        entry.cmd_chat(parser.parse_args(["chat", "-q", "hi", "--format", "text", "--worktree", *options]))
    except SystemExit as exc:
        code = exc.code
    else:
        code = 0
    stdout, stderr = capsys.readouterr()
    assert not chat.calls and not hasattr(chat, "agent")
    assert target.read_text(encoding="utf-8-sig") == "old artifact"
    assert stdout == ""
    assert code != 0, f"{failure} reported success without running the agent"
    assert "final response" in stderr
