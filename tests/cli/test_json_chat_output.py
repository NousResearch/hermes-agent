"""Behavior tests for the versioned ``hermes chat --json`` contract."""

import json
from types import SimpleNamespace

import pytest

import cli as cli_module


CALLBACKS = (
    "reasoning_callback",
    "tool_progress_callback",
    "tool_start_callback",
    "tool_complete_callback",
    "stream_delta_callback",
    "tool_gen_callback",
)


class _FakeAgent:
    def __init__(self, result, *, rotated_session_id=None):
        self.result = result
        self.session_id = rotated_session_id
        for callback in CALLBACKS:
            setattr(self, callback, lambda: print("callback leak"))

    def run_conversation(self, **_kwargs):
        assert all(getattr(self, callback) is None for callback in CALLBACKS)
        print("reasoning leak")
        raise_or_result = self.result
        if isinstance(raise_or_result, Exception):
            raise raise_or_result
        return raise_or_result


def _fake_cli_class(result, *, session_id="20260714_new", rotated_session_id=None):
    class FakeCLI:
        def __init__(self, **_kwargs):
            self.session_id = session_id
            self.conversation_history = []
            self._active_agent_route_signature = None
            self.agent = None
            self.provider = "openrouter"
            self.model = "z-ai/glm-5.2"

        def _claim_active_session(self, *_args, **_kwargs):
            print("claim leak")
            return True

        def _ensure_runtime_credentials(self):
            print("credential leak")
            return True

        def _resolve_turn_agent_config(self, _query):
            return {
                "signature": "json-test",
                "model": "z-ai/glm-5.2",
                "runtime": None,
            }

        def _init_agent(self, **_kwargs):
            print("initialization leak")
            self.agent = _FakeAgent(
                result,
                rotated_session_id=rotated_session_id or self.session_id,
            )
            return True

        def _release_active_session(self):
            return None

    return FakeCLI


def _run_json(monkeypatch, capsys, result, **fake_kwargs):
    monkeypatch.setattr(cli_module, "HermesCLI", _fake_cli_class(result, **fake_kwargs))
    monkeypatch.setattr(cli_module, "_finalize_single_query", lambda _cli: None)
    monkeypatch.setattr(cli_module.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("signal.signal", lambda *_args, **_kwargs: None)

    with pytest.raises(SystemExit) as exc:
        cli_module.main(
            query="Reply exactly: EMAIL-BRIDGE-OK",
            toolsets="composio",
            json_output=True,
        )

    return exc.value.code, capsys.readouterr()


def test_json_output_is_one_exact_object_despite_noisy_agent(monkeypatch, capsys):
    code, captured = _run_json(
        monkeypatch,
        capsys,
        {"final_response": "EMAIL-BRIDGE-OK"},
    )

    assert code == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out) == {
        "protocol": "hermes.chat.result.v1",
        "reply": "EMAIL-BRIDGE-OK",
        "session_id": "20260714_new",
    }


def test_json_output_reports_rotated_resumed_session(monkeypatch, capsys):
    code, captured = _run_json(
        monkeypatch,
        capsys,
        {"final_response": "continued"},
        session_id="20260714_parent",
        rotated_session_id="20260714_child",
    )

    assert code == 0
    assert json.loads(captured.out)["session_id"] == "20260714_child"


@pytest.mark.parametrize(
    "result",
    [
        {"final_response": "", "failed": True, "error": "provider secret"},
        RuntimeError("secret traceback detail"),
    ],
)
def test_json_failure_emits_no_success_object(monkeypatch, capsys, result):
    code, captured = _run_json(monkeypatch, capsys, result)

    assert code == 1
    assert captured.out == ""
    assert captured.err == "Hermes JSON chat failed\n"


def test_json_allows_an_empty_reply(monkeypatch, capsys):
    code, captured = _run_json(monkeypatch, capsys, {"final_response": ""})

    assert code == 0
    assert json.loads(captured.out)["reply"] == ""
