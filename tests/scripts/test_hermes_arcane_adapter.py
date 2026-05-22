import io
import json

import pytest

from scripts import hermes_arcane_adapter as adapter


def _run_adapter(monkeypatch, request):
    stdin = io.StringIO(json.dumps(request))
    stdout = io.StringIO()
    monkeypatch.setattr(adapter.sys, "stdin", stdin)
    monkeypatch.setattr(adapter.sys, "stdout", stdout)
    rc = adapter.main([])
    events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    return rc, events


def test_slash_help_emits_assistant_message_without_agent(monkeypatch):
    def fail_build_agent(*_args, **_kwargs):
        pytest.fail("slash commands must not construct AIAgent")

    monkeypatch.setattr(adapter, "build_agent", fail_build_agent)
    rc, events = _run_adapter(
        monkeypatch,
        {
            "sessionId": "arcane-session",
            "runId": "run-help",
            "content": "/help",
        },
    )

    assert rc == 0
    assert [event["type"] for event in events] == ["assistant.message", "run.done"]
    assert events[0]["sessionId"] == "arcane-session"
    assert events[0]["runId"] == "run-help"
    assert "Hermes commands available in Arcane" in events[0]["content"]
    assert "`/status`" in events[0]["content"]


def test_unknown_slash_command_returns_clear_message(monkeypatch):
    monkeypatch.setattr(adapter, "build_agent", lambda *_args, **_kwargs: pytest.fail("unexpected model path"))
    rc, events = _run_adapter(
        monkeypatch,
        {
            "sessionId": "arcane-session",
            "runId": "run-unknown",
            "content": "/does-not-exist",
        },
    )

    assert rc == 0
    assert [event["type"] for event in events] == ["assistant.message", "run.done"]
    assert "Unknown Hermes command `/does-not-exist`" in events[0]["content"]


def test_adapter_sets_arcane_environment_for_tools(monkeypatch):
    monkeypatch.delenv("ARCANE_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(adapter, "build_agent", lambda *_args, **_kwargs: pytest.fail("unexpected model path"))

    rc, _events = _run_adapter(
        monkeypatch,
        {
            "sessionId": "sess-env",
            "runId": "run-env",
            "content": "/commands 1",
            "arcaneBaseUrl": "http://arcane.local/",
            "arcaneAccessToken": "token-from-request",
        },
    )

    assert rc == 0
    assert adapter.os.environ["ARCANE_SESSION_ID"] == "sess-env"
    assert adapter.os.environ["ARCANE_RUN_ID"] == "run-env"
    assert adapter.os.environ["ARCANE_BASE_URL"] == "http://arcane.local"
    assert adapter.os.environ["ARCANE_ACCESS_TOKEN"] == "token-from-request"


def test_stream_emitter_uses_stable_message_id_and_incrementing_index(monkeypatch):
    stdout = io.StringIO()
    monkeypatch.setattr(adapter.sys, "stdout", stdout)
    sink = adapter.StdoutArcaneEventSink("sess-stream", "run-stream")
    emitter = adapter.ArcaneStreamEmitter(sink, "run-stream")

    emitter.emit("Hello")
    emitter.emit(None)
    emitter.emit(" world")

    events = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [event["type"] for event in events] == ["assistant.delta", "assistant.delta"]
    assert [event["messageId"] for event in events] == ["assistant-run-stream", "assistant-run-stream"]
    assert [event["index"] for event in events] == [0, 1]
    assert [event["delta"] for event in events] == ["Hello", " world"]
