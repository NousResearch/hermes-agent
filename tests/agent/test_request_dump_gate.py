"""A failing turn must not leave the conversation on disk unless the operator asked for it.

Both error paths (non-retryable 4xx, retries exhausted) wrote a full request body — system
prompt, conversation, every tool schema — on every failure, with only the preflight path gated.
The gate now lives on the writer, and ``diagnostics.request_dump.content`` decides how much of
the body survives: ``redacted`` (default) keeps the shape, ``full`` keeps the body as sent. The
on/off switch stays on ``HERMES_DUMP_REQUESTS``, which predates the config section.
"""

import json
from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import dump_api_request_debug
from agent.error_classifier import classify_api_error
from agent.turn_recovery import max_retries_exhausted_result, nonretryable_client_error_result

CONVERSATION = "my account number is 4417 please pay the rent on the 1st"
BASE_URL = "https://inference-api.nousresearch.com/v1"
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": "run shell commands",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    },
}


def _request_kwargs():
    return {
        "model": "stepfun/step-3.7-flash:free",
        "messages": [
            {"role": "system", "content": "you are hermes, a careful agent"},
            {"role": "user", "content": CONVERSATION},
        ],
        "tools": [TOOL_SCHEMA],
        "temperature": 0.2,
        "timeout": 60,
    }


def _agent(tmp_path):
    agent = SimpleNamespace(
        provider="nous", api_key="sk-test-1234567890abcdefghij", base_url=BASE_URL,
        model="stepfun/step-3.7-flash:free", api_mode="chat_completions", session_id="sess-1",
        client=SimpleNamespace(api_key="sk-test-key-abcdefghijklmnop"),
        logs_dir=tmp_path, verbose_logging=False, log_prefix="",
        _rate_limit_state=None, _has_pending_fallback=lambda: False,
        _flush_status_buffer=lambda: None, _summarize_api_error=lambda e: str(e),
        _emit_status=lambda *a, **kw: None, _persist_session=lambda *a, **kw: None,
        _plines=lambda *a, **kw: None, _vprint=lambda *a, **kw: None,
        _buffer_status=lambda *a, **kw: None, _buffer_vprint=lambda *a, **kw: None,
        _emit_diagnostic_status=lambda *a, **kw: None,
        _buffer_diagnostic_status=lambda *a, **kw: None,
        _try_activate_fallback=lambda: False,
        _mask_api_key_for_logs=lambda key: "sk-...redacted",
    )
    agent._dump_api_request_debug = lambda kwargs, **kw: dump_api_request_debug(agent, kwargs, **kw)
    return agent


def _classified(agent, error):
    return classify_api_error(error, provider="nous", model=agent.model, base_url=BASE_URL,
                              api_key=agent.api_key)


def _drive_nonretryable(agent):
    error = Exception("Your request was rejected: maximum context length exceeded")
    error.status_code = 400
    return nonretryable_client_error_result(
        agent, error, _classified(agent, error), status_code=400,
        api_kwargs=_request_kwargs(), api_messages=_request_kwargs()["messages"], messages=[],
        conversation_history=[], api_call_count=1, approx_tokens=10, provider="nous",
        base_url=BASE_URL, model=agent.model,
    )


def _drive_exhausted(agent):
    error = Exception("connection reset by peer")
    return max_retries_exhausted_result(
        agent, error, _classified(agent, error), attempts=3, is_rate_limited=False,
        error_msg="connection reset by peer", api_kwargs=_request_kwargs(),
        api_messages=_request_kwargs()["messages"], messages=[], conversation_history=[],
        api_call_count=3, approx_tokens=10, provider="nous", base_url=BASE_URL, model=agent.model,
    )


def _dumps(tmp_path):
    return sorted(tmp_path.glob("request_dump_*.json"))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HERMES_DUMP_REQUESTS", raising=False)
    monkeypatch.delenv("HERMES_DUMP_REQUESTS_CONTENT", raising=False)
    monkeypatch.delenv("HERMES_DUMP_REQUEST_STDOUT", raising=False)


def test_neither_error_path_writes_a_dump_without_the_env(tmp_path, monkeypatch):
    """Unset must mean no file — including the two error paths that used to ignore the gate."""
    monkeypatch.delenv("HERMES_DUMP_REQUESTS", raising=False)

    _drive_nonretryable(_agent(tmp_path))
    _drive_exhausted(_agent(tmp_path))

    assert _dumps(tmp_path) == []


def test_enabled_writes_one_redacted_dump_per_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_DUMP_REQUESTS", "1")

    _drive_exhausted(_agent(tmp_path))

    dumps = _dumps(tmp_path)
    assert len(dumps) == 1
    payload = json.loads(dumps[0].read_text(encoding="utf-8"))
    body = payload["request"]["body"]
    serialized = json.dumps(payload)
    assert "4417" not in serialized and "careful agent" not in serialized
    assert body["message_count"] == 2
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert body["messages"][1]["content"].startswith("<text ")
    assert body["tools"] == [{"type": "function", "name": "terminal", "parameters_chars": body["tools"][0]["parameters_chars"]}]
    assert body["temperature"] == 0.2
    assert payload["error"]["type"] == "Exception"


def test_full_content_mode_keeps_the_body_when_asked(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_DUMP_REQUESTS", "1")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"diagnostics": {"request_dump": {"content": "full"}}},
    )

    _drive_exhausted(_agent(tmp_path))

    dumps = _dumps(tmp_path)
    assert len(dumps) == 1
    assert "4417" in dumps[0].read_text(encoding="utf-8")


@pytest.mark.parametrize("value", ["redact", "REDACTED", "", "false"])
def test_unrecognized_content_value_stays_redacted(tmp_path, monkeypatch, value):
    """A typo must not silently turn on full-body dumps."""
    monkeypatch.setenv("HERMES_DUMP_REQUESTS", "1")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"diagnostics": {"request_dump": {"content": value}}},
    )

    _drive_nonretryable(_agent(tmp_path))

    dumps = _dumps(tmp_path)
    assert len(dumps) == 1
    assert "4417" not in dumps[0].read_text(encoding="utf-8")


def test_the_body_mode_is_not_reachable_from_an_env_var(tmp_path, monkeypatch):
    """Behaviour belongs in config.yaml: the old env var is gone and cannot re-enable a full body."""
    monkeypatch.setenv("HERMES_DUMP_REQUESTS", "1")
    monkeypatch.setenv("HERMES_DUMP_REQUESTS_CONTENT", "full")
    monkeypatch.setattr("hermes_cli.config.load_config", dict)

    _drive_exhausted(_agent(tmp_path))

    dumps = _dumps(tmp_path)
    assert len(dumps) == 1
    assert "4417" not in dumps[0].read_text(encoding="utf-8")


def test_a_config_without_the_key_still_redacts():
    """An install that never set diagnostics.request_dump.content gets the documented default."""
    from agent.request_dump import request_dump_content_mode

    assert request_dump_content_mode({}) == "redacted"
    assert request_dump_content_mode({"diagnostics": {}}) == "redacted"


def test_both_error_paths_consult_the_one_shared_predicate(tmp_path, monkeypatch):
    """One predicate decides for every dump path, so a new call site cannot reintroduce a bypass."""
    calls = []

    def _spy():
        calls.append("asked")
        return False

    from agent import request_dump

    monkeypatch.setattr(request_dump, "request_dump_enabled", _spy)

    _drive_nonretryable(_agent(tmp_path))
    _drive_exhausted(_agent(tmp_path))

    assert calls == ["asked", "asked"]
    assert _dumps(tmp_path) == []
