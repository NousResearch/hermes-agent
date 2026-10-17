"""A probe 401 from an overloaded engine must not replace the original 5xx (#136025).

The streaming-5xx unmask probe re-issues the request non-streaming into the same
condition that produced the 5xx. An overloaded engine (stepfun) can answer that probe
with 401 and auth-flavoured wording — "Incorrect API key provided" before any overload
mention. Replacing the 5xx with that artifact benches every credential-pool entry
sharing the key; overload wording means the ORIGINAL 5xx (backoff, same key) is the
truth. A structured ``error.code``/``error.type`` declaring an auth failure is the
provider's own verdict and always wins over the text heuristic.
"""
from types import SimpleNamespace

import agent.chat_completion_helpers as cch
import agent.chat_completion_helpers_probe as probe_mod


class _FakeHttpError(Exception):
    """Minimal SDK error shape: ``.status_code`` + ``.body``, message as str()."""

    def __init__(self, status_code, message, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body if body is not None else {"error": {"message": message}}


_INCIDENT_BODY = {
    "error": {
        "message": ("Incorrect API key provided. The engine is currently overloaded, "
                    "please try again later."),
    }
}


def _probe_401(body=None, message=None):
    body = body if body is not None else _INCIDENT_BODY
    message = message if message is not None else f"Error code: 401 - {body}"
    return _FakeHttpError(401, message, body)


# ---- artifact predicate ------------------------------------------------------------------------

def test_incident_shape_401_is_artifact():
    # The exact shape from the 2026-10-10 incident: auth-flavoured opening, overload truth.
    assert probe_mod._probe_401_is_overload_artifact(_probe_401(), 401) is True


def test_declared_auth_code_beats_overload_wording():
    # A structured ``error.code`` is a declaration and always wins, even when the body
    # text mentions overload too. Text-first matching was the wrong order: the incident
    # body opens with "Incorrect API key provided".
    err = _probe_401(body={"error": {"code": "invalid_api_key", "message": "engine is overloaded"}})
    assert probe_mod._probe_401_is_overload_artifact(err, 401) is False


def test_declared_auth_type_beats_overload_wording():
    # Anthropic-style ``error.type``.
    err = _probe_401(body={"error": {"type": "authentication_error", "message": "server is overloaded"}})
    assert probe_mod._probe_401_is_overload_artifact(err, 401) is False


def test_plain_auth_401_without_overload_wording_is_not_artifact():
    err = _probe_401(body={"error": {"message": "Incorrect API key provided"}})
    assert probe_mod._probe_401_is_overload_artifact(err, 401) is False


def test_overload_wording_without_structured_body_is_artifact():
    # An SDK that stringifies the body without exposing ``.body`` still classifies.
    err = _FakeHttpError(401, "Error code: 401 - the server is overloaded", None)
    assert probe_mod._probe_401_is_overload_artifact(err, 401) is True


# ---- probe branch: the original 5xx survives ---------------------------------------------------

def _call(monkeypatch, probe_exc):
    agent = SimpleNamespace(
        _interrupt_requested=False, api_mode="chat_completions", _stream_5xx_probe_ts=None)
    call = object.__new__(cch._StreamingCall)
    call.agent = agent
    call.api_kwargs = {"stream": True, "model": "m", "messages": []}
    call.result = {"response": None, "error": None, "partial_tool_names": []}
    call.deltas_were_sent = {"yes": False}
    call._stream_stale_timeout = 30.0

    def _probe(_agent, kwargs):
        assert "stream" not in kwargs  # the probe is the non-streaming re-issue
        raise probe_exc

    monkeypatch.setattr(cch, "interruptible_api_call", _probe)
    return call


def test_probe_overload_401_keeps_the_original_5xx(monkeypatch):
    call = _call(monkeypatch, _probe_401())
    original = _FakeHttpError(503, "Error code: 503 - something went wrong")

    handled = call._unmask_server_error_with_nonstreaming(original)

    assert handled is False  # propagate: the caller keeps the 5xx for retry/backoff
    # The artifact never entered result — on False the caller writes the original ``e``;
    # on the old path the 401 REPLACED the 5xx and drained the credential pool.
    assert call.result["error"] is None


def test_probe_declared_auth_401_still_replaces_the_5xx(monkeypatch):
    err = _FakeHttpError(401, "Error code: 401", {"error": {"code": "invalid_api_key"}})
    call = _call(monkeypatch, err)
    original = _FakeHttpError(503, "Error code: 503")

    handled = call._unmask_server_error_with_nonstreaming(original)

    assert handled is True
    assert call.result["error"] is err


def test_probe_plain_400_validation_still_replaces_the_5xx(monkeypatch):
    # Non-401 4xx replacements (the probe's original purpose) are untouched.
    err = _FakeHttpError(400, "Error code: 400", {"error": {"message": "max_tokens is too large"}})
    call = _call(monkeypatch, err)
    original = _FakeHttpError(503, "Error code: 503")

    handled = call._unmask_server_error_with_nonstreaming(original)

    assert handled is True
    assert call.result["error"] is err
