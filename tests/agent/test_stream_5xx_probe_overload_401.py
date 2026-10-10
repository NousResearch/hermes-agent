"""The 5xx unmask probe must not replace the 5xx with an overload-flavoured 401/403.

Incident (#136025, stepfun, 2026-10): the stream died with an opaque 503 "engine is
currently overloaded"; the non-streaming unmask probe hit the SAME overload window and
was answered ``401 Incorrect API key provided`` (the body only later admits the engine
is overloaded). The probe branch replaced any opaque 5xx with any probe 4xx, so
credential_pool marked the HEALTHY key exhausted (401 → 5 min TTL) and the single-key
pool drained — cron sessions minutes later failed with 401 while the same key kept
serving 200s before and after.

Contract:
- a structured auth ``error.type``/``code`` (invalid_api_key, authentication_error, …)
  is a declaration and always wins → the probe error still REPLACES the 5xx;
- only when the structured type is NOT an auth declaration do overload semantics in
  the body ("overloaded", "capacity", "try again later", 过载/繁忙…) mark the 4xx as
  an overload-window artifact → the ORIGINAL 5xx is kept (retry/backoff, never
  credential rotation);
- ordering is a hard requirement: stepfun's artifact body LEADS with the auth-flavoured
  sentence, so any text-first auth scan classifies every artifact as a real auth
  failure and fixes nothing;
- everything else is unchanged: a bare 401 stays a real validation error (conservative
  replace), a probe 5xx and a status-less probe error keep the original.
"""
from types import SimpleNamespace

from agent import chat_completion_helpers as h
from agent import error_classifier as ec


def _make_call(api_kwargs, *, deltas_sent=False, api_mode="chat_completions"):
    call = h._StreamingCall.__new__(h._StreamingCall)
    call.agent = SimpleNamespace(
        provider="stepfun", model="step-5-preview", api_mode=api_mode,
        _interrupt_requested=False,
        _is_provider_stream_parse_error=lambda e: False,
        _buffer_status=lambda text: call.buffered.append(text),
        _disable_streaming=False, _stream_5xx_probe_ts=None,
        _fire_stream_delta=lambda text: call.deltas.append(text),
    )
    call.buffered = []
    call.deltas = []
    call.first_delta_fired = {"done": True}
    call.on_first_delta = None
    call._stream_stale_timeout = 180.0
    call.api_kwargs = api_kwargs
    call.result = {"response": None, "error": None, "partial_tool_names": []}
    call.deltas_were_sent = {"yes": deltas_sent}
    call.provider_tool_in_flight = {"yes": False}
    call._request_cancelled = {"value": False}
    return call


class _StreamErr(Exception):
    """Stands in for the opaque 5xx the stream path raises (non-transient wording)."""

    def __init__(self, status):
        super().__init__(f"Error code: {status} - something went wrong")
        self.status_code = status


class _FakeHttpError(Exception):
    """Probe-shaped SDK error: numeric status + structured body."""

    def __init__(self, status_code, message, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body if body is not None else {"error": {"message": message}}


# The exact artifact shape from the #136025 incident: auth-flavoured sentence FIRST,
# overload admission in the structured body.
_INCIDENT_401 = _FakeHttpError(
    401, "Incorrect API key provided",
    {"error": {"message": "The engine is currently overloaded, please try again later",
               "type": "overloaded_error"}})


def _run_handle_stream_error(call, e):
    return call._handle_stream_error(e, attempt=2, max_retries=2)


# ---------------------------------------------------------------- classifier units

def test_incident_overload_401_body_is_artifact():
    assert ec._looks_like_overload_probe_error(_INCIDENT_401) is True


def test_stepfun_text_order_trap_is_still_artifact():
    # No structured type: the auth-flavoured sentence leads and the overload wording
    # trails. A text-first auth scan classifies this as a real auth failure — the
    # exact first-attempt bug the author hit (9/11 on the first run).
    err = _FakeHttpError(
        401,
        "Incorrect API key provided. The engine is currently overloaded, please try again later",
        {"error": {"message": "Incorrect API key provided. The engine is currently "
                              "overloaded, please try again later"}})
    assert ec._looks_like_overload_probe_error(err) is True


def test_structured_invalid_api_key_type_is_never_artifact():
    body = {"error": {"message": "Incorrect API key provided", "type": "invalid_api_key"}}
    assert ec._looks_like_overload_probe_error(_FakeHttpError(401, "Incorrect API key provided", body)) is False
    # OpenAI's real-401 shape carries the declaration in error.code instead.
    body = {"error": {"message": "Incorrect API key provided",
                      "type": "invalid_request_error", "code": "invalid_api_key"}}
    assert ec._looks_like_overload_probe_error(_FakeHttpError(401, "Incorrect API key provided", body)) is False


def test_structured_authentication_error_type_is_never_artifact():
    body = {"error": {"message": "authentication failed: bad key", "type": "authentication_error"}}
    assert ec._looks_like_overload_probe_error(_FakeHttpError(401, "authentication failed: bad key", body)) is False


def test_structured_auth_type_beats_overload_wording():
    # The declaration ALWAYS wins over text: a genuine auth error whose body ALSO
    # carries overload wording must never be excused (ordering regression = the
    # author's first attempt classified every artifact as a real auth failure).
    body = {"error": {"message": "The engine is currently overloaded, please try again later",
                      "type": "invalid_api_key"}}
    err = _FakeHttpError(401, "The engine is currently overloaded, please try again later", body)
    assert ec._looks_like_overload_probe_error(err) is False


def test_bare_401_without_overload_semantics_is_conservative():
    err = _FakeHttpError(401, "Incorrect API key provided")
    assert ec._looks_like_overload_probe_error(err) is False


def test_artifact_scope_is_401_403_only():
    overloadish = {"error": {"message": "The engine is currently overloaded, please try again later"}}
    assert ec._looks_like_overload_probe_error(_FakeHttpError(429, "The engine is currently overloaded, please try again later", overloadish)) is False
    assert ec._looks_like_overload_probe_error(_FakeHttpError(400, "The engine is currently overloaded, please try again later", overloadish)) is False
    assert ec._looks_like_overload_probe_error(_FakeHttpError(503, "The engine is currently overloaded, please try again later", overloadish)) is False


def test_chinese_overload_wording_is_artifact():
    err = _FakeHttpError(401, "API 密钥不正确", {"error": {"message": "服务器繁忙，请稍后重试"}})
    assert ec._looks_like_overload_probe_error(err) is True


def test_status_less_probe_error_is_never_artifact():
    err = Exception("The engine is currently overloaded, please try again later")
    assert ec._looks_like_overload_probe_error(err) is False


def test_body_error_type_reads_nested_and_top_level():
    assert ec._body_error_type({"error": {"type": "Overloaded_Error"}}) == "overloaded_error"
    assert ec._body_error_type({"type": "authentication_error"}) == "authentication_error"
    # Gemini puts the HTTP status in error.code as a NUMBER — not a type declaration.
    assert ec._body_error_type({"error": {"code": 401, "status": "UNAVAILABLE"}}) == ""
    assert ec._body_error_type("not a dict") == ""


# ---------------------------------------------------------------- handler branches

def test_probe_overload_401_artifact_keeps_original_5xx(monkeypatch):
    call = _make_call({"model": "step-5-preview", "messages": []})
    original = _StreamErr(503)

    def fake_probe(agent, kwargs):
        raise _INCIDENT_401

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)

    handled = _run_handle_stream_error(call, original)

    assert not handled  # propagate the ORIGINAL error to the retry loop
    assert call.result["error"] is original  # NOT the probe's 401 artifact
    assert call.result["response"] is None


def test_probe_genuine_invalid_api_key_still_replaces_5xx(monkeypatch):
    call = _make_call({"model": "step-5-preview", "messages": []})
    real = _FakeHttpError(401, "Incorrect API key provided",
                          {"error": {"message": "Incorrect API key provided",
                                     "type": "invalid_api_key"}})

    def fake_probe(agent, kwargs):
        raise real

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)

    handled = _run_handle_stream_error(call, _StreamErr(500))

    assert not handled  # the 4xx is handled: loop stops, `real` propagates
    assert call.result["error"] is real  # the REAL validation error replaces the 5xx
    assert call.result["response"] is None


def test_probe_bare_401_replaces_5xx_conservatively(monkeypatch):
    # No structured type, no overload wording: today's replace behaviour is preserved.
    call = _make_call({"model": "m", "messages": []})
    real = _FakeHttpError(401, "Incorrect API key provided")

    def fake_probe(agent, kwargs):
        raise real

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)

    handled = _run_handle_stream_error(call, _StreamErr(500))

    assert not handled  # the bare 401 is handled: `real` propagates
    assert call.result["error"] is real


def test_probe_5xx_and_status_less_keep_original(monkeypatch):
    call = _make_call({"model": "m", "messages": []})
    original = _StreamErr(500)

    def fake_probe_5xx(agent, kwargs):
        raise _StreamErr(503)

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe_5xx)
    assert not _run_handle_stream_error(call, original)
    assert call.result["error"] is original

    def fake_probe_status_less(agent, kwargs):
        raise Exception("real validation message")

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe_status_less)
    call2 = _make_call({"model": "m", "messages": []})
    assert not _run_handle_stream_error(call2, _StreamErr(500))
    assert call2.result["error"] is not None and call2.result["error"] is not fake_probe_status_less
