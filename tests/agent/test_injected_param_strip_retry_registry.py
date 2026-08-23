"""Registry walk for the injected-parameter strip-and-retry safety class.

Bug class (#90257, #89897, #91164, #89503): a parameter that Hermes (or the
provider's own gateway) injects into a request — ``prompt_cache_retention``,
reasoning translations, ``temperature``/``max_tokens``/``response_format``
adjustments — draws an HTTP 400 from the provider and the turn DIES instead
of stripping the parameter (or retrying the identical request when the field
was never ours) and completing.

Two request paths are pinned, each driving the REAL recovery code with only
the HTTP boundary faked (a client that 400s once, then succeeds):

* Classifier path (main conversation loop): ``classify_api_error`` +
  ``_SERVER_INJECTED_PARAM_SENDERS`` (fixed in PR #91643). The registry is
  walked at collection time, so adding a new injected param automatically
  extends coverage. The retry contract mirrors conversation_loop: retryable
  → resend identical request; non-retryable → abort.
* Auxiliary path: ``agent.auxiliary_client.call_llm``'s reactive strip-and-
  retry rungs for every parameter that path knows how to strip.

Guard: a 400 naming a param NOT in the registry must still fail — we never
blindly strip arbitrary params, because that would convert our own request
bug into a friendly lie (error-classifier policy, AGENTS.md).
"""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.error_classifier import (
    FailoverReason,
    _SERVER_INJECTED_PARAM_SENDERS,
    classify_api_error,
)
from agent.auxiliary_client import call_llm


class MockAPIError(Exception):
    """Simulates an OpenAI SDK APIStatusError (status_code + body)."""

    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}


# ---------------------------------------------------------------------------
# Path 1 — classifier path (main conversation loop contract)
# ---------------------------------------------------------------------------

REGISTRY_PARAMS = sorted(_SERVER_INJECTED_PARAM_SENDERS)


def _error_shapes(param):
    """Real provider 400 shapes naming *param* (see #90257 live capture)."""
    openai_body = {
        "error": {
            "message": f"Unsupported parameter: '{param}'",
            "type": "invalid_request_error",
            "param": param,
            "code": "unsupported_parameter",
        }
    }
    return [
        pytest.param(
            f"Error code: 400 - {openai_body!r}", openai_body,
            id=f"{param}-openai-structured",
        ),
        pytest.param(
            f"{param} is not supported on this model", {},
            id=f"{param}-message-only",
        ),
        pytest.param(
            f"Unknown parameter: {param}",
            {"detail": f"Unknown parameter: {param}"},
            id=f"{param}-terse-detail",
        ),
    ]


def _drive_conversation_retry(transport, *, provider, model, max_attempts=3):
    """Minimal loop honoring the conversation_loop classifier contract.

    retryable → resend the identical request; non-retryable → abort the turn.
    The classifier under test is the REAL one; only the transport is fake.
    """
    for _ in range(max_attempts):
        try:
            return transport()
        except Exception as exc:  # noqa: BLE001 — contract mirror
            verdict = classify_api_error(
                exc, provider=provider, model=model,
                approx_tokens=50, num_messages=3,
            )
            if not verdict.retryable:
                raise
    raise AssertionError("retries exhausted")


class _FlakyTransport:
    """400s exactly once, then succeeds — the HTTP boundary and nothing else."""

    def __init__(self, error):
        self.error = error
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls == 1:
            raise self.error
        return {"ok": True}


class TestClassifierPathRegistryWalk:
    """Every registered injected param × every real 400 shape must retry."""

    def test_registry_is_never_empty(self):
        assert REGISTRY_PARAMS, "injected-param registry must not be empty"

    @pytest.mark.parametrize("param", REGISTRY_PARAMS)
    def test_classified_retryable_without_compression(self, param):
        """#90257/#91164: injected-param 400 → retryable server_error,
        never routed into the compression loop (request shape was fine)."""
        for msg_param in _error_shapes(param):
            message, body = msg_param.values
            err = MockAPIError(message, status_code=400, body=copy.deepcopy(body))
            verdict = classify_api_error(
                err, provider="openai-codex", model="gpt-5.6-sol",
                approx_tokens=546912, context_length=272000, num_messages=576,
            )
            assert verdict.reason == FailoverReason.server_error, message
            assert verdict.retryable is True, message
            assert verdict.should_compress is False, message

    @pytest.mark.parametrize("param", REGISTRY_PARAMS)
    def test_turn_completes_via_retry(self, param):
        """The full contract: 400 once naming the injected param, retry the
        identical request, turn completes (#89503 — the turn used to die)."""
        for msg_param in _error_shapes(param):
            message, body = msg_param.values
            transport = _FlakyTransport(
                MockAPIError(message, status_code=400, body=copy.deepcopy(body))
            )
            result = _drive_conversation_retry(
                transport, provider="openai-codex", model="gpt-5.6-sol",
            )
            assert result == {"ok": True}, message
            assert transport.calls == 2, message

    @pytest.mark.parametrize("param", REGISTRY_PARAMS)
    def test_sender_route_still_fails_fast(self, param):
        """When the current provider IS a deliberate sender of the param, the
        400 is a real request bug — it must stay a non-retryable format_error."""
        sender = _SERVER_INJECTED_PARAM_SENDERS[param][0]
        err = MockAPIError(
            f"Unsupported parameter: '{param}'", status_code=400,
            body={"error": {"message": f"Unsupported parameter: '{param}'",
                            "code": "unsupported_parameter", "param": param}},
        )
        verdict = classify_api_error(err, provider=sender, model="any-model")
        assert verdict.reason == FailoverReason.format_error
        assert verdict.retryable is False

    def test_guard_unregistered_param_still_fails(self):
        """Policy guard: a 400 naming a param NOT in the registry must abort.
        Blindly stripping arbitrary params would mask real request bugs
        (never convert our own request bug into a friendly lie)."""
        err = MockAPIError(
            "Unsupported parameter: 'frobnication_level'", status_code=400,
            body={"error": {"message": "Unsupported parameter: 'frobnication_level'",
                            "code": "unsupported_parameter",
                            "param": "frobnication_level"}},
        )
        verdict = classify_api_error(
            err, provider="openai-codex", model="gpt-5.6-sol",
        )
        assert verdict.reason == FailoverReason.format_error
        assert verdict.retryable is False
        transport = _FlakyTransport(err)
        with pytest.raises(MockAPIError):
            _drive_conversation_retry(
                transport, provider="openai-codex", model="gpt-5.6-sol",
            )
        assert transport.calls == 1  # aborted on first attempt, no blind retry


# ---------------------------------------------------------------------------
# Path 2 — auxiliary call path (call_llm reactive strip-and-retry rungs)
# ---------------------------------------------------------------------------

class _FlakyClient:
    """Fake OpenAI-SDK client: chat.completions.create 400s once, then OK."""

    def __init__(self, error):
        self.base_url = "https://api.openai.com/v1"
        self.calls = []
        self._error = error
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if len(self.calls) == 1:
            raise self._error
        return {"ok": True}


def _aux_patches(client):
    return (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("openai-codex", "gpt-5.5", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client",
              return_value=(client, "gpt-5.5")),
        patch("agent.auxiliary_client._validate_llm_response",
              side_effect=lambda resp, _task, **_kw: resp),
    )


# Every parameter the auxiliary path knows how to strip, with a real provider
# phrasing and the kwargs that inject it. If a new strip rung is added to
# call_llm, extend this walk (the guard test below keeps the boundary honest).
AUX_STRIPPABLE = [
    pytest.param(
        {"temperature": 0.3},
        "HTTP 400: Unsupported parameter: temperature",
        lambda kw: "temperature" not in kw,
        id="temperature",
    ),
    pytest.param(
        {"max_tokens": 128},
        "Error code: 400 - {'error': {'message': \"Unsupported parameter: "
        "'max_tokens' is not supported with this model.\", "
        "'code': 'unsupported_parameter', 'param': 'max_tokens'}}",
        lambda kw: "max_tokens" not in kw and "max_completion_tokens" not in kw,
        id="max_tokens",
    ),
    pytest.param(
        {"extra_body": {"response_format": {"type": "json_object"}}},
        "HTTP 400: Unsupported parameter: response_format",
        lambda kw: "response_format"
        not in (kw.get("extra_body") or {}) and "response_format" not in kw,
        id="response_format",
    ),
]


class TestAuxiliaryPathStripRetryWalk:
    """call_llm must strip the rejected injected param and retry to success."""

    @pytest.mark.parametrize("inject_kwargs,error_msg,stripped_ok", AUX_STRIPPABLE)
    def test_strips_param_and_completes(self, inject_kwargs, error_msg, stripped_ok):
        """#89897/#90257 class: provider 400 names a param we injected — the
        aux call must retry once WITHOUT the param and return the response."""
        client = _FlakyClient(RuntimeError(error_msg))
        p1, p2, p3 = _aux_patches(client)
        with p1, p2, p3:
            result = call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hi"}],
                **inject_kwargs,
            )
        assert result == {"ok": True}
        assert len(client.calls) == 2, "expected exactly one strip-and-retry"
        assert stripped_ok(client.calls[1]), (
            f"retry still carried the rejected param: {client.calls[1].keys()}"
        )

    def test_guard_unregistered_param_still_raises(self):
        """A 400 naming a param the aux path has no strip rung for must
        surface — no blind stripping, no silent success (#91164 policy)."""
        client = _FlakyClient(
            RuntimeError("HTTP 400: Unsupported parameter: frobnication_level")
        )
        p1, p2, p3 = _aux_patches(client)
        with p1, p2, p3:
            with pytest.raises(RuntimeError, match="frobnication_level"):
                call_llm(
                    task="session_search",
                    messages=[{"role": "user", "content": "hi"}],
                )
        assert len(client.calls) == 1, "must not retry an unknown-param 400"
