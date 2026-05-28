"""Detector + call_llm fallback for provider-rejected aux models.

Covers the "user's main chat model is forwarded verbatim to an aux
endpoint that only supports a subset of that provider's catalog" case.

Two surfaces:
  * ``is_model_unsupported_error`` (unit): correct on 404 / 401 / 400 / 422
    with and without model-marker wording, and correct on pure auth /
    payment / connection / rate-limit errors.
  * ``call_llm`` fallback (integration): when the chosen provider rejects
    the requested model, the provider's default aux model
    (``_get_aux_model_for_provider``) is tried before re-raising. Pure
    auth errors (401 without model wording) continue to follow the
    existing auth-refresh / credential-pool chain.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.aux_unsupported_model import is_model_unsupported_error


# Build exceptions that carry the OpenAI client ``status_code`` + ``message``
# fields without tripping pyright's strict ``Exception`` typing.
class _APIExc(Exception):
    def __init__(self, text: str, *, status_code: int, message: str | None = None):
        super().__init__(text)
        self.status_code = status_code
        if message is not None:
            self.message = message
        self.error = None


def _exc(status_code: int, text: str, *, msg_attr=None):
    return _APIExc(text, status_code=status_code, message=msg_attr)


@pytest.mark.parametrize("exc,expected", [
    # 404 + model marker → model unsupported
    (_exc(404, "Model mimo-v2.5 not found"), True),
    (_exc(404, "The requested model does not exist in our catalog"), True),
    (_exc(404, "unknown model foo-bar"), True),
    # 401 + "not supported" wording (aggregator-specific)
    (_exc(401, "HTTP 401: Model mimo-v2.5 is not supported"), True),
    (_exc(401, "unsupported model: foo"), True),
    # 400 / 422 with "not supported" wording
    (_exc(400, "Model X is not supported on this tier"), True),
    (_exc(422, "The model foo-bar is not supported"), True),
    # Pure credential failures — must NOT classify as model unsupported
    (_exc(401, "missing or invalid API key"), False),
    (_exc(401, "AuthenticationError [HTTP 401]"), False),
    (_exc(401, "Invalid Authorization header"), False),
    # 404 / 400 lacking a model marker
    (_exc(404, "Path /foo/v1/bar not found"), False),
    (_exc(400, "Invalid request body: field X required"), False),
    (_exc(400, "unsupported_parameter: temperature"), False),
    # Payment / rate limit / connection — separate recovery paths
    (_exc(402, "Insufficient credits"), False),
    (_exc(429, "Rate limit exceeded"), False),
    (_exc(503, "Service unavailable"), False),
    # openai-style error object with ``.error.message``
    (_exc(404, "Error", msg_attr="Model does not exist in our catalog"), True),
    # No status_code attribute at all — detector must be conservative (False)
    (Exception("Model not found"), False),
])
def test_detector(exc, expected):
    assert is_model_unsupported_error(exc) is expected


# ── call_llm fallback integration ─────────────────────────────────────────

def _make_unsupported_exc():
    """404-style 'model rejected' exception."""
    return _APIExc("Model mimo-v2.5 not found", status_code=404)


def _make_pure_auth_exc():
    """401 with no model wording — should NOT trigger aux fallback."""
    return _APIExc("missing or invalid API key", status_code=401)


def test_call_llm_falls_back_to_provider_default_aux_model():
    """When the main model is rejected, fall back to the provider's default
    aux model (gemini-3-flash for opencode-zen) instead of re-raising."""
    from agent import auxiliary_client as ac

    rejected_model = "mimo-v2.5"
    aux_default = "gemini-3-flash"
    provider = "opencode-zen"

    primary_response = MagicMock()
    primary_response.choices = [MagicMock()]
    primary_response.choices[0].message.content = "ok via aux default"

    primary_client = MagicMock()
    primary_client.base_url = "https://opencode.ai/zen/go/v1"
    # First call (main model) raises; second call (fallback) succeeds.
    primary_client.chat.completions.create.side_effect = [
        _make_unsupported_exc(),
        primary_response,
    ]

    fallback_client = MagicMock()
    fallback_client.base_url = "https://opencode.ai/zen/go/v1"
    fallback_client.chat.completions.create.return_value = primary_response

    with patch.object(ac, "_get_cached_client", side_effect=[(primary_client, rejected_model), (fallback_client, aux_default)]) as gcc, \
         patch.object(ac, "_get_aux_model_for_provider", return_value=aux_default) as gaux, \
         patch.object(ac, "_resolve_task_provider_model",
                      return_value=(provider, rejected_model, "", "", "")), \
         patch.object(ac, "_get_task_extra_body", return_value={}):
        result = ac.call_llm(
            task="title_generation",
            messages=[{"role": "user", "content": "hi"}],
        )

    # Detector was consulted
    gaux.assert_called_once_with(provider)
    # The SECOND _get_cached_client call must have used the aux default model
    assert gcc.call_count == 2
    _, kwargs2 = gcc.call_args_list[1]
    # Positional second argument is the model
    _, model_arg = gcc.call_args_list[1][0][:2]
    assert model_arg == aux_default

    # The result was validated from the fallback client
    assert result.choices[0].message.content == "ok via aux default"


def test_call_llm_pure_auth_error_does_not_trigger_aux_fallback():
    """A 401 with no model wording must NOT be swallowed by the new fallback."""
    from agent import auxiliary_client as ac

    with patch.object(ac, "_get_cached_client") as gcc, \
         patch.object(ac, "_get_aux_model_for_provider", return_value="gemini-3-flash") as gaux, \
         patch.object(ac, "_resolve_task_provider_model",
                      return_value=("some-provider", "some-model", "", "", "")), \
         patch.object(ac, "_get_task_extra_body", return_value={}):
        primary = MagicMock()
        primary.base_url = "https://example/v1"
        primary.chat.completions.create.side_effect = _make_pure_auth_exc()
        gcc.return_value = (primary, "some-model")

        with pytest.raises(Exception, match="missing or invalid API key"):
            ac.call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hi"}],
            )

        # No fallback attempt for pure auth errors
        gaux.assert_not_called()


def test_call_llm_aux_fallback_no_op_when_rejected_is_default():
    """If the rejected model IS the provider's default aux model, no retry."""
    from agent import auxiliary_client as ac

    model = "gemini-3-flash"
    provider = "opencode-zen"

    with patch.object(ac, "_get_cached_client") as gcc, \
         patch.object(ac, "_get_aux_model_for_provider", return_value=model), \
         patch.object(ac, "_resolve_task_provider_model",
                      return_value=(provider, model, "", "", "")), \
         patch.object(ac, "_get_task_extra_body", return_value={}):
        primary = MagicMock()
        primary.base_url = "https://opencode.ai/zen/go/v1"
        primary.chat.completions.create.side_effect = _make_unsupported_exc()
        gcc.return_value = (primary, model)

        with pytest.raises(Exception, match="not found"):
            ac.call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hi"}],
            )

        # Only ONE client request (no fallback since rejected == aux default)
        assert gcc.call_count == 1


def test_call_llm_auto_user_skips_aux_model_fallback():
    """For ``provider: auto`` users the rejected model IS the aux model —
    retrying with the same provider's default would be a no-op. Skip."""
    from agent import auxiliary_client as ac

    with patch.object(ac, "_get_cached_client") as gcc, \
         patch.object(ac, "_get_aux_model_for_provider", return_value="gemini-3-flash") as gaux, \
         patch.object(ac, "_resolve_task_provider_model",
                      return_value=("auto", "mimo-v2.5", "", "", "")), \
         patch.object(ac, "_get_task_extra_body", return_value={}):
        primary = MagicMock()
        primary.base_url = "https://opencode.ai/zen/go/v1"
        primary.chat.completions.create.side_effect = _make_unsupported_exc()
        gcc.return_value = (primary, "mimo-v2.5")

        with pytest.raises(Exception, match="not found"):
            ac.call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hi"}],
            )

        # No default-aux fallback for auto users
        gaux.assert_not_called()
