"""Streaming-5xx unmask probe classification (sibling of ``chat_completion_helpers``).

The probe re-issues the request non-streaming into the same condition that produced the
5xx, so an overloaded engine can answer it with 401 and auth-flavoured wording (#136025).
"""
from agent.error_classifier import _OVERLOADED_PATTERNS, _extract_error_body, _extract_error_code

# One non-streaming re-issue per this window. Covers the outer retry loop (up to ~3
# attempts x backoff, well under 60s) so an outage doesn't double traffic every attempt,
# while later turns re-arm automatically.
_STREAM_5XX_PROBE_WINDOW_S = 60.0

# Structured ``error.code``/``error.type`` values that are a provider's declaration of an
# auth failure. A probe 401 carrying one of these is a real validation verdict; anything
# less specific is suspect while the original failure is a 5xx.
_PROBE_DECLARED_AUTH_CODES = frozenset({
    "invalid_api_key",       # OpenAI family
    "authentication_error",  # Anthropic
    "unauthenticated",       # Google/Gemini error.status
    "invalid_token",         # OAuth-style gateways
})


def _probe_401_is_overload_artifact(probe_err: Exception, probe_status: int) -> bool:
    """Whether a probe 4xx is the overload speaking, not the key failing (#136025).

    The probe re-issued into the same condition that produced the 5xx, so an overloaded
    engine can answer 401 with auth-flavoured wording (stepfun: the body opens with
    "Incorrect API key provided" and only then says the engine is overloaded). A
    structured error code declaring an auth failure always wins — text-first matching
    would misread that opening as a verdict. Otherwise overload wording in the error
    marks the 401 an artifact and the ORIGINAL 5xx must survive, or the credential pool
    benches every entry sharing the key on a verdict the provider never actually issued.
    """
    if probe_status != 401:
        return False
    body = _extract_error_body(probe_err)
    code = (_extract_error_code(body) or "").lower()
    if code in _PROBE_DECLARED_AUTH_CODES:
        return False
    return any(p in str(probe_err).lower() for p in _OVERLOADED_PATTERNS)
