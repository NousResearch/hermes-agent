"""Codex's ``usage_limit_reached`` quota wall is one verdict for the main loop and the aux ladder.

An exhausted plan quota is not a per-minute throttle: the key cannot serve another request until the
window resets. The main classifier already calls it ``billing`` (``error_classifier._status_429``),
but ``auxiliary_client._PAYMENT_KEYWORDS`` carried neither Codex's structured
``usage_limit_reached`` code nor its plain-text ``"the usage limit has been reached"`` wording, so
``_is_payment_error`` returned False and the aux ladder took the rate-limit branch at
``_run_recovery_ladder``: one extra same-provider retry against a key that is definitionally out of
quota, on every compression / session-summarization / title / curator call. Reported against
PR #34024 by @carltonawong, who saw it as repeated summarization failures while the main agent
looked alive.
"""
import pytest

from agent import auxiliary_client as ac
from agent.error_classifier import FailoverReason, classify_api_error

# The body Codex actually returns, as the SDK stringifies it into the exception (what the aux
# predicates see — they read ``str(exc)``, not ``.body``).
_CODEX_QUOTA_REPR = (
    "Error code: 429 - {'error': {'type': 'usage_limit_reached', 'message': 'The usage limit "
    "has been reached', 'plan_type': 'pro'}}"
)
_CODEX_QUOTA_CODE_REPR = (
    "Error code: 429 - {'error': {'code': 'usage_limit_reached', 'message': 'The usage limit "
    "has been reached'}}"
)


class _Err(Exception):
    def __init__(self, msg, status=429, body=None):
        super().__init__(msg)
        self.status_code = status
        if body is not None:
            self.body = body


@pytest.mark.parametrize("text", [_CODEX_QUOTA_REPR, _CODEX_QUOTA_CODE_REPR])
def test_codex_quota_wall_is_a_payment_error_for_the_aux_ladder(text):
    """Both payload shapes (``error.type`` and ``error.code``) are quota exhaustion, not a throttle."""
    exc = _Err(text)
    assert ac._is_payment_error(exc), "aux ladder must not treat an exhausted plan quota as a throttle"


def test_aux_and_main_loop_agree_on_the_codex_quota_wall():
    """One verdict, two surfaces: a body the main loop ends the turn on must not earn an aux retry."""
    exc = _Err(_CODEX_QUOTA_REPR, body={"error": {"type": "usage_limit_reached", "message": "The usage limit has been reached", "plan_type": "pro"}})
    assert classify_api_error(exc).reason is FailoverReason.billing
    assert ac._is_payment_error(exc)
    # ``_run_recovery_ladder`` gates its extra same-provider retry on exactly this conjunction
    # (``_is_rate_limit_error(err) and not _is_payment_error(err)``). It must be False here, or the
    # ladder burns a call on a key that cannot serve one.
    assert not (ac._is_rate_limit_error(exc) and not ac._is_payment_error(exc))


def test_a_reset_bearing_usage_limit_is_still_a_retryable_throttle():
    """Converse guard: a usage-limit body that names its reset window is transient, not exhaustion.

    ``_status_429`` deliberately keeps these retryable (``_has_usage_limit_transient_signal``), and
    the aux ladder must not start failing them closed just because the quota vocabulary widened.
    """
    transient = _Err(
        "Error code: 429 - {'error': {'type': 'usage_limit_reached', 'message': 'Usage limit "
        "reached, try again in 60 seconds', 'resets_in_seconds': 60}}",
        body={"error": {"type": "usage_limit_reached", "message": "Usage limit reached, try again in 60 seconds", "resets_in_seconds": 60}},
    )
    assert classify_api_error(transient).reason is FailoverReason.rate_limit
    assert ac._is_rate_limit_error(transient)


def test_an_ordinary_throttle_is_untouched():
    """A plain per-key 429 keeps its rate-limit verdict and its same-provider retry."""
    throttle = _Err("Error code: 429 - {'error': {'message': 'Rate limit exceeded for this API key', 'code': 429}}")
    assert not ac._is_payment_error(throttle)
    assert ac._is_rate_limit_error(throttle)
