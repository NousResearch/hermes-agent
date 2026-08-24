"""P5(b) at the CALLER: a connector egress decline must not fall back.

The adapter-level tests prove the decline REACHES the caller. These prove the
caller ACTS on it. Review round 1 blocker B-3: `_approval_send_outcome` had
only sent/failed/ambiguous, so a decline collapsed into `failed` — which is
precisely the cue to run the plain-text fallback into the chat the connector
had just refused. The adapter fix improved the error STRING while the
user-visible behaviour stayed identical to base.

These tests drive the real `gateway.run` classifier and the real
`tools.slash_confirm` registry; only the SendResult (the connector's answer)
is constructed.
"""

from __future__ import annotations

import concurrent.futures
from types import SimpleNamespace

import pytest

DECLINE_ERROR = (
    "discord egress declined: target is not an approved destination for this connection"
)
LANE_ERROR = "relay prompt op unavailable"


def _future(result):
    fut: concurrent.futures.Future = concurrent.futures.Future()
    fut.set_result(result)
    return fut


def _result(*, success: bool, error: str | None = None):
    return SimpleNamespace(success=success, error=error, message_id=None)


# ── the classifier ──────────────────────────────────────────────────────────


def test_decline_is_not_classified_as_failed():
    """`failed` is the fallback cue; a decline must not wear it."""
    from gateway.run import _approval_send_outcome

    outcome = _approval_send_outcome(
        _future(_result(success=False, error=DECLINE_ERROR)), timeout=5
    )
    assert outcome == "declined"


def test_genuine_lane_failure_still_falls_back():
    """The guard must not swallow real failures: those still re-ask."""
    from gateway.run import _approval_send_outcome

    outcome = _approval_send_outcome(
        _future(_result(success=False, error=LANE_ERROR)), timeout=5
    )
    assert outcome == "failed"


def test_success_and_timeout_verdicts_unchanged():
    """No collateral change to the two settled verdicts."""
    from gateway.run import _approval_send_outcome

    assert (
        _approval_send_outcome(_future(_result(success=True)), timeout=5) == "sent"
    )

    pending: concurrent.futures.Future = concurrent.futures.Future()
    assert _approval_send_outcome(pending, timeout=0.05) == "ambiguous"


@pytest.mark.parametrize(
    "error",
    [
        DECLINE_ERROR,
        "slack egress declined: destination not permitted",
        # Case-insensitivity is part of the contract (`.lower()` in
        # is_egress_decline), so a connector that capitalises still classifies.
        "WhatsApp Egress Declined: target refused",
    ],
)
def test_decline_recognised_across_lanes(error):
    """The verdict follows the decline CONTRACT, not one lane's wording.

    The contract is `EGRESS_DECLINE_MARKER` ("egress declined:") or a
    structured `code`; an invented sentence like "EGRESS_DECLINED: ..." is NOT
    a decline and must not be treated as one. My first version of this test
    asserted that invented form and failed — the test was wrong, not the code.
    """
    from gateway.run import _approval_send_outcome

    assert (
        _approval_send_outcome(_future(_result(success=False, error=error)), timeout=5)
        == "declined"
    )


def test_non_decline_error_text_is_not_laundered_into_declined():
    """Fail-closed the other way: only the real contract yields `declined`.

    Without this, a permissive marker check would silence genuine failures —
    turning a lane outage into a silent no-fallback.
    """
    from gateway.run import _approval_send_outcome

    for error in (
        "connection reset by peer",
        "declined",  # bare word, not the marker sentence
        "egress declined",  # no colon: not the uniform marker
    ):
        assert (
            _approval_send_outcome(
                _future(_result(success=False, error=error)), timeout=5
            )
            == "failed"
        ), error


# ── the STRUCTURED response, not the rendered string ────────────────────────
#
# Round-2 review: I fixed the text-marker path and tested only the text-marker
# path. The adapter preserves the connector's own dict in `raw_response`, and
# classifying the error STRING instead loses two contracts.


def _result_raw(raw, *, error=None):
    from gateway.relay.egress import decline_error

    return SimpleNamespace(
        success=False, error=error if error is not None else decline_error(raw),
        raw_response=raw,
    )


def test_code_only_decline_is_declined_not_failed():
    """A decline carrying `code` and NO text must not fall back.

    `decline_error()` renders it as "relay egress declined" — no marker colon —
    so a string-based check returns `failed`, which is the fallback cue. This
    is the shape the connector actually sends when it has no human text.
    """
    from gateway.run import _approval_send_outcome

    raw = {"success": False, "code": "egress_declined"}
    assert _approval_send_outcome(_future(_result_raw(raw)), timeout=5) == "declined"


def test_ambiguous_result_is_never_a_definite_failure():
    """Lost ack ≠ refusal ≠ failure.

    `is_egress_decline` deliberately excludes `ambiguous`, because the frame
    may well have been applied. Flattening it into `failed` re-sends a card
    that may already be on the user's screen — the duplicate-card bug the
    ambiguous verdict was introduced to prevent.
    """
    from gateway.run import _approval_send_outcome

    raw = {"success": False, "ambiguous": True, "error": "lost ack"}
    assert _approval_send_outcome(_future(_result_raw(raw)), timeout=5) == "ambiguous"


def test_ambiguous_wins_over_decline_text():
    """An ambiguous result carrying decline text is STILL ambiguous.

    Order matters: a mid-write drop whose partial error happens to contain the
    marker must not be reported as a definite refusal.
    """
    from gateway.run import _approval_send_outcome

    raw = {
        "success": False,
        "ambiguous": True,
        "error": "discord egress declined: target is not approved",
    }
    assert _approval_send_outcome(_future(_result_raw(raw)), timeout=5) == "ambiguous"


def test_structured_lane_failure_still_falls_back():
    """A structured NON-decline failure keeps its fallback."""
    from gateway.run import _approval_send_outcome

    raw = {"success": False, "error": "connection reset"}
    assert _approval_send_outcome(_future(_result_raw(raw)), timeout=5) == "failed"


def test_legacy_connector_without_raw_response_still_classified():
    """No `raw_response` (older connector) falls back to the wire sentence."""
    from gateway.run import _approval_send_outcome

    legacy = SimpleNamespace(success=False, error=DECLINE_ERROR, raw_response=None)
    assert _approval_send_outcome(_future(legacy), timeout=5) == "declined"


def test_declined_clarify_aborts_instead_of_waiting_for_a_reply():
    """Review round 3, finding 4.

    `_clarify_send_disposition` handled `failed` and `ambiguous`; `declined`
    fell through to `wait_for_response`, so the agent blocked until
    clarify_timeout for a card that was REFUSED and can never be answered.
    A decline is more definitive than a failure, not less.
    """
    from types import SimpleNamespace

    from gateway.relay.egress import EGRESS_DECLINE_CODE
    from gateway.run import _clarify_send_disposition

    cleared = []
    clarify_mod = SimpleNamespace(
        clear_session=lambda sk: cleared.append(sk),
        get_clarify_timeout=lambda: 600,
        wait_for_response=lambda *a, **k: pytest.fail(
            "waited for a reply to a prompt the connector refused"
        ),
    )

    class _Fut:
        def result(self, timeout=None):
            return SimpleNamespace(
                success=False,
                error=None,
                raw_response={"success": False, "code": EGRESS_DECLINE_CODE},
            )

    abort = _clarify_send_disposition(
        _Fut(), session_key="sk1", clarify_mod=clarify_mod
    )
    assert abort is not None
    assert cleared == ["sk1"]


def test_ambiguous_clarify_still_waits():
    """Control: a possibly-delivered card must STAY armed for a late reply."""
    from types import SimpleNamespace

    from gateway.run import _clarify_send_disposition

    cleared = []
    clarify_mod = SimpleNamespace(
        clear_session=lambda sk: cleared.append(sk),
        get_clarify_timeout=lambda: 600,
        wait_for_response=lambda *a, **k: "answer",
    )

    class _Fut:
        def result(self, timeout=None):
            return SimpleNamespace(
                success=False,
                error="lost ack",
                raw_response={"success": False, "error": "x", "ambiguous": True},
            )

    assert (
        _clarify_send_disposition(_Fut(), session_key="sk1", clarify_mod=clarify_mod)
        is None
    )
    assert cleared == []
