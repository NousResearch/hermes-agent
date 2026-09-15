"""#82154: an unverified billing verdict must carry its ambiguity through
every downstream surface — the returned terminal response, the structured
result fields, the credential-pool failure_reason, and the persisted entry —
not just the explanatory guidance text.

Anthropic returns the identical "out of extra usage" HTTP 400 body on a
subscription OAuth token both for genuine overage depletion and for a
server-side content-filter rejection of the request. The classifier marks
that verdict ``billing_unverified``; these tests pin that the marking is not
dropped on the way out.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.conversation_loop import _billing_failure_result, _billing_terminal_label
from agent.error_classifier import FailoverReason, classify_api_error


class MockAPIError(Exception):
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


_EXTRA_USAGE_BODY = (
    "You're out of extra usage. Add more at claude.ai/settings/usage and keep going."
)


def _classified_unverified():
    e = MockAPIError(
        _EXTRA_USAGE_BODY,
        status_code=400,
        body={"error": {"type": "invalid_request_error", "message": _EXTRA_USAGE_BODY}},
    )
    return classify_api_error(e, provider="anthropic")


def _classified_confirmed():
    e = MockAPIError(
        "Your credit balance is too low to access the Anthropic API.",
        status_code=400,
        body={"error": {
            "type": "invalid_request_error",
            "message": "Your credit balance is too low to access the Anthropic API.",
        }},
    )
    return classify_api_error(e, provider="anthropic")


# ── Returned terminal response ───────────────────────────────────────────────


class TestTerminalResponse:
    def test_unverified_terminal_response_does_not_assert_billing(self):
        """The exact ambiguous 400 must not produce an unhedged
        'Billing or credits exhausted' terminal response."""
        result = _billing_failure_result(
            classified=_classified_unverified(),
            summary="HTTP 400: out of extra usage",
            messages=[],
            api_call_count=3,
            provider="anthropic",
            base_url="https://api.anthropic.com",
            model="claude-opus-5",
        )
        final = result["final_response"]
        assert not final.startswith("Billing or credits exhausted")
        assert "unverified" in final
        assert "content-filter" in final or "content filter" in final
        # The guidance must ride along and hedge too.
        assert "still shows quota remaining" in final

    def test_unverified_terminal_response_structured_fields(self):
        """The structured result carries the ambiguity, not just the prose."""
        result = _billing_failure_result(
            classified=_classified_unverified(),
            summary="HTTP 400: out of extra usage",
            messages=[],
            api_call_count=3,
            provider="anthropic",
            base_url="https://api.anthropic.com",
            model="claude-opus-5",
        )
        assert result["failed"] is True
        assert result["failure_reason"] == "billing"
        assert result["billing_unverified"] is True
        block = result["billing_block"]
        if block is not None:  # None only if billing_links is unavailable
            assert block.get("unverified") is True

    def test_confirmed_terminal_response_stays_assertive(self):
        """A confirmed billing verdict keeps the original terminal label and
        carries no ambiguity flag."""
        result = _billing_failure_result(
            classified=_classified_confirmed(),
            summary="HTTP 400: credit balance too low",
            messages=[],
            api_call_count=1,
            provider="anthropic",
            base_url="https://api.anthropic.com",
            model="claude-opus-5",
        )
        assert result["final_response"].startswith("Billing or credits exhausted")
        assert result["billing_unverified"] is False
        block = result["billing_block"]
        if block is not None:
            assert "unverified" not in block

    def test_terminal_label_contract(self):
        assert _billing_terminal_label("boom", False) == "Billing or credits exhausted: boom"
        hedged = _billing_terminal_label("boom", True)
        assert "unverified" in hedged
        assert "content-filter" in hedged
        assert not hedged.startswith("Billing or credits exhausted")


# ── Credential-pool plumbing ─────────────────────────────────────────────────


class TestPoolFailureReason:
    def _run_recovery(self, *, billing_unverified: bool) -> dict:
        """Drive recover_with_credential_pool with a billing classification and
        capture what the pool is told."""
        from agent.agent_runtime_helpers import recover_with_credential_pool

        captured: dict = {}
        next_entry = SimpleNamespace(label="secondary")

        class _Pool:
            provider = "anthropic"

            def current(self):
                return None

            def entries(self):
                return []

            def mark_exhausted_and_rotate(self, **kwargs):
                captured.update(kwargs)
                return next_entry

        agent = SimpleNamespace(
            provider="anthropic",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-oat01-test",
            _credential_pool=_Pool(),
            _credential_pool_entry_id=None,
            _swap_credential=MagicMock(),
        )
        recovered, _ = recover_with_credential_pool(
            agent,
            status_code=400,
            has_retried_429=False,
            classified_reason=FailoverReason.billing,
            billing_unverified=billing_unverified,
        )
        assert recovered is True
        return captured

    def test_unverified_billing_reaches_pool_as_unverified(self):
        captured = self._run_recovery(billing_unverified=True)
        assert captured["failure_reason"] == "billing_unverified"

    def test_confirmed_billing_reaches_pool_as_billing(self):
        captured = self._run_recovery(billing_unverified=False)
        assert captured["failure_reason"] == "billing"
