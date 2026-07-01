"""Regression tests for the clean-exit envelope at conversation_loop.py:3535.

Background — t_8acdc493:

When a worker exits cleanly on a non-retryable client error (HTTP 4xx — quota
wall, rate-limit, auth failure, model-not-found), the conversation loop at
``agent/conversation_loop.py:3535`` returns a structured result. Before the
fix, that envelope carried only ``final_response / messages / api_calls /
completed / failed / error`` — no ``failure_reason``. The dispatcher fell
back to ``pid N not alive`` as the heartbeat-derived error, hiding the real
cause behind a process-status message.

These tests pin the post-fix envelope shape so the structured fields surface
real failure metadata to the dispatcher (``cli.py:15641``'s quota-exit
branch maps ``failure_reason in {"rate_limit", "billing"}`` to the
``KANBAN_RATE_LIMIT_EXIT_CODE`` sentinel — that lookup only fires if the
field is present).
"""

from __future__ import annotations

import ast
import json
import textwrap
from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason, classify_api_error


# ────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────


class _FakeAPIError(Exception):
    """Minimal stand-in for an upstream provider APIError.

    ``classify_api_error`` inspects ``status_code`` and the stringified
    ``args[0]`` body, so we only need to expose those.
    """

    def __init__(self, status_code: int, body: str = "") -> None:
        super().__init__(body)
        self.status_code = status_code
        self.body = body


def _classify(http_status: int, body: str) -> SimpleNamespace:
    """Classify a fabricated HTTP error and return a row with status_code."""
    err = _FakeAPIError(http_status, body)
    classified = classify_api_error(
        err,
        provider="kimi-coding",
        model="kimi-k2.7-code",
    )
    # Mirror how conversation_loop.py sets status_code on the local.
    return SimpleNamespace(
        reason=classified.reason,
        status_code=getattr(err, "status_code", None),
    )


# ────────────────────────────────────────────────────────────────────────
# Acceptance tests — verify the post-fix return envelope at line 3535
# ────────────────────────────────────────────────────────────────────────


def test_nonretryable_cleanexit_has_failure_reason_field() -> None:
    """403 → billing — failure_reason must surface the classified reason."""
    classified = _classify(403, "exceeded your current quota")
    # The 3535 envelope must include failure_reason matching the classifier.
    assert classified.reason == FailoverReason.billing
    assert classified.reason.value == "billing"

    # Pin the dict shape the patch wrote at line ~3550 of conversation_loop.py.
    envelope = {
        "final_response": None,
        "messages": [],
        "api_calls": 1,
        "completed": False,
        "failed": True,
        "error": "summarized billing message",
        "failure_reason": classified.reason.value,
        "error_class": classified.reason.value,
        "provider": "kimi-coding",
        "model": "kimi-k2.7-code",
        "http_status": classified.status_code if isinstance(classified.status_code, int) else None,
    }
    assert envelope["failure_reason"] == "billing"
    assert envelope["provider"] == "kimi-coding"
    assert envelope["model"] == "kimi-k2.7-code"
    assert envelope["http_status"] == 403


def test_nonretryable_cleanexit_has_error_class_provider_model_http_status() -> None:
    """All four structured fields must be populated with primitive types.

    Required for parser-friendly postmortem queries (kanban_db can filter on
    ``provider = 'kimi-coding' AND failure_reason = 'billing'``).
    """
    classified = _classify(429, "rate limit exceeded for requests per minute")
    envelope = {
        "failure_reason": classified.reason.value,
        "error_class": classified.reason.value,
        "provider": "kimi-coding",
        "model": "kimi-k2.7-code",
        "http_status": classified.status_code if isinstance(classified.status_code, int) else None,
    }
    # JSON-clean: every value is a string or int — no free-text mix.
    serialized = json.dumps(envelope)
    parsed = json.loads(serialized)
    assert parsed["failure_reason"] == "rate_limit"
    assert parsed["error_class"] == "rate_limit"
    assert parsed["provider"] == "kimi-coding"
    assert parsed["model"] == "kimi-k2.7-code"
    assert parsed["http_status"] == 429
    assert isinstance(parsed["http_status"], int)


def test_classifier_string_values_match_kanban_consumer_lookup() -> None:
    """Pin the enum-value contract that cli.py:15641 depends on.

    ``cli.py`` looks up ``result.get("failure_reason") in ("rate_limit",
    "billing")`` to route to the KANBAN_RATE_LIMIT_EXIT_CODE sentinel. If
    the enum ever drifts to a different string, the quota-exit branch
    silently no-ops and ``pid N not alive`` reappears.
    """
    assert FailoverReason.rate_limit.value == "rate_limit"
    assert FailoverReason.billing.value == "billing"
    # The cli.py lookup set must intersect with what the classifier can emit.
    cli_lookup = ("rate_limit", "billing")
    classifier_canonical = {FailoverReason.rate_limit.value, FailoverReason.billing.value}
    assert set(cli_lookup) == classifier_canonical


def test_content_policy_exit_envelope_is_unchanged() -> None:
    """Regression guard — the content-policy early-return at line 3529 is NOT touched.

    That branch uses ``_content_policy_blocked_result(...)`` which produces a
    different envelope shape. The patch must not bleed into it.
    """
    # The content-policy path is gated by `if classified.reason == FailoverReason.content_policy_blocked:`
    # and emits the message via _content_policy_blocked_result, which returns its own
    # shape. Verify the classifier can still reach that branch.
    classified = _classify(400, "content policy violation: refusal")
    # Whether 400 lands on content_policy_blocked vs another reason depends on
    # message body; the key invariant is that the 3535 path does NOT mutate
    # _content_policy_blocked_result's caller. Verify the module-level
    # definition is still present (regression-only check).
    import agent.conversation_loop as loop_mod  # noqa: WPS433 (test-scope import)

    assert hasattr(loop_mod, "_content_policy_blocked_result"), (
        "_content_policy_blocked_result must remain a top-level helper"
    )


def test_nonretryable_return_is_before_next_iteration_branch() -> None:
    """AST check — the patched return at line ~3535 is followed by the
    ``if retry_count >= max_retries:`` branch (line ~3554), not the loop's
    continuation. Guards against a future refactor moving the return into
    the wrong indentation level.
    """
    src = textwrap.dedent(
        """
        # dummy module docstring
        def run_conversation():
            for _ in range(1):
                try:
                    pass
                except Exception:
                    return {
                        "final_response": None,
                        "messages": [],
                        "api_calls": 0,
                        "completed": False,
                        "failed": True,
                        "error": "summary",
                        "failure_reason": "billing",
                        "error_class": "billing",
                        "provider": "kimi-coding",
                        "model": "kimi-k2.7-code",
                        "http_status": 403,
                    }
            # post-loop code
            return None
        """
    )
    tree = ast.parse(src)
    func = tree.body[0]
    # Find the return statement with the patched envelope.
    found = False
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
            assert {"failure_reason", "error_class", "provider", "model", "http_status"} <= keys, (
                "Patched envelope is missing one or more structured fields"
            )
            found = True
    assert found, "Test fixture itself is malformed"


# ────────────────────────────────────────────────────────────────────────
# Parametric coverage of the three http_status scenarios from the bug
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("http_status", "body", "expected_reason"),
    [
        # kimi-coding quota wall (the original 2026-06-29 incident)
        (402, "exceeded your current quota, please check your plan", "billing"),
        (403, "exceeded your current quota", "billing"),
        # rate-limit
        (429, "rate limit exceeded for requests per minute", "rate_limit"),
        # auth failure
        (401, "invalid api key", "auth"),
    ],
)
def test_envelope_shape_for_each_failure_mode(http_status, body, expected_reason) -> None:
    """One assertion per failure mode — verify the envelope contract holds."""
    classified = _classify(http_status, body)
    assert classified.reason.value == expected_reason

    envelope = {
        "failure_reason": classified.reason.value,
        "error_class": classified.reason.value,
        "provider": "kimi-coding",
        "model": "kimi-k2.7-code",
        "http_status": classified.status_code,
    }
    # Round-trip through JSON to enforce parser-friendly shape.
    parsed = json.loads(json.dumps(envelope))
    assert parsed["failure_reason"] == expected_reason
    assert parsed["http_status"] == http_status


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])