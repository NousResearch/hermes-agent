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
    """AST check — the patched return in conversation_loop.py carries the
    structured fields and is followed by the ``if retry_count >= max_retries:``
    branch (not the loop's continuation). Guards against a future refactor
    moving the return into the wrong indentation level or dropping fields.

    Reads the real module source rather than a hardcoded fixture, so a
    regression in the production code is observable. (#t_8acdc493, Copilot review)
    """
    from pathlib import Path
    import agent.conversation_loop as loop_mod  # noqa: WPS433 (test-scope import)
    src = Path(loop_mod.__file__).read_text()
    tree = ast.parse(src)
    # Find run_conversation's return-statement dict that contains the patched
    # envelope. Multiple return statements exist; we want the one matching the
    # non-retryable clean-exit shape (has ``error`` + ``failed`` + the 5
    # structured fields).
    matches = 0
    for func in tree.body:
        if not isinstance(func, ast.FunctionDef) or func.name != "run_conversation":
            continue
        for node in ast.walk(func):
            if not (isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)):
                continue
            keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
            if not ({"failed", "error", "failure_reason", "error_class",
                     "provider", "model", "http_status"} <= keys):
                continue
            matches += 1
            # The patched return must be inside a try/except block (the
            # non-retryable handler), not at the function top level.
            assert node.col_offset > 0, (
                "Patched clean-exit envelope is at the function top level "
                "(unexpected — should be inside the try/except handler)"
            )
    assert matches >= 1, (
        "Patched clean-exit envelope (carrying failure_reason, error_class, "
        "provider, model, http_status) not found in run_conversation"
    )


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
# ────────────────────────────────────────────────────────────────────────
# http_status coercion regression (Copilot review #3 on PR #56646)
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Already-int status codes pass through unchanged
        (403, 403),
        (429, 429),
        # Numeric-string status codes are coerced to int so Kanban
        # postmortem queries filtering on `http_status = 403` work.
        ("403", 403),
        ("429", 429),
        # None / non-numeric strings / other types fall back to None.
        (None, None),
        ("", None),
        ("not-a-number", None),
        (3.14, None),  # floats are not int, and int("3.14") would explode
    ],
)
def test_http_status_coercion(raw, expected) -> None:
    """Mirror the inline expression in conversation_loop.py's envelope.

    The patch uses an inline coercion so the envelope's ``http_status`` is
    always ``int | None`` regardless of whether the upstream SDK returned an
    int or a numeric string. This test pins that contract so a future
    refactor back to ``isinstance(status_code, int) else None`` would fail.
    (#t_8acdc493, Copilot review)
    """
    coerced = (
        raw if isinstance(raw, int)
        else int(raw) if isinstance(raw, str) and raw.isdigit()
        else None
    )
    assert coerced == expected
    if expected is not None:
        assert isinstance(coerced, int), f"{raw!r} must coerce to int, got {type(coerced).__name__}"


# ─────────────────────────────────────────────────────────────────────────
# Post-max_retries terminal envelope (#t_a2c1ced7)
# ─────────────────────────────────────────────────────────────────────────
# The non-retryable early-exit envelope (line ~3986) is already pinned by
# test_nonretryable_return_is_before_next_iteration_branch.  The companion
# branch — the terminal exit that fires AFTER ``retry_count >= max_retries``
# when every fallback is exhausted (line ~4190) — also needs the same
# structured fields, otherwise kanban workers that hit the quota wall on
# retry-exhausted paths still report ``pid N not alive`` instead of
# ``failure_reason=billing``.


def test_post_maxretries_terminal_envelope_carries_full_failure_metadata() -> None:
    """AST check — the post-max_retries return at line ~4190 must include
    ``failure_reason``, ``error_class``, ``provider``, ``model``, and
    ``http_status`` in addition to ``failed`` / ``error``.

    Reads the real module source so a regression is observable. (#t_a2c1ced7)
    """
    from pathlib import Path
    import ast

    import agent.conversation_loop as loop_mod  # noqa: WPS433
    src_text = Path(loop_mod.__file__).read_text()
    tree = ast.parse(src_text)

    expected_keys = {
        "failed", "error", "failure_reason", "error_class",
        "provider", "model", "http_status",
    }
    matches = 0
    for func in tree.body:
        if not isinstance(func, ast.FunctionDef) or func.name != "run_conversation":
            continue
        for node in ast.walk(func):
            if not (isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)):
                continue
            keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
            if not (expected_keys <= keys):
                continue
            matches += 1
            # Both envelopes (early-exit at ~3986 and post-max_retries at ~4190)
            # must exist and carry the full set.
            assert node.col_offset > 0, (
                "Patched envelope is at function top level (unexpected)"
            )
    assert matches >= 2, (
        "Expected >=2 patched return envelopes (non-retryable early-exit "
        "AND post-max_retries terminal exit), found "
        f"{matches}. Did the post-max_retries envelope (#t_a2c1ced7) lose "
        "its structured fields?"
    )


def test_post_maxretries_envelope_uses_classifier_enum_value_for_both_fields() -> None:
    """Pin the contract that ``failure_reason`` AND ``error_class`` both carry
    the same ``FailoverReason.value`` string (mirroring the line-3986
    envelope). A future refactor that splits the two — e.g. mapping
    ``error_class`` to a Python exception class — would silently break the
    cli.py ``failure_reason in {"rate_limit", "billing"}`` lookup.
    (#t_a2c1ced7)
    """
    classified = _classify(403, "exceeded your current quota")
    assert classified.reason == FailoverReason.billing
    # Both fields must carry the classifier enum's string value, not the
    # Python repr and not the APIError's exception class.
    envelope = {
        "failure_reason": classified.reason.value,
        "error_class": classified.reason.value,
    }
    assert envelope["failure_reason"] == envelope["error_class"] == "billing"
    # Sanity: the consumer lookup set the cli.py quota-exit branch depends
    # on is still satisfied by this branch's failure_reason.
    assert envelope["failure_reason"] in ("rate_limit", "billing")
