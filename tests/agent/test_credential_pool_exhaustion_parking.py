"""BUILD-262: the credential pool's ``CredentialPoolExhausted`` must be
parked/aborted by the retry loop in ``agent.conversation_loop.run_conversation``
— not treated as a signal to loop calling rotation again.

Source-inspection style (matches ``test_nous_oauth_401_guidance.py`` and
``test_gemini_fast_fallback.py`` for this same giant function): asserting on
the source text pins the control-flow shape so a future refactor can't
silently reintroduce a `continue` here and reproduce the 2026-07-08 incident
(4,323 tight-loop iterations over 43 minutes, ~2 req/sec, zero backoff, all
because a bogus "successful rotation" kept making this exact call site
`continue`).

The pool-layer behavior (bounded retries, typed error, cooldown/backoff,
same-entry guard, log-once) is covered with real execution in
``tests/agent/test_credential_pool.py`` and
``tests/agent/test_credential_pool_routing.py``.
"""
from __future__ import annotations

import inspect
import re

from agent import conversation_loop


def _recover_with_pool_call_and_handler_source() -> str:
    """Return the source slice from the ``_recover_with_credential_pool``
    call through its exception handling, up to (but not including) the
    ``if recovered_with_pool:`` line that resumes normal flow."""
    source = inspect.getsource(conversation_loop.run_conversation)
    start = source.index("agent._recover_with_credential_pool(")
    end = source.index("if recovered_with_pool:", start)
    return source[start:end]


def test_credential_pool_exhausted_is_imported():
    assert "CredentialPoolExhausted" in inspect.getsource(conversation_loop)
    assert conversation_loop.CredentialPoolExhausted is not None


def test_recover_with_credential_pool_call_is_wrapped_in_try_except():
    source = inspect.getsource(conversation_loop.run_conversation)
    call_idx = source.index("agent._recover_with_credential_pool(")
    preceding = source[:call_idx]
    # The nearest preceding `try:`/`except` pair must be the one guarding
    # this call — i.e. a `try:` appears after the last `except` before it,
    # closer to the call than any unrelated try block.
    last_try = preceding.rfind("\n                try:\n")
    assert last_try != -1, (
        "Expected `agent._recover_with_credential_pool(...)` to be called "
        "inside a `try:` block."
    )
    assert "except CredentialPoolExhausted as" in source[call_idx:call_idx + 800], (
        "Expected a `except CredentialPoolExhausted` handler immediately "
        "after the _recover_with_credential_pool call."
    )


def test_pool_exhausted_handler_does_not_continue_the_retry_loop():
    """The actual spin-site regression guard: whatever the except block
    does, it must NOT contain a bare `continue` that resumes the tight
    retry loop with no backoff."""
    handler_source = _recover_with_pool_call_and_handler_source()
    except_idx = handler_source.index("except CredentialPoolExhausted")
    handler_body = handler_source[except_idx:]

    # No `continue` statement anywhere in the exception handler body.
    assert not re.search(r"^\s*continue\s*$", handler_body, re.MULTILINE), (
        "The CredentialPoolExhausted handler must not `continue` the retry "
        "loop — that reproduces the 2026-07-08 incident (spinning on "
        "rotation with zero backoff)."
    )


def test_pool_exhausted_handler_parks_with_a_failed_result():
    """Parking/aborting means returning a terminal, non-retrying result —
    matching the shape other terminal branches of run_conversation use
    (see the FailoverReason.content_policy_blocked / nonretryable-summary
    early returns a few hundred lines below this one in the same loop)."""
    handler_source = _recover_with_pool_call_and_handler_source()
    except_idx = handler_source.index("except CredentialPoolExhausted")
    handler_body = handler_source[except_idx:]

    assert "return {" in handler_body
    assert '"failed": True' in handler_body
    assert '"completed": False' in handler_body


def test_pool_exhausted_handler_logs_a_warning():
    handler_source = _recover_with_pool_call_and_handler_source()
    except_idx = handler_source.index("except CredentialPoolExhausted")
    handler_body = handler_source[except_idx:]

    assert "logger.warning(" in handler_body
    assert "credential pool exhausted" in handler_body.lower()
