"""BUILD-262 / BUILD-342: the credential pool's ``CredentialPoolExhausted``
must never be treated as a signal to loop calling *pool rotation* again
(that reproduces the 2026-07-08 spin incident) — but it MUST now escalate
to the cross-provider fallback chain before parking the turn (BUILD-342:
the handler's hard `return` previously fired ~550 lines before the eager
fallback block, so a configured `fallback_providers` chain was unreachable
on pool exhaustion).

Source-inspection style (matches ``test_nous_oauth_401_guidance.py`` and
``test_gemini_fast_fallback.py`` for this same giant function): asserting on
the source text pins the control-flow shape so a future refactor can't
silently reintroduce a direct re-call into pool rotation and reproduce the
2026-07-08 incident (4,323 tight-loop iterations over 43 minutes, ~2
req/sec, zero backoff, all because a bogus "successful rotation" kept
making this exact call site `continue` straight back into the dead pool).

The `continue` that DOES belong in this handler post-BUILD-342 only
follows a successful, bounded ``agent._try_activate_fallback(...)``
activation (see ``tests/run_agent/test_build342_pool_exhausted_fallback.py``
for real end-to-end execution of that path) — never a raw retry of
rotation on the same pool.

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


def test_pool_exhausted_handler_does_not_retry_pool_rotation_directly():
    """The actual spin-site regression guard: the handler must never call
    credential-pool rotation again itself — that reproduces the
    2026-07-08 incident (spinning on rotation with zero backoff). It MAY
    (post-BUILD-342) `continue` the retry loop, but only via a bounded
    cross-provider fallback activation, never by re-entering pool
    rotation on its own cadence."""
    handler_source = _recover_with_pool_call_and_handler_source()
    except_idx = handler_source.index("except CredentialPoolExhausted")
    handler_body = handler_source[except_idx:]

    assert "_recover_with_credential_pool(" not in handler_body
    assert "mark_exhausted_and_rotate" not in handler_body


def test_pool_exhausted_handler_attempts_cross_provider_fallback():
    """BUILD-342: pool exhaustion must give the configured fallback chain
    a bounded activation attempt before parking — the fix for the 2026-
    07-10 incident where a healthy `fallback_providers` chain was
    unreachable from this handler."""
    handler_source = _recover_with_pool_call_and_handler_source()
    except_idx = handler_source.index("except CredentialPoolExhausted")
    handler_body = handler_source[except_idx:]

    assert "_try_activate_fallback(" in handler_body


def test_pool_exhausted_handler_continue_is_gated_on_fallback_activation():
    """The only `continue` in this handler must sit inside the
    successful-activation branch of `_try_activate_fallback(...)` — never
    a bare/unconditional `continue` that would resume the loop
    regardless of whether a healthy fallback was actually found."""
    handler_source = _recover_with_pool_call_and_handler_source()
    except_idx = handler_source.index("except CredentialPoolExhausted")
    handler_body = handler_source[except_idx:]

    match = re.search(
        r"if agent\._try_activate_fallback\([^)]*\):\n(?:.+\n)*?\s*continue\s*\n",
        handler_body,
    )
    assert match is not None, (
        "Expected `continue` to be reachable only from inside "
        "`if agent._try_activate_fallback(...):` in the pool-exhausted handler."
    )

    # And there must be no OTHER, unguarded `continue` in the handler body.
    other_continues = [
        m for m in re.finditer(r"^\s*continue\s*$", handler_body, re.MULTILINE)
        if not (match.start() <= m.start() < match.end())
    ]
    assert not other_continues, (
        "Found a `continue` in the CredentialPoolExhausted handler outside "
        "the guarded fallback-activation branch."
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
