"""Per-attempt recovery bookkeeping for the conversation turn loop.

The inner retry loop in ``run_conversation`` (``while retry_count <
max_retries``) makes several distinct recovery attempts on a single model API
call: a credential-pool 429 retry, a per-provider OAuth refresh (codex,
anthropic, nous, copilot), a long-context compression restart, a length-
continuation restart, and a handful of format-recovery branches (thinking-
signature stripping, multimodal-tool-content stripping, llama.cpp grammar
fallback, image shrink, invalid-encrypted-content, 1M-beta header).

Each of those branches is guarded by a one-shot boolean so it fires at most
once per attempt. They used to be ~16 bare ``*_attempted`` / ``has_retried_*``
/ ``restart_with_*`` locals declared inline before the loop and threaded
through its 2,400-line body. ``TurnRetryState`` collapses them into one object
the loop mutates in place (``state.codex_auth_retry_attempted = True``), giving
the recovery bookkeeping a single named, testable home.

Loop-control variables (``retry_count``, ``max_retries``,
``max_compression_attempts``) intentionally stay as plain locals — they are the
``while`` mechanics, not recovery bookkeeping, and putting them on the object
would add indirection without clarifying anything.

This module is dependency-free so it can be unit-tested in isolation and
imported by the turn loop without an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover — type-only, keeps this module dependency-free
    from agent.error_classifier import ClassifiedError


@dataclass
class TurnRetryState:
    """One-shot recovery guards + restart signals for a single API-call attempt.

    A fresh instance is created for each iteration of the outer turn loop
    (once per ``api_call_count``). Each guard fires its recovery branch at most
    once; the ``restart_with_*`` signals are read by the loop after the attempt
    to decide whether to rebuild the request and retry.
    """

    # ── Per-provider OAuth / credential refresh guards ───────────────────
    codex_auth_retry_attempted: bool = False
    anthropic_auth_retry_attempted: bool = False
    nous_auth_retry_attempted: bool = False
    nous_paid_entitlement_refresh_attempted: bool = False
    copilot_auth_retry_attempted: bool = False
    vertex_auth_retry_attempted: bool = False

    # ── Format / payload recovery guards ─────────────────────────────────
    thinking_sig_retry_attempted: bool = False
    invalid_encrypted_content_retry_attempted: bool = False
    image_shrink_retry_attempted: bool = False
    multimodal_tool_content_retry_attempted: bool = False
    oauth_1m_beta_retry_attempted: bool = False
    llama_cpp_grammar_retry_attempted: bool = False

    # ── Transport / rate-limit recovery ──────────────────────────────────
    primary_recovery_attempted: bool = False
    has_retried_429: bool = False

    # ── Auth-failure provider failover ───────────────────────────────────
    # Set once we've escalated a persistent 401/403 (after the per-provider
    # credential-refresh attempt above failed) to the fallback chain, so we
    # don't loop on the same auth failover within one attempt.
    auth_failover_attempted: bool = False

    # ── Restart signals (read by the outer loop after the attempt) ───────
    restart_with_compressed_messages: bool = False
    restart_with_length_continuation: bool = False
    # Set when a content-filter stream stall (e.g. MiniMax "new_sensitive")
    # has been escalated to the fallback chain: the partial-stream content
    # was rolled back off ``messages`` and the loop should re-issue the API
    # call against the newly-activated provider (#32421).
    restart_with_rebuilt_messages: bool = False

    # ── Cross-hop quota-origin memory (BUILD-343) ────────────────────────
    # First quota-class (rate_limit / billing / upstream_rate_limit)
    # FailoverReason value seen this turn, set where the loop escalates to
    # a fallback/failover for such a reason. A fallback chain can end at a
    # local/transport-only tier (e.g. omlx-local) with no billing/rate-
    # limit concept of its own; if THAT hop is what ultimately exhausts
    # the turn, the terminal failure_reason prefers this recorded origin
    # over the last hop's transport-class classification — implementing
    # the documented contract in hermes-runtime-routing-debugging/
    # SKILL.md: "when the fallback chain exhausts, Hermes surfaces the
    # original primary error, not the last fallback error." First quota
    # reason wins; never overwritten once set.
    quota_origin_reason: Optional[str] = None

    def resolve_failure_reason(
        self, classified: ClassifiedError | None = None, *, reason: str | None = None
    ) -> str:
        """Resolve the terminal ``failure_reason`` value for a turn result.

        Implements the BUILD-343 origin-preference contract (previously
        duplicated inline at every terminal ``return`` in
        ``conversation_loop.py``): report the CURRENT site's own reason
        unless it is non-quota-class AND an earlier quota-class origin was
        recorded this turn via ``quota_origin_reason`` — in which case the
        earlier origin wins. A fallback chain's tail hop is often a local/
        transport-only tier with no billing/rate-limit concept of its own;
        without this preference its plain transport error would mask the
        real quota-wall cause and the kanban exit-75 gate would never fire.

        Exactly one of the two calling conventions applies:
          - ``classified``: pass the current attempt's ``ClassifiedError``
            — used by every site that has one.
          - ``reason``: a raw ``FailoverReason.value`` string for sites
            that die before any ``ClassifiedError`` exists (e.g. the Nous
            Portal preemptive rate-limit guard, which returns before an
            API call is even attempted). The caller vouches this reason is
            itself quota-class, so no origin lookup is needed.
        """
        if classified is not None and reason is not None:
            raise ValueError(
                "resolve_failure_reason: pass classified or reason, not both"
            )
        if classified is not None:
            if not classified.is_quota_exhaustion and self.quota_origin_reason:
                return self.quota_origin_reason
            return classified.reason.value
        if reason is None:
            raise ValueError("resolve_failure_reason requires classified or reason")
        return reason

    def __iter__(self):
        # Convenience for debugging / tests: iterate (name, value) pairs.
        for f in fields(self):
            yield f.name, getattr(self, f.name)
