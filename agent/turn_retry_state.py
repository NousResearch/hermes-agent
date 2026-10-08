"""Per-attempt recovery bookkeeping (``TurnRetryState``) for the conversation turn loop.
Dependency-free so it imports without a cycle."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnRetryState:
    """One-shot recovery guards + restart signals for a single API-call attempt.

    A fresh instance is created per ``api_call_count`` iteration; each guard fires at
    most once, and ``restart_with_*`` signals tell the loop to rebuild and retry.
    Loop-control (``retry_count``, ``max_retries``) stays as plain loop locals."""

    # Per-provider OAuth / credential refresh guards
    codex_auth_retry_attempted: bool = False
    anthropic_auth_retry_attempted: bool = False
    nous_auth_retry_attempted: bool = False
    nous_paid_entitlement_refresh_attempted: bool = False
    # Nous free tier: one model move onto the tier's own model after a ``model_not_free``
    # refusal, and one route re-read after a wrong-host refusal (``anon_on_paid_host``).
    welcome_model_switch_attempted: bool = False
    welcome_route_heal_attempted: bool = False
    copilot_auth_retry_attempted: bool = False
    copilot_403_retries_used: int = 0
    # Policy scheduling the current request; None denotes the initial attempt.
    # Keep this separate from the cumulative hard Copilot retry limit.
    scheduled_retry_policy: str | None = None
    # Copilot surfaces a stale credential as a 400 ``model_not_available_for_integrator``
    # / ``model_not_supported``, not a 401 — separate guard from the 401 one.
    copilot_stale_cred_retry_attempted: bool = False
    vertex_auth_retry_attempted: bool = False

    # Format / payload recovery guards
    thinking_sig_retry_attempted: bool = False
    invalid_encrypted_content_retry_attempted: bool = False
    native_compaction_reject_retry_attempted: bool = False
    image_shrink_retry_attempted: bool = False
    multimodal_tool_content_retry_attempted: bool = False
    reasoning_mandatory_retry_attempted: bool = False
    oauth_1m_beta_retry_attempted: bool = False
    llama_cpp_grammar_retry_attempted: bool = False

    # Transport / rate-limit recovery
    primary_recovery_attempted: bool = False
    has_retried_429: bool = False
    # Credit-limited 402 ("can only afford N tokens") already retried with a lowered output cap.
    affordable_402_clamp_attempted: bool = False
    # Persistent 401/403 already escalated to the fallback chain once this attempt.
    auth_failover_attempted: bool = False
    # Post-exhaustion auto-recovery cycles spent on this API call (agent.auto_recovery_cycles caps it).
    auto_recovery_cycles_used: int = 0

    # Restart signals (read by the outer loop after the attempt)
    restart_with_compressed_messages: bool = False
    restart_with_length_continuation: bool = False
    # A fallback activation (incl. content-filter stream stalls) rolled partial content
    # off ``messages``; re-issue the call against the new provider.
    restart_with_rebuilt_messages: bool = False
    # A user correction cancelled the in-flight request: append a role-safe checkpoint +
    # user message, rebuild the payload, and retry the same logical iteration.
    restart_with_redirected_messages: bool = False
