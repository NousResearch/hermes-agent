"""Billing-evidence accounting of ``ClaudeAgentSdkSession``: subscription vs
metered evidence observed on the stream and the fail-closed metered guard.
Extracted from ``claude_agent_sdk_session.py``; every method resolves through
``ClaudeAgentSdkSession``'s MRO unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.transports.claude_agent_sdk_session")


class ClaudeSdkBillingMixin:
    """Billing evidence and reported billing mode (see module docstring)."""

    def _observe_billing_evidence(self, message: Any) -> None:
        """Record the CLI's machine-readable billing lane and fail closed.

        Environment scrubbing protects against billing vectors Hermes knows
        about. The child remains the authority for what it actually selected:
        ``system/init.apiKeySource`` reports API-key use, while the pinned SDK's
        typed ``RateLimitEvent`` exposes subscription Extra Usage. Unless the
        operator explicitly set ``allow_metered_key``, either signal is a
        fatal configuration/account-state mismatch, not an "included" turn.
        """
        name = type(message).__name__
        if name == "SystemMessage" and getattr(message, "subtype", "") == "init":
            data = getattr(message, "data", None)
            if isinstance(data, dict) and (
                "apiKeySource" in data or "api_key_source" in data
            ):
                source = data.get("apiKeySource", data.get("api_key_source"))
                source_text = str(source or "none").strip() or "none"
                self._billing_evidence["api_key_source"] = source_text
                if (
                    not self._allow_metered
                    and source_text.lower() != "none"
                    and self._billing_guard_error is None
                ):
                    self._billing_guard_error = (
                        "claude-agent-sdk billing guard: the CLI reported "
                        f"API-key source {source_text!r}. Remove the metered "
                        "credential, or set agent.claude_agent_sdk."
                        "allow_metered_key: true to opt in explicitly."
                    )
            return

        if name != "RateLimitEvent":
            return
        info = getattr(message, "rate_limit_info", None)
        if info is None:
            return
        raw = getattr(info, "raw", None)
        raw = raw if isinstance(raw, dict) else {}
        is_using_overage = raw.get("isUsingOverage")
        if isinstance(is_using_overage, bool):
            self._billing_evidence["is_using_overage"] = is_using_overage
        overage_status = getattr(info, "overage_status", None)
        if overage_status is None:
            overage_status = raw.get("overageStatus")
        if overage_status is not None:
            self._billing_evidence["overage_status"] = str(overage_status)
        rate_limit_type = getattr(info, "rate_limit_type", None)
        if rate_limit_type is None:
            rate_limit_type = raw.get("rateLimitType")
        if rate_limit_type is not None:
            self._billing_evidence["rate_limit_type"] = str(rate_limit_type)

        if self._allow_metered or self._billing_guard_error is not None:
            return
        if is_using_overage is True:
            self._billing_guard_error = (
                "claude-agent-sdk billing guard: metered subscription Extra "
                "Usage is active. Disable Extra Usage in the Claude account, "
                "or set agent.claude_agent_sdk.allow_metered_key: true to "
                "opt in explicitly."
            )
        elif str(overage_status or "").lower() in {"allowed", "allowed_warning"}:
            self._billing_guard_error = (
                "claude-agent-sdk billing guard: subscription extra usage is "
                "enabled and could silently become metered when the included "
                "limit is exhausted. Disable Extra Usage in the Claude "
                "account, or set agent.claude_agent_sdk.allow_metered_key: "
                "true to opt in explicitly."
            )

    def _reported_billing_mode(self) -> str:
        source = str(self._billing_evidence.get("api_key_source") or "").lower()
        using_overage = self._billing_evidence.get("is_using_overage")
        rate_limit_type = str(
            self._billing_evidence.get("rate_limit_type") or ""
        ).lower()
        if (source and source != "none") or using_overage is True:
            return "sdk_reported_metered"
        if rate_limit_type == "overage" and using_overage is not False:
            return "sdk_reported_metered"
        if not self._allow_metered:
            # Any contrary evidence trips the guard before accounting.
            return "subscription_included"
        if source == "none" and using_overage is False:
            return "subscription_included"
        return "unknown"
