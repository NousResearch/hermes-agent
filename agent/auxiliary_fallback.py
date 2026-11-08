"""Configured fallback identity and traversal after a candidate becomes unavailable."""
from __future__ import annotations

import re
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.auxiliary_client import _LadderRoute


def _fallback_provider_from_label(label: str) -> str:
    """Recover the provider identifier from a fallback display label."""
    match = re.match(r"(?:fallback_chain\[\d+\]|fallback_providers\[\d+\]|main-agent)\(([^)]+)\)$", label or "")
    return match.group(1).strip() if match else str(label or "").strip()



def _next_fallback_after_quarantine(
    task: Optional[str], resolved_provider: str, is_auto: bool, route: _LadderRoute,
    failed_model: Optional[str], failure_scope: Any, *, task_chain_only: bool = False,
    excluded_labels: Optional[set[str]] = None,
    configured_attempt_limit: Optional[int] = None,
) -> tuple[Optional[Any], Optional[str], str]:
    """Next candidate after a fallback entry was quarantined mid-request (dead credential or a
    capacity error): remaining configured entries (task chain, then main chain on auto) before the
    discovery chain. ``task_chain_only`` (explicit-provider auth error) stops at the task chain —
    the user never opted that task into discovery or the main model."""
    from agent.auxiliary_client import (
        _try_configured_fallback_chain, _try_main_fallback_chain, _try_payment_fallback,
    )

    reason = "fallback candidate unavailable"
    fb = (None, None, "")
    if configured_attempt_limit is None or len(excluded_labels or ()) < configured_attempt_limit:
        fb = _try_configured_fallback_chain(
            task, resolved_provider or "auto", reason=reason, failed_model=failed_model,
            failed_base_url=route.base_info, failure_scope=failure_scope,
            excluded_labels=excluded_labels)
    if task_chain_only:
        return fb
    if fb[0] is None and is_auto:
        fb = _try_main_fallback_chain(
            task, resolved_provider or "auto", reason=reason, failed_model=failed_model,
            failed_base_url=route.base_info, failure_scope=failure_scope)
    if fb[0] is None:
        fb = _try_payment_fallback(
            resolved_provider, task, reason=reason, failed_base_url=route.base_info,
            failure_scope=failure_scope, main_runtime=route.main_runtime)
    return fb

