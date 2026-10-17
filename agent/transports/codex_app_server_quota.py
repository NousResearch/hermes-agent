"""Narrow, pre-tool Codex App Server quota recovery on the selected OAuth pool."""

from __future__ import annotations

from typing import Any, Callable

from agent.credential_pool import PooledCredential
from agent.transports.codex_app_server_session import TurnResult


_TYPED_QUOTA_CODES = frozenset({"usageLimitExceeded", "rateLimitExceeded"})
_PRECONNECTION_FAILURES = frozenset({"httpConnectionFailed", "responseStreamConnectionFailed"})


def typed_native_quota_failure(error: Any) -> str | None:
    """Return quota evidence only from Codex's typed TurnError.codexErrorInfo field."""
    if not isinstance(error, dict):
        return None
    info = error.get("codexErrorInfo")
    if isinstance(info, str) and info in _TYPED_QUOTA_CODES:
        return info
    if not isinstance(info, dict) or len(info) != 1:
        return None
    variant, details = next(iter(info.items()))
    if variant not in _PRECONNECTION_FAILURES or not isinstance(details, dict):
        return None
    if set(details) != {"httpStatusCode"}:
        return None
    status = details.get("httpStatusCode")
    return "http429" if isinstance(status, int) and not isinstance(status, bool) and status == 429 else None


def is_replay_safe_pretool_quota_failure(turn: TurnResult) -> bool:
    """Only an explicit terminal FAILED quota result with no intervening activity is replayable."""
    quota_code = getattr(turn, "native_quota_failure", None)
    return bool(
        getattr(turn, "terminal_status", None) == "failed" and quota_code in _TYPED_QUOTA_CODES | {"http429"}
        and getattr(turn, "native_quota_scope_valid", False)
        and (not getattr(turn, "input_echo_started", False) or getattr(turn, "input_echo_completed", False))
        and not getattr(turn, "interrupted", False) and not getattr(turn, "saw_item", False)
        and not getattr(turn, "saw_server_request", False) and not getattr(turn, "saw_nonterminal_activity", False)
        and not getattr(turn, "should_retire", False)
    )


def recover_codex_app_server_quota(
    agent: Any, turn: TurnResult, *, retry_turn: Callable[[], TurnResult],
) -> TurnResult:
    """Rotate one exact failed row and run the same thread once; otherwise return the original result."""
    if not is_replay_safe_pretool_quota_failure(turn) or getattr(agent, "provider", None) != "openai-codex":
        return turn
    pool = getattr(agent, "_credential_pool", None)
    session = getattr(agent, "_codex_session", None)
    host_auth = getattr(session, "_host_auth", None)
    failed_id = getattr(agent, "_credential_pool_entry_id", None)
    if (pool is None or host_auth is None or not host_auth.ready or not failed_id
            or host_auth.credential_id != failed_id or session._client is None):
        return turn
    try:
        with pool._lock:
            host_auth.validate_current()
            rotated = pool.mark_exhausted_and_rotate(
                status_code=429, error_context={"code": getattr(turn, "native_quota_failure", None),
                                                "message": "Codex usage limit reached"},
                credential_id=failed_id, model=getattr(agent, "model", None),
            )
    except Exception:  # health: allow BLE001 -- stale/replaced bindings fail closed before pool rotation.
        return turn
    if not isinstance(rotated, PooledCredential) or rotated.id == failed_id:
        return turn
    if not host_auth.rotate_and_bind(session._client, rotated):
        turn.error = "Codex host authentication failed after quota rotation"
        turn.should_retire = True
        return turn
    agent._credential_pool_entry_id = rotated.id
    agent.api_key = rotated.runtime_api_key or rotated.access_token
    agent.base_url = rotated.runtime_base_url or rotated.base_url or agent.base_url
    retried = retry_turn()
    retried.api_calls = 2
    return retried
