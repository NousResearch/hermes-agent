"""Codex helpers extracted from AIAgent for modularity."""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def looks_like_codex_intermediate_ack(
    runner,
    user_message: str,
    assistant_content: str,
    messages: List[Dict[str, Any]],
) -> bool:
    """Forwarder — see ``agent.agent_runtime_helpers.looks_like_codex_intermediate_ack``."""
    from agent.agent_runtime_helpers import looks_like_codex_intermediate_ack as _impl
    return _impl(runner, user_message, assistant_content, messages)


def try_refresh_codex_client_credentials(runner, *, force: bool = True) -> bool:
    """Refresh Codex/oauth client credentials, avoiding silent account swaps."""
    if runner.api_mode != "codex_responses" or runner.provider not in {"openai-codex", "xai-oauth"}:
        return False

    # Guard against silent account swap.
    #
    # When an agent is using a non-singleton credential — e.g. a manual
    # pool entry (``hermes auth add xai-oauth``) whose tokens belong to
    # a different account than the loopback_pkce singleton, or an agent
    # constructed with an explicit ``api_key=`` arg — force-refreshing
    # the singleton here and adopting its tokens silently re-routes the
    # rest of the conversation onto the singleton's account.  The
    # credential pool's reactive recovery (``_recover_with_credential_pool``)
    # is the right channel for that case; this path is the
    # singleton-only fallback used when the pool can't recover, and
    # MUST only fire when the agent really is on singleton tokens.
    try:
        if runner.provider == "openai-codex":
            from hermes_cli.auth import resolve_codex_runtime_credentials

            singleton_now = resolve_codex_runtime_credentials(
                refresh_if_expiring=False,
            )
        else:
            from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

            singleton_now = resolve_xai_oauth_runtime_credentials(
                refresh_if_expiring=False,
            )
    except Exception as exc:
        logger.debug("%s singleton read failed: %s", runner.provider, exc)
        return False

    singleton_key = str(singleton_now.get("api_key") or "").strip()
    active_key = str(runner.api_key or "").strip()
    if singleton_key and active_key and singleton_key != active_key:
        logger.debug(
            "%s singleton tokens differ from the active api_key; "
            "skipping singleton force-refresh to avoid silent account swap. "
            "Reactive credential rotation should go through the pool.",
            runner.provider,
        )
        return False

    try:
        if runner.provider == "openai-codex":
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials(force_refresh=force)
        else:
            from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

            creds = resolve_xai_oauth_runtime_credentials(force_refresh=force)
    except Exception as exc:
        logger.debug("%s credential refresh failed: %s", runner.provider, exc)
        return False

    api_key = creds.get("api_key")
    base_url = creds.get("base_url")
    if not isinstance(api_key, str) or not api_key.strip():
        return False
    if not isinstance(base_url, str) or not base_url.strip():
        return False

    runner.api_key = api_key.strip()
    runner.base_url = base_url.strip().rstrip("/")
    runner._client_kwargs["api_key"] = runner.api_key
    runner._client_kwargs["base_url"] = runner.base_url

    if not runner._replace_primary_openai_client(reason=f"{runner.provider}_credential_refresh"):
        return False

    return True


def run_codex_app_server_turn(
    runner,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """Forwarder — see ``agent.codex_runtime.run_codex_app_server_turn``."""
    from agent.codex_runtime import run_codex_app_server_turn as _impl
    return _impl(
        runner,
        user_message=user_message,
        original_user_message=original_user_message,
        messages=messages,
        effective_task_id=effective_task_id,
        should_review_memory=should_review_memory,
    )
