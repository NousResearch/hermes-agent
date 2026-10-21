"""Turn-scoped suppression of sibling model probes after an exhausted credential fails."""

from agent.error_classifier import FailoverReason
import logging

logger = logging.getLogger(__name__)


def _candidate_identity(provider, entry):
    """Resolve without sending a completion; omitted fields mean provider defaults, not the live route."""
    from agent.auxiliary_client import resolve_provider_client
    from agent.chat_completion_helpers import _fallback_api_mode_hint
    from hermes_cli.fallback_config import resolve_entry_api_key

    endpoint = str(entry.get("base_url") or "").strip() or None
    key = resolve_entry_api_key(entry)
    _, mode = _fallback_api_mode_hint(entry, provider, endpoint)
    try:
        client, _model = resolve_provider_client(
            provider, model=entry["model"], raw_codex=True,
            explicit_base_url=endpoint, explicit_api_key=key, api_mode=mode,
        )
    except Exception:
        logger.debug("Could not prove a fallback credential identity", exc_info=True)
        return None
    resolved_key = getattr(client, "api_key", None)
    if client is None or not isinstance(resolved_key, str) or not resolved_key:
        return None
    return str(client.base_url).strip().rstrip("/"), resolved_key


def rejected_fallback_entries(agent, reason):
    """Share an auth verdict only across the same provider, endpoint and credential source.

    Pools retain their own key rotation policy. Explicit alternative keys and endpoints
    remain candidates. The set expires on the next user turn, allowing repaired auth to work.
    """
    turn = getattr(agent, "_current_turn_id", None)
    state = getattr(agent, "_auth_rejected_fallbacks", None)
    if state is None or state[0] != turn:
        state = (turn, set())
        agent._auth_rejected_fallbacks = state
    rejected = state[1]
    if reason not in (FailoverReason.auth, FailoverReason.auth_permanent) or getattr(agent, "_credential_pool", None) is not None:
        return rejected
    provider = str(agent.provider or "").strip().lower()
    endpoint = str(agent.base_url or "").strip().rstrip("/")
    key = getattr(agent, "api_key", None)
    if not provider or not isinstance(key, str) or not key:
        return rejected
    from hermes_cli.fallback_config import resolve_entry_api_key

    for index, entry in enumerate(agent._fallback_chain):
        if str(entry.get("provider") or "").strip().lower() != provider:
            continue
        explicit_endpoint = str(entry.get("base_url") or "").strip().rstrip("/")
        if explicit_endpoint and explicit_endpoint != endpoint:
            continue
        explicit_key = resolve_entry_api_key(entry)
        if explicit_key and explicit_key != key:
            continue
        if not explicit_key and any(entry.get(k) for k in ("api_key", "key_env", "api_key_env")):
            continue
        if _candidate_identity(provider, entry) == (endpoint, key):
            rejected.add(index)
    return rejected
