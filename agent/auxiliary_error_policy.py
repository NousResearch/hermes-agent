"""Provider-owned error verdicts at auxiliary retry and fallback boundaries."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def provider_error_verdict(error: Exception, provider: str, *, model: str = "", base_url: str = "") -> Any:
    """Consult only profiles with an error hook; other routes retain their existing policy."""
    from providers import get_provider_profile
    from agent.error_classifier import classify_api_error

    profile = get_provider_profile(provider)
    if profile is not None and profile.classify_api_error is not None:
        return classify_api_error(error, provider=provider, model=model, base_url=base_url)
    return None


def require_provider_recovery(error: Exception, provider: str, *, fallback: bool = False,
                              model: str = "", base_url: str = "") -> None:
    verdict = provider_error_verdict(error, provider, model=model, base_url=base_url)
    if verdict is None:
        return
    allowed = verdict.should_fallback if fallback else (
        verdict.retryable or verdict.should_rotate_credential or verdict.should_fallback or verdict.should_compress)
    if not allowed:
        raise error


def plugin_refresh_kwargs(client: Any, provider: str) -> dict[str, Any]:
    from providers import get_provider_profile

    profile = get_provider_profile(provider)
    if profile is None or profile.refresh_credential is None:
        return {}
    leaf = getattr(client, "_real_client", client)
    return {"failed_api_key": getattr(client, "api_key", ""),
            "credential_id": getattr(leaf, "_hermes_aux_credential_id", None)}


def _should_retry_same_provider(task: Optional[str], exc: Exception, tag: str, provider: str = "", *,
                                model: str = "", base_url: str = "") -> bool:
    """True when ``exc`` is a transient transport blip worth a same-provider retry; critical-path
    tasks skip it on a full-budget timeout (``_should_skip_same_provider_retry``) and go straight
    to fallback."""
    from agent.auxiliary_client import _is_transient_transport_error, _should_skip_same_provider_retry
    if not _is_transient_transport_error(exc, provider, model=model, base_url=base_url):
        return False
    if _should_skip_same_provider_retry(task, exc):
        logger.info("Auxiliary %s%s: timeout on the critical path; "
                    "skipping same-provider retry and falling back: %s", task, tag, exc)
        return False
    return True
