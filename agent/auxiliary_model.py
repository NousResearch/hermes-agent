"""Normalize auxiliary model IDs for the endpoint that will receive them."""
from typing import Optional


def _normalize_resolved_model(
    model_name: Optional[str], provider: str, base_url: Optional[str] = None,
) -> Optional[str]:
    """Mantle namespace dots are authoritative; other providers keep their normalization."""
    if not model_name:
        return model_name
    from agent.anthropic_endpoints import _is_bedrock_mantle_endpoint
    if _is_bedrock_mantle_endpoint(base_url):
        return model_name
    try:
        from hermes_cli.model_normalize import normalize_model_for_provider
        return normalize_model_for_provider(model_name, provider)
    except Exception:
        return model_name
