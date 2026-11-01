"""User-visible notices before auxiliary recovery switches services."""
import logging
import re
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


def _service(provider, client):
    provider = str(provider or "custom")
    # Configured names may be URLs; never display URL credentials or query strings.
    display = provider if re.fullmatch(r"[\w .:-]{1,80}", provider) else "custom service"
    identity = provider
    if provider in ("auto", "custom"):
        url = urlsplit(str(getattr(client, "base_url", "") or ""))
        identity = (provider, url.scheme, url.hostname, url.port, url.path.rstrip("/"))
    return display, identity


def warn_service_fallback(primary, fallback, *, source_provider, target_provider, runtime):
    """Observe a provider switch without inspecting credentials or changing recovery."""
    source, target = _service(source_provider, primary), _service(target_provider, fallback)
    if source[1] == target[1]:
        return
    logger.info("Auxiliary fallback: %s -> %s", source[0], target[0])
    callback = (runtime or {}).get("billing_warning_callback")
    if callable(callback):
        callback(
            f"⚠ Auxiliary: falling back from {source[0]} to {target[0]}. "
            "This may incur additional charges.", route_key=(source[1], target[1]),
        )
