"""Reply-image auto-delivery policy (#129975).

An agent reply containing ``![](url)`` or ``<img src>`` is fetched and delivered by the gateway with no
tool call and no approval, so a model-authored image URL is a zero-click exfiltration channel. Only URLs
on image-generation CDNs (where the model's image tools put their output, and where an attacker cannot
receive a request) are auto-delivered by default; any other image URL stays in the text as a link.

``security.reply_image_url_delivery``: ``generated`` (default, generator CDNs only), ``all`` (any
image URL), ``off`` (never auto-deliver). Unknown values => ``generated``.
``security.reply_image_allowed_hosts``: extra operator-trusted hostnames.
"""

import logging
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

GENERATOR_HOSTS = ("fal.media", "v2.fal.media", "v3.fal.media", "replicate.delivery",
                   "oaidalleapiprodscus.blob.core.windows.net")
_MODES = ("generated", "all", "off")


def _security_config() -> dict:
    """The ``security`` section; ``{}`` (=> ``generated``, no extra hosts) when config cannot be read."""
    try:
        from hermes_cli.config import load_config_readonly
        section = load_config_readonly().get("security")
    except Exception:
        logger.warning("reply-image delivery: config unreadable, using 'generated'", exc_info=True)
        return {}
    return section if isinstance(section, dict) else {}


def delivery_mode() -> str:
    raw = str(_security_config().get("reply_image_url_delivery", "generated")).strip().lower()
    return raw if raw in _MODES else "generated"


def extra_hosts() -> tuple[str, ...]:
    raw = _security_config().get("reply_image_allowed_hosts") or []
    return tuple(str(h).strip().lower().lstrip(".") for h in raw if str(h).strip())


def auto_deliverable(url: str) -> bool:
    """True when the gateway may fetch and auto-deliver *url* per the configured mode."""
    mode = delivery_mode()
    if mode != "generated":
        return mode == "all"
    try:
        host = (urlsplit(url).hostname or "").rstrip(".").lower()
    except ValueError:
        return False
    # FQDN suffix match only: ``host == s`` or a dotted-label boundary (``host`` ends with ``.s``).
    # Allowlist entries must themselves be dotted FQDNs, so a bare label can never match an attacker
    # subdomain (``fal-cdn.attacker.example`` does not match a ``fal-cdn`` entry).
    allowed = (*GENERATOR_HOSTS, *extra_hosts())
    return bool(host) and any(host == s or host.endswith("." + s) for s in allowed if s and "." in s)
