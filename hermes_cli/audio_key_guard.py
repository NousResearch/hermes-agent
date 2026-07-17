"""Shared credential guard for OpenAI-compatible & cloud audio providers.

A configured ``base_url`` may point at a self-hosted / LAN server. A real
*cloud* API key (XAI_API_KEY, ELEVENLABS_API_KEY, OPENAI_API_KEY, …) must
never be sent — in cleartext, for http — to such a target: it was not issued
for that server and anyone able to edit the config base_url could otherwise
harvest the env key. This module is the single source of truth for that
decision, shared by ``tools/transcription_tools.py`` and ``tools/tts_tool.py``
(previously two divergent private copies).
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit
from typing import Optional


# Sent as the token when a self-hosted base_url is configured without any key —
# auth-less OpenAI-compatible servers ignore it.
PLACEHOLDER_KEY = "sk-no-key-required"


def base_url_is_private(base_url: Optional[str]) -> bool:
    """True when ``base_url`` is plainly self-hosted: any ``http://`` URL, or an
    ``https://`` one whose host is localhost or a private/loopback/link-local/
    reserved IP literal. Public https hosts (api.openai.com, a hosted proxy)
    return False and keep the conventional cloud-key behaviour."""
    try:
        parts = urlsplit(str(base_url or ""))
    except ValueError:
        return False
    if parts.scheme == "http":
        return True
    host = (parts.hostname or "").strip().lower()
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved


def resolve_provider_key(
    cfg_api_key: Optional[str],
    env_key: Optional[str],
    base_url: Optional[str],
    *,
    placeholder: str = PLACEHOLDER_KEY,
) -> str:
    """Return the key to send to ``base_url``, mirroring the openai contract.

    * Private/self-hosted target: a config-supplied ``api_key`` is authoritative
      (self-hosted-with-auth), otherwise the placeholder — **never** the env
      cloud key.
    * Public host: config key wins, else the env key (conventional behaviour).
    """
    cfg = (cfg_api_key or "").strip()
    if base_url and base_url_is_private(base_url):
        return cfg or placeholder
    return cfg or (env_key or "").strip()
