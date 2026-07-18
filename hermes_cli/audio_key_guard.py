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
    ``https://`` one whose host is localhost, a private/loopback/link-local/
    reserved IP literal, **or a hostname that resolves to such an address**.
    Public https hosts (api.openai.com, a hosted proxy) return False and keep
    the conventional cloud-key behaviour.

    The DNS-resolution arm closes a key-leak path a literal-only check missed:
    a public-looking hostname (``https://sneaky.example.com``) that resolves to
    a LAN/loopback IP would otherwise pass as "public" and receive the real
    cloud key, which then travels to a private box. If ANY resolved address is
    private, the key is withheld (placeholder sent instead). Resolution failure
    is treated as non-private — the request would fail to connect anyway, so no
    key leaks, and a transient DNS hiccup must not silently break a legit host."""
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
        # Hostname, not an IP literal — resolve and check every answer.
        return _resolves_to_private(host)
    return _ip_is_private(ip)


def _ip_is_private(ip: "ipaddress._BaseAddress") -> bool:
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)


def _resolves_to_private(host: str) -> bool:
    """True if resolving ``host`` yields any private/loopback/link-local/
    reserved address. Best-effort with a short timeout; failure → False (the
    request would fail to connect, so no cloud key leaks regardless)."""
    if not host:
        return False
    import socket
    prev = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(3.0)
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except Exception:
        return False
    finally:
        try:
            socket.setdefaulttimeout(prev)
        except Exception:
            pass
    for info in infos:
        sockaddr = info[4] if len(info) > 4 else None
        if not sockaddr:
            continue
        try:
            ip = ipaddress.ip_address(str(sockaddr[0]))
        except (ValueError, IndexError):
            continue
        if _ip_is_private(ip):
            return True
    return False


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
