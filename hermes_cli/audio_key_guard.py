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


# RFC 6598 Carrier-Grade-NAT (100.64.0.0/10) is not flagged private by
# ipaddress, but is a shared/non-public range common in homelab / overlay
# setups — treat it as private so a cloud key never travels there.
_CGNAT_NET = ipaddress.ip_network("100.64.0.0/10")


def _ip_is_private(ip: "ipaddress._BaseAddress") -> bool:
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        return True
    return ip.version == 4 and ip in _CGNAT_NET


def _resolves_to_private(host: str) -> bool:
    """True if resolving ``host`` yields any private/loopback/link-local/
    reserved/CGNAT address.

    The lookup runs in a daemon thread with a real 3s join deadline: unlike
    ``socket.setdefaulttimeout`` (which does NOT bound ``getaddrinfo`` — that
    calls the blocking C resolver directly), this actually caps the wait, and
    it touches no process-global socket state (thread-safe). A slow/hung DNS
    answer → we give up and return False; the real connection would then also
    fail, so no cloud key leaks by proceeding.

    Residual risk (accepted): the real HTTP client resolves the host a second
    time, so an attacker who controls the host's DNS zone and flips the answer
    between the two lookups (DNS rebinding) could still route the connection to
    a private IP after the guard saw a public one. Fully closing that needs
    IP-pinned connections (as the WebUI TTS proxy does); here the guard covers
    the realistic config-injection case — a base_url hostname that statically
    resolves to a LAN address."""
    if not host:
        return False
    import socket
    import threading

    box: dict = {}

    def _lookup() -> None:
        try:
            box["infos"] = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        except Exception:
            box["infos"] = []

    t = threading.Thread(target=_lookup, daemon=True)
    t.start()
    t.join(3.0)
    if t.is_alive():  # DNS is hanging — don't block the audio call, don't leak
        return False
    for info in box.get("infos", []):
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
