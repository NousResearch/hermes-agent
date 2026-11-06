"""Yuanbao WS dial proxy repair: websockets' system-proxy auto-detection can pick a SOCKS proxy
it cannot dial (python-socks is not a dependency); swap in the HTTP(S) proxy or a direct dial."""

from __future__ import annotations

from typing import Optional


def _socks_runtime_available() -> bool:
    """Return True when the ``python-socks`` runtime websockets needs for SOCKS proxies is importable."""
    try:
        import python_socks
    except ImportError:
        return False
    return True


def _proxy_log_mode(proxy_url: Optional[str]) -> str:
    """Describe a proxy for logging using only its scheme — never its host or userinfo.

    A proxy URL can embed ``user:password@`` credentials that the global log formatter
    leaves unmasked, so only the scheme (or ``direct connection``) is safe to emit.
    """
    if not proxy_url:
        return "direct connection"
    scheme = proxy_url.split("://", 1)[0].strip().lower()
    return f"{scheme} proxy" if scheme else "proxy"


def _resolve_ws_proxy_override(ws_url: str) -> tuple[bool, Optional[str]]:
    """Repair the SOCKS proxy websockets would auto-derive for the Yuanbao WS dial (#122708).

    ``websockets.connect()`` called without an explicit ``proxy=`` picks its proxy from
    ``urllib.request.getproxies()`` — on macOS this reads the system proxy (SystemConfiguration),
    so a Clash/Surge/V2RayU SOCKS entry is selected even with no proxy env var set. ``python-socks``
    is not part of the dependency set, so the dial always raises
    ``ImportError: python-socks is required to use a SOCKS proxy`` and the platform
    reconnect-loops forever, while the sign-token HTTP call connects fine.

    Mirror websockets' own selection (``websockets.uri.get_proxy``) and intervene only when it
    would land on a SOCKS proxy websockets cannot dial — the ``python-socks`` runtime is missing,
    or the bare ``socks://`` spelling fails ``parse_proxy()`` outright. Hand it the system
    HTTP(S) proxy instead (websockets tunnels through those with a plain CONNECT, no extra
    dependency), else a direct connection.

    Returns ``(False, None)`` to leave websockets' default behaviour untouched, or
    ``(True, proxy_url)`` to pass ``proxy=proxy_url`` (``None`` forces a direct connection).
    """
    import urllib.request
    from urllib.parse import urlparse

    try:
        parsed = urlparse(ws_url)
    except ValueError:
        return False, None
    secure = parsed.scheme == "wss"
    host = parsed.hostname or ""
    port = parsed.port or (443 if secure else 80)
    try:
        if host and urllib.request.proxy_bypass(f"{host}:{port}"):
            return False, None
    except (ValueError, OSError):
        pass

    proxies = urllib.request.getproxies()
    # websockets.uri.get_proxy() scheme priority — SOCKS outranks HTTPS for a secure WS.
    ws_schemes = ["wss", "socks", "https"] if secure else ["ws", "socks", "https", "http"]
    selected_key = next((s for s in ws_schemes if proxies.get(s)), None)
    if selected_key is None:
        return False, None
    selected = proxies[selected_key]
    socks_dial = selected_key == "socks" or selected.lower().split("://", 1)[0].startswith("socks")
    if not socks_dial:
        # An HTTP(S) proxy websockets tunnels through on its own — leave the default alone.
        return False, None
    if _socks_runtime_available() and not selected.lower().startswith("socks://"):
        # websockets can dial this SOCKS proxy itself (it rewrites an ``http://`` spelling
        # of the ``socks`` entry to ``socks5h://`` before parsing).
        return False, None
    # Unsupported SOCKS spelling or no SOCKS runtime: reuse the HTTP(S) proxy entry, else
    # connect directly rather than failing every dial attempt forever.
    for scheme in ("https", "http"):
        http_proxy = proxies.get(scheme, "")
        if http_proxy and not http_proxy.lower().split("://", 1)[0].startswith("socks"):
            return True, http_proxy
    return True, None
