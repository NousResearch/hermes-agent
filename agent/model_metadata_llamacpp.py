"""llama.cpp server capability queries.

Split out of :mod:`agent.model_metadata` (which is over its line cap and may only
shrink) so the llama.cpp ``/props`` vision probe lives beside the Ollama queries it
mirrors. Private helpers are borrowed from the parent module on call, the way
:mod:`agent.image_token_cost` borrows the disk-cache helpers.
"""

from __future__ import annotations

from typing import Optional


def _is_llamacpp_server(base_url: str, api_key: str) -> bool:
    from agent.model_metadata import detect_local_server_type

    try:
        return detect_local_server_type(base_url, api_key=api_key) == "llamacpp"
    except Exception:  # health: allow BLE001 -- probe boundary: a broken detect reads as not-llamacpp, never breaks the chain
        return False


def query_llamacpp_supports_vision(base_url: str, api_key: str = "") -> Optional[bool]:
    """True/False when llama.cpp ``/props`` reports ``modalities`` (``{"vision": true}`` with
    ``--mmproj`` loaded; older builds emitted the list ``["vision"]``); None when unreachable,
    not a llama.cpp server, or the build predates the field (stays fail-closed to aux routing)."""
    if not base_url or not _is_llamacpp_server(base_url, api_key):
        return None
    from agent.model_metadata import (
        _auth_headers,
        _endpoint_blackholed,
        _is_connect_timeout,
        _localhost_to_ipv4,
        _note_endpoint_blackholed,
        _normalize_base_url,
        _server_root,
        model_metadata_http,
    )

    server_url = _server_root(_localhost_to_ipv4(_normalize_base_url(base_url)))
    if _endpoint_blackholed(server_url):
        return None
    import httpx
    try:
        with httpx.Client(timeout=2.0, headers=_auth_headers(api_key), verify=model_metadata_http.resolve_verify(server_url)) as client:
            resp = None
            for url in (f"{server_url}/v1/props", f"{server_url}/props"):  # older builds serve /props without the /v1 prefix
                try:
                    resp = client.get(url)
                except (httpx.HTTPError, OSError) as exc:
                    if _is_connect_timeout(exc):
                        _note_endpoint_blackholed(server_url)
                    continue
                if resp.status_code == 200:
                    break
        if resp is None or resp.status_code != 200:
            return None
        modalities = (resp.json() or {}).get("modalities")
    except (ValueError, AttributeError):
        return None  # malformed props body: no verdict, aux routing stays
    if isinstance(modalities, dict):
        vision = modalities.get("vision")
        return vision if isinstance(vision, bool) else None
    if isinstance(modalities, list) and modalities:
        return any(str(m).strip().lower() == "vision" for m in modalities)
    return None
