"""Inference connection tests shared by provider setup surfaces."""

import re
import httpx
from typing import Any

_TRANSPORT_ROUTES = {"chat_completions": "/chat/completions", "codex_responses": "/responses",
                     "anthropic_messages": "/messages"}
_TRANSPORT_LABELS = {"chat_completions": "Chat Completions", "codex_responses": "Responses API",
                     "anthropic_messages": "Anthropic Messages"}


def _endpoint_probe_failure(resp: Any, route: str, headers: dict[str, str]) -> str:
    """Expose a bounded provider explanation, redacting the credential we sent."""
    try:
        payload = resp.json()
    except ValueError:
        payload = None
    error = payload.get("error") if isinstance(payload, dict) else None
    detail = error.get("message") if isinstance(error, dict) else None
    if not isinstance(detail, str):
        detail = ""
    authorization = next((value for key, value in headers.items() if key.lower() == "authorization"), "")
    secrets = {*headers.values(), authorization.partition(" ")[2]}
    for secret in sorted(secrets, key=len, reverse=True):
        if secret:
            detail = detail.replace(secret, "[redacted]")
    detail = " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", detail).split())[:500]
    if resp.status_code == 403:
        reason = "access denied"
        fallback = "Check account permissions and endpoint network access policies."
    elif resp.status_code == 401:
        reason = "credential rejected"
        fallback = "Check the selected account and credential."
    else:
        reason = "request failed"
        fallback = "Check the deployment name and API mode."
    return f"POST {route} returned HTTP {resp.status_code} ({reason}). {detail or fallback}"


async def _probe_transport_route(
    client, base_url: str, mode: str, model: str, headers: dict[str, str], *, require_success: bool = False, bearer_only: bool = False,
) -> str:
    """Probe the chosen inference route without executing any returned tools.

    Explicit auth selections require a successful inference response. Older clients retain
    the route-existence check, which accepts request validation failures as inconclusive.
    """
    route = _TRANSPORT_ROUTES.get(mode)
    if route is None:
        return ""
    if mode == "anthropic_messages":
        payload = {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}
        protocol_headers = httpx.Headers(headers)
        token = protocol_headers.get("Authorization", "").removeprefix("Bearer ")
        protocol_headers.setdefault("anthropic-version", "2023-06-01")
        if token and not bearer_only:
            protocol_headers.setdefault("x-api-key", token)
        headers = dict(protocol_headers)
    elif mode == "codex_responses":
        payload = {"model": model, "input": "hi", "max_output_tokens": 16}
    else:
        from utils import base_url_host_matches, base_url_hostname, model_forces_max_completion_tokens

        # Match AIAgent._max_tokens_param: Azure deployment aliases may hide the model family.
        completion_limit = (model_forces_max_completion_tokens(model) or base_url_hostname(base_url) == "api.openai.com"
                            or base_url_host_matches(base_url, "openai.azure.com")
                            or base_url_host_matches(base_url, "githubcopilot.com"))
        limit_field = "max_completion_tokens" if completion_limit else "max_tokens"
        payload = {"model": model, limit_field: 1, "messages": [{"role": "user", "content": "hi"}]}
    try:
        resp = await client.post(base_url + route, json=payload, headers=headers)
    except (httpx.HTTPError, httpx.InvalidURL, ValueError):
        return "Could not validate the inference route." if require_success else ""
    if resp.status_code in (401, 403) or (require_success and not resp.is_success):
        return _endpoint_probe_failure(resp, route, headers)
    if resp.status_code not in (404, 405, 501):
        return ""
    return (f"{base_url}/models answered, but POST {route} returned HTTP {resp.status_code}: this host "
            f"does not serve the {_TRANSPORT_LABELS[mode]} API. Pick the API mode it does serve.")


def _endpoint_probe_client(url: str, timeout: float):
    """httpx client for a user-entered endpoint probe. Local endpoints (loopback, LAN, Tailscale)
    ignore ``HTTP(S)_PROXY``: httpx honours the env/system proxy but not its bypass list, so a
    system proxy (Clash on Windows, corporate) answered the ``127.0.0.1`` probe with its own error
    page and the GUI reported "advertised no models" while the CLI saw the model (#63472)."""
    from agent.model_metadata import is_local_endpoint
    return httpx.AsyncClient(timeout=httpx.Timeout(timeout), trust_env=not is_local_endpoint(url))
