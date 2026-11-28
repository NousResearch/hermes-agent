"""The single authenticated read that proves an API key works at an endpoint."""

from __future__ import annotations

from dataclasses import dataclass

from agent.anthropic_adapter import _COMMON_BETAS, _OAUTH_ONLY_BETAS
from agent.anthropic_credentials import _is_oauth_token, is_native_anthropic_host
from agent.anthropic_endpoints import _requires_bearer_auth
from agent.auxiliary_openai_url import _to_openai_base_url
from hermes_cli.auth_zai_kimi import _resolve_kimi_base_url
from hermes_cli.config import get_custom_provider_extra_headers
from hermes_cli.version_info import hermes_user_agent
from utils import base_url_host_matches, base_url_hostname

ANTHROPIC_MESSAGES = "anthropic_messages"

# Hosts whose ``/models`` answers 200 with no key at all: a 200 there proves nothing, so the
# check names the key unverified instead of calling it good. A key endpoint (below) wins.
_PUBLIC_CATALOG_HOSTS = frozenset({
    "inference-api.nousresearch.com", "router.huggingface.co", "integrate.api.nvidia.com", "api.kilo.ai"})
# Hosts whose ``/models`` is public but that expose an authenticated key-info read instead.
_KEY_INFO_URLS = {"openrouter.ai": "https://openrouter.ai/api/v1/key"}


@dataclass(frozen=True)
class ProbeRequest:
    url: str
    headers: dict[str, str]
    auth_mode: str = ""  # e.g. "Anthropic Console API key" vs "Anthropic OAuth token"


def _anthropic_request(base: str, key: str) -> ProbeRequest:
    headers = {"anthropic-version": "2023-06-01", "User-Agent": hermes_user_agent()}
    native = is_native_anthropic_host(base)
    if native and _is_oauth_token(key):
        headers.update({"Authorization": f"Bearer {key}", "anthropic-beta": ",".join(_COMMON_BETAS + _OAUTH_ONLY_BETAS)})
        mode = "Anthropic OAuth token"
    elif not native and _requires_bearer_auth(base):
        headers["Authorization"] = f"Bearer {key}"
        mode = ""
    else:
        headers["x-api-key"] = key
        mode = "Anthropic Console API key" if native else ""
    root = (base or "https://api.anthropic.com").rstrip("/")
    return ProbeRequest(root + ("/models" if root.endswith("/v1") else "/v1/models"), headers, mode)


def _openai_request(base: str, key: str) -> ProbeRequest:
    headers = {"Authorization": f"Bearer {key}", "User-Agent": hermes_user_agent(), "Accept": "application/json"}
    if base_url_host_matches(base, "generativelanguage.googleapis.com"):
        # Google rejects ``Bearer <api-key>`` (reserved for OAuth 2); plain keys ride x-goog-api-key.
        headers.pop("Authorization")
        headers["x-goog-api-key"] = key
    return ProbeRequest(_KEY_INFO_URLS.get(base_url_hostname(base)) or base.rstrip("/") + "/models", headers)


def build_probe_request(provider: str, key: str, base_url: str, api_mode: str = "") -> ProbeRequest | None:
    """The one request that proves *key* authenticates at *base_url*; None when no such read exists.

    *api_mode* is the configured transport of a ``providers:`` entry: an ``anthropic_messages``
    endpoint gets the Anthropic auth family whatever its path looks like, as chat does.
    """
    configured = (base_url or "").strip().rstrip("/")
    probe_base = configured
    if provider.startswith("kimi-coding"):
        probe_base = _resolve_kimi_base_url(key, probe_base, "")  # sk-kimi-* keys live on api.kimi.com/coding
    sibling = probe_base
    if provider != "anthropic" and (probe_base.endswith("/anthropic") or base_url_host_matches(probe_base, "api.kimi.com")):
        # Dual-surface hosts list models on their OpenAI sibling (MiniMax ``/anthropic`` -> ``/v1``);
        # an Anthropic-only gateway keeps its path and gets the Anthropic auth family below.
        sibling = _to_openai_base_url(probe_base)
    on_anthropic_route = sibling == probe_base and (sibling.endswith("/anthropic") or api_mode == ANTHROPIC_MESSAGES)
    probe_base = sibling
    if provider == "anthropic" or on_anthropic_route:
        request = _anthropic_request(probe_base, key)
    elif not probe_base or base_url_hostname(probe_base) in _PUBLIC_CATALOG_HOSTS:
        return None
    else:
        request = _openai_request(probe_base, key)
    # Headers are configured per route: look them up by the URL the user configured, not the
    # rewritten probe URL (a MiniMax ``/anthropic`` entry's headers must still ride its ``/v1`` read).
    request.headers.update(get_custom_provider_extra_headers(configured))
    return request
