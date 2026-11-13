"""Provider inference from a base URL: the host table, the profile claims, and the route label.

A route's provider id is what every catalog/profile lookup keys on, so it has to be derived
from the endpoint the request actually goes to. Two things decide it, in order: a registered
provider profile that claims the host (``hostnames``/``metadata_hostnames``, per home), then
the static vendor-host table below. ``custom:<name>`` picker ids are config rows, not
profiles, and are normalized here for that lookup (``resolve_route_provider``).
"""

from __future__ import annotations

import logging
from typing import Optional

from utils import base_url_host_matches, base_url_hostname


# Hostname -> provider: a key matches that host or a subdomain; regional <region>-aiplatform Vertex follows.
# Profiles claim more hosts via ``hostnames``/``metadata_hostnames`` (``providers.profile_for_host``), per home.
_URL_TO_PROVIDER: dict[str, str] = {
    "api.openai.com": "openai", "chatgpt.com": "openai", "api.anthropic.com": "anthropic",
    "api.z.ai": "zai", "open.bigmodel.cn": "zai",
    "api.moonshot.ai": "kimi-coding", "api.moonshot.cn": "kimi-coding-cn", "api.kimi.com": "kimi-coding",
    "api.stepfun.ai": "stepfun", "api.stepfun.com": "stepfun", "api.arcee.ai": "arcee",
    "api.minimax.io": "minimax", "api.minimaxi.com": "minimax", "api.minimax.chat": "minimax",
    "dashscope.aliyuncs.com": "alibaba", "dashscope-intl.aliyuncs.com": "alibaba", "portal.qwen.ai": "qwen-oauth",
    "openrouter.ai": "openrouter", "generativelanguage.googleapis.com": "gemini",
    "inference-api.nousresearch.com": "nous", "api.deepseek.com": "deepseek",
    "api.fireworks.ai": "fireworks", "opencode.ai": "opencode-go", "api.x.ai": "xai",
    "integrate.api.nvidia.com": "nvidia", "api.xiaomimimo.com": "xiaomi", "xiaomimimo.com": "xiaomi",
    "api.gmi-serving.com": "gmi", "api.novita.ai": "novita",
    "tokenhub.tencentmaas.com": "tencent-tokenhub", "api.lkeap.cloud.tencent.com": "tencent-tokenplan",
    "ollama.com": "ollama-cloud",
}

# Auto-extend with provider-profile hostnames not already mapped.
try:
    from providers import list_providers as _list_providers

    for _pp in _list_providers():
        _host = _pp.get_hostname()
        if _host and _host not in _URL_TO_PROVIDER:
            _URL_TO_PROVIDER[_host] = _pp.name
except Exception:  # a broken profile registry must not break import; the vendor table still answers
    logging.getLogger(__name__).debug("provider-profile hostnames not merged into the URL table", exc_info=True)


def _infer_provider_from_url(base_url: str) -> Optional[str]:
    """models.dev provider name for a base URL (custom endpoints need no explicit provider)."""
    host = base_url_hostname(base_url)
    if not host:
        return None
    from providers import profile_for_host
    if (profile := profile_for_host(base_url, metadata=True)) is not None:  # profile claims beat table rows
        return profile.name
    for domain, provider in _URL_TO_PROVIDER.items():
        if base_url_host_matches(host, domain):
            return provider
    return "vertex" if host.endswith("-aiplatform.googleapis.com") else None


def _is_known_provider_base_url(base_url: str) -> bool:
    return _infer_provider_from_url(base_url) is not None


def resolve_route_provider(provider: str, base_url: str = "") -> str:
    """Provider id to consult for a route's profile and catalog lookups.

    ``custom:<name>`` is the picker/route id of a user-registered relay: a
    ``providers.<name>`` config row, not a provider profile. For lookups it therefore
    behaves exactly like plain ``custom`` — the endpoint's host decides which registered
    profile (and models.dev provider) owns the route, so a relay whose host a profile
    already claims keeps that profile's live context seam instead of falling through to
    the hardcoded family defaults. A route id whose host nothing claims keeps its own
    name, because its ``catalog_provider`` alias is keyed on it. Every other label is
    returned unchanged.
    """
    label = (provider or "").strip()
    if not base_url or not label.lower().startswith("custom:"):
        return label
    return _infer_provider_from_url(base_url) or label
