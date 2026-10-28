"""OpenCode Zen / Go endpoint URLs: the per-API inference base for each wire API, and the model catalog."""

from __future__ import annotations

import re
import urllib.parse
from typing import Optional

from utils import base_url_host_matches

# Inference root per OpenCode family on opencode.ai hosts.
_OPENCODE_FAMILY_PATHS = {"opencode-zen": "/inference", "opencode-go": "/inference/go"}
# Any OpenCode base path on opencode.ai: the per-API paths, the shared ``/inference(/go)/v1`` base and
# the ``/zen(/go)/v1`` alias. Group 1 is the Go segment.
_OPENCODE_BASE_PATH = re.compile(r"/(?:zen|inference)(/go)?(?:/openai|/anthropic)?(?:/v1)?")


def opencode_catalog_base(base_url: Optional[str]) -> Optional[str]:
    """Base of the model catalog (``{base}/models``) for an opencode.ai URL, else None. The catalog has no
    per-API path, so it is never under a runtime base; the family comes from the URL's ``/go`` segment."""
    url = str(base_url or "")
    if not base_url_host_matches(url, "opencode.ai"):
        return None
    path = urllib.parse.urlparse(url).path.lower()
    go = "/inference/go" in path or "/zen/go" in path
    return "https://opencode.ai/inference/go/v1" if go else "https://opencode.ai/inference/v1"


def normalize_opencode_base_url(
    provider_id: Optional[str], api_mode: Optional[str], base_url: Optional[str]) -> str:
    """Normalize an OpenCode Zen / Go base URL for the API mode. On opencode.ai hosts each family has
    one base per wire API: ``/inference/openai/v1`` for chat/completions and Responses,
    ``/inference/anthropic`` for Messages (the Anthropic SDK appends ``/v1/messages``), and the same
    under ``/inference/go`` for Go. The path is rebuilt for the mode on every call because the
    Messages URL gets persisted to ``model.base_url`` after switching into an anthropic-routed model,
    and a later chat/codex model must get the OpenAI base back. The deprecated ``/zen/v1`` and
    ``/zen/go/v1`` aliases and the shared ``/inference(/go)/v1`` base are rewritten too, so saved
    configs move over. The family segment follows the resolved family for the BUILT-IN providers:
    Zen and Go serve different model sets, so a ``model.base_url`` carried over from the other family
    401s ("Model ... is not supported"). A custom provider merely named after a family
    (``opencode-go-bridge``) keeps the family its URL declares. Other hosts (``OPENCODE_*_BASE_URL``
    proxies) only lose ``/v1`` for Messages. Only the path is edited, so a port, userinfo, query or
    fragment round-trips untouched."""
    from hermes_cli.models import _OPENCODE_FAMILIES, normalize_provider, opencode_provider_family

    url = str(base_url or "").strip().rstrip("/")
    family = opencode_provider_family(provider_id)
    if not url or family is None:
        return url
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    official = host == "opencode.ai" or host.endswith(".opencode.ai")
    path = parsed.path.rstrip("/")
    match = _OPENCODE_BASE_PATH.fullmatch(path) if official else None
    if match:
        builtin = normalize_provider(provider_id) in _OPENCODE_FAMILIES
        root = _OPENCODE_FAMILY_PATHS[family] if builtin else "/inference" + (match.group(1) or "")
        path = root + ("/anthropic" if api_mode == "anthropic_messages" else "/openai/v1")
    elif api_mode == "anthropic_messages":
        path = re.sub(r"/v1$", "", path)
    elif official and not path.endswith("/v1"):
        path += "/v1"
    return urllib.parse.urlunparse(parsed._replace(path=path))
