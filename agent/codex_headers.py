"""Codex request identity helpers shared by agent client builders.

Leaf module with no dependency on the large auxiliary-client router, so a
long-lived process can import a newly added client builder without resolving a
new symbol from an older cached ``auxiliary_client`` module.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict
from urllib.parse import urlparse


CODEX_AUX_BASE_URL = "https://chatgpt.com/backend-api/codex"


def is_official_codex_base_url(base_url: str) -> bool:
    """Identify OpenAI's Codex endpoint without matching custom proxies."""
    try:
        parsed = urlparse(base_url)
        path = parsed.path.rstrip("/")
        return (
            parsed.scheme == "https"
            and parsed.hostname == "chatgpt.com"
            and parsed.port in (None, 443)
            and (path == "/backend-api/codex" or path.startswith("/backend-api/codex/"))
        )
    except (TypeError, ValueError):
        return False


def codex_cloudflare_headers(access_token: str, *, base_url: str = CODEX_AUX_BASE_URL) -> Dict[str, str]:
    """Identity and account headers for chatgpt.com/backend-api/codex.

    OpenAI requires third-party harnesses to identify themselves: the official
    endpoint gets Hermes' originator and version, custom endpoints keep the
    codex_cli_rs compatibility identity. The account headers come from the
    OAuth JWT (see :func:`codex_account_headers`).
    """
    if is_official_codex_base_url(base_url):
        from hermes_cli import __version__
        headers = {"User-Agent": f"HermesAgent/{__version__}", "originator": "hermes-agent"}
    else:
        headers = {"User-Agent": "codex_cli_rs/0.0.0 (Hermes Agent)", "originator": "codex_cli_rs"}
    headers.update(codex_account_headers(access_token))
    return headers


def codex_account_headers(access_token: str) -> Dict[str, str]:
    """Workspace headers the Codex backend derives from the OAuth JWT.

    ``ChatGPT-Account-ID`` (canonical casing, from codex-rs ``auth.rs``) comes from
    ``chatgpt_account_id``; ``x-openai-internal-codex-residency`` from
    ``chatgpt_data_residency`` (fallback ``chatgpt_compute_residency``) — without
    it residency-enforced workspaces answer 401 "Workspace is not authorized in
    this region". A malformed token drops the headers rather than raising, so it
    surfaces as a 401 instead of a crash at client construction.
    """
    headers: Dict[str, str] = {}
    if not isinstance(access_token, str) or not access_token.strip():
        return headers
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return headers
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        auth = json.loads(base64.urlsafe_b64decode(payload_b64)).get("https://api.openai.com/auth", {})
        acct_id = auth.get("chatgpt_account_id")
        if isinstance(acct_id, str) and acct_id:
            headers["ChatGPT-Account-ID"] = acct_id
        residency = auth.get("chatgpt_data_residency") or auth.get("chatgpt_compute_residency")
        if isinstance(residency, str) and residency.strip():
            headers["x-openai-internal-codex-residency"] = residency.strip()
    except Exception:
        pass
    return headers


def apply_required_codex_headers(client_kwargs: Dict[str, Any], *, access_token: str, base_url: str) -> None:
    """Keep required Codex identity after user/provider header overrides."""
    if not is_official_codex_base_url(base_url):
        return
    required = codex_cloudflare_headers(access_token, base_url=base_url)
    required_names = {name.lower() for name in required}
    existing = client_kwargs.get("default_headers") or {}
    client_kwargs["default_headers"] = {
        **{name: value for name, value in existing.items() if str(name).lower() not in required_names},
        **required,
    }
