"""Kari cloud model-provider integration.

The workflow login writes ``~/.hermes/workflow-secrets.json`` for Langflow
relay.  Hermes's main model picker/runtime also needs the same token to expose
the cloud-hosted Kari LLM tiers without persisting generated provider config.
"""

from __future__ import annotations

import os
from typing import Any

KARI_PROVIDER_SLUG = "kari-cloud"
KARI_PROVIDER_NAME = "Kari 云端"
KARI_TIER_MODELS = ("极致", "性能", "DeepSeek")


def _workflow_secrets() -> dict[str, Any]:
    try:
        from hermes_cli.workflow_backend import read_secrets

        data = read_secrets()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _auth_from_env() -> tuple[str, str]:
    return (
        (os.getenv("KARI_HUB_URL") or "").strip().rstrip("/"),
        (os.getenv("KARI_WORKSPACE_TOKEN") or "").strip(),
    )


def kari_cloud_auth() -> tuple[str, str] | None:
    """Return ``(cloud_base_url, token)`` when the user is logged into Kari."""
    base, token = _auth_from_env()
    if not (base and token):
        secrets = _workflow_secrets()
        kari = secrets.get("kari")
        if isinstance(kari, dict):
            base = str(kari.get("cloudBaseURL") or kari.get("cloudBaseUrl") or "").strip().rstrip("/")
            token = str(kari.get("token") or "").strip()
    if not (base and token):
        return None
    return base, token


def kari_cloud_openai_base_url(cloud_base_url: str) -> str:
    return f"{cloud_base_url.rstrip('/')}/api/v1/kari/llm/v1"


def kari_cloud_provider_config() -> dict[str, Any] | None:
    auth = kari_cloud_auth()
    if not auth:
        return None
    cloud_base_url, token = auth
    return {
        "name": KARI_PROVIDER_NAME,
        "base_url": kari_cloud_openai_base_url(cloud_base_url),
        "api_key": token,
        "api_mode": "chat_completions",
        "models": {model: {} for model in KARI_TIER_MODELS},
    }


def merge_kari_cloud_provider(user_providers: Any) -> dict[str, Any]:
    """Return a copy of ``user_providers`` with the synthetic Kari row added."""
    merged = dict(user_providers) if isinstance(user_providers, dict) else {}
    if KARI_PROVIDER_SLUG in merged:
        return merged
    cfg = kari_cloud_provider_config()
    if cfg:
        merged[KARI_PROVIDER_SLUG] = cfg
    return merged
