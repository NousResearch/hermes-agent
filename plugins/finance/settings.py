"""Config access for the finance plugin.

Non-secret settings live in ``config.yaml`` under the ``finance:`` section
(provider, sync interval, privacy mode, Plaid environment). Secrets
(``PLAID_CLIENT_ID`` / ``PLAID_SECRET``) live in ``.env`` and are read straight
from the environment — never from here.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

# Mirrors the ``finance`` block added to DEFAULT_CONFIG; used as a fallback when
# the config file cannot be loaded (e.g. minimal test environments).
_DEFAULTS: Dict[str, Any] = {
    "provider": "plaid",
    "sync_interval": "6h",
    "privacy_mode": "full",
    "plaid": {"environment": "sandbox", "webhook_url": ""},
    "categorization": {"llm_fallback": False},
}


def load_finance_settings() -> Dict[str, Any]:
    """Return the merged ``finance`` config section (defaults + user config)."""
    merged = copy.deepcopy(_DEFAULTS)
    try:
        from hermes_cli.config import load_config

        finance = (load_config() or {}).get("finance") or {}
    except Exception:
        finance = {}
    _deep_merge(merged, finance)
    return merged


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def get_provider_name() -> str:
    return str(load_finance_settings().get("provider") or "plaid").strip().lower()


def get_privacy_mode() -> str:
    return str(load_finance_settings().get("privacy_mode") or "full").strip().lower()


def get_plaid_environment() -> str:
    plaid = load_finance_settings().get("plaid") or {}
    return str(plaid.get("environment") or "sandbox").strip().lower()


def get_plaid_webhook_url() -> str:
    plaid = load_finance_settings().get("plaid") or {}
    return str(plaid.get("webhook_url") or "").strip()


def llm_fallback_enabled() -> bool:
    cat = load_finance_settings().get("categorization") or {}
    return bool(cat.get("llm_fallback"))
