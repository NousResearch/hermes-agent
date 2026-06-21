"""Shared Apify SDK client — lazy import, token validation, and cache.

Used by the Apify Actor execution tools (plugins/apify/tools.py). The
``apify-client`` SDK is installed on demand via ``tools.lazy_deps`` so the
dependency is only pulled when the user actually enables the plugin and runs
an Actor.
"""
from __future__ import annotations

import os
from typing import Any, Optional

# Sent with every request so Apify can attribute traffic to this integration.
_HERMES_HEADERS = {"x-apify-integration-platform": "hermes-agent"}

_CLIENT_CLS: Optional[type] = None
_CLIENT: Optional[Any] = None
_CLIENT_CONFIG: Optional[Any] = None


def _load_client_cls() -> type:
    global _CLIENT_CLS
    if _CLIENT_CLS is None:
        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("search.apify", prompt=False)
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            raise ImportError(str(exc))
        from apify_client import ApifyClient
        _CLIENT_CLS = ApifyClient
    return _CLIENT_CLS


def check_apify_api_key() -> bool:
    """Return True when APIFY_API_TOKEN is configured."""
    return bool(os.getenv("APIFY_API_TOKEN", "").strip())


def get_apify_client() -> Any:
    """Return a cached ApifyClient built from APIFY_API_TOKEN.

    Raises ValueError when the token is not set.
    """
    global _CLIENT, _CLIENT_CONFIG
    api_token = os.getenv("APIFY_API_TOKEN", "").strip()
    if not api_token:
        raise ValueError(
            "Apify tools are not configured. "
            "Set APIFY_API_TOKEN (get one at https://apify.com/account/integrations)."
        )
    client_config = ("direct", api_token)
    if _CLIENT is not None and _CLIENT_CONFIG == client_config:
        return _CLIENT
    _CLIENT = _load_client_cls()(token=api_token, headers=_HERMES_HEADERS)
    _CLIENT_CONFIG = client_config
    return _CLIENT


def _reset_client_for_tests() -> None:
    """Drop cached client so tests can re-instantiate cleanly."""
    global _CLIENT, _CLIENT_CONFIG
    _CLIENT = None
    _CLIENT_CONFIG = None
