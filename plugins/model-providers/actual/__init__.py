"""Actual Computer provider profile."""

from __future__ import annotations

import json
import logging
import os
from urllib.parse import urlparse
import urllib.request

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

DEFAULT_ACTUAL_BASE_URL = "https://api.actual.inc/v1"
DEFAULT_ACTUAL_LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"


def _normalize_actual_base_url(base_url: str) -> str:
    url = str(base_url or "").strip().rstrip("/")
    if not url:
        return DEFAULT_ACTUAL_BASE_URL
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        path = parsed.path.rstrip("/")
    except Exception:
        return url
    if host == "api.actual.inc" and path in {"", "/"}:
        return url + "/v1"
    if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} and path in {"", "/"}:
        return url + "/v1"
    return url


class ActualProfile(ProviderProfile):
    """Actual Computer provider.

    Hosted inference defaults to api.actual.inc. Local inference is exposed by
    the Actual client only when it runs in offline mode, so users opt into it by
    setting ACTUAL_BASE_URL to the local API URL.
    """

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        base_url = _normalize_actual_base_url(
            os.getenv("ACTUAL_BASE_URL", "").strip() or self.base_url
        )
        if not base_url:
            return None

        req = urllib.request.Request(base_url + "/models")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            items = data if isinstance(data, list) else data.get("data", [])
            return [m["id"] for m in items if isinstance(m, dict) and "id" in m]
        except Exception as exc:
            logger.debug("fetch_models(actual): %s", exc)
            return None


actual = ActualProfile(
    name="actual",
    aliases=("actual-computer", "actualcomputer", "aci"),
    display_name="Actual Computer",
    description=(
        "Actual Computer - hosted inference via api.actual.inc, or local "
        "offline inference via ACTUAL_BASE_URL"
    ),
    signup_url="https://actual.inc",
    env_vars=("ACTUAL_API_KEY", "ACTUAL_BASE_URL"),
    base_url=DEFAULT_ACTUAL_BASE_URL,
    auth_type="api_key",
    api_mode="codex_responses",
)

register_provider(actual)
