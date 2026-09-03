"""Actual Computer provider profile."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any
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

    def build_client_kwargs_extras(self, **context: Any) -> dict[str, Any]:
        base_url = str(context.get("base_url") or self.base_url or "")
        try:
            hostname = (urlparse(base_url).hostname or "").lower().rstrip(".")
        except Exception:
            return {}
        if sys.platform != "darwin" or hostname != "api.actual.inc":
            return {}

        try:
            import certifi
        except ImportError:
            return {}
        return {"ssl_ca_cert": certifi.where()}

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...] | None:
        from agent.reasoning_effort import ACTUAL_RELAY_EFFORTS

        return ACTUAL_RELAY_EFFORTS

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(reasoning_config, dict):
            return {}, {}

        from agent.reasoning_effort import clamp_effort, requested_effort

        enabled = reasoning_config.get("enabled") is not False
        if str(reasoning_config.get("effort") or "").strip().lower() == "none":
            enabled = False
        extra_body = {"thinking": {"type": "enabled" if enabled else "disabled"}}

        top_level: dict[str, Any] = {}
        effort = requested_effort(reasoning_config)
        if effort is not None:
            supported = self.supported_reasoning_efforts(context.get("model"))
            clamped = clamp_effort(effort, supported)
            if clamped in (supported or ()):
                top_level["reasoning_effort"] = clamped

        return extra_body, top_level

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        base_url = _normalize_actual_base_url(
            os.getenv("ACTUAL_BASE_URL", "").strip() or base_url or self.base_url
        )
        if not base_url:
            return None

        req = urllib.request.Request(base_url + "/models")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())

        from hermes_cli.urllib_security import open_credentialed_url

        try:
            with open_credentialed_url(req, timeout=timeout) as resp:
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
    api_mode="chat_completions",
)

register_provider(actual)
