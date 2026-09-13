"""Actual Computer provider profile."""

import os
import sys
from typing import Any
from urllib.parse import urlparse

from providers import register_provider
from providers.base import ProviderProfile

DEFAULT_ACTUAL_BASE_URL = "https://api.actual.inc/v1"


class ActualProfile(ProviderProfile):
    """Actual Computer: hosted at api.actual.inc; local (offline-mode client)
    inference opted into via model.base_url in config.yaml."""

    def build_client_kwargs_extras(self, **context: Any) -> dict[str, Any]:
        base_url = str(context.get("base_url") or self.base_url or "")
        try:
            hostname = (urlparse(base_url).hostname or "").lower().rstrip(".")
        except ValueError:
            return {}
        if sys.platform != "darwin" or hostname != "api.actual.inc":
            return {}
        if any(
            os.getenv(key)
            for key in (
                "HERMES_CA_BUNDLE",
                "SSL_CERT_FILE",
                "REQUESTS_CA_BUNDLE",
                "CURL_CA_BUNDLE",
            )
        ):
            return {}
        import certifi

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
        """Use the selected route, then config.yaml, then the legacy environment override."""
        from hermes_cli.auth import (
            normalize_actual_base_url,
            resolve_api_key_provider_credentials,
        )

        base_url = normalize_actual_base_url(
            base_url or resolve_api_key_provider_credentials("actual")["base_url"]
        )
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


actual = ActualProfile(
    name="actual",
    aliases=("actual-computer", "actualcomputer", "aci"),
    display_name="Actual Computer",
    description="Actual Computer - hosted inference via api.actual.inc, or local "
    "offline inference via model.base_url in config.yaml",
    signup_url="https://actual.inc",
    env_vars=("ACTUAL_API_KEY",),
    base_url=DEFAULT_ACTUAL_BASE_URL,
    auth_type="api_key",
    api_mode="chat_completions",
)

register_provider(actual)
