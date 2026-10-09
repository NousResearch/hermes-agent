"""DeepInfra provider profile: provider-specific catalogue HTTP and chat capabilities."""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

from agent.reasoning_effort import OPENAI_COMPAT_WIRE_EFFORTS, clamp_effort, requested_effort
from hermes_cli.urllib_security import open_credentialed_url
from hermes_cli.version_info import get_version_info
from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)


class _DeepInfraProfile(ProviderProfile):
    """DeepInfra's HTTP source; the shared tagged cache belongs to models.catalog_deepinfra."""

    def fetch_catalog(
        self, *, api_key: str = "", base_url: str | None = None, timeout: float = 5.0
    ) -> list[dict] | None:
        """Fetch all model surfaces in one guarded request; None distinguishes transport failure."""
        endpoint = str(base_url or self.base_url).strip().rstrip("/")
        req = urllib.request.Request(
            endpoint + "/models?filter=true&sort_by=hermes",
            headers={"User-Agent": f"hermes-cli/{get_version_info().base_version}",
                     **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
        )
        try:
            with open_credentialed_url(req, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            rows = payload.get("data") if isinstance(payload, dict) else None
            return rows if isinstance(rows, list) else None
        except Exception as exc:
            logger.debug("DeepInfra catalogue request failed: %s", exc)
            return None

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """Use the same tagged catalogue as image, video, pricing and auxiliary discovery."""
        from models.catalog_deepinfra import models_by_tag

        rows = models_by_tag(
            "chat", api_key=api_key or "", base_url=base_url or self.base_url, timeout=timeout
        )
        return [row["id"] for row in rows] or None if rows is not None else None

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Preserve DeepInfra's explicit reasoning-off wire and absent-effort defaults."""
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            return {}, {"reasoning_effort": "none"}
        effort = requested_effort(reasoning_config)
        clamped = clamp_effort(effort, OPENAI_COMPAT_WIRE_EFFORTS)
        return ({}, {"reasoning_effort": clamped}) if clamped in OPENAI_COMPAT_WIRE_EFFORTS else ({}, {})

    def default_vision_model(self):  # type: ignore[override]
        """Only vision-capable chat models qualify; never image-gen surface models."""
        from agent.secret_scope import get_secret
        from application_deepinfra_catalog import models_by_tag

        token = (get_secret("DEEPINFRA_API_KEY") or "").strip()
        if not token:
            return None
        for row in models_by_tag("chat", api_key=token) or []:
            meta = row.get("metadata") or {}
            if "vision" in (meta.get("tags") if isinstance(meta.get("tags"), list) else []):
                return row["id"]
        return None


deepinfra = _DeepInfraProfile(
    name="deepinfra", aliases=("deep-infra", "deepinfra-ai"), display_name="DeepInfra",
    description="DeepInfra — 100+ open models, pay-per-use",
    signup_url="https://deepinfra.com/dash/api_keys",
    env_vars=("DEEPINFRA_API_KEY",), base_url="https://api.deepinfra.com/v1/openai",
    base_url_env_var="DEEPINFRA_BASE_URL", auth_type="api_key",
    default_max_tokens=None, default_aux_model="deepseek-ai/DeepSeek-V4-Flash",
    fallback_models=(),
)
register_provider(deepinfra)
