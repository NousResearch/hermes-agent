"""MiniMax provider profiles (international, China, OAuth).

Default routes use anthropic_messages (base URLs end in /anthropic). Users can
opt MiniMax-M3 into the OpenAI-compatible https://api.minimax.io/v1 route,
which needs MiniMax-specific reasoning controls in extra_body.
"""

import json
import logging
import urllib.request
from typing import Any
from urllib.parse import urlparse

from hermes_cli.urllib_security import open_credentialed_url
from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)


def _is_minimax_global_openai_base_url(base_url: str | None) -> bool:
    parsed = urlparse(str(base_url or "").strip())
    return (parsed.hostname or "").lower() == "api.minimax.io" and parsed.path.rstrip("/").lower() == "/v1"


def _minimax_models_url(base_url: str | None) -> str:
    """Return the models endpoint for a MiniMax base URL.

    The OpenAI-compatible route serves a plain OpenAI-shaped list at
    ``{base}/models``; the anthropic route serves the same list at
    ``{base}/v1/models``.
    """
    endpoint = str(base_url or "").strip().rstrip("/")
    if not endpoint:
        return ""
    return endpoint + ("/models" if endpoint.endswith("/v1") else "/v1/models")


class MiniMaxProfile(ProviderProfile):
    """MiniMax — M3 OpenAI-compatible reasoning controls."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None,
        base_url: str | None = None, **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """M3 on api.minimax.io/v1 keeps thinking inline unless ``reasoning_split``
        is sent; effort levels only select adaptive vs disabled ``thinking``."""
        is_m3 = str(model or "").strip().lower() in {"minimax-m3", "minimax/minimax-m3"}
        if not _is_minimax_global_openai_base_url(base_url) or not is_m3:
            return {}, {}
        extra_body: dict[str, Any] = {"reasoning_split": True}
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            extra_body["thinking"] = {"type": "disabled"}
        elif reasoning_config is not None:
            extra_body["thinking"] = {"type": "adaptive"}
        return extra_body, {}

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """List the live MiniMax catalog from the ``/v1/models`` endpoint.

        The base implementation requests ``{base_url}/models`` with
        ``Authorization: Bearer`` auth. MiniMax's anthropic_messages routes
        expose the catalog at ``{base_url}/v1/models`` and authenticate with
        the ``x-api-key`` header, so the base request 404s and the picker
        silently falls back to the static catalog — which is why a newly
        released model never appeared in the model picker.

        Returns ``None`` on any failure so the caller falls back to the static
        list.
        """
        if not api_key:
            return None
        endpoint = str(base_url or self.base_url or "").strip().rstrip("/")
        url = _minimax_models_url(endpoint)
        if not url:
            return None
        req = urllib.request.Request(url)
        for k, v in (("x-api-key", api_key), ("anthropic-version", "2023-06-01"), ("Accept", "application/json")):
            req.add_header(k, v)
        try:
            with open_credentialed_url(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("fetch_models(minimax): %s", exc)
            return None
        models = [m["id"] for m in data.get("data", []) if isinstance(m, dict) and "id" in m]
        return list(dict.fromkeys(models)) or None


minimax = MiniMaxProfile(
    name="minimax", aliases=("mini-max",), api_mode="anthropic_messages", env_vars=("MINIMAX_API_KEY",),
    base_url="https://api.minimax.io/anthropic", auth_type="api_key", default_aux_model="MiniMax-M3",
)

minimax_cn = MiniMaxProfile(
    name="minimax-cn", aliases=("minimax-china", "minimax_cn"), api_mode="anthropic_messages",
    env_vars=("MINIMAX_CN_API_KEY",), base_url="https://api.minimaxi.com/anthropic", auth_type="api_key",
    default_aux_model="MiniMax-M3",
)

minimax_oauth = MiniMaxProfile(
    name="minimax-oauth", aliases=("minimax_oauth", "minimax-portal", "minimax-global", "minimax-oauth-io"), api_mode="anthropic_messages",
    display_name="MiniMax (OAuth)", description="MiniMax via OAuth browser flow — no API key required",
    signup_url="https://api.minimax.io/",
    env_vars=(),  # OAuth — tokens in auth.json, not env
    base_url="https://api.minimax.io/anthropic", auth_type="oauth_external", default_aux_model="MiniMax-M2.7",
)

register_provider(minimax)
register_provider(minimax_cn)
register_provider(minimax_oauth)
