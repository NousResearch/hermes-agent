"""MiniMax provider profiles (international, China, OAuth).

Default routes use anthropic_messages (base URLs end in /anthropic). Users can
opt MiniMax-M3 into the OpenAI-compatible https://api.minimax.io/v1 route,
which needs MiniMax-specific reasoning controls in extra_body.
"""

from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from providers import register_provider
from providers.base import ProviderProfile

from .quota import fetch_quota, quota_origin


def _is_minimax_global_openai_base_url(base_url: str | None) -> bool:
    parsed = urlparse(str(base_url or "").strip())
    return (parsed.hostname or "").lower() == "api.minimax.io" and parsed.path.rstrip("/").lower() == "/v1"


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

    def fetch_account_usage(self, *, base_url: str | None = None, api_key: str | None = None):
        """MiniMax token-plan windows (the rolling 5-hour limit and the weekly limit).

        The plan API answers at the API root (``/v1/token_plan/remains``), not under the
        Anthropic-compatible route the profile's base URL points at, so the origin comes
        from the profile rather than the runtime base URL. Every window is reported as
        percent USED (the generic contract); the MiniMax wire reports percent REMAINING.
        """
        from agent.account_usage import AccountUsageSnapshot

        runtime = self._runtime_credentials(base_url=base_url, api_key=api_key)
        if not runtime:
            return None
        token, origin = runtime
        windows = fetch_quota(token, origin)
        if not windows:
            return None
        return AccountUsageSnapshot(
            provider=self.name, source="token_plan_api", fetched_at=datetime.now(UTC),
            title="MiniMax plan", windows=tuple(windows))

    def _runtime_credentials(self, *, base_url: str | None, api_key: str | None):
        """``(api key, plan API origin)`` for this session, or None when there is no key.

        The live agent's own key wins over the configured one so the readout describes
        the account actually being billed. The origin comes from the PROFILE, never the
        runtime base URL: the plan API answers at MiniMax's API root, which a custom
        base URL (a proxy, an OpenAI-compatible route) does not carry.
        """
        token = str(api_key or "").strip()
        if not token:
            try:
                from hermes_cli.runtime_provider import resolve_runtime_provider

                runtime = resolve_runtime_provider(
                    requested=self.name, explicit_base_url=base_url, explicit_api_key=api_key)
                token = str(runtime.get("api_key") or "").strip()
            except Exception:  # health: allow BLE001 -- no resolvable credential means no readout, not a crash
                return None
        if not token:
            return None
        origin = quota_origin(self.base_url)
        return (token, origin) if origin else None


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
