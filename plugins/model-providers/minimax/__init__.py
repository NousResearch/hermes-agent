"""MiniMax provider profiles (international, China, OAuth).

Default routes use anthropic_messages (base URLs end in /anthropic). Users can
opt MiniMax-M3 into the OpenAI-compatible https://api.minimax.io/v1 route,
which needs MiniMax-specific reasoning controls in extra_body.
"""

from typing import Any
from urllib.parse import urlparse

from providers import register_provider
from providers.base import ProviderProfile


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


# The MiniMax catalog, in the order MiniMax serves it.
#
# MiniMax publishes its model catalog at ``<base>/v1/models`` — the Anthropic-compatible route
# answers there with an ``x-api-key`` header — while the generic ``<base>/models`` probe the
# framework issues does not answer, and the failure is swallowed, so every picker surface
# silently degraded to the curated floor and a newly released model never appeared (#142648).
#
# Declaring the catalog as the profile's ``fallback_models`` fixes both halves of that:
#   * the picker lists the current models when the live probe does not answer, and
#   * ``merge_profile_catalog`` wraps the fallback in ``CuratedFallbackModels``, which the disk
#     cache treats as a PLACEHOLDER — it expires on the short fallback TTL and is never
#     stale-served. Without it the degraded list was cached as the account's own catalog for a
#     full hour and stale-served for the whole SWR window, so a stale floor outlived the outage
#     that produced it.
#
# The list mirrors the curated policy: the flagship plus the models the curated catalog lists.
# The ``-highspeed`` variants are deliberately NOT here — the curated catalog leaves them out
# (they are the 2x-priced tier) and the live catalog contributes them when it is reachable.
#
# Verified 2026-10-26 against the live endpoint: ``GET https://api.minimax.io/anthropic/v1/models``
# with the ``x-api-key`` header returns HTTP 200 and nine ids, ``MiniMax-M3.1-Flash`` first.
# The China route serves the same catalog (https://api.minimaxi.com/anthropic/v1/models), and
# models.dev lists ``MiniMax-M3.1-Flash`` for both ``minimax`` and ``minimax-cn``.
_MINIMAX_CATALOG = (
    "MiniMax-M3.1-Flash", "MiniMax-M3",
    "MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1", "MiniMax-M2",
)

minimax = MiniMaxProfile(
    name="minimax", aliases=("mini-max",), api_mode="anthropic_messages", env_vars=("MINIMAX_API_KEY",),
    base_url="https://api.minimax.io/anthropic", auth_type="api_key", default_aux_model="MiniMax-M3",
    fallback_models=_MINIMAX_CATALOG,
)

minimax_cn = MiniMaxProfile(
    name="minimax-cn", aliases=("minimax-china", "minimax_cn"), api_mode="anthropic_messages",
    env_vars=("MINIMAX_CN_API_KEY",), base_url="https://api.minimaxi.com/anthropic", auth_type="api_key",
    default_aux_model="MiniMax-M3", fallback_models=_MINIMAX_CATALOG,
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
