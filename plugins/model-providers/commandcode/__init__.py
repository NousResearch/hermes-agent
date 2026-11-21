"""CommandCode provider profiles: ``commandcode`` (chat_completions) and
``commandcode-anthropic`` (anthropic_messages, Bearer auth — see
``agent/anthropic_adapter.py``). Same key and base URL for both.

CommandCode publishes ONE catalog for both wires, and its rows declare in
``supported_endpoints`` which wires accept them: every ``claude-*`` row is
``/messages``-only, so the chat_completions profile offering a Claude model gave
a 400 ``unsupported_model`` on every selection. Each profile therefore declares
the wire it speaks (``catalog_endpoint``) and the shared catalog filter keeps a
model the API would reject off the picker, instead of guessing families by id
prefix.
"""

import json
import logging
import urllib.request

from hermes_cli.urllib_security import open_credentialed_url
from providers import get_provider_profile, register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

_COMMANDCODE_BASE = "https://api.commandcode.ai/provider/v1"
_COMMANDCODE_MODELS_URL = f"{_COMMANDCODE_BASE}/models"


def _commandcode_catalog_items(
    self: ProviderProfile, *, api_key: str | None, base_url: str | None, timeout: float,
) -> list[dict] | None:
    """Raw ``/models`` items for CommandCode, or None when the fetch fails."""
    caller_base = (base_url or "").strip().rstrip("/")
    custom = caller_base and caller_base != _COMMANDCODE_BASE
    models_url = caller_base + "/models" if custom else _COMMANDCODE_MODELS_URL
    try:
        req = urllib.request.Request(models_url)
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())
        with open_credentialed_url(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        items = data.get("data", []) if isinstance(data, dict) else data
        return [item for item in items if isinstance(item, dict) and "id" in item]
    except Exception as exc:  # health: allow BLE001 -- a failed catalog probe degrades to fallback_models, never breaks the picker
        logger.debug("fetch_models(commandcode): %s", exc)
        return None


class CommandCodeProfile(ProviderProfile):
    """CommandCode — one catalog, filtered to whichever wire the profile speaks."""

    def fetch_catalog_items(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[dict] | None:
        """CommandCode's ``/models`` endpoint, rows intact (no filtering here).

        The picker passes ``base_url`` unconditionally, so only a value differing
        from the profile default is a custom endpoint. Wire filtering happens in
        :meth:`ProviderProfile.fetch_models` via ``catalog_endpoint``.
        """
        return _commandcode_catalog_items(
            self, api_key=api_key, base_url=base_url, timeout=timeout)

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict, dict]:
        """DeepSeek ids (``deepseek/deepseek-v4-flash``) get the native DeepSeek wire
        controls: DeepSeek V4+ defaults to thinking when ``thinking`` is omitted, so
        without them ``/reasoning`` never reaches the request (#95232). Other model
        families stay a no-op — CommandCode declares no reasoning vocabulary for them."""
        m = (model or "").strip()
        if not m.lower().startswith("deepseek/"):
            return {}, {}
        # Registry lookup, not a module import: the deepseek shim is only a loader-injected
        # sys.modules entry, and the registry honours a user override of the profile.
        native = get_provider_profile("deepseek")
        if native is None:
            return {}, {}
        return native.build_api_kwargs_extras(
            reasoning_config=reasoning_config, model=m.split("/", 1)[1], **context,
        )


commandcode = CommandCodeProfile(
    name="commandcode", aliases=("commandcode-chat",), api_mode="chat_completions",
    # Same key as the anthropic profile; distinct base-URL override vars so each
    # profile renders its own card on the desktop Keys tab (rows keyed by env var).
    env_vars=("COMMANDCODE_API_KEY", "COMMANDCODE_BASE_URL"),
    display_name="CommandCode", description="CommandCode — 20+ models via OpenAI-compatible API",
    signup_url="https://commandcode.ai/", base_url=_COMMANDCODE_BASE, models_url=_COMMANDCODE_MODELS_URL,
    fallback_models=(
        "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash", "Qwen/Qwen3.7-Max", "Qwen/Qwen3.6-Plus",
        "moonshotai/Kimi-K2.6", "zai-org/GLM-5.1", "MiniMaxAI/MiniMax-M2.7", "stepfun/Step-3.5-Flash",
        "xiaomi/mimo-v2.5-pro", "google/gemini-3.5-flash", "gpt-5.5",
    ),
    default_aux_model="deepseek/deepseek-v4-flash",
    catalog_endpoint="/chat/completions",
)

commandcode_anthropic = CommandCodeProfile(
    name="commandcode-anthropic", aliases=("commandcode-claude",), api_mode="anthropic_messages",
    env_vars=("COMMANDCODE_API_KEY", "COMMANDCODE_ANTHROPIC_BASE_URL"),
    display_name="CommandCode (Anthropic)",
    description="CommandCode — Claude models via Anthropic Messages API",
    signup_url="https://commandcode.ai/", base_url=_COMMANDCODE_BASE, models_url=_COMMANDCODE_MODELS_URL,
    fallback_models=("claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5-20251001"),
    default_aux_model="claude-haiku-4-5-20251001",
    catalog_endpoint="/messages",
)

register_provider(commandcode)
register_provider(commandcode_anthropic)
