"""Kimi / Moonshot provider profiles (chat_completions path; sk-kimi-* keys are
redirected to api.kimi.com/coding by core)."""

from typing import Any
from urllib.parse import urlparse

from agent.reasoning_effort import KIMI_K3_EFFORTS, KIMI_K3_OVERRIDES, thinking_toggle_extras
from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import OMIT_TEMPERATURE, ProviderProfile

_HEADERS = {
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-Title": "Hermes Agent",
    "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
}


def _is_confirmed_kimi_coding_url(base_url: str) -> bool:
    """True only for Kimi Code's canonical HTTPS API surfaces."""
    try:
        p = urlparse(base_url)
        port = p.port
    except ValueError:
        return False
    return (
        p.scheme.lower() == "https" and (p.hostname or "").lower() == "api.kimi.com" and port in (None, 443)
        and p.username is None and p.password is None
        and p.path.rstrip("/") in {"/coding", "/coding/v1"} and not p.query and not p.fragment
    )


class KimiProfile(ProviderProfile):
    """Kimi/Moonshot — temperature omitted, thinking xor reasoning_effort."""

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """Use Kimi Code's OpenAI-compatible surface for model discovery; the bare
        ``k3`` slug is only served there, so it is filtered off other endpoints."""
        effective_base = (base_url or self.base_url or "").rstrip("/")
        confirmed_coding_endpoint = _is_confirmed_kimi_coding_url(effective_base)
        if confirmed_coding_endpoint and urlparse(effective_base).path.rstrip("/") == "/coding":
            effective_base += "/v1"
        models = super().fetch_models(api_key=api_key, base_url=effective_base or None, timeout=timeout)
        if models is None or confirmed_coding_endpoint:
            return models
        return [model for model in models if model.strip().lower() != "k3"]

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Kimi reasoning controls.

        Moonshot's wire shape treats ``extra_body.thinking`` (a binary toggle)
        and a top-level ``reasoning_effort`` as mutually exclusive — sending
        both is at best redundant and risks "cannot specify both 'thinking' and
        'reasoning_effort'" (HTTP 400). This mirrors the kimi-k2 handling on the
        opencode-go relay: send effort when one is requested, otherwise fall
        back to ``extra_body.thinking`` — never both.
        """
        extra_body = {}
        top_level = {}

        if not reasoning_config or not isinstance(reasoning_config, dict):
            # No config → thinking enabled, let the server pick the depth.
            # (Previously also sent reasoning_effort="medium", which paired
            # thinking + effort on every default call.)
            extra_body["thinking"] = {"type": "enabled"}
            return extra_body, top_level

        enabled = reasoning_config.get("enabled", True)
        if enabled is False:
            extra_body["thinking"] = {"type": "disabled"}
            return extra_body, top_level

        # Enabled: prefer an explicit effort; only fall back to extra_body
        # thinking when no recognized effort is requested.
        # K3's vocabulary (low/high/max, default high) and its documented
        # rounding (medium→high, xhigh→max) are declared in
        # agent.reasoning_effort — shared with the chat-completions
        # transport's Kimi path so both stay in sync.
        from agent.reasoning_effort import (
            KIMI_K3_EFFORTS,
            KIMI_K3_OVERRIDES,
            clamp_effort,
        )

        effort = (reasoning_config.get("effort") or "").strip().lower()
        if effort and effort != "none":
            k3_effort = clamp_effort(effort, KIMI_K3_EFFORTS, KIMI_K3_OVERRIDES)
        else:
            k3_effort = None
        if k3_effort in KIMI_K3_EFFORTS:
            top_level["reasoning_effort"] = k3_effort
        else:
            extra_body["thinking"] = {"type": "enabled"}

        return extra_body, top_level


def _kimi(name: str, aliases: tuple, env_vars: tuple, base_url: str) -> KimiProfile:
    return KimiProfile(
        name=name, aliases=aliases, env_vars=env_vars, base_url=base_url,
        fixed_temperature=OMIT_TEMPERATURE, default_max_tokens=32000,
        default_headers=dict(_HEADERS), default_aux_model="kimi-k2-turbo-preview",
    )


kimi = _kimi("kimi-coding", ("kimi", "moonshot", "kimi-for-coding"), ("KIMI_API_KEY", "KIMI_CODING_API_KEY"),
             "https://api.moonshot.ai/v1")
kimi_cn = _kimi("kimi-coding-cn", ("kimi-cn", "moonshot-cn"), ("KIMI_CN_API_KEY",), "https://api.moonshot.cn/v1")

register_provider(kimi)
register_provider(kimi_cn)
