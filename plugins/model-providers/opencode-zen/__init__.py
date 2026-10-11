"""OpenCode provider profiles (Zen + Go).

Both route api_mode per model in core; these profiles carry the
chat_completions reasoning translations (Ox Alpha on Zen; the per-model table on Go).
"""

from typing import Any, Callable

from agent import reasoning_effort as re_
from hermes_cli.version_info import get_version_info
from providers import register_provider
from providers.base import ProviderProfile
from datetime import UTC

# Attribution headers (same values as OpenRouter / Vercel / Fireworks); via
# default_headers so they survive model switches and credential rotation.
_ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-Title": "Hermes Agent",
    "User-Agent": f"HermesAgent/{get_version_info().base_version}",
}


def _flat_model_name(model: str | None) -> str:
    """Bare OpenCode model ID, tolerating aggregator prefixes."""
    return (model or "").strip().rsplit("/", 1)[-1].lower()


# Version-less DeepSeek ids that still carry the thinking/effort knobs on this wire: the retired
# ``deepseek-reasoner`` alias and the canonical ``deepseek-flash`` (2026-09 Flash refresh), for
# which the Go relay honours the same top-level ``reasoning_effort``/``thinking`` contract.
_THINKING_CAPABLE_IDS: frozenset[str] = frozenset({"deepseek-reasoner", "deepseek-flash"})


def _is_deepseek_thinking_model(model: str | None) -> bool:
    m = _flat_model_name(model)
    return (m.startswith("deepseek-v") and not m.startswith("deepseek-v3")) or m in _THINKING_CAPABLE_IDS


_ExtrasFn = Callable[[dict | None, str | None], tuple[dict, dict]]

_GLM_5_3_TOKENS = ("glm-5.3", "glm-5-3", "glm-5p3")
_GLM_5_2_PLUS_TOKENS = ("glm-5.2", "glm-5-2", "glm-5p2") + _GLM_5_3_TOKENS


def _has_token(model: str | None, tokens: tuple[str, ...]) -> bool:
    m = _flat_model_name(model)
    return any(token in m for token in tokens)


def _is_glm_5_3_flash(model: str | None) -> bool:
    return _has_token(model, _GLM_5_3_TOKENS) and "flash" in _flat_model_name(model)


def _prefixed(*prefixes: str) -> Callable[[str | None], bool]:
    return lambda model: _flat_model_name(model).startswith(prefixes)


def _effort_extras(efforts: tuple[str, ...], overrides: dict[str, str] | None = None) -> _ExtrasFn:
    """Top-level ``reasoning_effort`` clamped onto a relay vocabulary. Disabled asks for ``none``;
    where the model has no ``none`` (thinking-only) the field is omitted and the server default holds."""

    def extras(reasoning_config: dict | None, model: str | None) -> tuple[dict, dict]:
        disabled = isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False
        effort = "none" if disabled else re_.requested_effort(reasoning_config)
        if effort is None or (effort == "none" and "none" not in efforts):
            return {}, {}
        clamped = re_.clamp_effort(effort, efforts, overrides)
        return ({}, {"reasoning_effort": clamped}) if clamped in efforts else ({}, {})

    return extras


# Kimi K2 models that are thinking-only on the relay: ``thinking: disabled`` 400s
# ("only type=enabled is allowed"), so a disable leaves the server default.
_THINKING_ONLY_KIMI_PREFIXES = ("kimi-k2.7",)


def _kimi_k2_extras(reasoning_config: dict | None, model: str | None) -> tuple[dict, dict]:
    if not isinstance(reasoning_config, dict):
        return {}, {}
    if reasoning_config.get("enabled") is False and _flat_model_name(model).startswith(_THINKING_ONLY_KIMI_PREFIXES):
        return {}, {}
    return re_.thinking_toggle_extras(reasoning_config, re_.KIMI_K2_EFFORTS)


def _deepseek_extras(reasoning_config: dict | None, model: str | None) -> tuple[dict, dict]:
    return re_.thinking_toggle_extras(reasoning_config, re_.DEEPSEEK_V4_EFFORTS, re_.DEEPSEEK_V4_OVERRIDES)


# First match owns the model; unmatched models send nothing (relay default). Vocabularies are
# the relay's accepted levels from live probes (agent.reasoning_effort OPENCODE_GO_*), so a new
# Go model needs a probe and a row here. Flash must precede the wider GLM-5.2/5.3 row.
_GO_REASONING_ROUTES: tuple[tuple[Callable[[str | None], bool], _ExtrasFn], ...] = (
    (_is_glm_5_3_flash, _effort_extras(re_.OPENAI_COMPAT_WIRE_EFFORTS)),
    (lambda model: _has_token(model, _GLM_5_2_PLUS_TOKENS), _effort_extras(re_.OPENCODE_GO_GLM_EFFORTS)),
    (_prefixed("kimi-k3"), _effort_extras(re_.OPENAI_COMPAT_WIRE_EFFORTS)),
    (_prefixed("kimi-k2"), _kimi_k2_extras),
    (_is_deepseek_thinking_model, _deepseek_extras),
    (_prefixed("mimo-v2.6-flash"), _effort_extras(re_.OPENCODE_GO_MIMO26_FLASH_EFFORTS)),
    (_prefixed("mimo-v2.5", "longcat-", "hy3", "hy4"), _effort_extras(re_.OPENAI_COMPAT_WIRE_EFFORTS)),
    (_prefixed("space-bunny"), _effort_extras(re_.OPENCODE_GO_SPACE_BUNNY_EFFORTS)),
)


class OpenCodeGoProfile(ProviderProfile):
    """OpenCode Go - model-specific reasoning controls."""

    # The relay's default max_tokens (262144) exceeds what Xiaomi accepts for
    # mimo-v2.5-pro and 400s; keys are normalized via _flat_model_name().
    _MODEL_MAX_TOKENS: dict[str, int] = {"mimo-v2.5-pro": 131072}

    def get_max_tokens(self, model: str | None) -> int | None:
        cap = self._MODEL_MAX_TOKENS.get(_flat_model_name(model))
        return self.default_max_tokens if cap is None else cap

    def fetch_account_usage(self, *, base_url: str | None = None, api_key: str | None = None):
        """Go subscription windows for /usage via ``/zen/go/v1/usage`` (anomalyco/opencode#16513).

        Percent-based payload ``{"usage": {"rolling"|"weekly"|"monthly": {"percent", "resetsAt"}}}``.
        Literal endpoint, NOT the runtime base_url: that one loses its /v1 suffix in
        anthropic_messages mode and /usage only exists under /v1.
        """
        from datetime import datetime, timezone

        import httpx

        from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested=self.name, explicit_base_url=base_url, explicit_api_key=api_key)
        token = str(runtime.get("api_key", "") or "").strip()
        if not token:
            return None
        with httpx.Client(timeout=10.0) as client:
            response = client.get("https://opencode.ai/zen/go/v1/usage",
                                  headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
            response.raise_for_status()
        usage = (response.json() or {}).get("usage") or {}
        windows = []
        for key, label in (("rolling", "Rolling window"), ("weekly", "Weekly"), ("monthly", "Monthly")):
            window = usage.get(key) or {}
            if window.get("percent") is None:
                continue
            reset_raw = str(window.get("resetsAt") or "").replace("Z", "+00:00")
            reset_at = datetime.fromisoformat(reset_raw) if reset_raw else None
            windows.append(AccountUsageWindow(label=label, used_percent=float(window["percent"]), reset_at=reset_at))
        return AccountUsageSnapshot(provider=self.name, source="go_usage_api",
                                    fetched_at=datetime.now(UTC), windows=tuple(windows))

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        for matches, extras in _GO_REASONING_ROUTES:
            if matches(model):
                return extras(reasoning_config, model)
        return {}, {}


class OpenCodeZenProfile(ProviderProfile):
    """OpenCode Zen - model-specific reasoning controls."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return re_.ox_alpha_reasoning_extras(reasoning_config, model)


opencode_zen = OpenCodeZenProfile(
    name="opencode-zen", aliases=("opencode", "opencode_zen", "zen"), env_vars=("OPENCODE_ZEN_API_KEY",),
    base_url="https://opencode.ai/zen/v1", default_headers=dict(_ATTRIBUTION_HEADERS),
    default_aux_model="gemini-3-flash",
)

opencode_go = OpenCodeGoProfile(
    name="opencode-go", aliases=("opencode_go", "go", "opencode-go-sub"), env_vars=("OPENCODE_GO_API_KEY",),
    base_url="https://opencode.ai/zen/go/v1", default_headers=dict(_ATTRIBUTION_HEADERS),
    default_aux_model="glm-5",
    # The Go relay's upstream validates tool content as a strict string: list-type tool
    # content (native vision embeds) 422s with ``messages.N.tool.content.str Input should
    # be a valid string`` (Console Go, #104731) or 400s ``text is not set`` (MiMo, #47026),
    # and the rejected row stays in history so every later call dies too. Images in user
    # messages are fine, so vision itself keeps working via the text-summary downgrade.
    supports_vision_tool_messages=False,
)

register_provider(opencode_zen)
register_provider(opencode_go)
