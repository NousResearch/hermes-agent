"""OpenCode provider profiles (Zen + Go).

Both route api_mode per model in core; these profiles carry the
chat_completions reasoning translations (GLM-5.2, Kimi K2, DeepSeek, Ox Alpha).
"""

from typing import Any
from urllib.parse import urlparse

from agent import reasoning_effort as re_
from hermes_cli.version_info import get_version_info
from providers import register_provider
from providers.base import ProviderProfile

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


def _keyless_model_id(model: str | None, prefixes: tuple = ()) -> str:
    """Bare model id for the keyless check, or "" when the spelling is not one we authorize.

    Accepts only the two forms that reach the relay as the bare keyless slug: the id itself,
    and the id behind a prefix that :func:`normalize_opencode_model_id` strips before the wire
    call. That normalizer strips ``<provider_id>/`` and ``<family>/``, which for this profile is
    ``opencode-zen/`` and ``opencode/`` — NOT every alias (``zen/`` and ``opencode_zen/`` survive
    it), so the stripped prefixes are spelled out rather than taken from ``aliases``. Anything
    else is refused, because the id sent to the relay would then differ from the id authorized
    here.
    """
    current = (model or "").strip()
    if not current:
        return ""
    head, sep, tail = current.rpartition("/")
    if not sep:
        return current.lower()
    # One prefix segment only, and it must be one the wire normalizer actually strips.
    if "/" in head:
        return ""
    return tail.lower() if head.lower() in {p.lower() for p in prefixes} else ""


# Version-less DeepSeek ids that still carry the thinking/effort knobs on this wire: the retired
# ``deepseek-reasoner`` alias and the canonical ``deepseek-flash`` (2026-09 Flash refresh), for
# which the Go relay honours the same top-level ``reasoning_effort``/``thinking`` contract.
_THINKING_CAPABLE_IDS: frozenset[str] = frozenset({"deepseek-reasoner", "deepseek-flash"})


def _is_deepseek_thinking_model(model: str | None) -> bool:
    m = _flat_model_name(model)
    return (m.startswith("deepseek-v") and not m.startswith("deepseek-v3")) or m in _THINKING_CAPABLE_IDS


def _is_glm_5_2_model(model: str | None) -> bool:
    """GLM-5.2 across alias spellings (glm-5.2 / glm-5-2 / glm-5p2)."""
    m = _flat_model_name(model)
    return any(token in m for token in ("glm-5.2", "glm-5-2", "glm-5p2"))


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
                                    fetched_at=datetime.now(timezone.utc), windows=tuple(windows))

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if _is_glm_5_2_model(model):
            # Native reasoning_effort knob (high/max); server default when unset/disabled.
            effort = re_.requested_effort(reasoning_config)
            if effort is None or effort == "none":
                return {}, {}
            clamped = re_.clamp_effort(effort, re_.GLM52_EFFORTS, re_.GLM52_OVERRIDES)
            return {}, {"reasoning_effort": clamped if clamped in re_.GLM52_EFFORTS else "high"}
        if _flat_model_name(model).startswith("kimi-k2"):
            if not isinstance(reasoning_config, dict):
                return {}, {}
            return re_.thinking_toggle_extras(reasoning_config, re_.KIMI_K2_EFFORTS)
        if _is_deepseek_thinking_model(model):
            return re_.thinking_toggle_extras(reasoning_config, re_.DEEPSEEK_V4_EFFORTS, re_.DEEPSEEK_V4_OVERRIDES)
        return {}, {}


class OpenCodeZenProfile(ProviderProfile):
    """OpenCode Zen - model-specific reasoning and anonymous-access controls."""

    # Only the prefixes normalize_opencode_model_id strips for this family; see _keyless_model_id.
    _KEYLESS_PREFIXES = ("opencode-zen", "opencode")

    def supports_anonymous_access(self, *, model: str | None, base_url: str | None = None) -> bool:
        """Allow only the documented free chat model on Zen's official HTTPS endpoint.

        A public ``/models`` response or a ``-free`` suffix is not proof that a model
        accepts anonymous inference. Keep this exact and endpoint-scoped so a proxy or
        future paid model cannot inherit keyless access accidentally.

        ``base_url`` is the endpoint the caller will actually use, so an absent one is
        refused rather than defaulted to :attr:`base_url`: substituting our own endpoint
        would decide the grant against a URL the caller never validated. Fail-closed like
        every other condition here.
        """
        # Only the prefixes normalize_opencode_model_id strips for this family; see _keyless_model_id.
        if _keyless_model_id(model, self._KEYLESS_PREFIXES) not in self.keyless_model_ids:
            return False
        candidate = str(base_url or "").strip()
        if not candidate:
            return False
        try:
            parsed = urlparse(candidate)
            return bool(
                parsed.scheme == "https"
                and parsed.hostname == "opencode.ai"
                and parsed.port in (None, 443)
                and not parsed.username
                and not parsed.password
                and parsed.path.rstrip("/") == "/zen/v1"
                and not parsed.params
                and not parsed.query
                and not parsed.fragment
            )
        except ValueError:
            return False

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return re_.ox_alpha_reasoning_extras(reasoning_config, model)


opencode_zen = OpenCodeZenProfile(
    name="opencode-zen", aliases=("opencode", "opencode_zen", "zen"), env_vars=("OPENCODE_ZEN_API_KEY",),
    keyless_model_ids=frozenset({"space-bunny-free"}),
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
