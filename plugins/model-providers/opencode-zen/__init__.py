"""OpenCode provider profiles (Zen + Go).

Both route api_mode per model in core; these profiles carry the
chat_completions reasoning translations (GLM-5.2, Kimi K2, DeepSeek, Ox Alpha).
"""

from typing import Any

from agent import reasoning_effort as re_
from providers import register_provider
from providers.base import ProviderProfile
from datetime import UTC

# Attribution headers matching the OpenCode CLI's wire format (Issue #49433).
# User-Agent must be opencode/latest/<version>/cli (NOT opencode/<version>/cli).
# x-opencode-project: global is now validated by the relay.
_ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-Title": "Hermes Agent",
    "User-Agent": "opencode/latest/2.0.29/cli",
    "x-opencode-client": "cli",
    "x-opencode-project": "global",
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


def _is_glm_5_2_model(model: str | None) -> bool:
    """GLM-5.2 across alias spellings (glm-5.2 / glm-5-2 / glm-5p2)."""
    m = _flat_model_name(model)
    return any(token in m for token in ("glm-5.2", "glm-5-2", "glm-5p2"))


# --- OpenCode free-tier relay contract (Issue #49433) ---
# The Zen/Go relay's authenticate() (zen.go) checks: User-Agent format
# ``opencode/<channel>/<version>/cli`` (satisfied by _ATTRIBUTION_HEADERS above),
# ``x-opencode-session`` in ``ses_<12hex><14base62>`` format (produced by core's
# ``merge_session_affinity_headers``, which still runs in build_api_kwargs),
# ``body.stream=true``, and ``body.tools`` containing ``bash`` and ``read``.
# Absent any of these -> FreeTierError 403. All of this lives in the profile
# overrides (OpenCodeFreeTierMixin) so it does not block ``hermes update``; core
# only invokes the generic ProviderProfile.prepare_tools / prepare_request_extras
# hooks from build_api_kwargs.
# OpenAI SDK telemetry headers that identify the Python SDK to the relay.
# Stripped via Omit() so the request appears to come from the official OpenCode CLI.
_STAINLESS_HEADERS = (
    "X-Stainless-Lang", "X-Stainless-Package-Version", "X-Stainless-OS",
    "X-Stainless-Arch", "X-Stainless-Runtime", "X-Stainless-Runtime-Version",
    "X-Stainless-Async", "x-stainless-retry-count", "x-stainless-read-timeout",
)


def _is_opencode_free_tier_model(model: str | None) -> bool:
    """True when ``model`` is a free-tier OpenCode model requiring the bypass."""
    return "-free" in (_flat_model_name(model) or "")


# Tool stubs the relay expects on free-tier requests. Named ``bash``/``read`` per
# zen.go (NOT Hermes' real ``terminal``/``read_file`` tools, which the relay
# rejects). Non-empty ``parameters`` are required (empty ``{}`` bodies 400).
_OPENCODE_FREE_TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Executes a shell command in the user's terminal. Output is sandboxed to the project directory; file writes require explicit user confirmation. Always quote arguments.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The shell command to execute (e.g. ls -la, git status)."},
            "is_background": {"type": "boolean", "description": "Run in background and return output later."},
            "timeout_in_seconds": {"type": "number", "description": "Maximum execution time in seconds."},
            "mode": {"type": "string", "enum": ["foreground", "background"]},
        }, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read",
        "description": "Reads a file from the local filesystem. Supports partial reads with offset and limit.",
        "parameters": {"type": "object", "properties": {
            "filePath": {"type": "string", "description": "The path to the file to read."},
            "offset": {"type": "number", "description": "Line number to start reading from (0-based)."},
            "limit": {"type": "number", "description": "Number of lines to read."},
        }, "required": ["filePath"]}}},
]


class OpenCodeFreeTierMixin(ProviderProfile):
    """Free-tier bypass for OpenCode relays, kept in the plugin (not core).

    ``prepare_tools`` injects the ``bash``/``read`` tool stubs the relay's
    authenticate() requires (Hermes' real tools are ``terminal``/``read_file``);
    ``prepare_request_extras`` sets ``stream=true`` + ``Bearer public`` and strips
    ``X-Stainless-*`` telemetry so the request looks like the official CLI. Only
    free-tier (``*-free``) models are shaped; paid models are pass-through.
    """

    def _is_free_tier(self, model: str | None) -> bool:
        return _is_opencode_free_tier_model(model)

    def prepare_tools(
        self,
        tools: list,
        *,
        model: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        **_context: Any,
    ) -> list:
        """Append relay-expected ``bash``/``read`` stubs for free-tier models."""
        if not self._is_free_tier(model):
            return tools
        existing = list(tools or [])
        names = {t.get("function", {}).get("name") for t in existing if isinstance(t, dict)}
        for stub in _OPENCODE_FREE_TOOLS:
            if stub["function"]["name"] not in names:
                existing.append(stub)
        return existing

    def prepare_request_extras(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        api_key: str | None = None,
        **_context: Any,
    ) -> dict[str, Any]:
        """``stream=true`` + ``X-Stainless-*`` omission for free-tier; ``Bearer public`` ONLY when no key.

        Mirrors the OpenCode CLI: fall back to the anonymous ``Bearer public``
        credential only when no real ``api_key`` is configured (free-tier quota),
        and leave ``Authorization`` untouched when a key is present so the real
        key is used (Issue #49433). Using ``Bearer public`` unconditionally is
        faulty (it masks a configured paid-tier key).
        """
        if not self._is_free_tier(model):
            return {}
        try:
            from openai._types import Omit as _Omit
        except ImportError:
            _Omit = None  # sandbox without openai: Omit() values become None (dropped by SDK)
        headers: dict[str, Any] = {}
        if not api_key:
            # No real key configured -> free-tier anonymous access via the CLI's
            # "Bearer public" credential.
            headers["Authorization"] = "Bearer public"
        if _Omit is not None:
            for _h in _STAINLESS_HEADERS:
                headers.setdefault(_h, _Omit())
        return {
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_headers": headers,
        }


class OpenCodeGoProfile(OpenCodeFreeTierMixin, ProviderProfile):
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


class OpenCodeZenProfile(OpenCodeFreeTierMixin, ProviderProfile):
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
