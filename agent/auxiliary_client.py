"""Shared auxiliary client router for side tasks.

Provides a single resolution chain so every consumer (context compression,
session search, web extraction, vision analysis, browser vision) picks up
the best available backend without duplicating fallback logic.

Resolution order for text tasks (auto mode):
  1. User's main provider + main model (used regardless of provider type —
     aggregators, direct API-key providers, native Anthropic, Codex, etc.)
  2. OpenRouter  (OPENROUTER_API_KEY)
  3. Nous Portal (~/.hermes/auth.json active provider)
  4. Custom endpoint (config.yaml model.base_url + OPENAI_API_KEY)
  5. Native Anthropic
  6. Direct API-key providers (z.ai/GLM, Kimi/Moonshot, MiniMax, MiniMax-CN)
  7. None

Resolution order for vision/multimodal tasks (auto mode):
  1. Selected main provider, if it is one of the supported vision backends below
  2. OpenRouter
  3. Nous Portal
  4. Native Anthropic
  5. Custom endpoint (for local vision models: Qwen-VL, LLaVA, Pixtral, etc.)
  6. None

Codex OAuth (ChatGPT-account auth) is intentionally NOT in either
fallback chain: OpenAI gates this endpoint behind an undocumented,
shifting model allow-list, so "just try Codex with a hardcoded model"
rots on its own.  Codex is used only when the user's main provider *is*
openai-codex (Step 1 above) or when a caller explicitly requests it with
a model (auxiliary.<task>.provider + auxiliary.<task>.model).

Per-task overrides are configured in config.yaml under the ``auxiliary:`` section
(e.g. ``auxiliary.vision.provider``, ``auxiliary.compression.model``).
Default "auto" follows the chains above.

Payment / credit exhaustion fallback:
  When a resolved provider returns HTTP 402 or a credit-related error,
  call_llm() automatically retries with the next available provider in the
  auto-detection chain.  This handles the common case where a user depletes
  their OpenRouter balance but has Codex OAuth or another provider available.
"""

import contextlib
import json
import logging
import os
import threading
import time
from pathlib import Path  # noqa: F401 — used by test mocks
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse, parse_qs, urlunparse

# NOTE: `from openai import OpenAI` is deliberately NOT at module top — the
# openai SDK pulls a large type tree (~240 ms cold, including responses/*,
# graders/*). We expose `OpenAI` here as a thin proxy that imports the SDK on
# first call and forwards, so:
#   (a) the 15+ in-module `OpenAI(...)` construction sites work unchanged
#       (Python's function-scope name lookup resolves `OpenAI` to the proxy
#       object bound in module globals here, without triggering any import);
#   (b) external code can still do `auxiliary_client.OpenAI` or
#       `patch("agent.auxiliary_client.OpenAI", ...)` — tests see the proxy,
#       and patch replaces the module attribute as usual;
#   (c) `OpenAI` as a type annotation resolves at runtime to the proxy class
#       (which is harmless — annotations aren't type-checked at runtime).
# See tests/agent/test_auxiliary_client.py for patch patterns this supports.
if TYPE_CHECKING:
    from openai import OpenAI  # noqa: F401 — type hints only

_OPENAI_CLS_CACHE: Optional[type] = None


def _load_openai_cls() -> type:
    """Import and cache ``openai.OpenAI``."""
    global _OPENAI_CLS_CACHE
    if _OPENAI_CLS_CACHE is None:
        from openai import OpenAI as _cls
        _OPENAI_CLS_CACHE = _cls
    return _OPENAI_CLS_CACHE


class _OpenAIProxy:
    """Module-level proxy that looks like the ``openai.OpenAI`` class.

    Forwards ``OpenAI(...)`` calls and ``isinstance(x, OpenAI)`` checks to the
    real SDK class, importing the SDK lazily on first use.
    """

    __slots__ = ()

    def __call__(self, *args, **kwargs):
        return _load_openai_cls()(*args, **kwargs)

    def __instancecheck__(self, obj):
        return isinstance(obj, _load_openai_cls())

    def __repr__(self):
        return "<lazy openai.OpenAI proxy>"


OpenAI = _OpenAIProxy()  # module-level name, resolves lazily on call/isinstance

from agent.credential_pool import load_pool
from hermes_cli.config import get_hermes_home
from hermes_constants import OPENROUTER_BASE_URL
from utils import base_url_host_matches, base_url_hostname, env_float, model_forces_max_completion_tokens, normalize_proxy_env_vars

logger = logging.getLogger(__name__)


# ── Interrupt protection for atomic auxiliary tasks ──────────────────────
# Some auxiliary tasks must NOT be aborted mid-flight by a gateway interrupt
# (e.g. an incoming user message while the agent is busy). Context
# compression is the prime case: if the summary LLM call is interrupted
# part-way, compression falls back to a static "summary unavailable" marker
# and the real handoff is lost (#23975). A thread-local flag lets such a
# task mark its in-flight LLM call as interrupt-protected; the Codex
# Responses stream's cancellation check honors it. TIMEOUTS still fire
# (a hung call must die), and all OTHER aux tasks (vision, web_extract,
# title_generation, …) remain freely interruptible.
_aux_interrupt_protection = threading.local()


def _aux_interrupt_protected() -> bool:
    return bool(getattr(_aux_interrupt_protection, "active", False))


@contextlib.contextmanager
def aux_interrupt_protection(active: bool = True):
    """Mark the current thread's auxiliary LLM call as interrupt-protected.

    Used by atomic aux tasks (compression) so a mid-flight gateway interrupt
    doesn't abort the call and trigger a degraded fallback. Re-entrant-safe:
    restores the previous value on exit.
    """
    prev = getattr(_aux_interrupt_protection, "active", False)
    _aux_interrupt_protection.active = active
    try:
        yield
    finally:
        _aux_interrupt_protection.active = prev


def _safe_isinstance(obj: Any, maybe_type: Any) -> bool:
    """Return False instead of raising when a patched symbol is not a type."""
    try:
        return isinstance(obj, maybe_type)
    except TypeError:
        return False


def _extract_url_query_params(url: str):
    """Extract query params from URL, return (clean_url, default_query dict or None)."""
    parsed = urlparse(url)
    if parsed.query:
        clean = urlunparse(parsed._replace(query=""))
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        return clean, params
    return url, None


# Module-level flag: only warn once per process about stale OPENAI_BASE_URL.
_stale_base_url_warned = False

_PROVIDER_ALIASES = {
    "google": "gemini",
    "google-gemini": "gemini",
    "google-ai-studio": "gemini",
    "x-ai": "xai",
    "x.ai": "xai",
    "grok": "xai",
    "glm": "zai",
    "z-ai": "zai",
    "z.ai": "zai",
    "zhipu": "zai",
    "kimi": "kimi-coding",
    "moonshot": "kimi-coding",
    "kimi-cn": "kimi-coding-cn",
    "moonshot-cn": "kimi-coding-cn",
    "gmi-cloud": "gmi",
    "gmicloud": "gmi",
    "minimax-china": "minimax-cn",
    "minimax_cn": "minimax-cn",
    "claude": "anthropic",
    "claude-code": "anthropic",
    "github": "copilot",
    "github-copilot": "copilot",
    "github-model": "copilot",
    "github-models": "copilot",
    "github-copilot-acp": "copilot-acp",
    "copilot-acp-agent": "copilot-acp",
    "tencent": "tencent-tokenhub",
    "tokenhub": "tencent-tokenhub",
    "tencent-cloud": "tencent-tokenhub",
    "tencentmaas": "tencent-tokenhub",
}


def _normalize_aux_provider(provider: Optional[str]) -> str:
    normalized = (provider or "auto").strip().lower()
    if normalized.startswith("custom:"):
        suffix = normalized.split(":", 1)[1].strip()
        if not suffix:
            return "custom"
        normalized = suffix
    if normalized == "codex":
        return "openai-codex"
    if normalized == "main":
        # Resolve to the user's actual main provider so named custom providers
        # and non-aggregator providers (DeepSeek, Alibaba, etc.) work correctly.
        main_prov = (_read_main_provider() or "").strip().lower()
        if main_prov and main_prov not in {"auto", "main", ""}:
            normalized = main_prov
        else:
            return "custom"
    return _PROVIDER_ALIASES.get(normalized, normalized)


# Sentinel: when returned by _fixed_temperature_for_model(), callers must
# strip the ``temperature`` key from API kwargs entirely so the provider's
# server-side default applies.  Kimi/Moonshot models manage temperature
# internally — sending *any* value (even the "correct" one) can conflict
# with gateway-side mode selection (thinking → 1.0, non-thinking → 0.6).
OMIT_TEMPERATURE: object = object()


def _is_kimi_model(model: Optional[str]) -> bool:
    """True for any Kimi / Moonshot model that manages temperature server-side."""
    bare = (model or "").strip().lower().rsplit("/", 1)[-1]
    return bare.startswith("kimi-") or bare == "kimi"


def _is_arcee_trinity_thinking(model: Optional[str]) -> bool:
    """True for Arcee Trinity Large Thinking (direct or via OpenRouter)."""
    bare = (model or "").strip().lower().rsplit("/", 1)[-1]
    return bare == "trinity-large-thinking"


# Context window enforced by ChatGPT's Codex OAuth backend for gpt-5.5.
# The raw OpenAI API and OpenRouter expose 1.05M for the same slug, but the
# Codex backend hard-caps at 272K (verified live: a ~330K-token request to
# chatgpt.com/backend-api/codex/responses is rejected with
# ``context_length_exceeded`` while ~250K succeeds). With a 272K ceiling the
# default 50% compaction trigger fires at ~136K — wasteful, since the model
# can hold far more raw context before summarization actually buys anything.
# We raise the trigger to 85% (~231K) on this exact route so Codex gpt-5.5
# sessions use the window they actually have.
_CODEX_GPT55_COMPACTION_THRESHOLD = 0.85


def _is_codex_gpt55(model: Optional[str], provider: Optional[str] = None) -> bool:
    """True for gpt-5.5 accessed through the ChatGPT Codex OAuth backend.

    Matches only the Codex OAuth route (provider ``openai-codex``), not the
    direct OpenAI API, OpenRouter, or GitHub Copilot paths — those expose a
    larger context window for the same slug and must keep the user's default
    compaction threshold. ``gpt-5.5-pro`` and dated snapshots
    (``gpt-5.5-2026-04-23``) are matched via prefix so the override tracks the
    family without re-listing every variant.
    """
    prov = (provider or "").strip().lower()
    if prov != "openai-codex":
        return False
    bare = (model or "").strip().lower().rsplit("/", 1)[-1]
    return bare == "gpt-5.5" or bare.startswith("gpt-5.5-") or bare.startswith("gpt-5.5.")


def _fixed_temperature_for_model(
    model: Optional[str],
    base_url: Optional[str] = None,
) -> "Optional[float] | object":
    """Return a temperature directive for models with strict contracts.

    Returns:
        ``OMIT_TEMPERATURE`` — caller must remove the ``temperature`` key so the
            provider chooses its own default.  Used for all Kimi / Moonshot
            models whose gateway selects temperature server-side.
        ``float`` — a specific value the caller must use (reserved for future
            models with fixed-temperature contracts).
        ``None`` — no override; caller should use its own default.
    """
    if _is_kimi_model(model):
        logger.debug("Omitting temperature for Kimi model %r (server-managed)", model)
        return OMIT_TEMPERATURE
    if _is_arcee_trinity_thinking(model):
        return 0.5
    return None


def _compression_threshold_for_model(
    model: Optional[str],
    provider: Optional[str] = None,
    *,
    allow_codex_gpt55_autoraise: bool = True,
) -> Optional[float]:
    """Return a context-compression threshold override for specific models.

    The threshold is the fraction of the model's context window that must be
    consumed before Hermes triggers summarization.  Higher values delay
    compression and preserve more raw context.

    Per-model/route overrides:
      - Arcee Trinity Large Thinking → 0.75 (preserve reasoning context).
      - gpt-5.5 on the Codex OAuth route → 0.85, because Codex caps the window
        at 272K and the default 50% trigger would compact at ~136K. Gated by
        ``allow_codex_gpt55_autoraise`` so the user can opt back down to the
        global default (the caller passes the config flag through here).

    Returns a float in (0, 1] to override the global ``compression.threshold``
    config value, or ``None`` to leave the user's config value unchanged.
    """
    if _is_arcee_trinity_thinking(model):
        return 0.75
    if allow_codex_gpt55_autoraise and _is_codex_gpt55(model, provider):
        return _CODEX_GPT55_COMPACTION_THRESHOLD
    return None

# Default auxiliary models for direct API-key providers (cheap/fast for side tasks)
def _get_aux_model_for_provider(provider_id: str) -> str:
    """Return the cheap auxiliary model for a provider.

    Reads from ProviderProfile.default_aux_model first, falling back to the
    legacy hardcoded dict for providers that predate the profiles system.
    """
    try:
        from providers import get_provider_profile
        _p = get_provider_profile(provider_id)
        if _p and _p.default_aux_model:
            return _p.default_aux_model
    except Exception:
        pass
    return _API_KEY_PROVIDER_AUX_MODELS_FALLBACK.get(provider_id, "")


# Fallback for providers not yet migrated to ProviderProfile.default_aux_model,
# plus providers we intentionally keep pinned here (e.g. Anthropic predates
# profiles). New providers should set default_aux_model on their profile instead.
_API_KEY_PROVIDER_AUX_MODELS_FALLBACK: Dict[str, str] = {
    "gemini": "gemini-3-flash-preview",
    "zai": "glm-4.5-flash",
    "kimi-coding": "kimi-k2-turbo-preview",
    "stepfun": "step-3.5-flash",
    "kimi-coding-cn": "kimi-k2-turbo-preview",
    "gmi": "google/gemini-3.1-flash-lite-preview",
    "anthropic": "claude-haiku-4-5-20251001",
    "opencode-zen": "gemini-3-flash",
    "opencode-go": "glm-5",
    "kilocode": "google/gemini-3-flash-preview",
    "ollama-cloud": "nemotron-3-nano:30b",
    "tencent-tokenhub": "hy3-preview",
}

# Legacy alias — callers that haven't been updated to _get_aux_model_for_provider()
# can still use this dict directly. Kept in sync with _FALLBACK above.
_API_KEY_PROVIDER_AUX_MODELS: Dict[str, str] = _API_KEY_PROVIDER_AUX_MODELS_FALLBACK

# Vision-specific model overrides for direct providers.
# When the user's main provider has a dedicated vision/multimodal model that
# differs from their main chat model, map it here.  The vision auto-detect
# "exotic provider" branch checks this before falling back to the main model.
_PROVIDER_VISION_MODELS: Dict[str, str] = {
    "xiaomi": "mimo-v2.5",
    "zai": "glm-5v-turbo",
}

# Providers whose endpoint does not accept image input, even though the
# provider's broader ecosystem has vision models available elsewhere.  When
# `auxiliary.vision.provider: auto` sees one of these as the main provider,
# it must skip straight to the aggregator chain instead of returning a client
# that will 404 on every vision request.
#
# kimi-coding / kimi-coding-cn: the Kimi Coding Plan routes through
# api.kimi.com/coding (Anthropic Messages wire) which Kimi's own docs
# describe as having no image_in capability. Vision lives on the separate
# Kimi Platform (api.moonshot.ai, OpenAI-wire, pay-as-you-go).  See #17076.
#
# deepseek: DeepSeek's chat API (api.deepseek.com/v1) returns HTTP 401 on
# vision requests for all current models (deepseek-chat, deepseek-reasoner).
# The provider has no vision-capable endpoint, so auto-detection must skip it
# and fall through to the aggregator chain.  See #50712.
_PROVIDERS_WITHOUT_VISION: frozenset = frozenset({
    "kimi-coding",
    "kimi-coding-cn",
    "deepseek",
})

# OpenRouter app attribution headers (base — always sent).
# `X-Title` is the canonical attribution header OpenRouter's dashboard
# reads; the previous `X-OpenRouter-Title` label was not recognized there.
_OR_HEADERS_BASE = {
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-Title": "Hermes Agent",
    "X-OpenRouter-Categories": "productivity,cli-agent",
}

# Truthy values for boolean env-var parsing.
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def _apply_user_default_headers(headers: dict | None) -> dict | None:
    """Merge user-configured ``model.default_headers`` onto resolved headers.

    User values take precedence over provider/SDK defaults, mirroring the main
    agent client (``AIAgent._apply_user_default_headers``). This lets a
    ``custom`` OpenAI-compatible endpoint behind a gateway/WAF that rejects the
    OpenAI SDK's identifying headers (``User-Agent: OpenAI/Python ...``,
    ``X-Stainless-*``) override them for auxiliary calls too — otherwise the
    main turn would succeed but title/compression/vision calls to the same
    endpoint would still fail. (#40033)

    Returns the merged dict, or the original ``headers`` (possibly ``None``)
    when nothing is configured. No allocation when there are no overrides.
    """
    try:
        from hermes_cli.config import cfg_get, load_config
        user_headers = cfg_get(load_config(), "model", "default_headers")
    except Exception:
        return headers
    if not isinstance(user_headers, dict) or not user_headers:
        return headers
    merged = dict(headers or {})
    for key, value in user_headers.items():
        if value is None:
            continue
        merged[str(key)] = str(value)
    return merged or headers


def build_or_headers(or_config: dict | None = None) -> dict:
    """Build OpenRouter headers, optionally including response-cache headers.

    Precedence for response cache: env var > config.yaml > default (enabled).

    Environment variables:
        ``HERMES_OPENROUTER_CACHE`` — truthy (``1``/``true``/``yes``/``on``)
            enables caching; ``0``/``false``/``no``/``off`` disables.
            Overrides ``openrouter.response_cache`` in config.yaml.
        ``HERMES_OPENROUTER_CACHE_TTL`` — integer seconds (1-86400).
            Overrides ``openrouter.response_cache_ttl`` in config.yaml.

    *or_config* is the ``openrouter`` section from config.yaml.  When *None*,
    falls back to reading config from disk via ``load_config()``.
    """
    headers = dict(_OR_HEADERS_BASE)

    # Resolve config from disk if not provided.
    if or_config is None:
        try:
            from hermes_cli.config import load_config
            or_config = load_config().get("openrouter", {})
        except Exception:
            or_config = {}

    # Determine cache enabled: env var overrides config.
    env_cache = os.environ.get("HERMES_OPENROUTER_CACHE", "").strip().lower()
    if env_cache:
        cache_enabled = env_cache in _TRUTHY_ENV_VALUES
    else:
        cache_enabled = or_config.get("response_cache", False)

    if not cache_enabled:
        return headers

    headers["X-OpenRouter-Cache"] = "true"

    # Determine TTL: env var overrides config.
    env_ttl = os.environ.get("HERMES_OPENROUTER_CACHE_TTL", "").strip()
    if env_ttl:
        if env_ttl.isdigit():
            ttl = int(env_ttl)
            if 1 <= ttl <= 86400:
                headers["X-OpenRouter-Cache-TTL"] = str(ttl)
    else:
        ttl = or_config.get("response_cache_ttl", 300)
        if isinstance(ttl, (int, float)) and 1 <= ttl <= 86400:
            headers["X-OpenRouter-Cache-TTL"] = str(int(ttl))

    return headers


# NVIDIA NIM cloud billing attribution.  Keep this host-gated because the
# nvidia provider also supports local/on-prem NIM endpoints via NVIDIA_BASE_URL.
_NVIDIA_NIM_CLOUD_HEADERS = {
    "X-BILLING-INVOKE-ORIGIN": "HermesAgent",
}


def build_nvidia_nim_headers(base_url: str | None) -> dict:
    """Return NVIDIA NIM cloud attribution headers for build.nvidia.com traffic."""
    if base_url_host_matches(str(base_url or ""), "integrate.api.nvidia.com"):
        return dict(_NVIDIA_NIM_CLOUD_HEADERS)
    return {}



# Nous Portal extra_body for product attribution.
# Callers should pass this as extra_body in chat.completions.create()
# when the auxiliary client is backed by Nous Portal.
#
# The tags are computed from agent.portal_tags so the client= marker stays
# in lockstep with hermes_cli.__version__ across every Portal call site
# (main loop, aux, compression, web_extract). Do not inline a literal here;
# see agent/portal_tags.py for the rationale.
from agent.portal_tags import nous_portal_tags as _nous_portal_tags


def _nous_extra_body() -> dict:
    """Return a fresh Nous Portal ``extra_body`` dict.

    Computed at call time so a hot-reloaded ``hermes_cli.__version__`` is
    reflected without restarting long-running processes.
    """
    return {"tags": _nous_portal_tags()}


# Backwards-compatible module attribute. Some callers (tests, third-party
# plugins) read ``NOUS_EXTRA_BODY`` directly; keep it as a snapshot of the
# current tags. Callers that need the freshest value should call
# ``_nous_extra_body()`` or import ``nous_portal_tags`` directly.
NOUS_EXTRA_BODY = _nous_extra_body()

# Set at resolve time — True if the auxiliary client points to Nous Portal
auxiliary_is_nous: bool = False

# Default auxiliary models per provider
_OPENROUTER_MODEL = "google/gemini-3-flash-preview"
_NOUS_MODEL = "google/gemini-3-flash-preview"
_NOUS_DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"
_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
_AUTH_JSON_PATH = get_hermes_home() / "auth.json"

# Codex OAuth endpoint used when a caller explicitly requests
# provider="openai-codex".  There is deliberately no hardcoded default
# model: the set of models OpenAI accepts on this endpoint for
# ChatGPT-account auth is an undocumented, shifting allow-list, and
# pinning one here has drifted silently twice (gpt-5.3-codex → gpt-5.2-codex
# → gpt-5.4 over 6 weeks in early 2026).  Callers must pass the model
# they want explicitly (from config.yaml model.model, auxiliary.<task>.model,
# or the user's active Codex model selection).
_CODEX_AUX_BASE_URL = "https://chatgpt.com/backend-api/codex"


def _codex_cloudflare_headers(access_token: str) -> Dict[str, str]:
    """Headers required to avoid Cloudflare 403s on chatgpt.com/backend-api/codex.

    The Cloudflare layer in front of the Codex endpoint whitelists a small set of
    first-party originators (``codex_cli_rs``, ``codex_vscode``, ``codex_sdk_ts``,
    anything starting with ``Codex``). Requests from non-residential IPs (VPS,
    server-hosted agents) that don't advertise an allowed originator are served
    a 403 with ``cf-mitigated: challenge`` regardless of auth correctness.

    We pin ``originator: codex_cli_rs`` to match the upstream codex-rs CLI, set
    ``User-Agent`` to a codex_cli_rs-shaped string (beats SDK fingerprinting),
    and extract ``ChatGPT-Account-ID`` (canonical casing, from codex-rs
    ``auth.rs``) out of the OAuth JWT's ``chatgpt_account_id`` claim.

    Malformed tokens are tolerated — we drop the account-ID header rather than
    raise, so a bad token still surfaces as an auth error (401) instead of a
    crash at client construction.
    """
    headers = {
        "User-Agent": "codex_cli_rs/0.0.0 (Hermes Agent)",
        "originator": "codex_cli_rs",
    }
    if not isinstance(access_token, str) or not access_token.strip():
        return headers
    try:
        import base64
        parts = access_token.split(".")
        if len(parts) < 2:
            return headers
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        acct_id = claims.get("https://api.openai.com/auth", {}).get("chatgpt_account_id")
        if isinstance(acct_id, str) and acct_id:
            headers["ChatGPT-Account-ID"] = acct_id
    except Exception:
        pass
    return headers


def _to_openai_base_url(base_url: str) -> str:
    """Normalize an Anthropic-style base URL to OpenAI-compatible format.

    Some providers (MiniMax, MiniMax-CN) expose an ``/anthropic`` endpoint for
    the Anthropic Messages API and a separate ``/v1`` endpoint for OpenAI chat
    completions.  The auxiliary client uses the OpenAI SDK, so it must hit the
    ``/v1`` surface.  Passing the raw ``inference_base_url`` causes requests to
    land on ``/anthropic/chat/completions`` — a 404.
    """
    url = str(base_url or "").strip().rstrip("/")
    if url.endswith("/anthropic"):
        # ZAI (open.bigmodel.cn) uses /api/anthropic for Anthropic wire
        # but /api/paas/v4 for OpenAI wire — the generic /v1 rewrite is wrong.
        if "open.bigmodel.cn" in url or "bigmodel" in url:
            rewritten = url[: -len("/anthropic")] + "/paas/v4"
            logger.debug("Auxiliary client: rewrote ZAI base URL %s → %s", url, rewritten)
            return rewritten
        rewritten = url[: -len("/anthropic")] + "/v1"
        logger.debug("Auxiliary client: rewrote base URL %s → %s", url, rewritten)
        return rewritten
    if "api.kimi.com" in url and url.endswith("/coding"):
        # Kimi Code uses /coding/v1/messages for Anthropic SDK (appends /v1/messages)
        # but /coding/v1/chat/completions for OpenAI SDK (appends /chat/completions)
        # Without /v1 here, OpenAI SDK hits /coding/chat/completions — a 404.
        rewritten = url + "/v1"
        logger.debug("Auxiliary client: rewrote Kimi base URL %s → %s", url, rewritten)
        return rewritten
    return url


def _select_pool_entry(provider: str) -> Tuple[bool, Optional[Any]]:
    """Return (pool_exists_for_provider, selected_entry)."""
    try:
        pool = load_pool(provider)
    except Exception as exc:
        logger.debug("Auxiliary client: could not load pool for %s: %s", provider, exc)
        return False, None
    if not pool or not pool.has_credentials():
        return False, None
    try:
        return True, pool.select()
    except Exception as exc:
        logger.debug("Auxiliary client: could not select pool entry for %s: %s", provider, exc)
        return True, None


def _peek_pool_entry(provider: str) -> Optional[Any]:
    """Best-effort current/next pool entry without mutating selection order."""
    try:
        pool = load_pool(provider)
    except Exception as exc:
        logger.debug("Auxiliary client: could not load pool for %s (peek): %s", provider, exc)
        return None
    if not pool or not pool.has_credentials():
        return None
    try:
        current_fn = getattr(pool, "current", None)
        if callable(current_fn):
            current = current_fn()
            if current is not None:
                return current
        peek_fn = getattr(pool, "peek", None)
        if callable(peek_fn):
            return peek_fn()
    except Exception as exc:
        logger.debug("Auxiliary client: could not peek pool entry for %s: %s", provider, exc)
    return None


def _pool_runtime_api_key(entry: Any) -> str:
    if entry is None:
        return ""
    # Use the PooledCredential.runtime_api_key property which handles
    # provider-specific fallback (e.g. agent_key for nous).
    key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    return str(key or "").strip()


def _pool_runtime_base_url(entry: Any, fallback: str = "") -> str:
    if entry is None:
        return str(fallback or "").strip().rstrip("/")
    # runtime_base_url handles provider-specific logic (e.g. nous prefers inference_base_url).
    # Fall back through inference_base_url and base_url for non-PooledCredential entries.
    url = (
        getattr(entry, "runtime_base_url", None)
        or getattr(entry, "inference_base_url", None)
        or getattr(entry, "base_url", None)
        or fallback
    )
    return str(url or "").strip().rstrip("/")


def _nous_min_key_ttl_seconds() -> int:
    try:
        return max(60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800")))
    except (TypeError, ValueError):
        return 1800


# ── Codex Responses → chat.completions adapter ─────────────────────────────
# All auxiliary consumers call client.chat.completions.create(**kwargs) and
# read response.choices[0].message.content. This adapter translates those
# calls to the Codex Responses API so callers don't need any changes.


class _CodexCompletionsAdapter:
    """Drop-in shim that accepts chat.completions.create() kwargs and
    routes them through the Codex Responses streaming API."""

    def __init__(self, real_client: OpenAI, model: str):
        self._client = real_client
        self._model = model

    def create(self, **kwargs) -> Any:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", self._model)

        # Separate system/instructions from replayable conversation messages,
        # then route the rest through the SINGLE shared chat->Responses
        # converter used by the main agent transport
        # (agent/transports/codex.py). Maintaining a private conversion loop
        # here let chat-style messages with role="tool" leak straight into
        # Responses input[] — which the Responses API rejects with
        # "Invalid value: 'tool'. Supported values are: 'assistant', 'system',
        # 'developer', and 'user'." (issue #5709, hit hard by flush_memories()
        # / compression replaying real session history that includes assistant
        # tool_calls + role="tool" results). The shared converter encodes
        # assistant tool calls as `function_call` items and tool results as
        # `function_call_output` items with a valid call_id, so every
        # Responses path normalizes tool history identically and cannot drift.
        from agent.codex_responses_adapter import _chat_messages_to_responses_input

        instructions = "You are a helpful assistant."
        replay_messages: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            if role == "system":
                instructions = content if isinstance(content, str) else str(content)
            else:
                replay_messages.append(msg)

        input_items = _chat_messages_to_responses_input(replay_messages)

        resp_kwargs: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": input_items or [{"role": "user", "content": ""}],
            "store": False,
        }

        # Preserve the chat.completions timeout contract. This adapter is used
        # by auxiliary calls such as context compression; if the timeout is not
        # forwarded and enforced, a Codex Responses stream can sit behind a
        # dead-looking CLI until the user force-interrupts the whole session.
        timeout = kwargs.get("timeout")
        if timeout is not None:
            resp_kwargs["timeout"] = timeout

        # Note: the Codex endpoint (chatgpt.com/backend-api/codex) does NOT
        # support max_output_tokens or temperature — omit to avoid 400 errors.

        # Translate extra_body.reasoning (chat.completions shape) into the
        # Responses API's top-level reasoning + include fields.  Mirrors
        # agent/transports/codex.py::build_kwargs() so auxiliary callers
        # that configure reasoning via auxiliary.<task>.extra_body get the
        # same behavior as the main agent's Codex transport.
        extra_body = kwargs.get("extra_body") or {}
        if isinstance(extra_body, dict):
            reasoning_cfg = extra_body.get("reasoning")
            if isinstance(reasoning_cfg, dict):
                if reasoning_cfg.get("enabled") is False:
                    # Reasoning explicitly disabled — do not set reasoning
                    # or include.  The Codex backend still thinks by
                    # default, but we honor the caller's intent where the
                    # API allows it.
                    pass
                else:
                    # Truthy-only check mirrors agent/transports/codex.py
                    # build_kwargs(): falsy values (None, "", 0) fall back
                    # to the default rather than being forwarded to the
                    # Codex backend, which rejects e.g. {"effort": null}
                    # with a 400.
                    effort = reasoning_cfg.get("effort") or "medium"
                    # Codex backend rejects "minimal"; clamp to "low" to
                    # match the main-agent Codex transport behavior.
                    if effort == "minimal":
                        effort = "low"
                    resp_kwargs["reasoning"] = {
                        "effort": effort,
                        "summary": "auto",
                    }
                    resp_kwargs["include"] = ["reasoning.encrypted_content"]

        # Tools support for auxiliary callers (e.g. skills_hub) that pass function schemas
        tools = kwargs.get("tools")
        if tools:
            # xAI's Responses endpoint rejects ``pattern`` and ``format`` JSON Schema
            # keywords (HTTP 400). Strip them here to match the parity guarantee that
            # chat_completion_helpers.py provides for the main-agent xAI path.
            #
            # Deep-copy before sanitizing — ``list(tools)`` is only a shallow
            # copy of the outer list, but the sanitizers mutate the inner
            # parameter dicts in place.  Without a deep copy the caller's
            # tool registry permanently loses its slash-containing enum
            # constraints after the first auxiliary xAI call.  See #27907.
            try:
                import copy as _copy
                from tools.schema_sanitizer import (
                    strip_pattern_and_format,
                    strip_slash_enum,
                )
                tools = _copy.deepcopy(list(tools))
                tools, _ = strip_pattern_and_format(tools)
                tools, _ = strip_slash_enum(tools)
            except Exception as exc:
                logger.warning(
                    "Auxiliary client: failed to sanitize tool schemas for "
                    "Codex/xAI Responses path: %s", exc,
                )
            converted = []
            for t in tools:
                fn = t.get("function", {}) if isinstance(t, dict) else {}
                name = fn.get("name")
                if not name:
                    continue
                converted.append({
                    "type": "function",
                    "name": name,
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
            if converted:
                resp_kwargs["tools"] = converted

        # Stream and collect the response
        text_parts: List[str] = []
        tool_calls_raw: List[Any] = []
        usage = None
        total_timeout = timeout if isinstance(timeout, (int, float)) and timeout > 0 else None
        deadline = time.monotonic() + float(total_timeout) if total_timeout else None
        timed_out = threading.Event()
        timeout_timer: Optional[threading.Timer] = None

        def _timeout_message() -> str:
            return f"Codex auxiliary Responses stream exceeded {float(total_timeout):.1f}s total timeout"

        def _close_client_on_timeout() -> None:
            timed_out.set()
            close = getattr(self._client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("Codex auxiliary: client close during timeout failed", exc_info=True)
            # The cached auxiliary client wraps this same ``self._client``
            # (or *is* a ``CodexAuxiliaryClient`` whose ``_real_client`` is
            # this instance).  After we close the httpx transport above, the
            # cache must drop that entry — otherwise the next auxiliary call
            # (compression retry, memory flush, etc.) reuses the dead client
            # and fails fast with a connection error.  See issue #23432.
            try:
                _evict_cached_client_instance(self._client)
            except Exception:
                logger.debug("Codex auxiliary: cache eviction on timeout failed", exc_info=True)

        def _check_cancelled() -> None:
            if deadline is not None and time.monotonic() >= deadline:
                if not timed_out.is_set():
                    _close_client_on_timeout()
                raise TimeoutError(_timeout_message())
            try:
                from tools.interrupt import is_interrupted
                # Honor interrupt protection for atomic aux tasks (compression):
                # a mid-flight gateway interrupt must NOT abort the summary call
                # and trigger a degraded fallback marker (#23975). Timeouts above
                # still fire; other aux tasks remain interruptible.
                if is_interrupted() and not _aux_interrupt_protected():
                    raise InterruptedError("Codex auxiliary Responses stream interrupted")
            except InterruptedError:
                raise
            except Exception:
                # Interrupt state is a best-effort UX hook; never make it a
                # new failure mode for auxiliary calls.
                pass

        try:
            if total_timeout:
                timeout_timer = threading.Timer(float(total_timeout), _close_client_on_timeout)
                timeout_timer.daemon = True
                timeout_timer.start()
            _check_cancelled()

            # Event-driven Responses streaming via the low-level
            # ``responses.create(stream=True)`` path.  The high-level
            # ``responses.stream(...)`` helper does post-hoc typed
            # reconstruction from ``response.completed.response.output``,
            # which the chatgpt.com Codex backend has been observed to
            # return as ``null`` (gpt-5.5, May 2026) — that crashes the SDK
            # with ``TypeError: 'NoneType' object is not iterable``.
            # Consuming raw events and assembling the final response
            # ourselves from ``response.output_item.done`` makes us
            # structurally immune to that drift.
            from agent.codex_runtime import _consume_codex_event_stream

            stream_kwargs = dict(resp_kwargs)
            stream_kwargs["stream"] = True

            def _on_each_event(_event: Any) -> None:
                # Re-check timeout/cancellation per event, matching the
                # cadence the old in-line ``_check_cancelled()`` used.
                _check_cancelled()

            event_stream = self._client.responses.create(**stream_kwargs)
            try:
                final = _consume_codex_event_stream(
                    event_stream,
                    model=resp_kwargs.get("model"),
                    on_event=_on_each_event,
                )
            finally:
                close_fn = getattr(event_stream, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass

            if final is None:
                raise RuntimeError("Codex auxiliary Responses stream did not return a final response")

            # Extract text and tool calls from the Responses output.
            # Items may be SimpleNamespace (raw-event path) or dicts
            # (some legacy fallback paths), so handle both shapes.
            def _item_get(obj: Any, key: str, default: Any = None) -> Any:
                val = getattr(obj, key, None)
                if val is None and isinstance(obj, dict):
                    val = obj.get(key, default)
                return val if val is not None else default

            for item in (getattr(final, "output", None) or []):
                item_type = _item_get(item, "type")
                if item_type == "message":
                    for part in (_item_get(item, "content") or []):
                        ptype = _item_get(part, "type")
                        if ptype in {"output_text", "text"}:
                            text_parts.append(_item_get(part, "text", ""))
                elif item_type == "function_call":
                    tool_calls_raw.append(SimpleNamespace(
                        id=_item_get(item, "call_id", ""),
                        type="function",
                        function=SimpleNamespace(
                            name=_item_get(item, "name", ""),
                            arguments=_item_get(item, "arguments", "{}"),
                        ),
                    ))

            resp_usage = getattr(final, "usage", None)
            if resp_usage:
                usage = SimpleNamespace(
                    prompt_tokens=getattr(resp_usage, "input_tokens", 0)
                        or (resp_usage.get("input_tokens", 0) if isinstance(resp_usage, dict) else 0),
                    completion_tokens=getattr(resp_usage, "output_tokens", 0)
                        or (resp_usage.get("output_tokens", 0) if isinstance(resp_usage, dict) else 0),
                    total_tokens=getattr(resp_usage, "total_tokens", 0)
                        or (resp_usage.get("total_tokens", 0) if isinstance(resp_usage, dict) else 0),
                )
        except Exception as exc:
            if timed_out.is_set():
                raise TimeoutError(_timeout_message()) from exc
            logger.debug("Codex auxiliary Responses API call failed: %s", exc)
            raise
        finally:
            if timeout_timer is not None:
                timeout_timer.cancel()

        content = "".join(text_parts).strip() or None

        # Build a response that looks like chat.completions
        message = SimpleNamespace(
            role="assistant",
            content=content,
            tool_calls=tool_calls_raw or None,
        )
        choice = SimpleNamespace(
            index=0,
            message=message,
            finish_reason="stop" if not tool_calls_raw else "tool_calls",
        )
        return SimpleNamespace(
            choices=[choice],
            model=model,
            usage=usage,
        )


class _CodexChatShim:
    """Wraps the adapter to provide client.chat.completions.create()."""

    def __init__(self, adapter: _CodexCompletionsAdapter):
        self.completions = adapter


class CodexAuxiliaryClient:
    """OpenAI-client-compatible wrapper that routes through Codex Responses API.

    Consumers can call client.chat.completions.create(**kwargs) as normal.
    Also exposes .api_key and .base_url for introspection by async wrappers.
    """

    def __init__(self, real_client: OpenAI, model: str):
        self._real_client = real_client
        adapter = _CodexCompletionsAdapter(real_client, model)
        self.chat = _CodexChatShim(adapter)
        self.api_key = real_client.api_key
        self.base_url = real_client.base_url

    def close(self):
        self._real_client.close()


class _AsyncCodexCompletionsAdapter:
    """Async version of the Codex Responses adapter.

    Wraps the sync adapter via asyncio.to_thread() so async consumers
    (web_tools, session_search) can await it as normal.
    """

    def __init__(self, sync_adapter: _CodexCompletionsAdapter):
        self._sync = sync_adapter

    async def create(self, **kwargs) -> Any:
        import asyncio
        return await asyncio.to_thread(self._sync.create, **kwargs)


class _AsyncCodexChatShim:
    def __init__(self, adapter: _AsyncCodexCompletionsAdapter):
        self.completions = adapter


class AsyncCodexAuxiliaryClient:
    """Async-compatible wrapper matching AsyncOpenAI.chat.completions.create()."""

    def __init__(self, sync_wrapper: "CodexAuxiliaryClient"):
        sync_adapter = sync_wrapper.chat.completions
        async_adapter = _AsyncCodexCompletionsAdapter(sync_adapter)
        self.chat = _AsyncCodexChatShim(async_adapter)
        self.api_key = sync_wrapper.api_key
        self.base_url = sync_wrapper.base_url
        # Mirror the sync wrapper's _real_client so cache eviction by leaf
        # OpenAI client (e.g. _close_client_on_timeout in #23482) drops
        # this async entry too. Without this, sync and async cache entries
        # diverge on poisoning: the sync entry is evicted but the async
        # entry keeps reusing the closed transport, failing every
        # subsequent async aux call with 'Connection error' until the
        # gateway restarts.
        self._real_client = sync_wrapper._real_client


class _AnthropicCompletionsAdapter:
    """OpenAI-client-compatible adapter for Anthropic Messages API."""

    def __init__(self, real_client: Any, model: str, is_oauth: bool = False):
        self._client = real_client
        self._model = model
        self._is_oauth = is_oauth

    def create(self, **kwargs) -> Any:
        from agent.anthropic_adapter import build_anthropic_kwargs, create_anthropic_message
        from agent.transports import get_transport

        messages = kwargs.get("messages", [])
        model = kwargs.get("model", self._model)
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        # ZAI's Anthropic-compatible endpoint rejects max_tokens on vision
        # models (glm-4v-flash etc.) with error code 1210.  When the caller
        # signals this by setting _skip_zai_max_tokens in kwargs, omit it.
        _skip_mt = kwargs.pop("_skip_zai_max_tokens", False)
        if _skip_mt:
            max_tokens = None
        else:
            max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 2000
        temperature = kwargs.get("temperature")

        normalized_tool_choice = None
        if isinstance(tool_choice, str):
            normalized_tool_choice = tool_choice
        elif isinstance(tool_choice, dict):
            choice_type = str(tool_choice.get("type", "")).lower()
            if choice_type == "function":
                normalized_tool_choice = tool_choice.get("function", {}).get("name")
            elif choice_type in {"auto", "required", "none"}:
                normalized_tool_choice = choice_type

        anthropic_kwargs = build_anthropic_kwargs(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            reasoning_config=None,
            tool_choice=normalized_tool_choice,
            is_oauth=self._is_oauth,
        )
        # Opus 4.7+ rejects any non-default temperature/top_p/top_k; only set
        # temperature for models that still accept it. build_anthropic_kwargs
        # additionally strips these keys as a safety net — keep both layers.
        if temperature is not None:
            from agent.anthropic_adapter import _forbids_sampling_params
            if not _forbids_sampling_params(model):
                anthropic_kwargs["temperature"] = temperature

        response = create_anthropic_message(self._client, anthropic_kwargs)
        _transport = get_transport("anthropic_messages")
        _nr = _transport.normalize_response(
            response, strip_tool_prefix=self._is_oauth
        )

        # ToolCall already duck-types as OpenAI shape (.type, .function.name,
        # .function.arguments) via properties, so no wrapping needed.
        assistant_message = SimpleNamespace(
            content=_nr.content,
            tool_calls=_nr.tool_calls,
            reasoning=_nr.reasoning,
        )
        finish_reason = _nr.finish_reason

        usage = None
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "input_tokens", 0) or 0
            completion_tokens = getattr(response.usage, "output_tokens", 0) or 0
            total_tokens = getattr(response.usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
            usage = SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        choice = SimpleNamespace(
            index=0,
            message=assistant_message,
            finish_reason=finish_reason,
        )
        return SimpleNamespace(
            choices=[choice],
            model=model,
            usage=usage,
        )


class _AnthropicChatShim:
    def __init__(self, adapter: _AnthropicCompletionsAdapter):
        self.completions = adapter


class AnthropicAuxiliaryClient:
    """OpenAI-client-compatible wrapper over a native Anthropic client."""

    def __init__(self, real_client: Any, model: str, api_key: str, base_url: str, is_oauth: bool = False):
        self._real_client = real_client
        adapter = _AnthropicCompletionsAdapter(real_client, model, is_oauth=is_oauth)
        self.chat = _AnthropicChatShim(adapter)
        self.api_key = api_key
        self.base_url = base_url

    def close(self):
        close_fn = getattr(self._real_client, "close", None)
        if callable(close_fn):
            close_fn()


class _AsyncAnthropicCompletionsAdapter:
    def __init__(self, sync_adapter: _AnthropicCompletionsAdapter):
        self._sync = sync_adapter

    async def create(self, **kwargs) -> Any:
        import asyncio
        return await asyncio.to_thread(self._sync.create, **kwargs)


class _AsyncAnthropicChatShim:
    def __init__(self, adapter: _AsyncAnthropicCompletionsAdapter):
        self.completions = adapter


class AsyncAnthropicAuxiliaryClient:
    def __init__(self, sync_wrapper: "AnthropicAuxiliaryClient"):
        sync_adapter = sync_wrapper.chat.completions
        async_adapter = _AsyncAnthropicCompletionsAdapter(sync_adapter)
        self.chat = _AsyncAnthropicChatShim(async_adapter)
        self.api_key = sync_wrapper.api_key
        self.base_url = sync_wrapper.base_url
        # See AsyncCodexAuxiliaryClient: mirror _real_client so cache
        # eviction on a poisoned underlying client also drops this entry.
        self._real_client = sync_wrapper._real_client


def _endpoint_speaks_anthropic_messages(base_url: str) -> bool:
    """True if the endpoint at ``base_url`` speaks the Anthropic Messages
    protocol instead of OpenAI chat.completions.

    Mirrors ``hermes_cli.runtime_provider._detect_api_mode_for_url`` so the
    auxiliary client and the main agent stay in sync on transport selection.
    Covers:

    - Any URL ending in ``/anthropic`` (MiniMax, Zhipu GLM, LiteLLM proxies,
      Anthropic-compatible gateways).
    - ``api.kimi.com/coding`` (Kimi Coding Plan — the /coding route only
      speaks Claude-Code's native Anthropic shape; ``chat.completions``
      returns 404 on Anthropic-only model aliases like ``kimi-for-coding``).
    - ``api.anthropic.com`` (native Anthropic).
    """
    normalized = (base_url or "").strip().lower().rstrip("/")
    if not normalized:
        return False
    path = urlparse(normalized).path.rstrip("/")
    if path.endswith("/anthropic") or path.endswith("/anthropic/v1"):
        return True
    hostname = base_url_hostname(normalized)
    if hostname == "api.anthropic.com":
        return True
    if hostname == "api.kimi.com" and "/coding" in normalized:
        return True
    return False


def _maybe_wrap_anthropic(
    client_obj: Any,
    model: str,
    api_key: str,
    base_url: str,
    api_mode: Optional[str] = None,
) -> Any:
    """Rewrap a plain OpenAI client in ``AnthropicAuxiliaryClient`` when
    the endpoint actually speaks Anthropic Messages.

    This is the single chokepoint for aux-client transport correction.
    Runs at the end of every ``resolve_provider_client`` branch so that
    api_key providers (Kimi Coding Plan), the ``custom`` endpoint, and
    future /anthropic gateways all land on the right wire format
    regardless of which branch built the client.

    Returns ``client_obj`` unchanged when:

    - It's already an Anthropic/Codex/Gemini/CopilotACP wrapper.
    - The endpoint is an OpenAI-wire endpoint.
    - ``api_mode`` is explicitly set to a non-Anthropic transport.
    - The ``anthropic`` SDK is not installed (falls back to OpenAI wire).
    """
    # Already wrapped — don't double-wrap.
    if _safe_isinstance(client_obj, AnthropicAuxiliaryClient):
        return client_obj
    # Other specialized adapters we should never re-dispatch.
    if _safe_isinstance(client_obj, CodexAuxiliaryClient):
        return client_obj
    try:
        from agent.gemini_native_adapter import GeminiNativeClient
        if _safe_isinstance(client_obj, GeminiNativeClient):
            return client_obj
    except ImportError:
        pass
    try:
        from agent.copilot_acp_client import CopilotACPClient
        if _safe_isinstance(client_obj, CopilotACPClient):
            return client_obj
    except ImportError:
        pass

    # Explicit non-anthropic api_mode wins over URL heuristics.
    if api_mode and api_mode != "anthropic_messages":
        return client_obj

    should_wrap = (
        api_mode == "anthropic_messages"
        or _endpoint_speaks_anthropic_messages(base_url)
    )
    if not should_wrap:
        return client_obj

    try:
        from agent.anthropic_adapter import build_anthropic_client
    except ImportError:
        logger.warning(
            "Endpoint %s speaks Anthropic Messages but the anthropic SDK is "
            "not installed — falling back to OpenAI-wire (will likely 404).",
            base_url,
        )
        return client_obj

    try:
        real_client = build_anthropic_client(api_key, base_url)
    except Exception as exc:
        logger.warning(
            "Failed to build Anthropic client for %s (%s) — falling back to "
            "OpenAI-wire client.", base_url, exc,
        )
        return client_obj

    logger.debug(
        "Auxiliary transport: wrapping client in AnthropicAuxiliaryClient "
        "(model=%s, base_url=%s, api_mode=%s)",
        model, base_url[:60] if base_url else "", api_mode or "auto-detected",
    )
    return AnthropicAuxiliaryClient(
        real_client, model, api_key, base_url, is_oauth=False,
    )


def _read_nous_auth() -> Optional[dict]:
    """Read and validate ~/.hermes/auth.json for an active Nous provider.

    Returns the provider state dict if Nous is active with tokens,
    otherwise None.
    """
    pool_present, entry = _select_pool_entry("nous")
    if pool_present:
        if entry is None:
            return None
        return {
            "access_token": getattr(entry, "access_token", ""),
            "refresh_token": getattr(entry, "refresh_token", None),
            "agent_key": getattr(entry, "agent_key", None),
            "inference_base_url": _pool_runtime_base_url(entry, _NOUS_DEFAULT_BASE_URL),
            "portal_base_url": getattr(entry, "portal_base_url", None),
            "client_id": getattr(entry, "client_id", None),
            "scope": getattr(entry, "scope", None),
            "token_type": getattr(entry, "token_type", "Bearer"),
            "source": "pool",
        }

    try:
        if not _AUTH_JSON_PATH.is_file():
            return None
        data = json.loads(_AUTH_JSON_PATH.read_text())
        if data.get("active_provider") != "nous":
            return None
        provider = data.get("providers", {}).get("nous", {})
        # Must have at least an access_token or agent_key
        if not provider.get("agent_key") and not provider.get("access_token"):
            return None
        return provider
    except Exception as exc:
        logger.debug("Could not read Nous auth: %s", exc)
        return None


def _nous_api_key(provider: dict) -> str:
    """Extract a usable Nous inference JWT from stored auth state."""
    from hermes_cli.auth import _nous_invoke_jwt_is_usable

    for token_key, expiry_key in (
        ("agent_key", "agent_key_expires_at"),
        ("access_token", "expires_at"),
    ):
        token = provider.get(token_key)
        if not isinstance(token, str) or not token.strip():
            continue
        if _nous_invoke_jwt_is_usable(
            token,
            scope=provider.get("scope"),
            expires_at=provider.get(expiry_key),
        ):
            return token
    return ""


def _nous_base_url() -> str:
    """Resolve the Nous inference base URL from env or default."""
    return os.getenv("NOUS_INFERENCE_BASE_URL", _NOUS_DEFAULT_BASE_URL)


def _resolve_nous_pool_runtime_api(*, force_refresh: bool = False) -> Optional[tuple[str, str]]:
    """Resolve Nous auxiliary credentials from the selected pool entry."""
    try:
        from hermes_cli.auth import _agent_key_is_usable

        pool = load_pool("nous")
    except Exception as exc:
        logger.debug("Auxiliary Nous pool credential resolution failed: %s", exc)
        return None

    if not pool or not pool.has_credentials():
        return None

    try:
        entry = pool.select()
    except Exception as exc:
        logger.debug("Auxiliary Nous pool selection failed: %s", exc)
        return None

    if entry is None:
        return None

    state = {
        "agent_key": getattr(entry, "agent_key", None),
        "agent_key_expires_at": getattr(entry, "agent_key_expires_at", None),
        "scope": getattr(entry, "scope", None),
    }
    if force_refresh or not _agent_key_is_usable(state, _nous_min_key_ttl_seconds()):
        try:
            refreshed = pool.try_refresh_current()
        except Exception as exc:
            logger.debug("Auxiliary Nous pool refresh failed: %s", exc)
            refreshed = None
        if refreshed is None:
            return None
        entry = refreshed

    provider = {
        "agent_key": getattr(entry, "agent_key", None),
        "agent_key_expires_at": getattr(entry, "agent_key_expires_at", None),
        "access_token": getattr(entry, "access_token", None),
        "expires_at": getattr(entry, "expires_at", None),
        "scope": getattr(entry, "scope", None),
    }
    api_key = _nous_api_key(provider)
    base_url = _pool_runtime_base_url(entry, _NOUS_DEFAULT_BASE_URL)
    if not api_key or not base_url:
        return None
    return api_key, base_url


def _resolve_nous_runtime_api(*, force_refresh: bool = False) -> Optional[tuple[str, str]]:
    """Return fresh Nous runtime credentials when available.

    This mirrors the main agent's 401 recovery path and keeps auxiliary
    clients aligned with the singleton auth store + JWT refresh flow instead of
    relying only on whatever raw tokens happen to be sitting in auth.json
    or the credential pool.
    """
    pooled = _resolve_nous_pool_runtime_api(force_refresh=force_refresh)
    if pooled is not None:
        return pooled

    try:
        from hermes_cli.auth import resolve_nous_runtime_credentials

        creds = resolve_nous_runtime_credentials(
            timeout_seconds=env_float("HERMES_NOUS_TIMEOUT_SECONDS", 15),
            force_refresh=force_refresh,
        )
    except Exception as exc:
        logger.debug("Auxiliary Nous runtime credential resolution failed: %s", exc)
        return None

    api_key = str(creds.get("api_key") or "").strip()
    base_url = str(creds.get("base_url") or "").strip().rstrip("/")
    if not api_key or not base_url:
        return None
    return api_key, base_url


def _resolve_xai_oauth_for_aux() -> Optional[Tuple[str, str]]:
    """Resolve a fresh xAI OAuth (api_key, base_url) for auxiliary clients.

    Prefer the credential pool, matching the main runtime/provider status
    path.  Some xAI OAuth logins live only as pool entries; falling straight
    to the singleton auth-store resolver would make auxiliary tasks such as
    compression report "no provider configured" even though ``hermes auth
    status`` shows xAI OAuth as logged in.

    Falls back to ``hermes_cli.auth``'s singleton runtime resolver for older
    auth-store-only logins. Returns ``None`` if the user is not authenticated
    with xAI Grok OAuth.
    """
    try:
        from hermes_cli.auth import (
            DEFAULT_XAI_OAUTH_BASE_URL,
            _xai_validate_inference_base_url,
        )

        pool = load_pool("xai-oauth")
        if pool and pool.has_credentials():
            entry = pool.select()
            if entry is not None:
                api_key = str(
                    getattr(entry, "runtime_api_key", None)
                    or getattr(entry, "access_token", "")
                    or ""
                ).strip()
                base_url = _xai_validate_inference_base_url(
                    os.getenv("HERMES_XAI_BASE_URL", "").strip().rstrip("/")
                    or os.getenv("XAI_BASE_URL", "").strip().rstrip("/")
                    or str(getattr(entry, "runtime_base_url", None) or "").strip().rstrip("/")
                    or str(getattr(entry, "base_url", None) or "").strip().rstrip("/"),
                    fallback=DEFAULT_XAI_OAUTH_BASE_URL,
                )
                if api_key and base_url:
                    return api_key, base_url
    except Exception as exc:
        logger.debug("Auxiliary xAI OAuth pool credential resolution failed: %s", exc)

    try:
        from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

        creds = resolve_xai_oauth_runtime_credentials()
    except Exception as exc:
        logger.debug("Auxiliary xAI OAuth runtime credential resolution failed: %s", exc)
        return None

    api_key = str(creds.get("api_key") or "").strip()
    base_url = str(creds.get("base_url") or "").strip().rstrip("/")
    if not api_key or not base_url:
        return None
    return api_key, base_url


def _read_codex_access_token() -> Optional[str]:
    """Read a valid, non-expired Codex OAuth access token from Hermes auth store.

    If a credential pool exists but currently has no selectable runtime entry
    (for example all pool slots are marked exhausted), fall back to the
    profile's auth.json token instead of hard-failing. This keeps explicit
    fallback-to-Codex working when the pool state is stale but the stored OAuth
    token is still valid.
    """
    pool_present, entry = _select_pool_entry("openai-codex")
    if pool_present:
        token = _pool_runtime_api_key(entry)
        if token:
            return token

    try:
        from hermes_cli.auth import _read_codex_tokens
        data = _read_codex_tokens()
        tokens = data.get("tokens", {})
        access_token = tokens.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            return None

        # Check JWT expiry — expired tokens block the auto chain and
        # prevent fallback to working providers (e.g. Anthropic).
        try:
            import base64
            payload = access_token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload))
            exp = claims.get("exp", 0)
            if exp and time.time() > exp:
                logger.debug("Codex access token expired (exp=%s), skipping", exp)
                return None
        except Exception:
            pass  # Non-JWT token or decode error — use as-is

        return access_token.strip()
    except Exception as exc:
        logger.debug("Could not read Codex auth for auxiliary client: %s", exc)
        return None


def _resolve_api_key_provider() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Try each API-key provider in PROVIDER_REGISTRY order.

    Returns (client, model) for the first provider with usable runtime
    credentials, or (None, None) if none are configured.
    """
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY, resolve_api_key_provider_credentials
    except ImportError:
        logger.debug("Could not import PROVIDER_REGISTRY for API-key fallback")
        return None, None

    for provider_id, pconfig in PROVIDER_REGISTRY.items():
        if pconfig.auth_type != "api_key":
            continue
        if _is_provider_unhealthy(provider_id):
            logger.debug("Auxiliary api-key chain: %s is unhealthy, skipping", provider_id)
            continue
        if provider_id == "anthropic":
            # Only try anthropic when the user has explicitly configured it.
            # Without this gate, Claude Code credentials get silently used
            # as auxiliary fallback when the user's primary provider fails.
            try:
                from hermes_cli.auth import is_provider_explicitly_configured
                if not is_provider_explicitly_configured("anthropic"):
                    continue
            except ImportError:
                pass
            return _try_anthropic()

        pool_present, entry = _select_pool_entry(provider_id)
        if pool_present:
            api_key = _pool_runtime_api_key(entry)
            if not api_key:
                continue

            raw_base_url = _pool_runtime_base_url(entry, pconfig.inference_base_url) or pconfig.inference_base_url
            base_url = _to_openai_base_url(raw_base_url)
            model = _get_aux_model_for_provider(provider_id) or None
            if model is None:
                continue  # skip provider if we don't know a valid aux model
            logger.debug("Auxiliary text client: %s (%s) via pool", pconfig.name, model)
            if provider_id == "gemini":
                from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url

                if is_native_gemini_base_url(base_url):
                    return GeminiNativeClient(api_key=api_key, base_url=base_url), model
            extra = {}
            if base_url_host_matches(base_url, "api.kimi.com"):
                extra["default_headers"] = {"User-Agent": "claude-code/0.1.0"}
            elif base_url_host_matches(base_url, "api.githubcopilot.com"):
                from hermes_cli.models import copilot_default_headers

                extra["default_headers"] = copilot_default_headers()
            elif base_url_host_matches(base_url, "integrate.api.nvidia.com"):
                extra["default_headers"] = build_nvidia_nim_headers(base_url)
            else:
                try:
                    from providers import get_provider_profile as _gpf_aux
                    _ph_aux = _gpf_aux(provider_id)
                    if _ph_aux and _ph_aux.default_headers:
                        extra["default_headers"] = dict(_ph_aux.default_headers)
                except Exception:
                    pass
            _merged_aux = _apply_user_default_headers(extra.get("default_headers"))
            if _merged_aux:
                extra["default_headers"] = _merged_aux
            _client = OpenAI(api_key=api_key, base_url=base_url, **extra)
            _client = _maybe_wrap_anthropic(_client, model, api_key, raw_base_url)
            return _client, model

        creds = resolve_api_key_provider_credentials(provider_id)
        api_key = str(creds.get("api_key", "")).strip()
        if not api_key:
            continue

        raw_base_url = str(creds.get("base_url", "")).strip().rstrip("/") or pconfig.inference_base_url
        base_url = _to_openai_base_url(raw_base_url)
        model = _get_aux_model_for_provider(provider_id) or None
        if model is None:
            continue  # skip provider if we don't know a valid aux model
        logger.debug("Auxiliary text client: %s (%s)", pconfig.name, model)
        if provider_id == "gemini":
            from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url

            if is_native_gemini_base_url(base_url):
                return GeminiNativeClient(api_key=api_key, base_url=base_url), model
        extra = {}
        if base_url_host_matches(base_url, "api.kimi.com"):
            extra["default_headers"] = {"User-Agent": "claude-code/0.1.0"}
        elif base_url_host_matches(base_url, "api.githubcopilot.com"):
            from hermes_cli.models import copilot_default_headers

            extra["default_headers"] = copilot_default_headers()
        elif base_url_host_matches(base_url, "integrate.api.nvidia.com"):
            extra["default_headers"] = build_nvidia_nim_headers(base_url)
        else:
            try:
                from providers import get_provider_profile as _gpf_aux2
                _ph_aux2 = _gpf_aux2(provider_id)
                if _ph_aux2 and _ph_aux2.default_headers:
                    extra["default_headers"] = dict(_ph_aux2.default_headers)
            except Exception:
                pass
        _merged_aux2 = _apply_user_default_headers(extra.get("default_headers"))
        if _merged_aux2:
            extra["default_headers"] = _merged_aux2
        _client = OpenAI(api_key=api_key, base_url=base_url, **extra)
        _client = _maybe_wrap_anthropic(_client, model, api_key, raw_base_url)
        return _client, model

    return None, None


# ── Provider resolution helpers ─────────────────────────────────────────────



def _try_openrouter(explicit_api_key: str = None, model: str = None) -> Tuple[Optional[OpenAI], Optional[str]]:
    pool_present, entry = _select_pool_entry("openrouter")
    if pool_present:
        or_key = explicit_api_key or _pool_runtime_api_key(entry)
        if not or_key:
            _mark_provider_unhealthy("openrouter", ttl=60)
            return None, None
        base_url = _pool_runtime_base_url(entry, OPENROUTER_BASE_URL) or OPENROUTER_BASE_URL
        logger.debug("Auxiliary client: OpenRouter via pool")
        return OpenAI(api_key=or_key, base_url=base_url,
                       default_headers=build_or_headers()), model or _OPENROUTER_MODEL

    or_key = explicit_api_key or os.getenv("OPENROUTER_API_KEY")
    if not or_key:
        _mark_provider_unhealthy("openrouter", ttl=60)
        return None, None
    logger.debug("Auxiliary client: OpenRouter")
    return OpenAI(api_key=or_key, base_url=OPENROUTER_BASE_URL,
                   default_headers=build_or_headers()), model or _OPENROUTER_MODEL


def _describe_openrouter_unavailable() -> str:
    """Return a more precise OpenRouter auth failure reason for logs."""
    pool_present, entry = _select_pool_entry("openrouter")
    if pool_present:
        if entry is None:
            return "OpenRouter credential pool has no usable entries (credentials may be exhausted)"
        if not _pool_runtime_api_key(entry):
            return "OpenRouter credential pool entry is missing a runtime API key"
    if not str(os.getenv("OPENROUTER_API_KEY") or "").strip():
        return "OPENROUTER_API_KEY not set"
    return "no usable OpenRouter credentials found"


def _try_nous(vision: bool = False) -> Tuple[Optional[OpenAI], Optional[str]]:
    # Check cross-session rate limit guard before attempting Nous —
    # if another session already recorded a 429, skip Nous entirely
    # to avoid piling more requests onto the tapped RPH bucket.
    try:
        from agent.nous_rate_guard import nous_rate_limit_remaining
        _remaining = nous_rate_limit_remaining()
        if _remaining is not None and _remaining > 0:
            logger.debug(
                "Auxiliary: skipping Nous Portal (rate-limited, resets in %.0fs)",
                _remaining,
            )
            _mark_provider_unhealthy("nous", ttl=_remaining)
            return None, None
    except Exception:
        pass

    nous = _read_nous_auth()
    runtime = _resolve_nous_runtime_api(force_refresh=False)
    if runtime is None and not nous:
        logger.warning(
            "Auxiliary Nous client unavailable: no Nous authentication found "
            "(run: hermes auth)."
        )
        _mark_provider_unhealthy("nous", ttl=60)
        return None, None
    if runtime is None and nous:
        logger.debug(
            "Auxiliary Nous: runtime JWT refresh failed; checking stored "
            "auth.json token."
        )
    global auxiliary_is_nous
    auxiliary_is_nous = True
    logger.debug("Auxiliary client: Nous Portal")

    # Ask the Portal which model it currently recommends for this task type.
    # The /api/nous/recommended-models endpoint is the authoritative source:
    # it distinguishes paid vs free tier recommendations, and get_nous_recommended_aux_model
    # auto-detects the caller's tier via check_nous_free_tier().  Fall back to
    # _NOUS_MODEL (google/gemini-3-flash-preview) when the Portal is unreachable
    # or returns a null recommendation for this task type.
    model = _NOUS_MODEL
    try:
        from hermes_cli.models import get_nous_recommended_aux_model
        recommended = get_nous_recommended_aux_model(vision=vision)
        if recommended:
            model = recommended
            logger.debug(
                "Auxiliary/%s: using Portal-recommended model %s",
                "vision" if vision else "text", model,
            )
        else:
            logger.debug(
                "Auxiliary/%s: no Portal recommendation, falling back to %s",
                "vision" if vision else "text", model,
            )
    except Exception as exc:
        logger.debug(
            "Auxiliary/%s: recommended-models lookup failed (%s); "
            "falling back to %s",
            "vision" if vision else "text", exc, model,
        )

    if runtime is not None:
        api_key, base_url = runtime
    else:
        api_key = _nous_api_key(nous or {})
        if not api_key:
            logger.warning(
                "Auxiliary Nous client unavailable: no usable inference JWT found "
                "(run: hermes auth add nous)."
            )
            _mark_provider_unhealthy("nous", ttl=60)
            return None, None
        base_url = str((nous or {}).get("inference_base_url") or _nous_base_url()).rstrip("/")
    return (
        OpenAI(
            api_key=api_key,
            base_url=base_url,
        ),
        model,
    )


def _refresh_nous_recommended_model(
    *, vision: bool, stale_model: Optional[str]
) -> Optional[str]:
    """Re-fetch the Nous Portal's recommended model after a stale-model 404.

    Long-lived processes (gateway, watchers) cache the Portal's
    ``recommended-models`` payload for 10 minutes and, in practice, can pin a
    model for the whole process lifetime. When that model is later dropped from
    the Nous → OpenRouter catalog, every auxiliary call 404s with
    "model does not exist". This forces a fresh Portal fetch and returns a
    model name to retry with:

      * the Portal's current recommendation for the task, if it differs from
        the model that just failed; otherwise
      * ``_NOUS_MODEL`` (google/gemini-3-flash-preview), the known-good default,
        if it too differs from the failed model.

    Returns ``None`` when no usable alternative is available (e.g. the Portal
    still recommends the exact model that just 404'd and the default also
    matches it) — callers should then let the original error propagate.
    """
    stale = (stale_model or "").strip().lower()
    fresh: Optional[str] = None
    try:
        from hermes_cli.models import get_nous_recommended_aux_model

        fresh = get_nous_recommended_aux_model(vision=vision, force_refresh=True)
    except Exception as exc:
        logger.debug(
            "Nous recommended-model refresh failed (%s); using default %s",
            exc, _NOUS_MODEL,
        )
    if fresh and fresh.strip().lower() != stale:
        return fresh
    # Portal recommendation unchanged or unavailable — fall back to the
    # hardcoded known-good default, but only if it's actually different.
    if _NOUS_MODEL.strip().lower() != stale:
        return _NOUS_MODEL
    return None


def _read_main_model() -> str:
    """Read the user's configured main model from config.yaml.

    config.yaml model.default is the single source of truth for the active
    model. Environment variables are no longer consulted.

    Runtime override: when an AIAgent is active with a CLI/gateway-provided
    model that differs from config.yaml, ``set_runtime_main()`` records the
    override in a process-local global. This is consulted FIRST so tools
    that gate on "the active main model" (e.g. ``vision_analyze``'s native
    fast path) see the live runtime, not the persisted config default.
    """
    override = _RUNTIME_MAIN_MODEL
    if isinstance(override, str) and override.strip():
        return override.strip()
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, str) and model_cfg.strip():
            return model_cfg.strip()
        if isinstance(model_cfg, dict):
            default = model_cfg.get("default", "")
            if isinstance(default, str) and default.strip():
                return default.strip()
    except Exception:
        pass
    return ""


def _read_main_provider() -> str:
    """Read the user's configured main provider from config.yaml.

    Returns the lowercase provider id (e.g. "alibaba", "openrouter") or ""
    if not configured.

    Runtime override: see ``_read_main_model`` — same mechanism for the
    provider half of the runtime tuple.
    """
    override = _RUNTIME_MAIN_PROVIDER
    if isinstance(override, str) and override.strip():
        return override.strip().lower()
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, dict):
            provider = model_cfg.get("provider", "")
            if isinstance(provider, str) and provider.strip():
                return provider.strip().lower()
    except Exception:
        pass
    return ""


# Process-local override set by AIAgent at session/turn start. Single-threaded
# per turn — no lock needed. Cleared by ``clear_runtime_main()``.
_RUNTIME_MAIN_PROVIDER: str = ""
_RUNTIME_MAIN_MODEL: str = ""
_RUNTIME_MAIN_BASE_URL: str = ""
_RUNTIME_MAIN_API_KEY: str = ""
_RUNTIME_MAIN_API_MODE: str = ""


def set_runtime_main(
    provider: str,
    model: str,
    *,
    base_url: str = "",
    api_key: str = "",
    api_mode: str = "",
) -> None:
    """Record the live runtime provider/model/credentials for the current AIAgent.

    Called by ``run_agent.AIAgent._sync_runtime_main_for_aux_routing`` (or
    equivalent setter) at the top of each turn so that
    ``_read_main_provider`` / ``_read_main_model`` reflect CLI/gateway
    overrides instead of the stale config.yaml default.

    For ``custom:`` providers, ``base_url`` and ``api_key`` must also be
    recorded so that ``_resolve_auto`` can construct a valid client in
    Step 1 instead of falling through to the aggregator chain.
    """
    global _RUNTIME_MAIN_PROVIDER, _RUNTIME_MAIN_MODEL
    global _RUNTIME_MAIN_BASE_URL, _RUNTIME_MAIN_API_KEY, _RUNTIME_MAIN_API_MODE
    _RUNTIME_MAIN_PROVIDER = (provider or "").strip().lower()
    _RUNTIME_MAIN_MODEL = (model or "").strip()
    _RUNTIME_MAIN_BASE_URL = (base_url or "").strip()
    _RUNTIME_MAIN_API_KEY = api_key.strip() if isinstance(api_key, str) else ""
    _RUNTIME_MAIN_API_MODE = (api_mode or "").strip()


def clear_runtime_main() -> None:
    """Clear the runtime override (e.g. on session end)."""
    global _RUNTIME_MAIN_PROVIDER, _RUNTIME_MAIN_MODEL
    global _RUNTIME_MAIN_BASE_URL, _RUNTIME_MAIN_API_KEY, _RUNTIME_MAIN_API_MODE
    _RUNTIME_MAIN_PROVIDER = ""
    _RUNTIME_MAIN_MODEL = ""
    _RUNTIME_MAIN_BASE_URL = ""
    _RUNTIME_MAIN_API_KEY = ""
    _RUNTIME_MAIN_API_MODE = ""


def _resolve_custom_runtime() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve the active custom/main endpoint the same way the main CLI does.

    This covers both env-driven OPENAI_BASE_URL setups and config-saved custom
    endpoints where the base URL lives in config.yaml instead of the live
    environment.
    """
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested="custom")
    except Exception as exc:
        logger.debug("Auxiliary client: custom runtime resolution failed: %s", exc)
        runtime = None

    if not isinstance(runtime, dict):
        openai_base = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_base:
            return None, None, None
        runtime = {
            "base_url": openai_base,
            "api_key": openai_key,
        }

    custom_base = runtime.get("base_url")
    custom_key = runtime.get("api_key")
    custom_mode = runtime.get("api_mode")
    if not isinstance(custom_base, str) or not custom_base.strip():
        return None, None, None

    custom_base = custom_base.strip().rstrip("/")
    if base_url_host_matches(custom_base, "openrouter.ai"):
        # requested='custom' falls back to OpenRouter when no custom endpoint is
        # configured. Treat that as "no custom endpoint" for auxiliary routing.
        return None, None, None

    # Local servers (Ollama, llama.cpp, vLLM, LM Studio) don't require auth.
    # Use a placeholder key — the OpenAI SDK requires a non-empty string but
    # local servers ignore the Authorization header.  Same fix as cli.py
    # _ensure_runtime_credentials() (PR #2556).
    if not isinstance(custom_key, str) or not custom_key.strip():
        custom_key = "no-key-required"

    if not isinstance(custom_mode, str) or not custom_mode.strip():
        custom_mode = None

    return custom_base, custom_key.strip(), custom_mode


def _current_custom_base_url() -> str:
    custom_base, _, _ = _resolve_custom_runtime()
    return custom_base or ""


def _validate_proxy_env_urls() -> None:
    """Fail fast with a clear error when proxy env vars have malformed URLs.

    Common cause: shell config (e.g. .zshrc) with a typo like
    ``export HTTP_PROXY=http://127.0.0.1:6153export NEXT_VAR=...``
    which concatenates 'export' into the port number.  Without this
    check the OpenAI/httpx client raises a cryptic ``Invalid port``
    error that doesn't name the offending env var.
    """
    from urllib.parse import urlparse

    normalize_proxy_env_vars()

    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        value = str(os.environ.get(key) or "").strip()
        if not value:
            continue
        try:
            parsed = urlparse(value)
            if parsed.scheme:
                _ = parsed.port          # raises ValueError for e.g. '6153export'
        except ValueError as exc:
            raise RuntimeError(
                f"Malformed proxy environment variable {key}={value!r}. "
                "Fix or unset your proxy settings and try again."
            ) from exc


def _validate_base_url(base_url: str) -> None:
    """Reject obviously broken custom endpoint URLs before they reach httpx."""
    from urllib.parse import urlparse

    candidate = str(base_url or "").strip()
    if not candidate or candidate.startswith("acp://"):
        return
    try:
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"}:
            _ = parsed.port              # raises ValueError for malformed ports
    except ValueError as exc:
        raise RuntimeError(
            f"Malformed custom endpoint URL: {candidate!r}. "
            "Run `hermes setup` or `hermes model` and enter a valid http(s) base URL."
        ) from exc


def _try_custom_endpoint() -> Tuple[Optional[Any], Optional[str]]:
    runtime = _resolve_custom_runtime()
    if len(runtime) == 2:
        custom_base, custom_key = runtime
        custom_mode = None
    else:
        custom_base, custom_key, custom_mode = runtime
    if not custom_base or not custom_key:
        return None, None
    if custom_base.lower().startswith(_CODEX_AUX_BASE_URL.lower()):
        return None, None
    model = _read_main_model() or "gpt-4o-mini"
    logger.debug("Auxiliary client: custom endpoint (%s, api_mode=%s)", model, custom_mode or "chat_completions")
    _clean_base, _dq = _extract_url_query_params(custom_base)
    _extra = {"default_query": _dq} if _dq else {}
    # User-configured model.default_headers override the SDK's identifying
    # headers (User-Agent: OpenAI/Python ..., X-Stainless-*) on this custom
    # endpoint's auxiliary calls too — matching the main agent client so the
    # whole session reaches a gateway/WAF that rejects the SDK fingerprint. (#40033)
    _custom_headers = _apply_user_default_headers(None)
    if _custom_headers:
        _extra["default_headers"] = _custom_headers
    if custom_mode == "codex_responses":
        real_client = OpenAI(api_key=custom_key, base_url=_clean_base, **_extra)
        return CodexAuxiliaryClient(real_client, model), model
    if custom_mode == "anthropic_messages":
        # Third-party Anthropic-compatible gateway (MiniMax, Zhipu GLM,
        # LiteLLM proxies, etc.).  Must NEVER be treated as OAuth —
        # Anthropic OAuth claims only apply to api.anthropic.com.
        try:
            from agent.anthropic_adapter import build_anthropic_client
            real_client = build_anthropic_client(custom_key, custom_base)
        except ImportError:
            logger.warning(
                "Custom endpoint declares api_mode=anthropic_messages but the "
                "anthropic SDK is not installed — falling back to OpenAI-wire."
            )
            return OpenAI(api_key=custom_key, base_url=_clean_base, **_extra), model
        return (
            AnthropicAuxiliaryClient(real_client, model, custom_key, custom_base, is_oauth=False),
            model,
        )
    # URL-based anthropic detection for custom endpoints that didn't set
    # api_mode explicitly (e.g. kimi.com/coding reached via custom config).
    _fallback_client = OpenAI(api_key=custom_key, base_url=_clean_base, **_extra)
    _fallback_client = _maybe_wrap_anthropic(
        _fallback_client, model, custom_key, custom_base, custom_mode,
    )
    return _fallback_client, model


def _build_xai_oauth_aux_client(model: str) -> Tuple[Optional[Any], Optional[str]]:
    """Build a CodexAuxiliaryClient for an xAI Grok OAuth-authenticated session.

    xAI's ``/v1/responses`` endpoint speaks the OpenAI Responses API, so we
    wrap a plain ``OpenAI`` client in ``CodexAuxiliaryClient`` to translate
    ``chat.completions.create()`` calls into ``responses.stream()`` requests.

    The caller must pass an explicit model — pinning a default for Grok
    would silently rot when xAI's allowlist drifts.  Returns ``(None, None)``
    when the user has not authenticated with xAI Grok OAuth.
    """
    if not model:
        logger.warning(
            "Auxiliary client: xai-oauth requested without a model; "
            "pass model explicitly (auxiliary.<task>.model in config.yaml)."
        )
        return None, None
    resolved = _resolve_xai_oauth_for_aux()
    if resolved is None:
        return None, None
    api_key, base_url = resolved
    logger.debug("Auxiliary client: xAI OAuth (%s via Responses API)", model)
    real_client = OpenAI(api_key=api_key, base_url=base_url)
    return CodexAuxiliaryClient(real_client, model), model


def _build_codex_client(model: str) -> Tuple[Optional[Any], Optional[str]]:
    """Build a CodexAuxiliaryClient for an explicitly-requested model.

    There is no auto-selection of the Codex model: the ChatGPT-account
    Codex endpoint's accepted model list is an undocumented, drifting
    allow-list, so any hardcoded default we pick goes stale.  The caller
    is responsible for passing the model (e.g. from the user's own
    ``model.model`` or ``auxiliary.<task>.model`` config).

    Returns (None, None) when no Codex OAuth token is available.
    """
    if not model:
        logger.warning(
            "Auxiliary client: openai-codex requested without a model; "
            "pass model explicitly (auxiliary.<task>.model in config.yaml)."
        )
        return None, None
    pool_present, entry = _select_pool_entry("openai-codex")
    if pool_present:
        codex_token = _pool_runtime_api_key(entry)
        if codex_token:
            base_url = _pool_runtime_base_url(entry, _CODEX_AUX_BASE_URL) or _CODEX_AUX_BASE_URL
        else:
            codex_token = _read_codex_access_token()
            if not codex_token:
                return None, None
            base_url = _CODEX_AUX_BASE_URL
    else:
        codex_token = _read_codex_access_token()
        if not codex_token:
            return None, None
        base_url = _CODEX_AUX_BASE_URL
    logger.debug("Auxiliary client: Codex OAuth (%s via Responses API)", model)
    real_client = OpenAI(
        api_key=codex_token,
        base_url=base_url,
        default_headers=_codex_cloudflare_headers(codex_token),
    )
    return CodexAuxiliaryClient(real_client, model), model


def _try_azure_foundry(
    *,
    model: Optional[str] = None,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Resolve an Azure Foundry auxiliary client via the runtime resolver.

    Mirrors the ``_try_anthropic`` / ``_try_nous`` shape but delegates to
    :func:`hermes_cli.runtime_provider._resolve_azure_foundry_runtime` —
    the same resolver the main agent uses — so:

    * ``auth_mode: api_key`` (default) gets the static
      ``AZURE_FOUNDRY_API_KEY`` string.
    * ``auth_mode: entra_id`` gets a callable bearer-token provider
      (``Callable[[], str]`` from
      :mod:`agent.azure_identity_adapter`).
    * Per-model ``api_mode`` auto-routing for GPT-5.x / o-series /
      codex models works.
    * ``model.entra.{tenant_id,client_id,authority,scope}`` config
      fields propagate.
    * Non-default ``model.base_url`` overrides are honored.

    The OpenAI SDK accepts both shapes for ``api_key`` so the caller
    can forward the result without coercion.

    Returns ``(client, model)`` or ``(None, None)`` on failure.
    """
    try:
        from hermes_cli.runtime_provider import _resolve_azure_foundry_runtime
        from hermes_cli.auth import AuthError
        from hermes_cli.config import load_config
    except ImportError:
        return None, None

    try:
        cfg = load_config()
        model_cfg = cfg.get("model") if isinstance(cfg, dict) else {}
        if not isinstance(model_cfg, dict):
            model_cfg = {}
    except Exception:
        model_cfg = {}

    try:
        runtime = _resolve_azure_foundry_runtime(
            requested_provider="azure-foundry",
            model_cfg=model_cfg,
            explicit_api_key=explicit_api_key,
            explicit_base_url=explicit_base_url,
            target_model=model,
        )
    except AuthError as exc:
        logger.debug("Auxiliary azure-foundry: %s", exc)
        return None, None
    except Exception as exc:
        logger.debug("Auxiliary azure-foundry runtime error: %s", exc)
        return None, None

    api_key = runtime.get("api_key")
    base_url = str(runtime.get("base_url", "") or "")
    runtime_api_mode = api_mode or runtime.get("api_mode") or "chat_completions"

    # Empty-string check on api_key here would be wrong for callable
    # token providers (callables are truthy and non-empty by definition).
    # Bail only when api_key is None / empty string.
    _has_key = bool(api_key) if not callable(api_key) else True
    if not _has_key or not base_url:
        return None, None

    final_model = _normalize_resolved_model(
        model or str(model_cfg.get("default") or ""),
        "azure-foundry",
    )
    if not final_model:
        # No fallback aux model for Azure — the user must have a
        # deployment name. Surface that as "no client" so the auto
        # chain falls through to the next provider rather than 404ing.
        logger.debug(
            "Auxiliary azure-foundry: no model resolved (model=%r, default=%r)",
            model, model_cfg.get("default"),
        )
        return None, None

    # Azure pre-v1 endpoints sometimes carry api-version query params
    # in the base URL; the OpenAI SDK drops them when joining paths,
    # so lift them out and pass via default_query.
    extra: Dict[str, Any] = {}
    _clean_base, _dq = _extract_url_query_params(base_url)
    if _dq:
        extra["default_query"] = _dq

    client = OpenAI(api_key=api_key, base_url=_clean_base, **extra)

    if runtime_api_mode == "codex_responses":
        # GPT-5.x / o-series / codex models on Azure Foundry are
        # Responses-API-only — wrap so chat.completions.create() is
        # translated to /responses behind the scenes.
        return CodexAuxiliaryClient(client, final_model), final_model

    if runtime_api_mode == "anthropic_messages":
        # Forward ``api_key`` verbatim — for static keys it's a string,
        # for Entra ID it's a callable. ``_maybe_wrap_anthropic`` →
        # ``build_anthropic_client`` detects the callable and installs
        # the bearer-injecting httpx hook.
        return _maybe_wrap_anthropic(
            client, final_model, api_key,
            base_url, runtime_api_mode,
        ), final_model

    # chat_completions — return the plain OpenAI client.
    return client, final_model


def _try_anthropic(explicit_api_key: str = None) -> Tuple[Optional[Any], Optional[str]]:
    try:
        from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token
    except ImportError:
        return None, None

    pool_present, entry = _select_pool_entry("anthropic")
    if pool_present:
        if entry is None:
            return None, None
        token = explicit_api_key or _pool_runtime_api_key(entry)
    else:
        entry = None
        token = explicit_api_key or resolve_anthropic_token()
    if not token:
        return None, None

    # Allow base URL override from config.yaml model.base_url, but only
    # when the configured provider is anthropic — otherwise a non-Anthropic
    # base_url (e.g. Codex endpoint) would leak into Anthropic requests.
    base_url = _pool_runtime_base_url(entry, _ANTHROPIC_DEFAULT_BASE_URL) if pool_present else _ANTHROPIC_DEFAULT_BASE_URL
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get("model")
        if isinstance(model_cfg, dict):
            cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
            if cfg_provider == "anthropic":
                cfg_base_url = (model_cfg.get("base_url") or "").strip().rstrip("/")
                if cfg_base_url:
                    base_url = cfg_base_url
    except Exception:
        pass

    from agent.anthropic_adapter import _is_oauth_token
    is_oauth = _is_oauth_token(token)
    model = _get_aux_model_for_provider("anthropic") or "claude-haiku-4-5-20251001"
    logger.debug("Auxiliary client: Anthropic native (%s) at %s (oauth=%s)", model, base_url, is_oauth)
    try:
        real_client = build_anthropic_client(token, base_url)
    except ImportError:
        # The anthropic_adapter module imports fine but the SDK itself is
        # missing — build_anthropic_client raises ImportError at call time
        # when _anthropic_sdk is None.  Treat as unavailable.
        return None, None
    return AnthropicAuxiliaryClient(real_client, model, token, base_url, is_oauth=is_oauth), model


_AUTO_PROVIDER_LABELS = {
    "_try_openrouter": "openrouter",
    "_try_nous": "nous",
    "_try_custom_endpoint": "local/custom",
    "_resolve_api_key_provider": "api-key",
}

_MAIN_RUNTIME_FIELDS = ("provider", "model", "base_url", "api_key", "api_mode", "auth_mode")


def _normalize_main_runtime(main_runtime: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a sanitized copy of a live main-runtime override.

    Most fields are stripped strings. ``api_key`` may legitimately be a
    zero-arg callable (Azure Foundry Entra ID token provider) — preserve
    those as-is so auxiliary clients inherit the same authentication
    surface as the main agent. The OpenAI SDK accepts ``Callable[[], str]``
    for ``api_key`` and calls it before every request.
    """
    if not isinstance(main_runtime, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for field in _MAIN_RUNTIME_FIELDS:
        value = main_runtime.get(field)
        # Preserve a callable api_key (Entra ID bearer provider) unchanged.
        if field == "api_key" and callable(value) and not isinstance(value, str):
            normalized[field] = value
            continue
        if isinstance(value, str) and value.strip():
            normalized[field] = value.strip()
    provider = normalized.get("provider")
    if isinstance(provider, str):
        normalized["provider"] = provider.lower()
    return normalized


def _get_provider_chain() -> List[tuple]:
    """Return the ordered provider detection chain.

    Built at call time (not module level) so that test patches
    on the ``_try_*`` functions are picked up correctly.

    NOTE: ``openai-codex`` is deliberately NOT in this chain.  The
    ChatGPT-account Codex endpoint only accepts a shifting, undocumented
    allow-list of model IDs, so falling back to it with a guessed model
    fails more often than not.  Codex is used only when the user's main
    provider *is* openai-codex (see Step 1 of ``_resolve_auto``) or when
    a caller explicitly requests it with a model.
    """
    return [
        ("openrouter", _try_openrouter),
        ("nous", _try_nous),
        ("local/custom", _try_custom_endpoint),
        ("api-key", _resolve_api_key_provider),
    ]


# ── Auxiliary "recently 402'd" unhealthy-provider cache ────────────────────
#
# When an auxiliary provider returns HTTP 402 (Payment Required / credit
# exhaustion), retrying it on every subsequent aux call is wasteful — the
# provider stays depleted for hours or days, but the chain re-tries it as
# the FIRST entry on every compression/title-gen/session-search call,
# burns ~1 RTT, gets 402 again, then falls back. On a long Discord/LCM
# session that adds up to dozens of doomed 402s.
#
# Solution: when ANY caller observes a payment error against a provider,
# mark it unhealthy for ``_AUX_UNHEALTHY_TTL_SECONDS``. ``_resolve_auto``
# Step-2 and ``_try_payment_fallback`` both consult this cache and skip
# unhealthy entries (logging once per skip-reason so the user sees what
# happened). Entries auto-expire so a topped-up account recovers without
# manual intervention.
#
# Failure isolation: the cache is in-process only. A second hermes
# process won't inherit the unhealthy mark — that's intentional, since
# the user might be running two profiles with different OpenRouter keys.

_AUX_UNHEALTHY_TTL_SECONDS = 600  # 10 minutes
_aux_unhealthy_until: Dict[str, float] = {}
_aux_unhealthy_logged_at: Dict[str, float] = {}

# Map provider names that show up in resolved_provider / explicit-config
# back to the chain labels used by _get_provider_chain(). Keep in sync
# with the alias map in _try_payment_fallback below.
_AUX_UNHEALTHY_LABEL_ALIASES = {
    "openrouter": "openrouter",
    "nous": "nous",
    "custom": "local/custom",
    "local/custom": "local/custom",
    "openai-codex": "openai-codex",
    "codex": "openai-codex",
}


def _normalize_chain_label(provider: str) -> str:
    """Normalize a resolved_provider value to a chain label used by
    ``_get_provider_chain()``. Falls back to the lowercased input for
    direct API-key providers (deepseek, alibaba, minimax, etc.) which
    each report their own provider name from the api-key chain.
    """
    if not provider:
        return ""
    p = str(provider).strip().lower()
    return _AUX_UNHEALTHY_LABEL_ALIASES.get(p, p)


def _mark_provider_unhealthy(provider: str, ttl: Optional[float] = None) -> None:
    """Mark ``provider`` as recently-402'd, hidden from chain iteration
    until the TTL expires. Called from the payment-fallback branches in
    ``call_llm`` and ``acall_llm`` after a confirmed payment error.
    """
    label = _normalize_chain_label(provider)
    if not label:
        return
    expires_at = time.time() + (ttl if ttl is not None else _AUX_UNHEALTHY_TTL_SECONDS)
    _aux_unhealthy_until[label] = expires_at
    logger.warning(
        "Auxiliary: marking %s unhealthy for %ds (payment / credit error). "
        "Subsequent auxiliary calls will skip it until %s.",
        label,
        int(ttl if ttl is not None else _AUX_UNHEALTHY_TTL_SECONDS),
        time.strftime("%H:%M:%S", time.localtime(expires_at)),
    )


def _is_provider_unhealthy(label: str) -> bool:
    """True iff ``label`` is in the unhealthy cache and the TTL hasn't expired.
    Lazily evicts expired entries so the cache stays small.
    """
    if not label:
        return False
    expires_at = _aux_unhealthy_until.get(label)
    if expires_at is None:
        return False
    if time.time() >= expires_at:
        _aux_unhealthy_until.pop(label, None)
        _aux_unhealthy_logged_at.pop(label, None)
        return False
    return True


def _log_skip_unhealthy(label: str, task: Optional[str] = None) -> None:
    """Emit a single info-level log per minute when we skip an unhealthy
    provider. Avoids spamming the log on bursty sessions while still
    giving the user a trail.
    """
    now = time.time()
    last = _aux_unhealthy_logged_at.get(label, 0.0)
    if now - last >= 60:
        _aux_unhealthy_logged_at[label] = now
        expires_at = _aux_unhealthy_until.get(label, now)
        logger.info(
            "Auxiliary %s: skipping %s (recently returned payment error, retry in %ds)",
            task or "call", label, max(0, int(expires_at - now)),
        )


def _reset_aux_unhealthy_cache() -> None:
    """Clear the unhealthy cache. Used by tests and by a future explicit
    user trigger (e.g. ``hermes config aux reset``)."""
    _aux_unhealthy_until.clear()
    _aux_unhealthy_logged_at.clear()


def _is_payment_error(exc: Exception) -> bool:
    """Detect payment/credit/quota exhaustion errors.

    Returns True for HTTP 402 (Payment Required) and for 429/other errors
    whose message indicates billing exhaustion or daily quota exhaustion
    rather than transient rate limiting.

    Daily token quota errors (e.g. Bedrock "Too many tokens per day",
    Vertex AI "quota exceeded") are functionally equivalent to credit
    exhaustion — the provider cannot serve the request until the quota
    resets — and should trigger the same provider-fallback logic.
    """
    status = getattr(exc, "status_code", None)
    if status == 402:
        return True
    err_lower = str(exc).lower()
    # OpenRouter and other providers include "credits" or "afford" in 402 bodies,
    # but sometimes wrap them in 429 or other codes.
    # Daily quota exhaustion from Bedrock, Vertex AI, and similar providers
    # uses different language but is semantically identical to credit exhaustion.
    if status in {402, 404, 429, None}:
        if any(kw in err_lower for kw in (
            "credits", "insufficient funds",
            "can only afford", "billing",
            "payment required",
            "out of funds", "run out of funds",
            "balance_depleted", "no usable credits",
            "model_not_supported_on_free_tier",
            "not available on the free tier",
            # Daily / monthly / weekly quota exhaustion keywords
            "quota exceeded", "quota_exceeded",
            "too many tokens per day", "daily limit",
            "tokens per day", "daily quota",
            "resource exhausted",  # Vertex AI / gRPC quota errors
            "weekly usage limit", "weekly limit",  # OpenCode Go weekly subscription cap
        )):
            return True
    return False


def _nous_portal_account_has_fresh_paid_access() -> bool:
    """Return True only when the fresh Nous account API says paid access is allowed."""
    try:
        from hermes_cli.nous_account import get_nous_portal_account_info

        account_info = get_nous_portal_account_info(force_fresh=True)
        return account_info.paid_service_access is True
    except Exception as exc:
        logger.debug("Auxiliary Nous paid-entitlement refresh check failed: %s", exc)
        return False


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect rate-limit errors that warrant provider fallback.

    Returns True for HTTP 429 errors whose message indicates rate limiting
    (as opposed to billing/quota exhaustion, which _is_payment_error handles).
    Also catches OpenAI SDK RateLimitError instances that may not set
    .status_code on the exception object.
    """
    status = getattr(exc, "status_code", None)
    err_lower = str(exc).lower()

    # OpenAI SDK's RateLimitError sometimes omits .status_code —
    # detect by class name so we don't miss these.  (PR #8023 pattern)
    if type(exc).__name__ == "RateLimitError":
        return True

    if status == 429:
        # Distinguish rate-limit from billing: billing keywords are handled
        # by _is_payment_error, everything else on 429 is a rate limit.
        if any(kw in err_lower for kw in (
            "rate limit", "rate_limit", "too many requests",
            "try again", "retry after", "resets in",
        )):
            return True
        # Generic 429 without billing keywords = likely a rate limit
        if not any(kw in err_lower for kw in (
            "credits", "insufficient funds", "billing",
            "payment required", "can only afford",
            "out of funds", "run out of funds",
            "balance_depleted", "no usable credits",
            "model_not_supported_on_free_tier",
            "not available on the free tier",
        )):
            return True
    return False


def _is_connection_error(exc: Exception) -> bool:
    """Detect connection/network errors that warrant provider fallback.

    Returns True for errors indicating the provider endpoint is unreachable
    (DNS failure, connection refused, TLS errors, timeouts).  These are
    distinct from API errors (4xx/5xx) which indicate the provider IS
    reachable but returned an error.
    """
    try:
        from openai import APIConnectionError, APITimeoutError
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
    except ImportError:
        pass
    # urllib3 / httpx / httpcore connection errors
    err_type = type(exc).__name__
    if any(kw in err_type for kw in ("Connection", "Timeout", "DNS", "SSL")):
        return True
    err_lower = str(exc).lower()
    if any(kw in err_lower for kw in (
        "connection refused", "name or service not known",
        "no route to host", "network is unreachable",
        "timed out", "connection reset",
        # httpcore / httpx streaming premature-close errors.  These surface
        # when a proxy or provider drops the connection mid-stream and are
        # transient by nature — the request should be retried or rerouted.
        # See issue #18458.
        "incomplete chunked read",
        "peer closed connection",
        "response ended prematurely",
        "unexpected eof",
        "remoteprotocolerror",
        "localprotocolerror",
    )):
        return True
    return False


def _is_transient_transport_error(exc: Exception) -> bool:
    """Return True for a one-off transport blip worth retrying ONCE on the
    same provider before any provider/model fallback.

    Covers connection/streaming-close errors (via the canonical
    ``_is_connection_error`` detector, shared so the two cannot drift) plus a
    pure 5xx/408 HTTP status. Deliberately narrow: this is the "retry the
    same target once" gate, distinct from ``_is_payment_error`` /
    ``_is_auth_error`` / ``_is_rate_limit_error`` which the except-chain
    handles by switching provider, refreshing creds, or rotating the pool.
    """
    if _is_connection_error(exc):
        return True
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    return isinstance(status, int) and (status == 408 or 500 <= status < 600)


def _is_auth_error(exc: Exception) -> bool:
    """Detect auth failures that should trigger provider-specific refresh."""
    status = getattr(exc, "status_code", None)
    if status == 401:
        return True
    err_lower = str(exc).lower()
    if "error code: 401" in err_lower or "authenticationerror" in type(exc).__name__.lower():
        return True
    # xAI returns HTTP 403 with "unauthenticated:bad-credentials" when an OAuth2
    # access token has expired or is invalid — semantically a 401 auth failure,
    # even though the status code is 403 (PermissionDenied).
    if status == 403 and "bad-credentials" in err_lower:
        return True
    if "unauthenticated" in err_lower and "bad-credentials" in err_lower:
        return True
    return False


def _is_unsupported_parameter_error(exc: Exception, param: str) -> bool:
    """Detect provider 400s for an unsupported request parameter.

    Different OpenAI-compatible endpoints phrase the same class of error a few
    ways: ``Unsupported parameter: X``, ``unsupported_parameter`` with a
    ``param`` field, ``X is not supported``, ``unknown parameter: X``,
    ``unrecognized request argument: X``.  We match on both the parameter
    name and a generic "unsupported/unknown/unrecognized parameter" marker so
    call sites can reactively retry without the offending key instead of
    surfacing a noisy auxiliary failure.

    Generalizes the temperature-specific detector that originally shipped
    with PR #15621 so the same retry strategy can cover ``max_tokens``,
    ``seed``, ``top_p``, and any future quirk. Credit @nicholasrae (PR #15416)
    for the generalization pattern.
    """
    param_lower = (param or "").lower()
    if not param_lower:
        return False
    err_lower = str(exc).lower()
    if param_lower not in err_lower:
        return False
    return any(marker in err_lower for marker in (
        "unsupported parameter",
        "unsupported_parameter",
        "not supported",
        "does not support",
        "unknown parameter",
        "unrecognized request argument",
        "unrecognized parameter",
        "invalid parameter",
    ))


def _is_unsupported_temperature_error(exc: Exception) -> bool:
    """Back-compat wrapper: detect API errors where the model rejects ``temperature``.

    Delegates to :func:`_is_unsupported_parameter_error`; kept as a separate
    public symbol because existing tests and call sites import it by name.
    """
    return _is_unsupported_parameter_error(exc, "temperature")


def _is_model_not_found_error(exc: Exception) -> bool:
    """Detect "the requested model doesn't exist" errors (404 / invalid model).

    This fires when a resolved model name is no longer served by the endpoint
    — most commonly when a long-lived process pinned a Portal-recommended model
    that has since been dropped from the Nous → OpenRouter catalog. The Nous
    proxy returns 404 with a body like::

        Model 'gpt-5.4-mini' not found. The requested model does not exist
        in our configuration or OpenRouter catalog.

    Distinct from :func:`_is_payment_error` (which also matches some 404s for
    free-tier/credit language) — this one keys on "does not exist / not found /
    not a valid model" phrasing, and explicitly excludes the billing keywords
    that the payment path already owns so the two predicates don't overlap.
    """
    status = getattr(exc, "status_code", None)
    err_lower = str(exc).lower()
    # Billing/quota 404s belong to _is_payment_error — don't claim them here.
    if any(kw in err_lower for kw in (
        "credits", "insufficient funds", "billing", "out of funds",
        "balance_depleted", "no usable credits", "free tier", "free-tier",
        "not available on the free tier",
    )):
        return False
    if status not in {404, 400, None}:
        return False
    return any(kw in err_lower for kw in (
        "model does not exist",
        "does not exist in our configuration",
        "openrouter catalog",
        "is not a valid model",
        "no such model",
        "model not found",
        "the model `",            # OpenAI-style: "The model `X` does not exist"
        "model_not_found",
        "unknown model",
    ))


def _evict_cached_clients(provider: str) -> None:
    """Drop cached auxiliary clients for a provider so fresh creds are used."""
    normalized = _normalize_aux_provider(provider)
    with _client_cache_lock:
        stale_keys = [
            key for key in _client_cache
            if _normalize_aux_provider(str(key[0])) == normalized
        ]
        for key in stale_keys:
            client = _client_cache.get(key, (None, None, None))[0]
            if client is not None:
                _force_close_async_httpx(client)
                try:
                    close_fn = getattr(client, "close", None)
                    if callable(close_fn):
                        close_fn()
                except Exception:
                    pass
            _client_cache.pop(key, None)


def _evict_cached_client_instance(target: Any) -> bool:
    """Drop the cache entry whose stored client is *target*.

    Used when a specific cached client has been poisoned (closed httpx
    transport after a timeout, broken streaming session, etc.) so the next
    auxiliary call rebuilds rather than reusing the dead instance.

    Walks both sync and async wrappers (``CodexAuxiliaryClient``,
    ``AnthropicAuxiliaryClient``, ``AsyncCodexAuxiliaryClient``, etc.) via
    their ``_real_client`` attribute so a timeout that closes the underlying
    ``OpenAI`` (or native provider) client evicts every cached shim that
    exposed it. Async wrappers must mirror their sync sibling's
    ``_real_client`` for this to work — otherwise the sync entry is evicted
    but the async entry survives and keeps reusing the dead transport.

    Returns True when at least one entry was evicted.
    """
    if target is None:
        return False
    evicted = False
    with _client_cache_lock:
        for key in list(_client_cache.keys()):
            entry = _client_cache.get(key)
            if entry is None:
                continue
            cached = entry[0]
            if cached is None:
                continue
            real = getattr(cached, "_real_client", None)
            if cached is target or real is target:
                del _client_cache[key]
                evicted = True
    return evicted


def _pool_cache_hint(
    provider: str,
    *,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> str:
    """Return a stable cache discriminator for pooled providers."""
    normalized = _normalize_aux_provider(provider)
    if normalized == "auto":
        runtime = _normalize_main_runtime(main_runtime)
        normalized = _normalize_aux_provider(runtime.get("provider") or _read_main_provider())
    if normalized in {"", "auto", "custom"}:
        return ""
    entry = _peek_pool_entry(normalized)
    if entry is None:
        return ""
    entry_id = str(getattr(entry, "id", "") or "").strip()
    if not entry_id:
        return ""
    return f"{normalized}:{entry_id}"


def _pool_error_context(exc: Exception) -> Dict[str, Any]:
    status = getattr(exc, "status_code", None)
    payload: Dict[str, Any] = {"message": str(exc)}
    if status is not None:
        payload["status_code"] = status
    return payload


def _recoverable_pool_provider(
    resolved_provider: str,
    client: Any,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Infer which provider pool can recover the current auxiliary client."""
    normalized = _normalize_aux_provider(resolved_provider)
    if normalized not in {"", "auto", "custom"}:
        return normalized
    base = str(getattr(client, "base_url", "") or "")
    if base_url_host_matches(base, "chatgpt.com"):
        return "openai-codex"
    if base_url_host_matches(base, "openrouter.ai"):
        return "openrouter"
    if base_url_host_matches(base, "inference-api.nousresearch.com"):
        return "nous"
    if base_url_host_matches(base, "api.anthropic.com"):
        return "anthropic"
    if base_url_host_matches(base, "api.githubcopilot.com"):
        return "copilot"
    if base_url_host_matches(base, "api.kimi.com"):
        return "kimi-coding"
    if base_url_host_matches(base, "api.x.ai"):
        return "xai-oauth"
    # For api_key providers not in the hardcoded list (e.g. opencode-go), match
    # the client base URL against all registered api_key providers so that
    # credential-pool rotation works for any provider the user configured.
    if main_runtime:
        rt = _normalize_main_runtime(main_runtime)
        rt_provider = rt.get("provider", "")
        if rt_provider and rt_provider not in {"", "auto", "custom"}:
            try:
                from hermes_cli.auth import PROVIDER_REGISTRY
                pconfig = PROVIDER_REGISTRY.get(rt_provider)
                if pconfig and getattr(pconfig, "auth_type", None) == "api_key":
                    rt_base = str(getattr(pconfig, "inference_base_url", "") or "").rstrip("/")
                    if rt_base and base_url_host_matches(base, base_url_hostname(rt_base)):
                        return rt_provider
            except Exception:
                pass
    return None


def _recover_provider_pool(provider: str, exc: Exception, *, failed_api_key: str = "") -> bool:
    """Try same-provider credential-pool recovery for auxiliary calls.

    ``failed_api_key`` is the API key that was actually used for the failing
    request.  Passing it lets mark_exhausted_and_rotate identify the correct
    pool entry even when another process has already rotated the pool (which
    would leave current() as None, causing the wrong entry to be marked).
    """
    normalized = _normalize_aux_provider(provider)
    try:
        pool = load_pool(normalized)
    except Exception as load_exc:
        logger.debug("Auxiliary client: could not load pool for %s recovery: %s", normalized, load_exc)
        return False
    if not pool or not pool.has_credentials():
        return False

    status_code = getattr(exc, "status_code", None)
    error_context = _pool_error_context(exc)
    hint = failed_api_key or None

    if _is_auth_error(exc):
        refreshed = pool.try_refresh_current()
        if refreshed is not None:
            _evict_cached_clients(normalized)
            return True
        next_entry = pool.mark_exhausted_and_rotate(
            status_code=status_code if status_code is not None else 401,
            error_context=error_context,
            api_key_hint=hint,
        )
        if next_entry is not None:
            _evict_cached_clients(normalized)
            return True
        return False

    if _is_payment_error(exc) or _is_rate_limit_error(exc):
        fallback_status = 402 if _is_payment_error(exc) else 429
        next_entry = pool.mark_exhausted_and_rotate(
            status_code=status_code if status_code is not None else fallback_status,
            error_context=error_context,
            api_key_hint=hint,
        )
        if next_entry is not None:
            _evict_cached_clients(normalized)
            return True
    return False


def _retry_same_provider_sync(
    *,
    task: Optional[str],
    resolved_provider: str,
    resolved_model: Optional[str],
    resolved_base_url: Optional[str],
    resolved_api_key: Optional[str],
    resolved_api_mode: Optional[str],
    main_runtime: Optional[Dict[str, Any]],
    final_model: Optional[str],
    messages: list,
    temperature: Optional[float],
    max_tokens: Optional[int],
    tools: Optional[list],
    effective_timeout: float,
    effective_extra_body: dict,
) -> Any:
    if task == "vision":
        _, retry_client, retry_model = resolve_vision_provider_client(
            provider=resolved_provider,
            model=final_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            async_mode=False,
        )
    else:
        retry_client, retry_model = _get_cached_client(
            resolved_provider,
            resolved_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            api_mode=resolved_api_mode,
            main_runtime=main_runtime,
        )
    if retry_client is None:
        raise RuntimeError(
            f"Auxiliary {task or 'call'}: provider {resolved_provider} could not be rebuilt after recovery"
        )

    retry_base = str(getattr(retry_client, "base_url", "") or "")
    retry_kwargs = _build_call_kwargs(
        resolved_provider,
        retry_model or final_model,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        timeout=effective_timeout,
        extra_body=effective_extra_body,
        base_url=retry_base or resolved_base_url,
    )
    if _is_anthropic_compat_endpoint(resolved_provider, retry_base):
        retry_kwargs["messages"] = _convert_openai_images_to_anthropic(retry_kwargs["messages"])
    return _validate_llm_response(
        retry_client.chat.completions.create(**retry_kwargs), task,
    )


async def _retry_same_provider_async(
    *,
    task: Optional[str],
    resolved_provider: str,
    resolved_model: Optional[str],
    resolved_base_url: Optional[str],
    resolved_api_key: Optional[str],
    resolved_api_mode: Optional[str],
    final_model: Optional[str],
    messages: list,
    temperature: Optional[float],
    max_tokens: Optional[int],
    tools: Optional[list],
    effective_timeout: float,
    effective_extra_body: dict,
) -> Any:
    if task == "vision":
        _, retry_client, retry_model = resolve_vision_provider_client(
            provider=resolved_provider,
            model=final_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            async_mode=True,
        )
    else:
        retry_client, retry_model = _get_cached_client(
            resolved_provider,
            resolved_model,
            async_mode=True,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            api_mode=resolved_api_mode,
        )
    if retry_client is None:
        raise RuntimeError(
            f"Auxiliary {task or 'call'}: provider {resolved_provider} could not be rebuilt after recovery"
        )

    retry_base = str(getattr(retry_client, "base_url", "") or "")
    retry_kwargs = _build_call_kwargs(
        resolved_provider,
        retry_model or final_model,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        timeout=effective_timeout,
        extra_body=effective_extra_body,
        base_url=retry_base or resolved_base_url,
    )
    if _is_anthropic_compat_endpoint(resolved_provider, retry_base):
        retry_kwargs["messages"] = _convert_openai_images_to_anthropic(retry_kwargs["messages"])
    return _validate_llm_response(
        await retry_client.chat.completions.create(**retry_kwargs), task,
    )


def _refresh_provider_credentials(provider: str) -> bool:
    """Refresh short-lived credentials for OAuth-backed auxiliary providers."""
    normalized = _normalize_aux_provider(provider)
    try:
        if normalized == "openai-codex":
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials(force_refresh=True)
            if not str(creds.get("api_key", "") or "").strip():
                return False
            _evict_cached_clients(normalized)
            return True
        if normalized == "nous":
            from hermes_cli.auth import resolve_nous_runtime_credentials

            creds = resolve_nous_runtime_credentials(
                timeout_seconds=env_float("HERMES_NOUS_TIMEOUT_SECONDS", 15),
                force_refresh=True,
            )
            if not str(creds.get("api_key", "") or "").strip():
                return False
            _evict_cached_clients(normalized)
            return True
        if normalized == "anthropic":
            from agent.anthropic_adapter import read_claude_code_credentials, _refresh_oauth_token, resolve_anthropic_token

            creds = read_claude_code_credentials()
            token = _refresh_oauth_token(creds) if isinstance(creds, dict) and creds.get("refreshToken") else None
            if not str(token or "").strip():
                token = resolve_anthropic_token()
            if not str(token or "").strip():
                return False
            _evict_cached_clients(normalized)
            return True
        if normalized == "xai-oauth":
            # Preference: pool-level refresh (uses refresh_token from pool entry),
            # then fall back to singleton auth-store resolver.
            pool = load_pool(normalized)
            if pool and pool.has_credentials():
                # Ensure a current entry is selected before trying to refresh.
                pool.select()
                refreshed = pool.try_refresh_current()
                if refreshed is not None and str(getattr(refreshed, "runtime_api_key", "") or "").strip():
                    _evict_cached_clients(normalized)
                    return True
            from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

            creds = resolve_xai_oauth_runtime_credentials(force_refresh=True)
            if not str(creds.get("api_key", "") or "").strip():
                return False
            _evict_cached_clients(normalized)
            return True
    except Exception as exc:
        logger.debug("Auxiliary provider credential refresh failed for %s: %s", normalized, exc)
        return False
    return False


def _try_payment_fallback(
    failed_provider: str,
    task: str = None,
    reason: str = "payment error",
) -> Tuple[Optional[Any], Optional[str], str]:
    """Try alternative providers after a payment/credit or connection error.

    Iterates the standard auto-detection chain, skipping the provider that
    failed.

    Returns:
        (client, model, provider_label) or (None, None, "") if no fallback.
    """
    # Normalise the failed provider label for matching.
    skip = failed_provider.lower().strip()
    # Also skip Step-1 main-provider path if it maps to the same backend.
    # (e.g. main_provider="openrouter" → skip "openrouter" in chain)
    main_provider = _read_main_provider()
    skip_labels = {skip}
    if main_provider and main_provider.lower() in skip:
        skip_labels.add(main_provider.lower())
    # Map common resolved_provider values back to chain labels.
    _alias_to_label = {"openrouter": "openrouter", "nous": "nous",
                       "openai-codex": "openai-codex", "codex": "openai-codex",
                       "custom": "local/custom", "local/custom": "local/custom"}
    skip_chain_labels = {_alias_to_label.get(s, s) for s in skip_labels}

    tried = []
    for label, try_fn in _get_provider_chain():
        if label in skip_chain_labels:
            continue
        if _is_provider_unhealthy(label):
            _log_skip_unhealthy(label, task)
            tried.append(f"{label} (unhealthy)")
            continue
        client, model = try_fn()
        if client is not None:
            logger.info(
                "Auxiliary %s: %s on %s — falling back to %s (%s)",
                task or "call", reason, failed_provider, label, model or "default",
            )
            return client, model, label
        tried.append(label)

    logger.warning(
        "Auxiliary %s: %s on %s and no fallback available (tried: %s)",
        task or "call", reason, failed_provider, ", ".join(tried),
    )
    return None, None, ""


def _try_main_agent_model_fallback(
    failed_provider: str,
    task: str = None,
    reason: str = "error",
) -> Tuple[Optional[Any], Optional[str], str]:
    """Last-resort fallback to the user's main agent provider + model.

    Used after the configured fallback_chain is exhausted (or empty) for
    users with an explicit auxiliary provider.  This is the "safety net"
    layer: if nothing the user asked for can serve the request, try the
    main chat model before giving up.

    Skips when the failed provider already IS the main provider (no point
    retrying the same backend that just failed).

    Returns:
        (client, model, provider_label) or (None, None, "") if no fallback.
    """
    main_provider = (_read_main_provider() or "").strip()
    main_model = (_read_main_model() or "").strip()
    if not main_provider or not main_model or main_provider.lower() in {"auto", ""}:
        return None, None, ""

    skip = (failed_provider or "").lower().strip()
    if main_provider.lower() == skip:
        # The thing that failed IS the main model — nothing to fall back to.
        return None, None, ""
    if _is_provider_unhealthy(main_provider):
        _log_skip_unhealthy(main_provider, task)
        return None, None, ""

    try:
        client, resolved_model = resolve_provider_client(
            provider=main_provider, model=main_model,
        )
    except Exception:
        client, resolved_model = None, None

    if client is None:
        return None, None, ""

    label = f"main-agent({main_provider})"
    logger.info(
        "Auxiliary %s: %s on %s — falling back to main agent model %s (%s)",
        task or "call", reason, failed_provider, label, resolved_model or main_model,
    )
    return client, resolved_model or main_model, label


def _try_configured_fallback_chain(
    task: str,
    failed_provider: str,
    reason: str = "error",
) -> Tuple[Optional[Any], Optional[str], str]:
    """Try user-configured fallback_chain for a specific auxiliary task.

    Reads auxiliary.<task>.fallback_chain from config.yaml and tries each
    entry in order.  Each entry must have at least ``provider``; ``model``,
    ``base_url``, and ``api_key`` are optional.

    Returns:
        (client, model, provider_label) or (None, None, "") if no fallback.
    """
    if not task:
        return None, None, ""

    task_config = _get_auxiliary_task_config(task)
    chain = task_config.get("fallback_chain")
    if not chain or not isinstance(chain, list):
        return None, None, ""

    skip = failed_provider.lower().strip()
    tried = []

    for i, entry in enumerate(chain):
        if not isinstance(entry, dict):
            continue
        fb_provider = str(entry.get("provider", "")).strip()
        if not fb_provider or fb_provider.lower() == skip:
            continue
        fb_model = str(entry.get("model", "")).strip() or None

        label = f"fallback_chain[{i}]({fb_provider})"

        try:
            fb_client, resolved_model = _resolve_fallback_entry(entry)
        except Exception:
            fb_client, resolved_model = None, None

        if fb_client is not None:
            logger.info(
                "Auxiliary %s: %s on %s — configured fallback to %s (%s)",
                task, reason, failed_provider, label, resolved_model or fb_model or "default",
            )
            return fb_client, resolved_model or fb_model, label
        tried.append(label)

    if tried:
        logger.debug(
            "Auxiliary %s: configured fallback_chain exhausted (tried: %s)",
            task, ", ".join(tried),
        )
    return None, None, ""


def _fallback_entry_api_key(entry: Dict[str, Any]) -> Optional[str]:
    """Resolve inline or env-backed API key from a fallback-chain entry."""
    explicit = str(entry.get("api_key") or "").strip()
    if explicit:
        return explicit
    key_env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
    if key_env:
        return os.getenv(key_env, "").strip() or None
    return None


def _resolve_fallback_entry(entry: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str]]:
    """Resolve one fallback entry through the central provider router."""
    provider = str(entry.get("provider") or "").strip()
    model = str(entry.get("model") or "").strip() or None
    if not provider or not model:
        return None, None
    base_url = str(entry.get("base_url") or "").strip() or None
    api_key = _fallback_entry_api_key(entry)
    api_mode = str(entry.get("api_mode") or entry.get("transport") or "").strip() or None
    return resolve_provider_client(
        provider,
        model=model,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
        api_mode=api_mode,
    )


def _try_main_fallback_chain(
    task: Optional[str],
    failed_provider: str = "",
    reason: str = "error",
) -> Tuple[Optional[Any], Optional[str], str]:
    """Try the top-level main-agent fallback chain for an auxiliary call.

    ``provider: auto`` auxiliary tasks should respect the user's declared
    main fallback policy before dropping into Hermes' built-in discovery
    chain. The top-level chain is read through ``get_fallback_chain`` so
    both modern ``fallback_providers`` and legacy ``fallback_model`` entries
    participate in the same order as the main agent.
    """
    try:
        from hermes_cli.config import load_config
        from hermes_cli.fallback_config import get_fallback_chain

        chain = get_fallback_chain(load_config())
    except Exception as exc:
        logger.debug("Auxiliary %s: could not load main fallback chain: %s", task or "call", exc)
        return None, None, ""

    if not chain:
        return None, None, ""

    failed_norm = (failed_provider or "").strip().lower()
    main_norm = (_read_main_provider() or "").strip().lower()
    skip = {p for p in (failed_norm, main_norm, "auto") if p}
    tried: List[str] = []

    for i, entry in enumerate(chain):
        if not isinstance(entry, dict):
            continue
        fb_provider = str(entry.get("provider") or "").strip()
        fb_model = str(entry.get("model") or "").strip()
        if not fb_provider or not fb_model:
            continue
        fb_norm = fb_provider.lower()
        label = f"fallback_providers[{i}]({fb_provider})"
        if fb_norm in skip:
            tried.append(f"{label} (skipped)")
            continue
        if _is_provider_unhealthy(fb_norm):
            _log_skip_unhealthy(fb_norm, task)
            tried.append(f"{label} (unhealthy)")
            continue
        try:
            fb_client, resolved_model = _resolve_fallback_entry(entry)
        except Exception as exc:
            logger.debug("Auxiliary %s: main fallback %s failed to resolve: %s", task or "call", label, exc)
            fb_client, resolved_model = None, None
        if fb_client is not None:
            logger.info(
                "Auxiliary %s: %s on %s — main fallback chain to %s (%s)",
                task or "call", reason, failed_provider or "auto", label,
                resolved_model or fb_model,
            )
            return fb_client, resolved_model or fb_model, fb_provider
        tried.append(label)

    if tried:
        logger.debug(
            "Auxiliary %s: main fallback chain exhausted (tried: %s)",
            task or "call", ", ".join(tried),
        )
    return None, None, ""


def _resolve_single_provider(
    provider: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[Any]:
    """Resolve a single provider entry from fallback_chain to an OpenAI client.

    Uses the existing provider resolution infrastructure where possible.
    """
    # Reuse resolve_provider_client which handles provider→client mapping.
    client, resolved_model = resolve_provider_client(
        provider=provider,
        model=model,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
    )
    return client

def _resolve_auto(
    main_runtime: Optional[Dict[str, Any]] = None,
    task: Optional[str] = None,
) -> Tuple[Optional[OpenAI], Optional[str]]:
    """Full auto-detection chain.

    Priority:
      1. User's main provider + main model, regardless of provider type.
         This means auxiliary tasks (compression, vision, web extraction,
         session search, etc.) use the same model the user configured for
         chat.  Users on OpenRouter/Nous get their chosen chat model; users
         on DeepSeek/ZAI/Alibaba get theirs; etc.  Running aux tasks on the
         user's picked model keeps behavior predictable — no surprise
         switches to a cheap fallback model for side tasks.
      2. OpenRouter → Nous → custom → Codex → API-key providers (fallback
         chain, only used when the main provider has no working client).
    """
    global auxiliary_is_nous, _stale_base_url_warned
    auxiliary_is_nous = False  # Reset — _try_nous() will set True if it wins
    runtime = _normalize_main_runtime(main_runtime)
    runtime_provider = runtime.get("provider", "")
    runtime_model = str(runtime.get("model") or "")
    runtime_base_url = str(runtime.get("base_url") or "")
    runtime_api_key = runtime.get("api_key", "")
    runtime_api_mode = str(runtime.get("api_mode") or "")

    # Fall back to process-local globals when main_runtime dict was not
    # provided or was incomplete.  ``set_runtime_main()`` now records
    # base_url/api_key/api_mode alongside provider/model, so custom:
    # providers get the full credential surface in Step 1 of the
    # auto-detect chain.
    if not runtime_base_url and _RUNTIME_MAIN_BASE_URL:
        runtime_base_url = _RUNTIME_MAIN_BASE_URL
    if not runtime_api_key and _RUNTIME_MAIN_API_KEY:
        runtime_api_key = _RUNTIME_MAIN_API_KEY
    if not runtime_api_mode and _RUNTIME_MAIN_API_MODE:
        runtime_api_mode = _RUNTIME_MAIN_API_MODE

    # ── Warn once if OPENAI_BASE_URL is set but config.yaml uses a named
    #    provider (not 'custom').  This catches the common "env poisoning"
    #    scenario where a user switches providers via `hermes model` but the
    #    old OPENAI_BASE_URL lingers in ~/.hermes/.env. ──
    if not _stale_base_url_warned:
        _env_base = os.getenv("OPENAI_BASE_URL", "").strip()
        _cfg_provider = runtime_provider or _read_main_provider()
        if (_env_base and _cfg_provider
                and _cfg_provider != "custom"
                and not _cfg_provider.startswith("custom:")):
            logger.warning(
                "OPENAI_BASE_URL is set (%s) but model.provider is '%s'. "
                "Auxiliary clients may route to the wrong endpoint. "
                "Run: hermes model to reconfigure, or remove "
                "OPENAI_BASE_URL from ~/.hermes/.env",
                _env_base, _cfg_provider,
            )
            _stale_base_url_warned = True

    # ── Step 1: main provider + main model → use them directly ──
    #
    # This is the primary aux backend for every user.  "auto" means
    # "use my main chat model for side tasks as well" — including users
    # on aggregators (OpenRouter, Nous) who previously got routed to a
    # cheap provider-side default.  Explicit per-task overrides set via
    # config.yaml (auxiliary.<task>.provider) still win over this.
    main_provider = str(runtime_provider or _read_main_provider() or "")
    main_model = str(runtime_model or _read_main_model() or "")
    if (main_provider and main_model
            and main_provider not in {"auto", ""}):
        resolved_provider = main_provider
        explicit_base_url = runtime_base_url or None
        explicit_api_key = None
        if runtime_base_url and (main_provider == "custom" or main_provider.startswith("custom:")):
            resolved_provider = "custom"
            explicit_base_url = runtime_base_url
            explicit_api_key = runtime_api_key or None
        elif runtime_api_key:
            # Pin auxiliary to the same api_key as the active main chat session
            # so that a working key is reused instead of re-selecting from the pool
            # (which might pick a different, potentially exhausted key).
            explicit_api_key = runtime_api_key
        # Skip Step-1 if the main provider was recently 402'd. The unhealthy
        # cache TTL bounds how long we bypass it, so a topped-up account
        # recovers automatically. If we tried Step-1 anyway, every aux call
        # on a depleted main provider would pay one doomed 402 RTT before
        # falling to Step-2.
        main_chain_label = _normalize_chain_label(resolved_provider)
        if main_chain_label and _is_provider_unhealthy(main_chain_label):
            _log_skip_unhealthy(main_chain_label)
        else:
            client, resolved = resolve_provider_client(
                resolved_provider,
                main_model,
                explicit_base_url=explicit_base_url,
                explicit_api_key=explicit_api_key,
                api_mode=runtime_api_mode or None,
            )
            if client is not None:
                logger.info("Auxiliary auto-detect: using main provider %s (%s)",
                            main_provider, resolved or main_model)
                return client, resolved or main_model

    # ── Step 2: user-configured fallback policy ─────────────────────────
    # In auto mode, respect the task-specific fallback chain first, then the
    # main agent's top-level fallback_providers/fallback_model chain. The
    # hardcoded provider discovery chain below is only the convenience default
    # for users who have not declared a fallback policy.
    if task:
        fb_client, fb_model, _fb_label = _try_configured_fallback_chain(
            task, main_provider or "auto", reason="main provider unavailable")
        if fb_client is not None:
            return fb_client, fb_model
    fb_client, fb_model, _fb_label = _try_main_fallback_chain(
        task, main_provider or "auto", reason="main provider unavailable")
    if fb_client is not None:
        return fb_client, fb_model

    # ── Step 3: aggregator / fallback chain ──────────────────────────────
    tried = []
    for label, try_fn in _get_provider_chain():
        if _is_provider_unhealthy(label):
            _log_skip_unhealthy(label)
            tried.append(f"{label} (unhealthy)")
            continue
        client, model = try_fn()
        if client is not None:
            if tried:
                logger.info("Auxiliary auto-detect: using %s (%s) — skipped: %s",
                            label, model or "default", ", ".join(tried))
            else:
                logger.info("Auxiliary auto-detect: using %s (%s)", label, model or "default")
            return client, model
        tried.append(label)
    logger.warning("Auxiliary auto-detect: no provider available (tried: %s). "
                   "Compression, summarization, and memory flush will not work. "
                   "Set OPENROUTER_API_KEY or configure a local model in config.yaml.",
                   ", ".join(tried))
    return None, None


# ── Centralized Provider Router ─────────────────────────────────────────────
#
# resolve_provider_client() is the single entry point for creating a properly
# configured client given a (provider, model) pair.  It handles auth lookup,
# base URL resolution, provider-specific headers, and API format differences
# (Chat Completions vs Responses API for Codex).
#
# All auxiliary consumer code should go through this or the public helpers
# below — never look up auth env vars ad-hoc.


def _to_async_client(sync_client, model: str, is_vision: bool = False):
    """Convert a sync client to its async counterpart, preserving Codex routing.

    When ``is_vision=True`` and the underlying base URL is Copilot, the
    resulting async client carries the ``Copilot-Vision-Request: true``
    header so the request is routed to Copilot's vision-capable
    infrastructure (otherwise vision payloads silently time out).
    """
    from openai import AsyncOpenAI

    if isinstance(sync_client, CodexAuxiliaryClient):
        return AsyncCodexAuxiliaryClient(sync_client), model
    if isinstance(sync_client, AnthropicAuxiliaryClient):
        return AsyncAnthropicAuxiliaryClient(sync_client), model
    try:
        from agent.gemini_native_adapter import GeminiNativeClient, AsyncGeminiNativeClient

        if isinstance(sync_client, GeminiNativeClient):
            return AsyncGeminiNativeClient(sync_client), model
    except ImportError:
        pass
    try:
        from agent.copilot_acp_client import CopilotACPClient
        if isinstance(sync_client, CopilotACPClient):
            return sync_client, model
    except ImportError:
        pass

    async_kwargs = {
        "api_key": sync_client.api_key,
        "base_url": str(sync_client.base_url),
    }
    sync_base_url = str(sync_client.base_url)
    if base_url_host_matches(sync_base_url, "openrouter.ai"):
        async_kwargs["default_headers"] = build_or_headers()
    elif base_url_host_matches(sync_base_url, "api.githubcopilot.com"):
        from hermes_cli.copilot_auth import copilot_request_headers

        async_kwargs["default_headers"] = copilot_request_headers(
            is_agent_turn=True, is_vision=is_vision
        )
    elif base_url_host_matches(sync_base_url, "api.kimi.com"):
        async_kwargs["default_headers"] = {"User-Agent": "claude-code/0.1.0"}
    elif base_url_host_matches(sync_base_url, "integrate.api.nvidia.com"):
        async_kwargs["default_headers"] = build_nvidia_nim_headers(sync_base_url)
    else:
        # Fall back to profile.default_headers for providers that declare
        # client-level headers on their ProviderProfile (e.g. attribution
        # User-Agent strings). Provider is inferred from the hostname.
        try:
            from agent.model_metadata import _infer_provider_from_url
            from providers import get_provider_profile as _gpf_async
            _inferred = _infer_provider_from_url(sync_base_url)
            if _inferred:
                _ph_async = _gpf_async(_inferred)
                if _ph_async and _ph_async.default_headers:
                    async_kwargs["default_headers"] = dict(_ph_async.default_headers)
        except Exception:
            pass
    _merged_async = _apply_user_default_headers(async_kwargs.get("default_headers"))
    if _merged_async:
        async_kwargs["default_headers"] = _merged_async
    return AsyncOpenAI(**async_kwargs), model


def _normalize_resolved_model(model_name: Optional[str], provider: str) -> Optional[str]:
    """Normalize a resolved model for the provider that will receive it."""
    if not model_name:
        return model_name
    try:
        from hermes_cli.model_normalize import normalize_model_for_provider

        return normalize_model_for_provider(model_name, provider)
    except Exception:
        return model_name


def resolve_provider_client(
    provider: str,
    model: str = None,
    async_mode: bool = False,
    raw_codex: bool = False,
    explicit_base_url: str = None,
    explicit_api_key: str = None,
    api_mode: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    is_vision: bool = False,
    task: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Central router: given a provider name and optional model, return a
    configured client with the correct auth, base URL, and API format.

    The returned client always exposes ``.chat.completions.create()`` — for
    Codex/Responses API providers, an adapter handles the translation
    transparently.

    Args:
        provider: Provider identifier.  One of:
            "openrouter", "nous", "openai-codex" (or "codex"),
            "zai", "kimi-coding", "minimax", "minimax-cn",
            "custom" (OPENAI_BASE_URL + OPENAI_API_KEY),
            "auto" (full auto-detection chain).
        model: Model slug override.  If None, uses the provider's default
               auxiliary model.
        async_mode: If True, return an async-compatible client.
        raw_codex: If True, return a raw OpenAI client for Codex providers
            instead of wrapping in CodexAuxiliaryClient.  Use this when
            the caller needs direct access to responses.stream() (e.g.,
            the main agent loop).
        explicit_base_url: Optional direct OpenAI-compatible endpoint.
        explicit_api_key: Optional API key paired with explicit_base_url.
        api_mode: API mode override.  One of "chat_completions",
            "codex_responses", or None (auto-detect).  When set to
            "codex_responses", the client is wrapped in
            CodexAuxiliaryClient to route through the Responses API.

    Returns:
        (client, resolved_model) or (None, None) if auth is unavailable.
    """
    _validate_proxy_env_urls()
    # Preserve the original provider name before alias normalization so a
    # user-declared ``custom_providers`` entry whose name coincidentally
    # matches a built-in alias (e.g. user names their custom provider "kimi"
    # which aliases to "kimi-coding") is still reachable via the named-custom
    # branch below.
    original_provider = (provider or "").strip().lower()
    # Normalise aliases
    provider = _normalize_aux_provider(provider)

    # Universal model-resolution fallback chain.  Callers (notably title
    # generation, vision, session search, and other auxiliary tasks) can
    # reach this function without an explicit model — the user picked their
    # main provider, didn't bother configuring a per-task ``auxiliary.<task>.model``,
    # and just expects "use my main model for side tasks too."  Resolve in
    # this order, stopping at the first non-empty answer:
    #
    #   1. ``model`` argument (caller knew what they wanted)
    #   2. Provider's catalog default — cheap/fast model the provider
    #      registered via ``ProviderProfile.default_aux_model`` or the
    #      legacy ``_API_KEY_PROVIDER_AUX_MODELS_FALLBACK`` dict.  Empty
    #      string for OAuth-gated providers (openai-codex, xai-oauth)
    #      whose accepted-model lists drift on the backend, so we don't
    #      pin a default that can silently rot.
    #   3. User's main model from ``model.model`` in config.yaml.  This is
    #      the load-bearing step for OAuth providers: an xai-oauth user
    #      with grok-4.3 configured gets grok-4.3 for title generation
    #      instead of silently dropping to whatever Step-2 fallback (#31845).
    #
    # Each provider branch below sees a non-empty ``model`` whenever the
    # user has *anything* configured — no provider-specific empty-model
    # guards needed.  When the user has NOTHING configured (fresh install,
    # main_model also empty), the branches still hit their own
    # missing-credentials returns and ``_resolve_auto`` falls through to
    # the Step-2 chain as before.
    if not model:
        model = _get_aux_model_for_provider(provider) or _read_main_model() or model

    def _needs_codex_wrap(client_obj, base_url_str: str, model_str: str) -> bool:
        """Decide if a plain OpenAI client should be wrapped for Responses API.

        Returns True when api_mode is explicitly "codex_responses", or when
        auto-detection (api.openai.com + codex-family model) suggests it.
        Already-wrapped clients (CodexAuxiliaryClient) are skipped.
        """
        if isinstance(client_obj, CodexAuxiliaryClient):
            return False
        if raw_codex:
            return False
        if api_mode == "codex_responses":
            return True
        # Auto-detect: api.openai.com + codex model name pattern
        if api_mode and api_mode != "codex_responses":
            return False  # explicit non-codex mode
        if base_url_hostname(base_url_str) == "api.openai.com":
            model_lower = (model_str or "").lower()
            if "codex" in model_lower:
                return True
        return False

    def _wrap_if_needed(client_obj, final_model_str: str, base_url_str: str = "",
                        api_key_str: str = ""):
        """Wrap a plain OpenAI client in the correct transport adapter.

        Handles two cases:
        - ``CodexAuxiliaryClient`` when the endpoint needs the Responses API
          (explicit ``api_mode=codex_responses`` or api.openai.com + codex
          model name).
        - ``AnthropicAuxiliaryClient`` when the endpoint speaks Anthropic
          Messages (explicit ``api_mode=anthropic_messages``, any ``/anthropic``
          suffix, ``api.kimi.com/coding``, or ``api.anthropic.com``).

        Clients that are already specialized wrappers pass through unchanged.
        """
        if _needs_codex_wrap(client_obj, base_url_str, final_model_str):
            logger.debug(
                "resolve_provider_client: wrapping client in CodexAuxiliaryClient "
                "(api_mode=%s, model=%s, base_url=%s)",
                api_mode or "auto-detected", final_model_str,
                base_url_str[:60] if base_url_str else "")
            return CodexAuxiliaryClient(client_obj, final_model_str)
        # Anthropic-wire endpoints: rewrap plain OpenAI clients so
        # chat.completions.create() is translated to /v1/messages.
        return _maybe_wrap_anthropic(
            client_obj, final_model_str, api_key_str, base_url_str, api_mode,
        )

    # ── Auto: try all providers in priority order ────────────────────
    if provider == "auto":
        client, resolved = _resolve_auto(main_runtime=main_runtime, task=task)
        if client is None:
            return None, None
        # When auto-detection lands on a non-OpenRouter provider (e.g. a
        # local server), an OpenRouter-formatted model override like
        # "google/gemini-3-flash-preview" won't work.  Drop it and use
        # the provider's own default model instead.
        if model and "/" in model and resolved and "/" not in resolved:
            logger.debug(
                "Dropping OpenRouter-format model %r for non-OpenRouter "
                "auxiliary provider (using %r instead)", model, resolved)
            model = None
        final_model = model or resolved
        return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                else (client, final_model))

    # ── OpenRouter ───────────────────────────────────────────
    if provider == "openrouter":
        client, default = _try_openrouter(explicit_api_key=explicit_api_key)
        if client is None:
            logger.warning(
                "resolve_provider_client: openrouter requested but %s",
                _describe_openrouter_unavailable(),
            )
            return None, None
        final_model = _normalize_resolved_model(model or default, provider)
        return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                else (client, final_model))

    # ── Nous Portal (OAuth) ──────────────────────────────────────────
    if provider == "nous":
        # Detect vision tasks: either explicit model override from
        # _PROVIDER_VISION_MODELS, or caller passed a known vision model.
        _is_vision = (
            model in _PROVIDER_VISION_MODELS.values()
            or (model or "").strip().lower() == "mimo-v2-omni"
        )
        client, default = _try_nous(vision=_is_vision)
        if client is None:
            logger.warning("resolve_provider_client: nous requested "
                           "but Nous Portal not configured (run: hermes auth)")
            return None, None
        final_model = _normalize_resolved_model(model or default, provider)
        return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                else (client, final_model))

    # ── OpenAI Codex (OAuth → Responses API) ─────────────────────────
    if provider == "openai-codex":
        if not model:
            logger.warning(
                "resolve_provider_client: openai-codex requested without a "
                "model; pass model explicitly (e.g. model.model in config.yaml "
                "or auxiliary.<task>.model for per-task aux routing)."
            )
            return None, None
        if raw_codex:
            # Return the raw OpenAI client for callers that need direct
            # access to responses.stream() (e.g., the main agent loop).
            codex_token = _read_codex_access_token()
            if not codex_token:
                logger.warning("resolve_provider_client: openai-codex requested "
                               "but no Codex OAuth token found (run: hermes model)")
                return None, None
            final_model = _normalize_resolved_model(model, provider)
            raw_client = OpenAI(
                api_key=codex_token,
                base_url=_CODEX_AUX_BASE_URL,
                default_headers=_codex_cloudflare_headers(codex_token),
            )
            return (raw_client, final_model)
        # Standard path: wrap in CodexAuxiliaryClient adapter
        client, default = _build_codex_client(model)
        if client is None:
            logger.warning("resolve_provider_client: openai-codex requested "
                           "but no Codex OAuth token found (run: hermes model)")
            return None, None
        final_model = _normalize_resolved_model(model or default, provider)
        return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                else (client, final_model))

    # ── xAI Grok OAuth (loopback PKCE → Responses API) ───────────────
    # Without this branch, an xai-oauth main provider falls through to the
    # generic ``oauth_external`` arm below and returns ``(None, None)``,
    # silently re-routing every auxiliary task (compression, web extract,
    # session search, curator, etc.) to whatever Step-2 fallback the user
    # has configured.  Users on xAI Grok OAuth would then see surprise
    # OpenRouter / Nous bills for side tasks they thought were running on
    # their xAI subscription.
    if provider == "xai-oauth":
        client, default = _build_xai_oauth_aux_client(model)
        if client is None:
            logger.warning(
                "resolve_provider_client: xai-oauth requested but no xAI "
                "OAuth token found (run: hermes model -> xAI Grok OAuth — SuperGrok / Premium+)"
            )
            return None, None
        final_model = _normalize_resolved_model(model or default, provider)
        return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                else (client, final_model))

    # ── Custom endpoint (OPENAI_BASE_URL + OPENAI_API_KEY) ───────────
    if provider == "custom":
        if explicit_base_url:
            custom_base = _to_openai_base_url(explicit_base_url).strip()
            custom_key = (
                (explicit_api_key or "").strip()
                or os.getenv("OPENAI_API_KEY", "").strip()
                or "no-key-required"  # local servers don't need auth
            )
            if not custom_base:
                logger.warning(
                    "resolve_provider_client: explicit custom endpoint requested "
                    "but base_url is empty"
                )
                return None, None
            final_model = _normalize_resolved_model(
                model or (main_runtime.get("model") if main_runtime else None) or "gpt-4o-mini",
                provider,
            )
            extra = {}
            _clean_base, _dq = _extract_url_query_params(custom_base)
            if _dq:
                extra["default_query"] = _dq
            if base_url_host_matches(custom_base, "api.kimi.com"):
                extra["default_headers"] = {"User-Agent": "claude-code/0.1.0"}
            elif base_url_host_matches(custom_base, "api.githubcopilot.com"):
                from hermes_cli.copilot_auth import copilot_request_headers
                extra["default_headers"] = copilot_request_headers(
                    is_agent_turn=True, is_vision=is_vision
                )
            elif base_url_host_matches(custom_base, "integrate.api.nvidia.com"):
                extra["default_headers"] = build_nvidia_nim_headers(custom_base)
            else:
                # Fall back to profile.default_headers for providers that
                # declare client-level attribution headers on their profile.
                try:
                    from providers import get_provider_profile as _gpf_custom
                    _ph_custom = _gpf_custom(provider)
                    if _ph_custom and _ph_custom.default_headers:
                        extra["default_headers"] = dict(_ph_custom.default_headers)
                except Exception:
                    pass
            _merged_custom = _apply_user_default_headers(extra.get("default_headers"))
            if _merged_custom:
                extra["default_headers"] = _merged_custom
            client = OpenAI(api_key=custom_key, base_url=_clean_base, **extra)
            client = _wrap_if_needed(client, final_model, custom_base, custom_key)
            return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                    else (client, final_model))
        # Try custom first, then API-key providers (Codex excluded here:
        # falling through to Codex with no model is a stale-constant trap).
        for try_fn in (_try_custom_endpoint, _resolve_api_key_provider):
            client, default = try_fn()
            if client is not None:
                final_model = _normalize_resolved_model(model or default, provider)
                _cbase = str(getattr(client, "base_url", "") or "")
                # ``client.api_key`` may be a callable (Azure Foundry Entra
                # bearer provider). Pass empty string for the wrapper-detection
                # path — wrapping decisions are based on base_url + api_mode.
                _raw_ckey = getattr(client, "api_key", "")
                _ckey = "" if (callable(_raw_ckey) and not isinstance(_raw_ckey, str)) else str(_raw_ckey or "")
                client = _wrap_if_needed(client, final_model, _cbase, _ckey)
                return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                        else (client, final_model))
        logger.warning("resolve_provider_client: custom/main requested "
                       "but no endpoint credentials found")
        return None, None

    # ── Named custom providers (config.yaml providers dict / custom_providers list) ───
    try:
        from hermes_cli.runtime_provider import _get_named_custom_provider
        # When the raw requested name is an alias (``kimi`` → ``kimi-coding``)
        # and the user defined a ``custom_providers`` entry under that alias
        # name, the custom entry is the intended target — the built-in alias
        # rewriting would otherwise hijack the request.  Only preferred when
        # the raw name is an alias (not a canonical provider name) so custom
        # entries that coincidentally match a canonical provider (e.g. ``nous``)
        # still defer to the built-in per `_get_named_custom_provider`'s guard.
        custom_entry = None
        if original_provider and original_provider != provider:
            custom_entry = _get_named_custom_provider(original_provider)
        if custom_entry is None:
            custom_entry = _get_named_custom_provider(provider)
        if custom_entry:
            custom_base = custom_entry.get("base_url", "").strip()
            custom_key = custom_entry.get("api_key", "").strip()
            custom_key_env = (custom_entry.get("key_env") or custom_entry.get("api_key_env") or "").strip()
            if not custom_key and custom_key_env:
                custom_key = os.getenv(custom_key_env, "").strip()
            custom_key = custom_key or "no-key-required"
            if custom_key == "no-key-required":
                logger.warning(
                    "resolve_provider_client: named custom provider %r has no resolvable "
                    "api_key — request will be sent with placeholder no-key-required "
                    "and will 401 on auth-required endpoints",
                    custom_entry.get("name") or provider,
                )
            # An explicit per-task api_mode override (from _resolve_task_provider_model)
            # wins; otherwise fall back to what the provider entry declared.
            entry_api_mode = (api_mode or custom_entry.get("api_mode") or "").strip()
            if custom_base:
                final_model = _normalize_resolved_model(
                    model
                    or custom_entry.get("model")
                    or (main_runtime.get("model") if main_runtime else None)
                    or _read_main_model()
                    or "gpt-4o-mini",
                    provider,
                )
                # anthropic_messages talks to the /anthropic surface directly;
                # OpenAI-wire paths (chat_completions / codex_responses) need the
                # /v1 equivalent.  Rewrite only on the OpenAI-wire path so the
                # Anthropic fallback SDK still sees the original URL.
                if entry_api_mode == "anthropic_messages":
                    openai_base = custom_base
                    raw_base_for_wrap = custom_base
                else:
                    openai_base = _to_openai_base_url(custom_base)
                    raw_base_for_wrap = custom_base
                _clean_base2, _dq2 = _extract_url_query_params(openai_base)
                _extra2 = {"default_query": _dq2} if _dq2 else {}
                _headers2 = _apply_user_default_headers(_extra2.get("default_headers"))
                if _headers2:
                    _extra2["default_headers"] = _headers2
                logger.debug(
                    "resolve_provider_client: named custom provider %r (%s, api_mode=%s)",
                    provider, final_model, entry_api_mode or "chat_completions")
                # anthropic_messages: route through the Anthropic Messages API
                # via AnthropicAuxiliaryClient. Mirrors the anonymous-custom
                # branch in _try_custom_endpoint(). See #15033.
                if entry_api_mode == "anthropic_messages":
                    try:
                        from agent.anthropic_adapter import build_anthropic_client
                        real_client = build_anthropic_client(custom_key, custom_base)
                    except ImportError:
                        logger.warning(
                            "Named custom provider %r declares api_mode="
                            "anthropic_messages but the anthropic SDK is not "
                            "installed — falling back to OpenAI-wire.",
                            provider,
                        )
                        # Fallback went OpenAI-wire after all — redo the query
                        # extraction against the rewritten /v1 URL.
                        _fallback_base = _to_openai_base_url(custom_base)
                        _fb_clean, _fb_dq = _extract_url_query_params(_fallback_base)
                        _fb_extra = {"default_query": _fb_dq} if _fb_dq else {}
                        _fb_headers = _apply_user_default_headers(_fb_extra.get("default_headers"))
                        if _fb_headers:
                            _fb_extra["default_headers"] = _fb_headers
                        client = OpenAI(api_key=custom_key, base_url=_fb_clean, **_fb_extra)
                        return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                                else (client, final_model))
                    sync_anthropic = AnthropicAuxiliaryClient(
                        real_client, final_model, custom_key, custom_base, is_oauth=False,
                    )
                    if async_mode:
                        return AsyncAnthropicAuxiliaryClient(sync_anthropic), final_model
                    return sync_anthropic, final_model
                client = OpenAI(api_key=custom_key, base_url=_clean_base2, **_extra2)
                # codex_responses or inherited auto-detect (via _wrap_if_needed).
                # _wrap_if_needed reads the closed-over `api_mode` (the task-level
                # override). Named-provider entry api_mode=codex_responses also
                # flows through here.
                if entry_api_mode == "codex_responses" and not isinstance(
                    client, CodexAuxiliaryClient
                ):
                    client = CodexAuxiliaryClient(client, final_model)
                else:
                    client = _wrap_if_needed(client, final_model, raw_base_for_wrap, custom_key)
                return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                        else (client, final_model))
            logger.warning(
                "resolve_provider_client: named custom provider %r has no base_url",
                provider)
            return None, None
    except ImportError:
        pass

    # ── Azure Foundry (delegates to runtime resolver for auth_mode-aware routing) ─
    #
    # The generic PROVIDER_REGISTRY path below uses
    # ``resolve_api_key_provider_credentials`` which only knows about the
    # static ``AZURE_FOUNDRY_API_KEY`` env var. That misses two important
    # cases for the ``azure-foundry`` provider:
    #
    #   1. ``model.auth_mode: entra_id`` — no static key exists; we need
    #      a callable bearer-token provider from ``azure_identity_adapter``.
    #   2. Non-default ``model.base_url`` (Foundry projects path) — the
    #      env-var-only resolver doesn't apply config-yaml-driven URL
    #      overrides.
    #
    # Delegate to the same runtime resolver the main agent uses so
    # auxiliary tasks (title generation, compression, vision, embedding,
    # session search) inherit the user's full Azure config.
    if provider == "azure-foundry":
        client, default_model = _try_azure_foundry(
            model=model,
            explicit_api_key=explicit_api_key,
            explicit_base_url=explicit_base_url,
            api_mode=api_mode,
        )
        if client is None:
            logger.warning(
                "resolve_provider_client: azure-foundry requested but "
                "runtime resolution failed (run: hermes doctor for "
                "diagnostics)"
            )
            return None, None
        final_model = _normalize_resolved_model(model or default_model, provider)
        return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                else (client, final_model))

    # ── API-key providers from PROVIDER_REGISTRY ─────────────────────
    try:
        from hermes_cli.auth import (
            PROVIDER_REGISTRY,
            resolve_api_key_provider_credentials,
            resolve_external_process_provider_credentials,
        )
    except ImportError:
        logger.debug("hermes_cli.auth not available for provider %s", provider)
        return None, None

    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig is None:
        logger.warning("resolve_provider_client: unknown provider %r", provider)
        return None, None

    if pconfig.auth_type == "api_key":
        if provider == "anthropic":
            client, default_model = _try_anthropic(explicit_api_key=explicit_api_key)
            if client is None:
                logger.warning("resolve_provider_client: anthropic requested but no Anthropic credentials found")
                return None, None
            final_model = _normalize_resolved_model(model or default_model, provider)
            return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode else (client, final_model))

        creds = resolve_api_key_provider_credentials(provider)
        api_key = str(creds.get("api_key", "")).strip()
        # Honour an explicit api_key override (e.g. from a fallback_model entry
        # or a custom_providers entry) so callers that pass an explicit
        # credential can authenticate against endpoints where no built-in
        # credential is registered for this provider alias.
        if explicit_api_key:
            api_key = explicit_api_key.strip() or api_key
        if not api_key:
            tried_sources = list(pconfig.api_key_env_vars)
            if provider == "copilot":
                tried_sources.append("gh auth token")
            logger.debug("resolve_provider_client: provider %s has no API "
                         "key configured (tried: %s)",
                         provider, ", ".join(tried_sources))
            return None, None

        raw_base_url = str(creds.get("base_url", "")).strip().rstrip("/") or pconfig.inference_base_url
        base_url = _to_openai_base_url(raw_base_url)
        # Honour an explicit base_url override from the caller — used when a
        # fallback_model entry (or custom_providers lookup) routes through a
        # built-in provider name but targets a user-specified endpoint.
        if explicit_base_url:
            base_url = _to_openai_base_url(explicit_base_url.strip().rstrip("/"))

        default_model = _get_aux_model_for_provider(provider)
        final_model = _normalize_resolved_model(model or default_model, provider)

        if provider == "gemini":
            from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url

            if is_native_gemini_base_url(base_url):
                client = GeminiNativeClient(api_key=api_key, base_url=base_url)
                logger.debug("resolve_provider_client: %s (%s)", provider, final_model)
                return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                        else (client, final_model))

        # Provider-specific headers
        headers = {}
        if base_url_host_matches(base_url, "api.kimi.com"):
            headers["User-Agent"] = "claude-code/0.1.0"
        elif base_url_host_matches(base_url, "api.githubcopilot.com"):
            from hermes_cli.copilot_auth import copilot_request_headers

            headers.update(copilot_request_headers(
                is_agent_turn=True, is_vision=is_vision
            ))
        elif base_url_host_matches(base_url, "integrate.api.nvidia.com"):
            headers.update(build_nvidia_nim_headers(base_url))
        else:
            # Fall back to profile.default_headers for providers that declare
            # client-level attribution headers on their profile (e.g. GMI
            # User-Agent for traffic identification).
            try:
                from providers import get_provider_profile as _gpf_main
                _ph_main = _gpf_main(provider)
                if _ph_main and _ph_main.default_headers:
                    headers.update(_ph_main.default_headers)
            except Exception:
                pass
        _merged_main = _apply_user_default_headers(headers)
        if _merged_main:
            headers = _merged_main
        client = OpenAI(api_key=api_key, base_url=base_url,
                        **({"default_headers": headers} if headers else {}))

        # Copilot GPT-5+ models (except gpt-5-mini) require the Responses
        # API — they are not accessible via /chat/completions.  Wrap the
        # plain client in CodexAuxiliaryClient so call_llm() transparently
        # routes through responses.stream().
        if provider == "copilot" and final_model and not raw_codex:
            try:
                from hermes_cli.models import _should_use_copilot_responses_api
                if _should_use_copilot_responses_api(final_model):
                    logger.debug(
                        "resolve_provider_client: copilot model %s needs "
                        "Responses API — wrapping with CodexAuxiliaryClient",
                        final_model)
                    client = CodexAuxiliaryClient(client, final_model)
            except ImportError:
                pass

        # Honor api_mode for any API-key provider (e.g. direct OpenAI with
        # codex-family models).  The copilot-specific wrapping above handles
        # copilot; this covers the general case (#6800).  Also rewraps
        # Anthropic-wire endpoints (Kimi Coding Plan api.kimi.com/coding,
        # /anthropic-suffixed gateways) so named providers like kimi-coding
        # land on the right transport without needing per-provider branches.
        client = _wrap_if_needed(client, final_model, raw_base_url, api_key)

        logger.debug("resolve_provider_client: %s (%s)", provider, final_model)
        return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                else (client, final_model))

    if pconfig.auth_type == "external_process":
        creds = resolve_external_process_provider_credentials(provider)
        final_model = _normalize_resolved_model(
            model
            or (main_runtime.get("model") if main_runtime else None)
            or _read_main_model(),
            provider,
        )
        if provider == "copilot-acp":
            api_key = str(creds.get("api_key", "")).strip()
            base_url = str(creds.get("base_url", "")).strip()
            command = str(creds.get("command", "")).strip() or None
            args = list(creds.get("args") or [])
            if not final_model:
                logger.warning(
                    "resolve_provider_client: copilot-acp requested but no model "
                    "was provided or configured"
                )
                return None, None
            if not api_key or not base_url:
                logger.warning(
                    "resolve_provider_client: copilot-acp requested but external "
                    "process credentials are incomplete"
                )
                return None, None
            from agent.copilot_acp_client import CopilotACPClient

            client = CopilotACPClient(
                api_key=api_key,
                base_url=base_url,
                command=command,
                args=args,
            )
            logger.debug("resolve_provider_client: %s (%s)", provider, final_model)
            return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                    else (client, final_model))
        logger.warning("resolve_provider_client: external-process provider %s not "
                       "directly supported", provider)
        return None, None

    elif pconfig.auth_type == "aws_sdk":
        # AWS SDK providers (Bedrock) — use the Anthropic Bedrock client via
        # boto3's credential chain (IAM roles, SSO, env vars, instance metadata).
        try:
            from agent.bedrock_adapter import has_aws_credentials, resolve_bedrock_region
            from agent.anthropic_adapter import build_anthropic_bedrock_client
        except ImportError:
            logger.warning("resolve_provider_client: bedrock requested but "
                           "boto3 or anthropic SDK not installed")
            return None, None

        if not has_aws_credentials():
            logger.debug("resolve_provider_client: bedrock requested but "
                         "no AWS credentials found")
            return None, None

        region = resolve_bedrock_region()
        default_model = "anthropic.claude-haiku-4-5-20251001-v1:0"
        final_model = _normalize_resolved_model(model or default_model, provider)
        try:
            real_client = build_anthropic_bedrock_client(region)
        except ImportError as exc:
            logger.warning("resolve_provider_client: cannot create Bedrock "
                           "client: %s", exc)
            return None, None
        client = AnthropicAuxiliaryClient(
            real_client, final_model, api_key="aws-sdk",
            base_url=f"https://bedrock-runtime.{region}.amazonaws.com",
        )
        logger.debug("resolve_provider_client: bedrock (%s, %s)", final_model, region)
        return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                else (client, final_model))

    elif pconfig.auth_type in {"oauth_device_code", "oauth_external"}:
        # OAuth providers — route through their specific try functions
        if provider == "nous":
            return resolve_provider_client("nous", model, async_mode)
        if provider == "openai-codex":
            return resolve_provider_client("openai-codex", model, async_mode)
        if provider == "xai-oauth":
            return resolve_provider_client("xai-oauth", model, async_mode)
        # Other OAuth providers not directly supported
        logger.warning("resolve_provider_client: OAuth provider %s not "
                       "directly supported, try 'auto'", provider)
        return None, None

    logger.warning("resolve_provider_client: unhandled auth_type %s for %s",
                   pconfig.auth_type, provider)
    return None, None


# ── Public API ──────────────────────────────────────────────────────────────

def get_text_auxiliary_client(
    task: str = "",
    *,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[OpenAI], Optional[str]]:
    """Return (client, default_model_slug) for text-only auxiliary tasks.

    Args:
        task: Optional task name ("compression", "web_extract") to check
              for a task-specific provider override.

    Callers may override the returned model via config.yaml
    (e.g. auxiliary.compression.model, auxiliary.web_extract.model).
    """
    provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(task or None)
    return resolve_provider_client(
        provider,
        model=model,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
    )


def get_async_text_auxiliary_client(task: str = "", *, main_runtime: Optional[Dict[str, Any]] = None):
    """Return (async_client, model_slug) for async consumers.

    For standard providers returns (AsyncOpenAI, model). For Codex returns
    (AsyncCodexAuxiliaryClient, model) which wraps the Responses API.
    Returns (None, None) when no provider is available.
    """
    provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(task or None)
    return resolve_provider_client(
        provider,
        model=model,
        async_mode=True,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
    )


_VISION_AUTO_PROVIDER_ORDER = (
    "openrouter",
    "nous",
)


def _main_model_supports_vision(provider: str, model: Optional[str]) -> bool:
    """Return True when ``provider``/``model`` is known to accept image input.

    Used by the vision auto-detect chain to skip the user's main provider
    when it's known to be text-only (e.g. DeepSeek, gpt-oss without vision).
    Without this guard, ``resolve_vision_provider_client(provider="auto")``
    would happily return the main-provider client and any subsequent image
    payload would surface as a cryptic provider-side error
    (``unknown variant `image_url`, expected `text```, #31179).

    Returns True when capability lookup is unknown — preserves the historical
    behaviour of attempting the call, so providers we haven't catalogued yet
    don't silently regress to text-only.
    """
    try:
        from agent.image_routing import _lookup_supports_vision
        from hermes_cli.config import load_config
    except ImportError:
        return True
    try:
        supports = _lookup_supports_vision(provider, model, load_config())
    except Exception:  # pragma: no cover - defensive
        return True
    if supports is None:
        # No capability data — keep current behaviour and let the call attempt
        # happen rather than silently skipping. This avoids false-positive
        # skips for new/custom providers.
        return True
    return bool(supports)


def _normalize_vision_provider(provider: Optional[str]) -> str:
    return _normalize_aux_provider(provider)


def _resolve_strict_vision_backend(
    provider: str,
    model: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    provider = _normalize_vision_provider(provider)
    if provider == "copilot":
        return resolve_provider_client("copilot", model, is_vision=True)
    if provider == "openrouter":
        return _try_openrouter(model=model)
    if provider == "nous":
        return _try_nous(vision=True)
    if provider == "openai-codex":
        # Route through resolve_provider_client so the caller's explicit
        # model is used.  There is no safe default Codex model (shifting
        # allow-list); callers must specify via auxiliary.<task>.model.
        return resolve_provider_client("openai-codex", model, is_vision=True)
    if provider == "anthropic":
        return _try_anthropic()
    if provider == "custom":
        return _try_custom_endpoint()
    return None, None


def _strict_vision_backend_available(provider: str) -> bool:
    return _resolve_strict_vision_backend(provider)[0] is not None


def get_available_vision_backends() -> List[str]:
    """Return the currently available vision backends in auto-selection order.

    Order: active provider → OpenRouter → Nous → stop.  This is the single
    source of truth for setup, tool gating, and runtime auto-routing of
    vision tasks.
    """
    available: List[str] = []
    # 1. Active provider — if the user configured a provider, try it first.
    main_provider = _read_main_provider()
    if main_provider and main_provider not in {"auto", ""}:
        if main_provider in _VISION_AUTO_PROVIDER_ORDER:
            if _strict_vision_backend_available(main_provider):
                available.append(main_provider)
        else:
            client, _ = resolve_provider_client(main_provider, _read_main_model())
            if client is not None:
                available.append(main_provider)
    # 2. OpenRouter, 3. Nous — skip if already covered by main provider.
    for p in _VISION_AUTO_PROVIDER_ORDER:
        if p not in available and _strict_vision_backend_available(p):
            available.append(p)
    return available


def resolve_vision_provider_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    async_mode: bool = False,
) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
    """Resolve the client actually used for vision tasks.

    Direct endpoint overrides take precedence over provider selection. Explicit
    provider overrides still use the generic provider router for non-standard
    backends, so users can intentionally force experimental providers. Auto mode
    stays conservative and only tries vision backends known to work today.
    """
    requested, resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode = _resolve_task_provider_model(
        "vision", provider, model, base_url, api_key
    )
    requested = _normalize_vision_provider(requested)

    def _finalize(resolved_provider: str, sync_client: Any, default_model: Optional[str]):
        if sync_client is None:
            return resolved_provider, None, None
        final_model = resolved_model or default_model
        if async_mode:
            async_client, async_model = _to_async_client(sync_client, final_model, is_vision=True)
            return resolved_provider, async_client, async_model
        return resolved_provider, sync_client, final_model

    if resolved_base_url:
        provider_for_base_override = (
            requested if requested and requested not in {"", "auto"} else "custom"
        )
        client, final_model = resolve_provider_client(
            provider_for_base_override,
            model=resolved_model,
            async_mode=async_mode,
            explicit_base_url=resolved_base_url,
            explicit_api_key=resolved_api_key,
            api_mode=resolved_api_mode,
        )
        if client is None:
            return provider_for_base_override, None, None
        return provider_for_base_override, client, final_model

    if requested == "auto":
        # Vision auto-detection order:
        #   1. User's main provider + main model (including aggregators).
        #      _PROVIDER_VISION_MODELS provides per-provider vision model
        #      overrides when the provider has a dedicated multimodal model
        #      that differs from the chat model (e.g. xiaomi → mimo-v2-omni,
        #      zai → glm-5v-turbo). Nous is the exception: it has a dedicated
        #      strict vision backend with tier-aware defaults, so it must not
        #      fall through to the user's text chat model here.
        #   2. OpenRouter  (vision-capable aggregator fallback)
        #   3. Nous Portal (vision-capable aggregator fallback)
        #   4. Stop
        main_provider = _read_main_provider()
        main_model = _read_main_model()
        if main_provider and main_provider not in {"auto", ""}:
            vision_model = _PROVIDER_VISION_MODELS.get(main_provider, main_model)
            if main_provider == "nous":
                sync_client, default_model = _resolve_strict_vision_backend(
                    main_provider, vision_model
                )
                if sync_client is not None:
                    logger.info(
                        "Vision auto-detect: using main provider %s (%s)",
                        main_provider, default_model or resolved_model or main_model,
                    )
                    return _finalize(main_provider, sync_client, default_model)
            elif main_provider in _PROVIDERS_WITHOUT_VISION:
                # Kimi Coding Plan's /coding endpoint (Anthropic Messages wire)
                # does not accept image input — Kimi's own docs say "Current
                # model does not support image input, switch to a model with
                # image_in capability" and vision lives on the separate Kimi
                # Platform (api.moonshot.ai). Skip the main provider and fall
                # through to the aggregator chain instead of returning a
                # client that will 404 on every vision request (#17076).
                logger.debug(
                    "Vision auto-detect: skipping main provider %s (no "
                    "vision support) — falling through to aggregator chain",
                    main_provider,
                )
            elif not _main_model_supports_vision(main_provider, vision_model):
                # The main model is known to be text-only (e.g. DeepSeek V4,
                # gpt-oss-120b without vision). Building a client and sending
                # an image would produce a cryptic provider-side error like
                # ``unknown variant `image_url`, expected `text``` (#31179).
                # Fall through to the aggregator chain instead.
                #
                # Only log the provider name (not the model) — mirrors the
                # sibling _PROVIDERS_WITHOUT_VISION branch above, and avoids
                # CodeQL py/clear-text-logging-sensitive-data heuristic false
                # positives on multi-value interpolations.
                logger.debug(
                    "Vision auto-detect: skipping main provider %s "
                    "(reports no vision capability) — falling through to "
                    "aggregator chain",
                    main_provider,
                )
            else:
                rpc_client, rpc_model = resolve_provider_client(
                    main_provider, vision_model,
                    api_mode=resolved_api_mode,
                    is_vision=True)
                if rpc_client is not None:
                    logger.info(
                        "Vision auto-detect: using main provider %s (%s)",
                        main_provider, rpc_model or vision_model,
                    )
                    return _finalize(
                        main_provider, rpc_client, rpc_model or vision_model)

        # Fall back through aggregators (uses their dedicated vision model,
        # not the user's main model) when main provider has no client.
        for candidate in _VISION_AUTO_PROVIDER_ORDER:
            if candidate == main_provider:
                continue  # already tried above
            sync_client, default_model = _resolve_strict_vision_backend(candidate)
            if sync_client is not None:
                return _finalize(candidate, sync_client, default_model)

        logger.debug("Auxiliary vision client: none available")
        return None, None, None

    if requested in _VISION_AUTO_PROVIDER_ORDER:
        sync_client, default_model = _resolve_strict_vision_backend(
            requested, resolved_model
        )
        return _finalize(requested, sync_client, default_model)

    # ZAI vision models must use the OpenAI-compatible endpoint, not the
    # Anthropic-compatible one (which may be the main-runtime default).
    # The Anthropic wire rejects max_tokens on multimodal calls (error 1210),
    # while the OpenAI wire handles it correctly.
    if requested == "zai" and not resolved_base_url:
        zai_openai_urls = [
            "https://open.bigmodel.cn/api/paas/v4",
            "https://api.z.ai/api/paas/v4",
        ]
        for _zai_url in zai_openai_urls:
            client, final_model = _get_cached_client(
                requested, resolved_model, async_mode,
                base_url=_zai_url,
                api_key=resolved_api_key or None,
                api_mode="chat_completions",
                is_vision=True,
            )
            if client is not None:
                return _finalize(requested, client, final_model)
        # Fallback: try without explicit base_url (old behavior)
        client, final_model = _get_cached_client(requested, resolved_model, async_mode,
                                                 api_mode=resolved_api_mode,
                                                 is_vision=True)
        if client is None:
            return requested, None, None
        return requested, client, final_model

    client, final_model = _get_cached_client(requested, resolved_model, async_mode,
                                             api_mode=resolved_api_mode,
                                             is_vision=True)
    if client is None:
        return requested, None, None
    return requested, client, final_model


def get_auxiliary_extra_body() -> dict:
    """Return extra_body kwargs for auxiliary API calls.
    
    Includes Nous Portal product tags when the auxiliary client is backed
    by Nous Portal. Returns empty dict otherwise.
    """
    return _nous_extra_body() if auxiliary_is_nous else {}


def auxiliary_max_tokens_param(value: int, *, model: Optional[str] = None) -> dict:
    """Return the correct max tokens kwarg for the auxiliary client's provider.

    OpenRouter and local models use 'max_tokens'. Direct OpenAI with newer
    models (gpt-4o, gpt-4.1, gpt-5+, o-series) requires 'max_completion_tokens'.
    The Codex adapter translates max_tokens internally, so we use max_tokens
    for it as well. Pass ``model`` so third-party OpenAI-compatible endpoints
    fronting the newer families are also recognised — URL-only detection
    misses the case where a custom base URL serves e.g. ``gpt-5.4``.
    """
    custom_base = _current_custom_base_url()
    or_key = os.getenv("OPENROUTER_API_KEY")
    # Use max_completion_tokens for direct OpenAI-compatible providers that reject
    # max_tokens on newer GPT-4o/o-series/GPT-5-style models.
    if (not or_key
            and _read_nous_auth() is None
            and base_url_hostname(custom_base) in {"api.openai.com", "api.githubcopilot.com"}):
        return {"max_completion_tokens": value}
    # ...and for any caller serving a newer OpenAI-family model by name.
    if model_forces_max_completion_tokens(model):
        return {"max_completion_tokens": value}
    return {"max_tokens": value}


# ── Centralized LLM Call API ────────────────────────────────────────────────
#
# call_llm() and async_call_llm() own the full request lifecycle:
#   1. Resolve provider + model from task config (or explicit args)
#   2. Get or create a cached client for that provider
#   3. Format request args for the provider + model (max_tokens handling, etc.)
#   4. Make the API call
#   5. Return the response
#
# Every auxiliary LLM consumer should use these instead of manually
# constructing clients and calling .chat.completions.create().

# Client cache: (provider, async_mode, base_url, api_key, api_mode, runtime_key) -> (client, default_model, loop)
# NOTE: loop identity is NOT part of the key.  On async cache hits we check
# whether the cached loop is the *current* loop; if not, the stale entry is
# replaced in-place.  This bounds cache growth to one entry per unique
# provider config rather than one per (config × event-loop), which previously
# caused unbounded fd accumulation in long-running gateway processes (#10200).
_client_cache: Dict[tuple, tuple] = {}
_client_cache_lock = threading.Lock()
_CLIENT_CACHE_MAX_SIZE = 64  # safety belt — evict oldest when exceeded


def _client_cache_key(
    provider: str,
    *,
    async_mode: bool,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_mode: Optional[str] = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    is_vision: bool = False,
    task: Optional[str] = None,
) -> tuple:
    runtime = _normalize_main_runtime(main_runtime)
    runtime_key = tuple(runtime.get(field, "") for field in _MAIN_RUNTIME_FIELDS) if provider == "auto" else ()
    # `auto` can now resolve through task-specific or main fallback policy,
    # so the task participates in the cache key. Non-auto providers keep the
    # old cache shape because the explicit provider/model tuple is sufficient.
    task_key = (task or "") if provider == "auto" else ""
    pool_hint = _pool_cache_hint(provider, main_runtime=main_runtime)
    return (provider, async_mode, base_url or "", api_key or "", api_mode or "", runtime_key, is_vision, task_key, pool_hint)


def _store_cached_client(cache_key: tuple, client: Any, default_model: Optional[str], *, bound_loop: Any = None) -> None:
    with _client_cache_lock:
        old_entry = _client_cache.get(cache_key)
        if old_entry is not None and old_entry[0] is not client:
            _force_close_async_httpx(old_entry[0])
            try:
                close_fn = getattr(old_entry[0], "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass
        _client_cache[cache_key] = (client, default_model, bound_loop)


def _refresh_nous_auxiliary_client(
    *,
    cache_provider: str,
    model: Optional[str],
    async_mode: bool,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_mode: Optional[str] = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    is_vision: bool = False,
) -> Tuple[Optional[Any], Optional[str]]:
    """Refresh Nous runtime creds, rebuild the client, and replace the cache entry."""
    runtime = _resolve_nous_runtime_api(force_refresh=True)
    if runtime is None:
        return None, model

    fresh_key, fresh_base_url = runtime
    sync_client = OpenAI(api_key=fresh_key, base_url=fresh_base_url)
    final_model = model

    current_loop = None
    if async_mode:
        try:
            import asyncio as _aio
            current_loop = _aio.get_event_loop()
        except RuntimeError:
            pass
        client, final_model = _to_async_client(sync_client, final_model or "", is_vision=is_vision)
    else:
        client = sync_client

    cache_key = _client_cache_key(
        cache_provider,
        async_mode=async_mode,
        base_url=base_url,
        api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
        is_vision=is_vision,
    )
    _store_cached_client(cache_key, client, final_model, bound_loop=current_loop)
    return client, final_model


def neuter_async_httpx_del() -> None:
    """Monkey-patch ``AsyncHttpxClientWrapper.__del__`` to be a no-op.

    The OpenAI SDK's ``AsyncHttpxClientWrapper.__del__`` schedules
    ``self.aclose()`` via ``asyncio.get_running_loop().create_task()``.
    When an ``AsyncOpenAI`` client is garbage-collected while
    prompt_toolkit's event loop is running (the common CLI idle state),
    the ``aclose()`` task runs on prompt_toolkit's loop but the
    underlying TCP transport is bound to a *different* loop (the worker
    thread's loop that the client was originally created on).  If that
    loop is closed or its thread is dead, the transport's
    ``self._loop.call_soon()`` raises ``RuntimeError("Event loop is
    closed")``, which prompt_toolkit surfaces as "Unhandled exception
    in event loop ... Press ENTER to continue...".

    Neutering ``__del__`` is safe because:
    - Cached clients are explicitly cleaned via ``_force_close_async_httpx``
      on stale-loop detection and ``shutdown_cached_clients`` on exit.
    - Uncached clients' TCP connections are cleaned up by the OS when the
      process exits.
    - The OpenAI SDK itself marks this as a TODO (``# TODO(someday):
      support non asyncio runtimes here``).

    Call this once at CLI startup, before any ``AsyncOpenAI`` clients are
    created.
    """
    try:
        from openai._base_client import AsyncHttpxClientWrapper
        AsyncHttpxClientWrapper.__del__ = lambda self: None  # type: ignore[assignment]
    except (ImportError, AttributeError):
        pass  # Graceful degradation if the SDK changes its internals


def _force_close_async_httpx(client: Any) -> None:
    """Mark the httpx AsyncClient inside an AsyncOpenAI client as closed.

    This prevents ``AsyncHttpxClientWrapper.__del__`` from scheduling
    ``aclose()`` on a (potentially closed) event loop, which causes
    ``RuntimeError: Event loop is closed`` → prompt_toolkit's
    "Press ENTER to continue..." handler.

    We intentionally do NOT run the full async close path — the
    connections will be dropped by the OS when the process exits.
    """
    try:
        from httpx._client import ClientState
        inner = getattr(client, "_client", None)
        if inner is not None and not getattr(inner, "is_closed", True):
            inner._state = ClientState.CLOSED
    except Exception:
        pass


def shutdown_cached_clients() -> None:
    """Close all cached clients (sync and async) to prevent event-loop errors.

    Call this during CLI shutdown, *before* the event loop is closed, to
    avoid ``AsyncHttpxClientWrapper.__del__`` raising on a dead loop.
    """
    import inspect

    with _client_cache_lock:
        for key, entry in list(_client_cache.items()):
            client = entry[0]
            if client is None:
                continue
            # Mark any async httpx transport as closed first (prevents __del__
            # from scheduling aclose() on a dead event loop).
            _force_close_async_httpx(client)
            # Sync clients: close the httpx connection pool cleanly.
            # Async clients: skip — we already neutered __del__ above.
            try:
                close_fn = getattr(client, "close", None)
                if close_fn and not inspect.iscoroutinefunction(close_fn):
                    close_fn()
            except Exception:
                pass
        _client_cache.clear()


def cleanup_stale_async_clients() -> None:
    """Force-close cached async clients whose event loop is closed.

    Call this after each agent turn to proactively clean up stale clients
    before GC can trigger ``AsyncHttpxClientWrapper.__del__`` on them.
    This is defense-in-depth — the primary fix is ``neuter_async_httpx_del``
    which disables ``__del__`` entirely.
    """
    with _client_cache_lock:
        stale_keys = []
        for key, entry in _client_cache.items():
            client, _default, cached_loop = entry
            if cached_loop is not None and cached_loop.is_closed():
                _force_close_async_httpx(client)
                stale_keys.append(key)
        for key in stale_keys:
            del _client_cache[key]


def _is_openrouter_client(client: Any) -> bool:
    for obj in (client, getattr(client, "_client", None), getattr(client, "client", None)):
        if obj and base_url_host_matches(str(getattr(obj, "base_url", "") or ""), "openrouter.ai"):
            return True
    return False


def _cached_client_accepts_slash_models(client: Any, cached_default: Optional[str]) -> bool:
    """Best-effort check for cached clients that accept ``vendor/model`` IDs."""
    if _is_openrouter_client(client):
        return True
    return bool(cached_default and "/" in cached_default)


def _compat_model(client: Any, model: Optional[str], cached_default: Optional[str]) -> Optional[str]:
    """Keep slash-bearing model IDs only for cached clients that support them.

    Mirrors the guard in resolve_provider_client() which is skipped on cache hits.
    """
    if model and "/" in model and not _cached_client_accepts_slash_models(client, cached_default):
        return cached_default
    return model or cached_default


def _get_cached_client(
    provider: str,
    model: str = None,
    async_mode: bool = False,
    base_url: str = None,
    api_key: str = None,
    api_mode: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    is_vision: bool = False,
    task: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Get or create a cached client for the given provider.

    Async clients (AsyncOpenAI) use httpx.AsyncClient internally, which
    binds to the event loop that was current when the client was created.
    Using such a client on a *different* loop causes deadlocks or
    RuntimeError.  To prevent cross-loop issues, the cache validates on
    every async hit that the cached loop is the *current, open* loop.
    If the loop changed (e.g. a new gateway worker-thread loop), the stale
    entry is replaced in-place rather than creating an additional entry.

    This keeps cache size bounded to one entry per unique provider config,
    preventing the fd-exhaustion that previously occurred in long-running
    gateways where recycled worker threads created unbounded entries (#10200).
    """
    # Resolve the current event loop for async clients so we can validate
    # cached entries.  Loop identity is NOT in the cache key — instead we
    # check at hit time whether the cached loop is still current and open.
    # This prevents unbounded cache growth from recycled worker-thread loops
    # while still guaranteeing we never reuse a client on the wrong loop
    # (which causes deadlocks, see #2681).
    current_loop = None
    if async_mode:
        try:
            import asyncio as _aio
            current_loop = _aio.get_event_loop()
        except RuntimeError:
            pass
    runtime = _normalize_main_runtime(main_runtime)
    cache_key = _client_cache_key(
        provider,
        async_mode=async_mode,
        base_url=base_url,
        api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
        is_vision=is_vision,
        task=task,
    )
    with _client_cache_lock:
        if cache_key in _client_cache:
            cached_client, cached_default, cached_loop = _client_cache[cache_key]
            if async_mode:
                # Validate: the cached client must be bound to the CURRENT,
                # OPEN loop.  If the loop changed or was closed, the httpx
                # transport inside is dead — force-close and replace.
                loop_ok = (
                    cached_loop is not None
                    and cached_loop is current_loop
                    and not cached_loop.is_closed()
                )
                if loop_ok:
                    effective = _compat_model(cached_client, model, cached_default)
                    return cached_client, effective
                # Stale — evict and fall through to create a new client.
                _force_close_async_httpx(cached_client)
                del _client_cache[cache_key]
            else:
                effective = _compat_model(cached_client, model, cached_default)
                return cached_client, effective
    # Build outside the lock.
    # For pool-backed api_key providers, derive the active API key from the
    # pool entry rather than from env vars.  resolve_api_key_provider_credentials
    # always prefers env vars (first-entry bias), which bypasses pool rotation:
    # after key #1 is marked exhausted the retry would still get key #1 from
    # the env var and fail again, causing the retry2_err handler to mark key #2.
    effective_api_key = api_key
    if not effective_api_key:
        _pe = _peek_pool_entry(_normalize_aux_provider(provider))
        if _pe is not None:
            _pk = _pool_runtime_api_key(_pe)
            if _pk:
                effective_api_key = _pk
    client, default_model = resolve_provider_client(
        provider,
        model,
        async_mode,
        explicit_base_url=base_url,
        explicit_api_key=effective_api_key,
        api_mode=api_mode,
        main_runtime=runtime,
        is_vision=is_vision,
        task=task,
    )
    if client is not None:
        # For async clients, remember which loop they were created on so we
        # can detect stale entries later.
        bound_loop = current_loop
        with _client_cache_lock:
            if cache_key not in _client_cache:
                # Safety belt: if the cache has grown beyond the max, evict
                # the oldest entries (FIFO — dict preserves insertion order).
                while len(_client_cache) >= _CLIENT_CACHE_MAX_SIZE:
                    evict_key, evict_entry = next(iter(_client_cache.items()))
                    _force_close_async_httpx(evict_entry[0])
                    del _client_cache[evict_key]
                _client_cache[cache_key] = (client, default_model, bound_loop)
            else:
                client, default_model, _ = _client_cache[cache_key]
    return client, model or default_model


# Aliases that target direct REST APIs not modeled as first-class providers
# in PROVIDER_REGISTRY. Used for ``auxiliary.<task>.provider`` so users can
# write the obvious name and have it resolve to a working ``custom`` endpoint
# without needing to know our internal provider IDs.
#
# Why these specifically: PROVIDER_REGISTRY has ``openai-codex`` (OAuth) and
# ``custom`` (manual base_url + OPENAI_API_KEY) but no plain ``openai`` for
# direct API-key access. Users predictably type ``provider: openai`` and
# expect it to use OPENAI_API_KEY against api.openai.com. Previously this
# silently fell back to the user's main provider, sending OpenAI model names
# to e.g. DeepSeek and producing cryptic ``unknown variant 'image_url'``
# errors (issue #31179).
_AUX_DIRECT_API_BASE_URLS: Dict[str, str] = {
    "openai": "https://api.openai.com/v1",
}


def _resolve_task_provider_model(
    task: str = None,
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Determine provider + model for a call.

    Priority:
      1. Explicit provider/model/base_url/api_key args (always win)
      2. Config file (auxiliary.{task}.provider/model/base_url)
      3. "auto" (full auto-detection chain)

    Returns (provider, model, base_url, api_key, api_mode) where model may
    be None (use provider default). When base_url is set, provider is forced
    to "custom" and the task uses that direct endpoint. api_mode is one of
    "chat_completions", "codex_responses", or None (auto-detect).
    """
    cfg_provider = None
    cfg_model = None
    cfg_base_url = None
    cfg_api_key = None
    cfg_api_mode = None

    if task:
        task_config = _get_auxiliary_task_config(task)
        cfg_provider = str(task_config.get("provider", "")).strip() or None
        cfg_model = str(task_config.get("model", "")).strip() or None
        cfg_base_url = str(task_config.get("base_url", "")).strip() or None
        cfg_api_key = str(task_config.get("api_key", "")).strip() or None
        cfg_api_mode = str(task_config.get("api_mode", "")).strip() or None

    resolved_model = model or cfg_model
    resolved_api_mode = cfg_api_mode

    # Convenience aliases for direct API-key endpoints that aren't first-class
    # providers (e.g. ``provider: openai`` → custom + api.openai.com/v1).
    # Applied to both explicit args and config-derived values. When the user
    # has already supplied a base_url we keep their endpoint but still rewrite
    # the provider to ``custom`` so resolution doesn't hit the
    # PROVIDER_REGISTRY-only path (which has no ``openai`` entry).
    def _expand_direct_api_alias(prov: Optional[str], existing_base: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if not prov:
            return prov, existing_base
        target_base = _AUX_DIRECT_API_BASE_URLS.get(prov.strip().lower())
        if target_base is None:
            return prov, existing_base
        return "custom", existing_base or target_base

    if provider:
        provider, base_url = _expand_direct_api_alias(provider, base_url)
    if cfg_provider:
        cfg_provider, cfg_base_url = _expand_direct_api_alias(cfg_provider, cfg_base_url)

    if base_url:
        return "custom", resolved_model, base_url, api_key, resolved_api_mode
    if provider:
        return provider, resolved_model, base_url, api_key, resolved_api_mode

    if task:
        # Config.yaml is the primary source for per-task overrides.
        if cfg_base_url and cfg_api_key:
            # Both base_url and api_key explicitly set → custom endpoint.
            return "custom", resolved_model, cfg_base_url, cfg_api_key, resolved_api_mode
        if cfg_base_url and cfg_provider and cfg_provider != "auto":
            # base_url set without api_key but with a known provider — use
            # the provider so it can resolve credentials from env vars
            # (e.g. OPENROUTER_API_KEY) instead of locking into "custom".
            return cfg_provider, resolved_model, cfg_base_url, None, resolved_api_mode
        if cfg_provider and cfg_provider != "auto":
            return cfg_provider, resolved_model, cfg_base_url, cfg_api_key, resolved_api_mode

        return "auto", resolved_model, None, None, resolved_api_mode

    return "auto", resolved_model, None, None, resolved_api_mode


_DEFAULT_AUX_TIMEOUT = 30.0


def _get_auxiliary_task_config(task: str) -> Dict[str, Any]:
    """Return the config dict for auxiliary.<task>, or {} when unavailable.

    For plugin-registered auxiliary tasks (see
    :meth:`hermes_cli.plugins.PluginContext.register_auxiliary_task`) the
    plugin's declared *defaults* are layered underneath the user's config
    so an unconfigured plugin task still works:

        plugin defaults  ←  config.yaml auxiliary.<task>  (user wins)

    Built-in tasks ignore this path (their defaults live in DEFAULT_CONFIG).
    """
    if not task:
        return {}
    try:
        from hermes_cli.config import load_config
        config = load_config()
    except ImportError:
        return {}
    aux = config.get("auxiliary", {}) if isinstance(config, dict) else {}
    task_config = aux.get(task, {}) if isinstance(aux, dict) else {}
    if not isinstance(task_config, dict):
        task_config = {}

    # Layer plugin-declared defaults underneath user config so
    # ctx.register_auxiliary_task(defaults={...}) takes effect without
    # forcing the user to write config.yaml entries.
    try:
        from hermes_cli.plugins import get_plugin_auxiliary_tasks
        for _entry in get_plugin_auxiliary_tasks():
            if _entry.get("key") == task:
                _defaults = _entry.get("defaults") or {}
                if isinstance(_defaults, dict):
                    merged = dict(_defaults)
                    merged.update(task_config)
                    return merged
                break
    except Exception:
        # Plugin discovery failure must not break aux task config reads.
        pass

    return task_config


def _get_task_timeout(task: str, default: float = _DEFAULT_AUX_TIMEOUT) -> float:
    """Read timeout from auxiliary.{task}.timeout in config, falling back to *default*."""
    if not task:
        return default
    task_config = _get_auxiliary_task_config(task)
    raw = task_config.get("timeout")
    if raw is not None:
        try:
            return float(raw)
        except (ValueError, TypeError):
            pass
    return default


def _get_task_extra_body(task: str) -> Dict[str, Any]:
    """Read auxiliary.<task>.extra_body and return a shallow copy when valid."""
    task_config = _get_auxiliary_task_config(task)
    raw = task_config.get("extra_body")
    if isinstance(raw, dict):
        return dict(raw)
    return {}


# ---------------------------------------------------------------------------
# Anthropic-compatible endpoint detection + image block conversion
# ---------------------------------------------------------------------------

# Providers that use Anthropic-compatible endpoints (via OpenAI SDK wrapper).
# Their image content blocks must use Anthropic format, not OpenAI format.
_ANTHROPIC_COMPAT_PROVIDERS = frozenset({"minimax", "minimax-oauth", "minimax-cn"})


def _is_anthropic_compat_endpoint(provider: str, base_url: str) -> bool:
    """Detect if an endpoint expects Anthropic-format content blocks.

    Returns True for known Anthropic-compatible providers (MiniMax) and
    any endpoint whose URL contains ``/anthropic`` in the path.
    """
    if provider in _ANTHROPIC_COMPAT_PROVIDERS:
        return True
    url_lower = (base_url or "").lower()
    return "/anthropic" in url_lower


def _convert_openai_images_to_anthropic(messages: list) -> list:
    """Convert OpenAI ``image_url``/``video_url`` blocks to Anthropic format.

    Converts:
    - ``image_url`` blocks to Anthropic ``image`` blocks
    - ``video_url`` blocks to Anthropic ``video`` blocks (MiniMax M3 compat)

    Only touches messages that have list-type content with ``image_url`` or
    ``video_url`` blocks; plain text messages pass through unchanged.
    """
    converted = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            converted.append(msg)
            continue
        new_content = []
        changed = False
        for block in content:
            if block.get("type") == "image_url":
                image_url_val = (block.get("image_url") or {}).get("url", "")
                if image_url_val.startswith("data:"):
                    # Parse data URI: data:<media_type>;base64,<data>
                    header, _, b64data = image_url_val.partition(",")
                    media_type = "image/png"
                    if ":" in header and ";" in header:
                        media_type = header.split(":", 1)[1].split(";", 1)[0]
                    new_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64data,
                        },
                    })
                else:
                    # URL-based image
                    new_content.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url_val,
                        },
                    })
                changed = True
            elif block.get("type") == "video_url":
                # MiniMax's Anthropic-compatible endpoint expects a "video"
                # block (not OpenAI's "video_url", and not "input_video").
                # See https://platform.minimax.io/docs/api-reference/text-anthropic-api
                # — the Messages-field table lists type="video" (M3 only,
                # URL/base64/mm_file://). The source shape mirrors the "image"
                # block: base64 → {type:"base64", media_type, data}, URL →
                # {type:"url", url}.
                video_url_val = (block.get("video_url") or {}).get("url", "")
                if video_url_val.startswith("data:"):
                    # Parse data URI: data:<media_type>;base64,<data>
                    header, _, b64data = video_url_val.partition(",")
                    media_type = "video/mp4"
                    if ":" in header and ";" in header:
                        media_type = header.split(":", 1)[1].split(";", 1)[0]
                    new_content.append({
                        "type": "video",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64data,
                        },
                    })
                else:
                    # URL-based video
                    new_content.append({
                        "type": "video",
                        "source": {
                            "type": "url",
                            "url": video_url_val,
                        },
                    })
                changed = True
            else:
                new_content.append(block)
        converted.append({**msg, "content": new_content} if changed else msg)
    return converted



def _build_call_kwargs(
    provider: str,
    model: str,
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[list] = None,
    timeout: float = 30.0,
    extra_body: Optional[dict] = None,
    base_url: Optional[str] = None,
) -> dict:
    """Build kwargs for .chat.completions.create() with model/provider adjustments."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
    }

    fixed_temperature = _fixed_temperature_for_model(model, base_url)
    if fixed_temperature is OMIT_TEMPERATURE:
        temperature = None  # strip — let server choose
    elif fixed_temperature is not None:
        temperature = fixed_temperature

    # Opus 4.7+ rejects any non-default temperature/top_p/top_k — silently
    # drop here so auxiliary callers that hardcode temperature (e.g. 0 on
    # structured-JSON extraction) don't 400 the moment
    # the aux model is flipped to 4.7.
    if temperature is not None:
        from agent.anthropic_adapter import _forbids_sampling_params
        if _forbids_sampling_params(model):
            temperature = None

    if temperature is not None:
        kwargs["temperature"] = temperature

    if max_tokens is not None:
        # We do NOT cap output by default. Most chat-completions providers treat
        # an omitted max_tokens as "use the model's max output", which is what we
        # want for auxiliary tasks (compression summaries, titles, vision, etc.) —
        # an explicit cap only risks truncating a summary or 400-ing on providers
        # that reject the parameter outright (e.g. GitHub Copilot / newer OpenAI
        # GPT-5 models require max_completion_tokens, not max_tokens; ZAI vision
        # models reject it entirely with error 1210). Omitting it sidesteps all of
        # those wire-format quirks at once.
        #
        # The one exception is the Anthropic Messages wire (MiniMax and any
        # ``/anthropic`` endpoint reached through the OpenAI SDK wrapper), where
        # max_tokens is a MANDATORY field — omitting it is a hard 400. Keep it only
        # there.
        _effective_base = base_url or (
            _current_custom_base_url() if provider == "custom" else ""
        )
        if _is_anthropic_compat_endpoint(provider, _effective_base):
            kwargs["max_tokens"] = max_tokens

    if tools:
        # Defensive dedup: providers like Google Vertex, Azure, and Bedrock
        # reject requests with duplicate tool names (HTTP 400).  The upstream
        # injection paths (run_agent.py) already dedup, but this guard
        # converts a hard API failure into a warning if an upstream regression
        # reintroduces duplicates.  See: #18478
        _seen: set = set()
        _deduped: list = []
        for _t in tools:
            _tname = (_t.get("function") or {}).get("name", "")
            if _tname and _tname in _seen:
                logger.warning(
                    "_build_call_kwargs: duplicate tool name '%s' removed "
                    "(provider=%s model=%s)",
                    _tname, provider, model,
                )
                continue
            if _tname:
                _seen.add(_tname)
            _deduped.append(_t)
        kwargs["tools"] = _deduped

    # Provider-specific extra_body
    merged_extra = dict(extra_body or {})
    if provider == "nous":
        merged_extra.setdefault("tags", []).extend(_nous_portal_tags())
    if merged_extra:
        kwargs["extra_body"] = merged_extra

    return kwargs


def _validate_llm_response(response: Any, task: str = None) -> Any:
    """Validate that an LLM response has the expected .choices[0].message shape.

    Fails fast with a clear error instead of letting malformed payloads
    propagate to downstream consumers where they crash with misleading
    AttributeError (e.g. "'str' object has no attribute 'choices'").

    See #7264.
    """
    if response is None:
        raise RuntimeError(
            f"Auxiliary {task or 'call'}: LLM returned None response"
        )
    # Allow SimpleNamespace responses from adapters (CodexAuxiliaryClient,
    # AnthropicAuxiliaryClient) — they have .choices[0].message.
    try:
        choices = response.choices
        if not choices or not hasattr(choices[0], "message"):
            raise AttributeError("missing choices[0].message")
    except (AttributeError, TypeError, IndexError) as exc:
        response_type = type(response).__name__
        response_preview = str(response)[:120]
        raise RuntimeError(
            f"Auxiliary {task or 'call'}: LLM returned invalid response "
            f"(type={response_type}): {response_preview!r}. "
            f"Expected object with .choices[0].message — check provider "
            f"adapter or custom endpoint compatibility."
        ) from exc
    return response


def call_llm(
    task: str = None,
    *,
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    messages: list,
    temperature: float = None,
    max_tokens: int = None,
    tools: list = None,
    timeout: float = None,
    extra_body: dict = None,
) -> Any:
    """Centralized synchronous LLM call.

    Resolves provider + model (from task config, explicit args, or auto-detect),
    handles auth, request formatting, and model-specific arg adjustments.

    Args:
        task: Auxiliary task name ("compression", "vision", "web_extract",
              "session_search", "skills_hub", "mcp", "title_generation").
              Reads provider:model from config/env. Ignored if provider is set.
        provider: Explicit provider override.
        model: Explicit model override.
        messages: Chat messages list.
        temperature: Sampling temperature (None = provider default).
        max_tokens: Max output tokens (handles max_tokens vs max_completion_tokens).
        tools: Tool definitions (for function calling).
        timeout: Request timeout in seconds (None = read from auxiliary.{task}.timeout config).
        extra_body: Additional request body fields.

    Returns:
        Response object with .choices[0].message.content

    Raises:
        RuntimeError: If no provider is configured.
    """
    resolved_provider, resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode = _resolve_task_provider_model(
        task, provider, model, base_url, api_key)
    effective_extra_body = _get_task_extra_body(task)
    effective_extra_body.update(extra_body or {})

    if task == "vision":
        effective_provider, client, final_model = resolve_vision_provider_client(
            provider=resolved_provider if resolved_provider != "auto" else provider,
            model=resolved_model or model,
            base_url=resolved_base_url or base_url,
            api_key=resolved_api_key or api_key,
            async_mode=False,
        )
        if client is None and resolved_provider != "auto" and not resolved_base_url:
            logger.warning(
                "Vision provider %s unavailable, falling back to auto vision backends",
                resolved_provider,
            )
            effective_provider, client, final_model = resolve_vision_provider_client(
                provider="auto",
                model=resolved_model,
                async_mode=False,
            )
        if client is None:
            raise RuntimeError(
                f"No LLM provider configured for task={task} provider={resolved_provider}. "
                f"Run: hermes setup"
            )
        resolved_provider = effective_provider or resolved_provider
    else:
        client, final_model = _get_cached_client(
            resolved_provider,
            resolved_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            api_mode=resolved_api_mode,
            main_runtime=main_runtime,
        )
        if client is None:
            # When the user explicitly chose a non-OpenRouter provider but no
            # credentials were found, fail fast instead of silently routing
            # through OpenRouter (which causes confusing 404s).
            _explicit = (resolved_provider or "").strip().lower()
            if _explicit and _explicit not in {"auto", "openrouter", "custom"}:
                raise RuntimeError(
                    f"Provider '{_explicit}' is set in config.yaml but no API key "
                    f"was found. Set the {_explicit.upper()}_API_KEY environment "
                    f"variable, or switch to a different provider with `hermes model`."
                )
            # For auto/custom with no credentials, try the full auto chain
            # rather than hardcoding OpenRouter (which may be depleted).
            # Pass model=None so each provider uses its own default —
            # resolved_model may be an OpenRouter-format slug that doesn't
            # work on other providers.
            if not resolved_base_url:
                logger.info("Auxiliary %s: provider %s unavailable, trying auto-detection chain",
                            task or "call", resolved_provider)
                client, final_model = _get_cached_client("auto", main_runtime=main_runtime, task=task)
        if client is None:
            raise RuntimeError(
                f"No LLM provider configured for task={task} provider={resolved_provider}. "
                f"Run: hermes setup")

    effective_timeout = timeout if timeout is not None else _get_task_timeout(task)

    # Log what we're about to do — makes auxiliary operations visible
    _base_info = str(getattr(client, "base_url", resolved_base_url) or "")
    if task:
        logger.info("Auxiliary %s: using %s (%s)%s",
                     task, resolved_provider or "auto", final_model or "default",
                     f" at {_base_info}" if _base_info and "openrouter" not in _base_info else "")

    # Pass the client's actual base_url (not just resolved_base_url) so
    # endpoint-specific temperature overrides can distinguish
    # api.moonshot.ai vs api.kimi.com/coding even on auto-detected routes.
    kwargs = _build_call_kwargs(
        resolved_provider, final_model, messages,
        temperature=temperature, max_tokens=max_tokens,
        tools=tools, timeout=effective_timeout, extra_body=effective_extra_body,
        base_url=_base_info or resolved_base_url)

    # Convert image blocks for Anthropic-compatible endpoints (e.g. MiniMax)
    _client_base = str(getattr(client, "base_url", "") or "")
    if _is_anthropic_compat_endpoint(resolved_provider, _client_base):
        kwargs["messages"] = _convert_openai_images_to_anthropic(kwargs["messages"])

    # Handle unsupported temperature, max_tokens vs max_completion_tokens retry,
    # then payment fallback.
    try:
        # Retry ONCE on the same provider for a one-off transient transport
        # blip (streaming-close / incomplete chunked read / 5xx / 408) before
        # the except-chain below escalates to provider/model fallback. A
        # single dropped connection shouldn't abandon an otherwise-healthy
        # provider. A second failure (or any non-transient error) falls
        # through to ``first_err`` and the existing fallback handling
        # unchanged. This is the unified home for the transient retry that
        # every auxiliary task (compression, memory flush, title-gen,
        # session-search, vision) shares. (PR #16587)
        try:
            return _validate_llm_response(
                client.chat.completions.create(**kwargs), task)
        except Exception as transient_err:
            if not _is_transient_transport_error(transient_err):
                raise
            logger.info(
                "Auxiliary %s: transient transport error; retrying once on "
                "the same provider before fallback: %s",
                task or "call", transient_err,
            )
            return _validate_llm_response(
                client.chat.completions.create(**kwargs), task)
    except Exception as first_err:
        if "temperature" in kwargs and _is_unsupported_temperature_error(first_err):
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("temperature", None)
            logger.info(
                "Auxiliary %s: provider rejected temperature; retrying once without it",
                task or "call",
            )
            try:
                return _validate_llm_response(
                    client.chat.completions.create(**retry_kwargs), task)
            except Exception as retry_err:
                retry_err_str = str(retry_err)
                # If retry still fails, fall through to the max_tokens /
                # payment / auth chains below using the temperature-stripped
                # kwargs.  Re-raise only if the retry hit something those
                # chains won't handle.
                if not (
                    _is_payment_error(retry_err)
                    or _is_connection_error(retry_err)
                    or _is_auth_error(retry_err)
                    or "max_tokens" in retry_err_str
                    or "unsupported_parameter" in retry_err_str
                ):
                    raise
                first_err = retry_err
                kwargs = retry_kwargs

        err_str = str(first_err)
        # ZAI vision models (glm-4v-flash etc.) return error code 1210
        # ("API 调用参数有误") when max_tokens is passed on multimodal
        # calls.  The error message does NOT contain "max_tokens" so the
        # generic retry below never fires.  Detect the ZAI-specific error
        # and strip max_tokens before retrying.
        _is_zai_param_error = (
            "1210" in err_str
            and "bigmodel" in str(getattr(client, "base_url", ""))
        )
        if max_tokens is not None and (
            "max_tokens" in err_str
            or "unsupported_parameter" in err_str
            or _is_unsupported_parameter_error(first_err, "max_tokens")
            or _is_zai_param_error
        ):
            kwargs.pop("max_tokens", None)
            kwargs.pop("max_completion_tokens", None)
            try:
                return _validate_llm_response(
                    client.chat.completions.create(**kwargs), task)
            except Exception as retry_err:
                # If the max_tokens retry also hits a payment or connection
                # error, fall through to the fallback chain below.
                if not (_is_payment_error(retry_err) or _is_connection_error(retry_err) or _is_rate_limit_error(retry_err)):
                    raise
                first_err = retry_err

        # ── Stale-model self-heal (Nous Portal recommendation drift) ───
        # A long-lived process can pin a Portal-recommended model that has
        # since been dropped from the Nous → OpenRouter catalog, so every
        # auxiliary call 404s with "model does not exist". Force a fresh
        # Portal fetch and retry once with the current recommendation (or the
        # known-good default). Only applies to Nous-routed calls.
        _heal_is_nous = (
            resolved_provider == "nous"
            or base_url_host_matches(_base_info, "inference-api.nousresearch.com")
        )
        if _is_model_not_found_error(first_err) and _heal_is_nous:
            healed_model = _refresh_nous_recommended_model(
                vision=(task == "vision"), stale_model=kwargs.get("model"))
            if healed_model and healed_model != kwargs.get("model"):
                logger.warning(
                    "Auxiliary %s: model %r no longer in Nous catalog; "
                    "retrying with refreshed recommendation %r",
                    task or "call", kwargs.get("model"), healed_model,
                )
                kwargs["model"] = healed_model
                try:
                    return _validate_llm_response(
                        client.chat.completions.create(**kwargs), task)
                except Exception as retry_err:
                    first_err = retry_err

        # ── Nous auth refresh parity with main agent ──────────────────
        client_is_nous = (
            resolved_provider == "nous"
            or base_url_host_matches(_base_info, "inference-api.nousresearch.com")
        )
        if (
            _is_payment_error(first_err)
            and client_is_nous
            and _nous_portal_account_has_fresh_paid_access()
        ):
            refreshed_client, refreshed_model = _refresh_nous_auxiliary_client(
                cache_provider=resolved_provider or "nous",
                model=final_model,
                async_mode=False,
                base_url=resolved_base_url,
                api_key=resolved_api_key,
                api_mode=resolved_api_mode,
                main_runtime=main_runtime,
                is_vision=(task == "vision"),
            )
            if refreshed_client is not None:
                logger.info(
                    "Auxiliary %s: refreshed Nous runtime credentials after paid account check, retrying",
                    task or "call",
                )
                if refreshed_model and refreshed_model != kwargs.get("model"):
                    kwargs["model"] = refreshed_model
                try:
                    return _validate_llm_response(
                        refreshed_client.chat.completions.create(**kwargs), task)
                except Exception as retry_err:
                    if not (
                        _is_auth_error(retry_err)
                        or _is_payment_error(retry_err)
                        or _is_connection_error(retry_err)
                        or _is_rate_limit_error(retry_err)
                    ):
                        raise
                    first_err = retry_err

        if _is_auth_error(first_err) and client_is_nous:
            refreshed_client, refreshed_model = _refresh_nous_auxiliary_client(
                cache_provider=resolved_provider or "nous",
                model=final_model,
                async_mode=False,
                base_url=resolved_base_url,
                api_key=resolved_api_key,
                api_mode=resolved_api_mode,
                main_runtime=main_runtime,
                is_vision=(task == "vision"),
            )
            if refreshed_client is not None:
                logger.info("Auxiliary %s: refreshed Nous runtime credentials after 401, retrying",
                            task or "call")
                if refreshed_model and refreshed_model != kwargs.get("model"):
                    kwargs["model"] = refreshed_model
                return _validate_llm_response(
                    refreshed_client.chat.completions.create(**kwargs), task)

        # ── Auth refresh retry ───────────────────────────────────────
        if (_is_auth_error(first_err)
                and resolved_provider not in {"auto", "", None}
                and not client_is_nous):
            if _refresh_provider_credentials(resolved_provider):
                logger.info(
                    "Auxiliary %s: refreshed %s credentials after auth error, retrying",
                    task or "call", resolved_provider,
                )
                return _retry_same_provider_sync(
                    task=task,
                    resolved_provider=resolved_provider,
                    resolved_model=resolved_model,
                    resolved_base_url=resolved_base_url,
                    resolved_api_key=resolved_api_key,
                    resolved_api_mode=resolved_api_mode,
                    main_runtime=main_runtime,
                    final_model=final_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    effective_timeout=effective_timeout,
                    effective_extra_body=effective_extra_body,
                )

        # ── Same-provider credential-pool recovery ─────────────────────
        pool_provider = _recoverable_pool_provider(resolved_provider, client, main_runtime=main_runtime)
        # Capture the exact API key used so mark_exhausted_and_rotate can find
        # the correct pool entry even when another process rotated the pool
        # between this call and recovery (which leaves current()=None and makes
        # _select_unlocked() return the NEXT key by mistake).
        _client_api_key = str(getattr(client, "api_key", "") or "")
        if pool_provider and (_is_auth_error(first_err) or _is_payment_error(first_err) or _is_rate_limit_error(first_err)):
            recovery_err = first_err
            # Skip the extra retry for clear payment/quota errors — the endpoint
            # won't accept another request with the same exhausted key.
            if _is_rate_limit_error(first_err) and not _is_payment_error(first_err):
                try:
                    return _validate_llm_response(
                        client.chat.completions.create(**kwargs), task)
                except Exception as retry_err:
                    if not (_is_auth_error(retry_err) or _is_payment_error(retry_err) or _is_rate_limit_error(retry_err)):
                        raise
                    recovery_err = retry_err
            if _recover_provider_pool(pool_provider, recovery_err, failed_api_key=_client_api_key):
                logger.info(
                    "Auxiliary %s: recovered %s via credential-pool rotation after %s",
                    task or "call", pool_provider, type(recovery_err).__name__,
                )
                try:
                    return _retry_same_provider_sync(
                        task=task,
                        resolved_provider=resolved_provider,
                        resolved_model=resolved_model,
                        resolved_base_url=resolved_base_url,
                        resolved_api_key=resolved_api_key,
                        resolved_api_mode=resolved_api_mode,
                        main_runtime=main_runtime,
                        final_model=final_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools,
                        effective_timeout=effective_timeout,
                        effective_extra_body=effective_extra_body,
                    )
                except Exception as retry2_err:
                    # The rotated key also hit a quota/auth wall.  Mark it
                    # immediately so concurrent processes don't make a
                    # redundant API call to discover it's exhausted too.
                    # Then fall through to the payment fallback below so
                    # alternative providers can still serve the request.
                    if (_is_payment_error(retry2_err) or _is_auth_error(retry2_err)
                            or _is_rate_limit_error(retry2_err)):
                        _recover_provider_pool(pool_provider, retry2_err)
                        first_err = retry2_err
                    else:
                        raise

        # ── Payment / credit exhaustion fallback ──────────────────────
        # When the resolved provider returns 402 or a credit-related error,
        # try alternative providers instead of giving up.  This handles the
        # common case where a user runs out of OpenRouter credits but has
        # Codex OAuth or another provider available.
        #
        # ── Connection error fallback ────────────────────────────────
        # When a provider endpoint is unreachable (DNS failure, connection
        # refused, timeout), try alternative providers.  This handles stale
        # Codex/OAuth tokens that authenticate but whose endpoint is down,
        # and providers the user never configured that got picked up by
        # the auto-detection chain.
        #
        # ── Rate-limit fallback (#13579) ─────────────────────────────
        # When the provider returns a 429 rate-limit (not billing), fall
        # back to an alternative provider instead of exhausting retries
        # against the same rate-limited endpoint.
        should_fallback = (
            _is_payment_error(first_err)
            or _is_connection_error(first_err)
            or _is_rate_limit_error(first_err)
        )
        # Respect explicit provider choice for transient errors (auth, request
        # validation, etc.) but allow fallback when the provider clearly cannot
        # serve the request due to capacity: payment/quota exhaustion and
        # connection failures are capacity problems, not request constraints.
        # See #26803: daily token quota (429 + "too many tokens per day") must
        # fall back just like a 402 credit error.
        is_auto = resolved_provider in {"auto", "", None}
        # Capacity errors bypass the explicit-provider gate: the provider
        # literally cannot serve this request regardless of user intent.
        is_capacity_error = _is_payment_error(first_err) or _is_connection_error(first_err)
        if should_fallback and (is_auto or is_capacity_error):
            if _is_payment_error(first_err):
                reason = "payment error"
                # Resolve the actual provider label (resolved_provider may be
                # "auto"; the client's base_url tells us which backend got the
                # 402). Mark THAT label unhealthy so subsequent aux calls
                # skip it instead of paying another doomed RTT.
                _mark_provider_unhealthy(
                    _recoverable_pool_provider(resolved_provider, client, main_runtime=main_runtime) or resolved_provider
                )
            elif _is_rate_limit_error(first_err):
                reason = "rate limit"
            else:
                reason = "connection error"
            logger.info("Auxiliary %s: %s on %s (%s), trying fallback",
                        task or "call", reason, resolved_provider, first_err)

            # Fallback order (#26882, #26803):
            #   1. User-configured fallback_chain (per-task) if set
            #   2. For auto: top-level main fallback_providers/fallback_model
            #   3. For auto: built-in auxiliary discovery chain
            #   4. For explicit aux providers: main agent model safety net
            fb_client, fb_model, fb_label = (None, None, "")
            if is_auto:
                fb_client, fb_model, fb_label = _try_configured_fallback_chain(
                    task, resolved_provider or "auto", reason=reason)
                if fb_client is None:
                    fb_client, fb_model, fb_label = _try_main_fallback_chain(
                        task, resolved_provider or "auto", reason=reason)
                if fb_client is None:
                    fb_client, fb_model, fb_label = _try_payment_fallback(
                        resolved_provider, task, reason=reason)
            else:
                fb_client, fb_model, fb_label = _try_configured_fallback_chain(
                    task, resolved_provider or "auto", reason=reason)
                if fb_client is None:
                    fb_client, fb_model, fb_label = _try_main_agent_model_fallback(
                        resolved_provider, task, reason=reason)

            if fb_client is not None:
                fb_kwargs = _build_call_kwargs(
                    fb_label, fb_model, messages,
                    temperature=temperature, max_tokens=max_tokens,
                    tools=tools, timeout=effective_timeout,
                    extra_body=effective_extra_body,
                    base_url=str(getattr(fb_client, "base_url", "") or ""))
                return _validate_llm_response(
                    fb_client.chat.completions.create(**fb_kwargs), task)
            # All fallback layers exhausted — emit a single user-visible
            # warning so the operator knows aux task is about to fail.
            # (#26882) The error itself is re-raised below.
            logger.warning(
                "Auxiliary %s: %s on %s and all fallbacks exhausted "
                "(fallback_chain + main agent model). Raising original error.",
                task or "call", reason, resolved_provider,
            )
        # Connection/timeout errors leave the cached client poisoned (closed
        # httpx transport, half-read stream, dead async loop).  Drop it from
        # the cache regardless of whether we found a fallback above so the
        # next auxiliary call rebuilds a fresh client instead of reusing the
        # dead one.  See issue #23432.
        if _is_connection_error(first_err):
            try:
                _evict_cached_client_instance(client)
            except Exception:
                logger.debug("Auxiliary: cache eviction after connection error failed",
                             exc_info=True)
        raise


def extract_content_or_reasoning(response) -> str:
    """Extract content from an LLM response, falling back to reasoning fields.

    Mirrors the main agent loop's behavior when a reasoning model (DeepSeek-R1,
    Qwen-QwQ, etc.) returns ``content=None`` with reasoning in structured fields.

    Resolution order:
      1. ``message.content`` — strip inline think/reasoning blocks, check for
         remaining non-whitespace text.
      2. ``message.reasoning`` / ``message.reasoning_content`` — direct
         structured reasoning fields (DeepSeek, Moonshot, NovitaAI, etc.).
      3. ``message.reasoning_details`` — OpenRouter unified array format.

    Returns the best available text, or ``""`` if nothing found.
    """
    import re

    msg = response.choices[0].message
    content = (msg.content or "").strip()

    if content:
        # Strip inline think/reasoning blocks (mirrors _strip_think_blocks)
        cleaned = re.sub(
            r"<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>"
            r".*?"
            r"</(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>",
            "", content, flags=re.DOTALL | re.IGNORECASE,
        ).strip()
        if cleaned:
            return cleaned

    # Content is empty or reasoning-only — try structured reasoning fields
    reasoning_parts: list[str] = []
    for field in ("reasoning", "reasoning_content"):
        val = getattr(msg, field, None)
        if val and isinstance(val, str) and val.strip() and val not in reasoning_parts:
            reasoning_parts.append(val.strip())

    details = getattr(msg, "reasoning_details", None)
    if details and isinstance(details, list):
        for detail in details:
            if isinstance(detail, dict):
                summary = (
                    detail.get("summary")
                    or detail.get("content")
                    or detail.get("text")
                )
                if summary and summary not in reasoning_parts:
                    reasoning_parts.append(summary.strip() if isinstance(summary, str) else str(summary))

    if reasoning_parts:
        return "\n\n".join(reasoning_parts)

    return ""


async def async_call_llm(
    task: str = None,
    *,
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    messages: list,
    temperature: float = None,
    max_tokens: int = None,
    tools: list = None,
    timeout: float = None,
    extra_body: dict = None,
) -> Any:
    """Centralized asynchronous LLM call.

    Same as call_llm() but async. See call_llm() for full documentation.
    """
    resolved_provider, resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode = _resolve_task_provider_model(
        task, provider, model, base_url, api_key)
    effective_extra_body = _get_task_extra_body(task)
    effective_extra_body.update(extra_body or {})

    if task == "vision":
        effective_provider, client, final_model = resolve_vision_provider_client(
            provider=resolved_provider if resolved_provider != "auto" else provider,
            model=resolved_model or model,
            base_url=resolved_base_url or base_url,
            api_key=resolved_api_key or api_key,
            async_mode=True,
        )
        if client is None and resolved_provider != "auto" and not resolved_base_url:
            logger.warning(
                "Vision provider %s unavailable, falling back to auto vision backends",
                resolved_provider,
            )
            effective_provider, client, final_model = resolve_vision_provider_client(
                provider="auto",
                model=resolved_model,
                async_mode=True,
            )
        if client is None:
            raise RuntimeError(
                f"No LLM provider configured for task={task} provider={resolved_provider}. "
                f"Run: hermes setup"
            )
        resolved_provider = effective_provider or resolved_provider
    else:
        client, final_model = _get_cached_client(
            resolved_provider,
            resolved_model,
            async_mode=True,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            api_mode=resolved_api_mode,
        )
        if client is None:
            _explicit = (resolved_provider or "").strip().lower()
            if _explicit and _explicit not in {"auto", "openrouter", "custom"}:
                raise RuntimeError(
                    f"Provider '{_explicit}' is set in config.yaml but no API key "
                    f"was found. Set the {_explicit.upper()}_API_KEY environment "
                    f"variable, or switch to a different provider with `hermes model`."
                )
            if not resolved_base_url:
                logger.info("Auxiliary %s: provider %s unavailable, trying auto-detection chain",
                            task or "call", resolved_provider)
                client, final_model = _get_cached_client("auto", async_mode=True, main_runtime=main_runtime, task=task)
        if client is None:
            raise RuntimeError(
                f"No LLM provider configured for task={task} provider={resolved_provider}. "
                f"Run: hermes setup")

    effective_timeout = timeout if timeout is not None else _get_task_timeout(task)

    # Pass the client's actual base_url (not just resolved_base_url) so
    # endpoint-specific temperature overrides can distinguish
    # api.moonshot.ai vs api.kimi.com/coding even on auto-detected routes.
    _client_base = str(getattr(client, "base_url", "") or "")
    kwargs = _build_call_kwargs(
        resolved_provider, final_model, messages,
        temperature=temperature, max_tokens=max_tokens,
        tools=tools, timeout=effective_timeout, extra_body=effective_extra_body,
        base_url=_client_base or resolved_base_url)

    # Convert image blocks for Anthropic-compatible endpoints (e.g. MiniMax)
    if _is_anthropic_compat_endpoint(resolved_provider, _client_base):
        kwargs["messages"] = _convert_openai_images_to_anthropic(kwargs["messages"])

    try:
        # Retry ONCE on the same provider for a transient transport blip
        # before the except-chain escalates to fallback — see call_llm()
        # for the rationale. (PR #16587)
        try:
            return _validate_llm_response(
                await client.chat.completions.create(**kwargs), task)
        except Exception as transient_err:
            if not _is_transient_transport_error(transient_err):
                raise
            logger.info(
                "Auxiliary %s (async): transient transport error; retrying "
                "once on the same provider before fallback: %s",
                task or "call", transient_err,
            )
            return _validate_llm_response(
                await client.chat.completions.create(**kwargs), task)
    except Exception as first_err:
        if "temperature" in kwargs and _is_unsupported_temperature_error(first_err):
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("temperature", None)
            logger.info(
                "Auxiliary %s (async): provider rejected temperature; retrying once without it",
                task or "call",
            )
            try:
                return _validate_llm_response(
                    await client.chat.completions.create(**retry_kwargs), task)
            except Exception as retry_err:
                retry_err_str = str(retry_err)
                if not (
                    _is_payment_error(retry_err)
                    or _is_connection_error(retry_err)
                    or _is_auth_error(retry_err)
                    or "max_tokens" in retry_err_str
                    or "unsupported_parameter" in retry_err_str
                ):
                    raise
                first_err = retry_err
                kwargs = retry_kwargs

        err_str = str(first_err)
        # ZAI vision models (glm-4v-flash etc.) return error code 1210
        # ("API 调用参数有误") when max_tokens is passed on multimodal
        # calls.  The error message does NOT contain "max_tokens" so the
        # generic retry below never fires.  Detect the ZAI-specific error
        # and strip max_tokens before retrying.
        _is_zai_param_error = (
            "1210" in err_str
            and "bigmodel" in str(getattr(client, "base_url", ""))
        )
        if max_tokens is not None and (
            "max_tokens" in err_str
            or "unsupported_parameter" in err_str
            or _is_unsupported_parameter_error(first_err, "max_tokens")
            or _is_zai_param_error
        ):
            kwargs.pop("max_tokens", None)
            kwargs.pop("max_completion_tokens", None)
            try:
                return _validate_llm_response(
                    await client.chat.completions.create(**kwargs), task)
            except Exception as retry_err:
                # If the max_tokens retry also hits a payment or connection
                # error, fall through to the fallback chain below.
                if not (_is_payment_error(retry_err) or _is_connection_error(retry_err) or _is_rate_limit_error(retry_err)):
                    raise
                first_err = retry_err

        # ── Stale-model self-heal (Nous Portal recommendation drift) ───
        # See the sync call_llm() path for the rationale: a long-lived process
        # can pin a Portal-recommended model that has since been dropped from
        # the Nous → OpenRouter catalog, 404'ing every auxiliary call. Force a
        # fresh Portal fetch and retry once with the current recommendation.
        _heal_is_nous = (
            resolved_provider == "nous"
            or base_url_host_matches(_client_base, "inference-api.nousresearch.com")
        )
        if _is_model_not_found_error(first_err) and _heal_is_nous:
            healed_model = _refresh_nous_recommended_model(
                vision=(task == "vision"), stale_model=kwargs.get("model"))
            if healed_model and healed_model != kwargs.get("model"):
                logger.warning(
                    "Auxiliary %s (async): model %r no longer in Nous catalog; "
                    "retrying with refreshed recommendation %r",
                    task or "call", kwargs.get("model"), healed_model,
                )
                kwargs["model"] = healed_model
                try:
                    return _validate_llm_response(
                        await client.chat.completions.create(**kwargs), task)
                except Exception as retry_err:
                    first_err = retry_err

        # ── Nous auth refresh parity with main agent ──────────────────
        client_is_nous = (
            resolved_provider == "nous"
            or base_url_host_matches(_client_base, "inference-api.nousresearch.com")
        )
        if (
            _is_payment_error(first_err)
            and client_is_nous
            and _nous_portal_account_has_fresh_paid_access()
        ):
            refreshed_client, refreshed_model = _refresh_nous_auxiliary_client(
                cache_provider=resolved_provider or "nous",
                model=final_model,
                async_mode=True,
                base_url=resolved_base_url,
                api_key=resolved_api_key,
                api_mode=resolved_api_mode,
                is_vision=(task == "vision"),
            )
            if refreshed_client is not None:
                logger.info(
                    "Auxiliary %s (async): refreshed Nous runtime credentials after paid account check, retrying",
                    task or "call",
                )
                if refreshed_model and refreshed_model != kwargs.get("model"):
                    kwargs["model"] = refreshed_model
                try:
                    return _validate_llm_response(
                        await refreshed_client.chat.completions.create(**kwargs), task)
                except Exception as retry_err:
                    if not (
                        _is_auth_error(retry_err)
                        or _is_payment_error(retry_err)
                        or _is_connection_error(retry_err)
                        or _is_rate_limit_error(retry_err)
                    ):
                        raise
                    first_err = retry_err

        if _is_auth_error(first_err) and client_is_nous:
            refreshed_client, refreshed_model = _refresh_nous_auxiliary_client(
                cache_provider=resolved_provider or "nous",
                model=final_model,
                async_mode=True,
                base_url=resolved_base_url,
                api_key=resolved_api_key,
                api_mode=resolved_api_mode,
                is_vision=(task == "vision"),
            )
            if refreshed_client is not None:
                logger.info("Auxiliary %s (async): refreshed Nous runtime credentials after 401, retrying",
                            task or "call")
                if refreshed_model and refreshed_model != kwargs.get("model"):
                    kwargs["model"] = refreshed_model
                return _validate_llm_response(
                    await refreshed_client.chat.completions.create(**kwargs), task)

        # ── Auth refresh retry (mirrors sync call_llm) ───────────────
        if (_is_auth_error(first_err)
                and resolved_provider not in {"auto", "", None}
                and not client_is_nous):
            if _refresh_provider_credentials(resolved_provider):
                logger.info(
                    "Auxiliary %s (async): refreshed %s credentials after auth error, retrying",
                    task or "call", resolved_provider,
                )
                return await _retry_same_provider_async(
                    task=task,
                    resolved_provider=resolved_provider,
                    resolved_model=resolved_model,
                    resolved_base_url=resolved_base_url,
                    resolved_api_key=resolved_api_key,
                    resolved_api_mode=resolved_api_mode,
                    final_model=final_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    effective_timeout=effective_timeout,
                    effective_extra_body=effective_extra_body,
                )

        # ── Same-provider credential-pool recovery (mirrors sync) ─────
        pool_provider = _recoverable_pool_provider(resolved_provider, client, main_runtime=main_runtime)
        _client_api_key = str(getattr(client, "api_key", "") or "")
        if pool_provider and (_is_auth_error(first_err) or _is_payment_error(first_err) or _is_rate_limit_error(first_err)):
            recovery_err = first_err
            # Skip the extra retry for clear payment/quota errors — the endpoint
            # won't accept another request with the same exhausted key.
            if _is_rate_limit_error(first_err) and not _is_payment_error(first_err):
                try:
                    return _validate_llm_response(
                        await client.chat.completions.create(**kwargs), task)
                except Exception as retry_err:
                    if not (_is_auth_error(retry_err) or _is_payment_error(retry_err) or _is_rate_limit_error(retry_err)):
                        raise
                    recovery_err = retry_err
            if _recover_provider_pool(pool_provider, recovery_err, failed_api_key=_client_api_key):
                logger.info(
                    "Auxiliary %s (async): recovered %s via credential-pool rotation after %s",
                    task or "call", pool_provider, type(recovery_err).__name__,
                )
                try:
                    return await _retry_same_provider_async(
                        task=task,
                        resolved_provider=resolved_provider,
                        resolved_model=resolved_model,
                        resolved_base_url=resolved_base_url,
                        resolved_api_key=resolved_api_key,
                        resolved_api_mode=resolved_api_mode,
                        final_model=final_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools,
                        effective_timeout=effective_timeout,
                        effective_extra_body=effective_extra_body,
                    )
                except Exception as retry2_err:
                    if (_is_payment_error(retry2_err) or _is_auth_error(retry2_err)
                            or _is_rate_limit_error(retry2_err)):
                        _recover_provider_pool(pool_provider, retry2_err)
                        first_err = retry2_err
                    else:
                        raise

        # ── Payment / connection / rate-limit fallback (mirrors sync call_llm) ──
        should_fallback = (
            _is_payment_error(first_err)
            or _is_connection_error(first_err)
            or _is_rate_limit_error(first_err)
        )
        # Capacity errors (payment/quota/connection) bypass the explicit-provider
        # gate — the provider cannot serve the request regardless of user intent.
        # See #26803: daily token quota must fall back like a 402 credit error.
        is_auto = resolved_provider in {"auto", "", None}
        is_capacity_error = _is_payment_error(first_err) or _is_connection_error(first_err)
        if should_fallback and (is_auto or is_capacity_error):
            if _is_payment_error(first_err):
                reason = "payment error"
                _mark_provider_unhealthy(
                    _recoverable_pool_provider(resolved_provider, client) or resolved_provider
                )
            elif _is_rate_limit_error(first_err):
                reason = "rate limit"
            else:
                reason = "connection error"
            logger.info("Auxiliary %s (async): %s on %s (%s), trying fallback",
                        task or "call", reason, resolved_provider, first_err)

            # Fallback order (#26882, #26803):
            #   1. User-configured fallback_chain (per-task) if set
            #   2. For auto: top-level main fallback_providers/fallback_model
            #   3. For auto: built-in auxiliary discovery chain
            #   4. For explicit aux providers: main agent model safety net
            fb_client, fb_model, fb_label = (None, None, "")
            if is_auto:
                fb_client, fb_model, fb_label = _try_configured_fallback_chain(
                    task, resolved_provider or "auto", reason=reason)
                if fb_client is None:
                    fb_client, fb_model, fb_label = _try_main_fallback_chain(
                        task, resolved_provider or "auto", reason=reason)
                if fb_client is None:
                    fb_client, fb_model, fb_label = _try_payment_fallback(
                        resolved_provider, task, reason=reason)
            else:
                fb_client, fb_model, fb_label = _try_configured_fallback_chain(
                    task, resolved_provider or "auto", reason=reason)
                if fb_client is None:
                    fb_client, fb_model, fb_label = _try_main_agent_model_fallback(
                        resolved_provider, task, reason=reason)

            if fb_client is not None:
                fb_kwargs = _build_call_kwargs(
                    fb_label, fb_model, messages,
                    temperature=temperature, max_tokens=max_tokens,
                    tools=tools, timeout=effective_timeout,
                    extra_body=effective_extra_body,
                    base_url=str(getattr(fb_client, "base_url", "") or ""))
                # Convert sync fallback client to async
                async_fb, async_fb_model = _to_async_client(
                    fb_client, fb_model or "", is_vision=(task == "vision")
                )
                if async_fb_model and async_fb_model != fb_kwargs.get("model"):
                    fb_kwargs["model"] = async_fb_model
                return _validate_llm_response(
                    await async_fb.chat.completions.create(**fb_kwargs), task)
            # All fallback layers exhausted — warn before re-raising. (#26882)
            logger.warning(
                "Auxiliary %s (async): %s on %s and all fallbacks exhausted "
                "(fallback_chain + main agent model). Raising original error.",
                task or "call", reason, resolved_provider,
            )
        # Mirror the sync path: drop poisoned clients on connection/timeout
        # so the next aux call rebuilds.  See issue #23432.
        if _is_connection_error(first_err):
            try:
                _evict_cached_client_instance(client)
            except Exception:
                logger.debug("Auxiliary (async): cache eviction after connection error failed",
                             exc_info=True)
        raise
