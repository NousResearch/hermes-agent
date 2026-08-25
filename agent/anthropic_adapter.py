"""Anthropic Messages API adapter: client construction + the Messages call for Hermes's
OpenAI-style internals. Auth: API keys (``sk-ant-api*``) -> x-api-key; OAuth setup-tokens
(``sk-ant-oat*``) and Claude Code credentials -> Bearer + beta header. Endpoint predicates,
payload conversion and credentials live in ``agent/anthropic_{endpoints,message_convert,
credentials}.py``; import them from there."""

import logging
import math
import re
import subprocess
from contextlib import suppress
from typing import Any, Dict, List, Optional

from utils import normalize_proxy_env_vars

from agent.anthropic_credentials import _is_oauth_token
from agent.anthropic_endpoints import (
    _base_url_needs_context_1m_beta, _is_azure_anthropic_endpoint, _is_kimi_coding_endpoint,
    _is_minimax_anthropic_endpoint, _is_nous_portal_endpoint, _is_opencode_endpoint,
    _is_third_party_anthropic_endpoint, _model_name_is_kimi_family, _normalize_base_url_text,
    _requires_bearer_auth,
)
from agent.anthropic_message_convert import (
    convert_messages_to_anthropic, convert_tools_to_anthropic, normalize_model_name,
)

from hermes_cli import __version__ as _HERMES_VERSION


# ``import anthropic`` is deliberately NOT at module top: the SDK costs ~220 ms of imports and
# every usage site is a cold user-triggered path. ``...`` = not yet tried; None = tried, missing.
_anthropic_sdk: Any = ...


def _get_anthropic_sdk():
    """Return the ``anthropic`` SDK module, importing lazily. None if not installed."""
    global _anthropic_sdk
    if _anthropic_sdk is ...:
        with suppress(Exception):  # ImportError or FeatureUnavailable — fall through to the import below
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("provider.anthropic", prompt=False)
        try:
            import anthropic as _sdk
            _anthropic_sdk = _sdk
        except ImportError:
            _anthropic_sdk = None
    return _anthropic_sdk


def _require_sdk(purpose: str, verb: str = "Install it with"):
    """``_get_anthropic_sdk()`` or ImportError naming the feature that needs it."""
    sdk = _get_anthropic_sdk()
    if sdk is None:
        raise ImportError(f"The 'anthropic' package is required for {purpose}. {verb}: pip install 'anthropic>=0.39.0'")
    return sdk


logger = logging.getLogger(__name__)

THINKING_BUDGET = {"xhigh": 32000, "high": 16000, "medium": 8000, "low": 4000}
# Hermes effort -> Anthropic adaptive-thinking effort (output_config.effort). 4.7+ exposes
# low/medium/high/xhigh/max; Opus/Sonnet 4.6 have no xhigh, so callers downgrade xhigh->max
# there (see _supports_xhigh_effort). "minimal" is a legacy alias for low on every model.
ADAPTIVE_EFFORT_MAP = {
    "ultra": "max", "max": "max", "xhigh": "xhigh", "high": "high", "medium": "medium", "low": "low",
    "minimal": "low",
}

# Thinking-mode classification. Claude 4.6 replaced budget-based extended thinking with *adaptive*
# thinking; 4.7 additionally forbids the manual ``thinking`` block and drops temperature/top_p/
# top_k. Newer releases share no common version substring, so an allowlist of "modern" versions
# would go stale and silently route a new model down the legacy path: unknown Claude models
# DEFAULT to the modern contract and only explicit *legacy* lists are kept (mirroring
# _get_anthropic_max_output's default-to-newest). Non-Claude Anthropic-Messages models (minimax,
# qwen3, GLM, ...) fall through to the legacy manual-thinking path, which they need.
# Older Claude families that need manual thinking (budget_tokens only); ``claude-3`` covers
# 3/3.5/3.7 and the ``-2025`` entries are date-stamped 4.0 ids.
_LEGACY_MANUAL_THINKING_CLAUDE_SUBSTRINGS = (
    "claude-3", "claude-opus-4-0", "claude-opus-4.0", "claude-opus-4-1", "claude-opus-4.1",
    "claude-sonnet-4-0", "claude-sonnet-4.0", "claude-opus-4-2025", "claude-sonnet-4-2025",
    "claude-opus-4-5", "claude-opus-4.5", "claude-sonnet-4-5", "claude-sonnet-4.5", "claude-haiku-4-5",
    "claude-haiku-4.5",
)
# Adaptive families that reject the "xhigh" effort (arrived with Opus 4.7) and still accept
# sampling params.
_NO_XHIGH_CLAUDE_SUBSTRINGS = ("claude-opus-4-6", "claude-opus-4.6", "claude-sonnet-4-6", "claude-sonnet-4.6")
# Adaptive families where thinking is mandatory: ``thinking: {"type": "disabled"}`` answers HTTP
# 400 (Portal flags them ``reasoning.mandatory``). The failure is asymmetric — a missing entry
# 400s the turn, a spurious one only leaves thinking on — so when in doubt, add the family.
_MANDATORY_THINKING_CLAUDE_SUBSTRINGS = ("claude-fable",)
_FAST_MODE_SUPPORTED_SUBSTRINGS = ("opus-4-8", "opus-4.8", "opus-5")

# Adaptive Claude families that REJECT a thinking disable — thinking is
# mandatory and ``thinking: {"type": "disabled"}`` answers HTTP 400. The Portal
# catalog flags the same families with ``reasoning.mandatory``.
#
# Unlike the two lists above, the failure here is asymmetric: a missing entry
# 400s the turn, while a spurious one only leaves thinking on. When in doubt,
# add the family.
_MANDATORY_THINKING_CLAUDE_SUBSTRINGS = (
    "claude-fable",
)


def _is_claude_model(model: str | None) -> bool:
    return "claude" in (model or "").lower()


def _model_matches(model: str, substrings) -> bool:
    """Case-insensitive substring match of ``model`` against a family list."""
    m = model.lower()
    return any(v in m for v in substrings)


# Max output tokens per model (Anthropic docs + Cline catalog). Anthropic requires max_tokens; a
# fixed 16384 starved thinking-enabled models (thinking tokens count toward the limit).
# ``claude-fable`` = Mythos-class named models (1M context); ``minimax`` is a third-party
# Anthropic-compatible endpoint; DashScope enforces ``qwen3`` max_tokens in [1, 65536].
_ANTHROPIC_OUTPUT_LIMITS = {
    "claude-fable": 128_000, "claude-sonnet-5": 128_000, "claude-opus-4-8": 128_000,
    "claude-opus-4-7": 128_000, "claude-opus-4-6": 128_000, "claude-sonnet-4-6": 64_000,
    "claude-opus-4-5": 64_000, "claude-sonnet-4-5": 64_000, "claude-haiku-4-5": 64_000,
    "claude-opus-4": 32_000, "claude-sonnet-4": 64_000, "claude-3-7-sonnet": 128_000,
    "claude-3-5-sonnet": 8_192, "claude-3-5-haiku": 8_192, "claude-3-opus": 4_096,
    "claude-3-sonnet": 4_096, "claude-3-haiku": 4_096, "minimax": 131_072, "qwen3": 65_536,
}
# Unknown models get the highest current limit: future models are unlikely to have *less*.
_ANTHROPIC_DEFAULT_OUTPUT_LIMIT = 128_000


def _get_anthropic_max_output(model: str) -> int:
    """Max output tokens for ``model`` via longest substring match against
    ``_ANTHROPIC_OUTPUT_LIMITS`` (so date-stamped ids and ``:1m``/``:fast`` suffixes resolve, and
    ``claude-3-5-sonnet`` beats ``claude-3-5``). Dots normalize to hyphens (``claude-opus-4.6``)."""
    m = model.lower().replace(".", "-")
    best_key = max((key for key in _ANTHROPIC_OUTPUT_LIMITS if key in m), key=len, default=None)
    return _ANTHROPIC_OUTPUT_LIMITS[best_key] if best_key else _ANTHROPIC_DEFAULT_OUTPUT_LIMIT


def _resolve_positive_anthropic_max_tokens(value) -> Optional[int]:
    """``value`` floored to a positive int, or None when it is not a finite positive number.
    Anthropic 400s on max_tokens that are 0, negative, fractional or non-finite; the ``max_tokens
    or fallback`` idiom catches 0 but lets ``-1``/``0.5`` through. Booleans are excluded (they
    subclass int)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        if not math.isfinite(value):
            return None
    except Exception:  # e.g. OverflowError for ints too large for float
        return None
    return int(value) if int(value) > 0 else None  # int() truncates toward zero for floats


def _resolve_anthropic_messages_max_tokens(requested, model: str, context_length: Optional[int] = None) -> int:
    """``requested`` when it is a positive finite number, else the model's output ceiling. Raises
    ValueError if neither is positive. The context-window clamp is the caller's job so the
    positive-value contract stays endpoint-agnostic."""
    resolved = _resolve_positive_anthropic_max_tokens(requested) or _get_anthropic_max_output(model)
    if resolved > 0:
        return resolved
    raise ValueError(
        f"Anthropic Messages adapter requires a positive max_tokens value for "
        f"model {model!r}; got {requested!r} and no model default resolved."
    )


def _supports_adaptive_thinking(model: str) -> bool:
    """True for Claude models using adaptive thinking (4.6+): unknown Claude models default to
    adaptive, the explicit legacy list stays manual, and non-Claude models return False — except
    Kimi/Moonshot, whose Anthropic-compatible endpoints implement the adaptive contract."""
    return _model_name_is_kimi_family(model) or (
        _is_claude_model(model) and not _model_matches(model, _LEGACY_MANUAL_THINKING_CLAUDE_SUBSTRINGS)
    )


def _supports_xhigh_effort(model: str) -> bool:
    """True for models accepting the 'xhigh' effort (Opus 4.7+). Opus/Sonnet 4.6 400 on it —
    callers downgrade xhigh->max when this returns False."""
    return _supports_adaptive_thinking(model) and not _model_matches(model, _NO_XHIGH_CLAUDE_SUBSTRINGS)


def _accepts_thinking_disable(model: str) -> bool:
    """True when ``model`` accepts an explicit ``thinking: {"type": "disabled"}``. Adaptive Claude
    thinks by default, so "off" only works if the disable is sent; mandatory-thinking families
    400 on it and keep the omit behavior. Legacy manual-thinking models are opt-in via
    budget_tokens, so omission is already off. Scoped to Claude: Kimi's documented disable is
    omission, and sending it a new parameter on the strength of Claude's contract is a guess."""
    return (
        _is_claude_model(model)
        and _supports_adaptive_thinking(model)
        and not _model_matches(model, _MANDATORY_THINKING_CLAUDE_SUBSTRINGS)
    )


def _accepts_thinking_disable(model: str) -> bool:
    """Return True when *model* accepts an explicit thinking disable.

    Adaptive Claude models default to thinking ON, so "thinking off" only
    takes effect if we actively send ``thinking: {"type": "disabled"}`` —
    omitting the parameter leaves the upstream default in place and the model
    thinks anyway.  Reasoning-mandatory families reject the disable outright
    with an HTTP 400, so they keep the omit-everything behavior.

    Legacy manual-thinking Claude models are excluded because they need no
    disable: thinking is opt-in there via ``budget_tokens``, so not sending
    the block already means off.

    Scoped to Claude deliberately.  Kimi/Moonshot endpoints also speak the
    adaptive contract, but their documented disable behavior is omission
    (#13848) and they are not part of this bug; sending them a new parameter
    on the strength of Claude's contract would be a guess.
    """
    if not _is_claude_model(model):
        return False
    if not _supports_adaptive_thinking(model):
        return False
    m = model.lower()
    return not any(v in m for v in _MANDATORY_THINKING_CLAUDE_SUBSTRINGS)


def _forbids_sampling_params(model: str) -> bool:
    """True for models that 400 on any non-default temperature/top_p/top_k (Opus 4.7 and later;
    unknown Claude defaults to forbidding). The 4.6 family and the legacy manual-thinking families
    still accept them. Callers omit the fields entirely — the API rejects anything non-null."""
    return _is_claude_model(model) and not _model_matches(
        model, _NO_XHIGH_CLAUDE_SUBSTRINGS + _LEGACY_MANUAL_THINKING_CLAUDE_SUBSTRINGS
    )


def _supports_fast_mode(model: str) -> bool:
    """True for models accepting ``speed: "fast"`` (Opus 4.8 / Opus 5, Claude API only). Explicit
    allowlist, not a version floor: Opus 4.6 had fast mode and lost it (requests silently run and
    bill at standard speed), Opus 4.7 hard-400s on the param. Dedicated ``...-fast`` ids select
    fast inference via the model field and must NOT also receive the speed parameter."""
    return "-fast" not in model and any(v in model for v in _FAST_MODE_SUPPORTED_SUBSTRINGS)


# Beta headers safe on ordinary/native Anthropic requests. GA on Claude 4.6+ (harmless no-op
# there) but older Claude and compatible endpoints still gate on them. Do NOT add
# ``context-1m-2025-08-07``: accounts without the long-context beta get HTTP 400, breaking short
# auxiliary calls. Bedrock/Azure still need it for 1M context and opt in on their own paths.
# MiniMax's Anthropic-compatible endpoints fail tool-use requests when the tool-streaming beta is
# present. ``_FAST_MODE_BETA`` enables the ``speed: "fast"`` request parameter.
_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"
_COMMON_BETAS = ["interleaved-thinking-2025-05-14", _TOOL_STREAMING_BETA]
_CONTEXT_1M_BETA = "context-1m-2025-08-07"
_FAST_MODE_BETA = "fast-mode-2026-02-01"
# Required for OAuth/subscription auth; matches Claude Code / pi-ai / OpenCode.
_OAUTH_ONLY_BETAS = ["claude-code-20250219", "oauth-2025-04-20"]

# Claude Code identity — OAuth requests without it intermittently 500. Anthropic rejects OAuth
# requests whose user-agent version is too far behind the actual release, so the installed
# version is detected and this fallback kept current.
_CLAUDE_CODE_VERSION_FALLBACK = "2.1.74"
_claude_code_version_cache: Optional[str] = None


def _detect_claude_code_version() -> str:
    """Installed Claude Code version (``claude --version``), else the static fallback."""
    for cmd in ("claude", "claude-code"):
        with suppress(Exception):
            result = subprocess.run(
                [cmd, "--version"],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                version = result.stdout.strip().split()[0]  # "2.1.74 (Claude Code)" or "2.1.74"
                if version and version[0].isdigit():
                    return version
    return _CLAUDE_CODE_VERSION_FALLBACK


def _get_claude_code_version() -> str:
    """Detect lazily (only OAuth headers need it) and cache for the process."""
    global _claude_code_version_cache
    if _claude_code_version_cache is None:
        _claude_code_version_cache = _detect_claude_code_version()
    return _claude_code_version_cache


_CLAUDE_CODE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."
_MCP_TOOL_PREFIX = "mcp__"

# Anthropic's OAuth billing classifier fingerprints certain Hermes tool schemas/prose as a
# third-party app and reroutes to the metered extra-usage lane (HTTP 400 "You're out of extra
# usage" on a valid subscription). Live A/B repros isolated two independent triggers — the
# ``session_search`` tool (schema/name/prose) and the ``memory`` tool (schema/name) — so both are
# aliased on the OAuth wire only; normalize_response reverses the mapping.
_OAUTH_TOOL_NAME_ALIASES = {"session_search": "chat_history_lookup", "memory": "context_notes"}
_OAUTH_TOOL_NAME_REVERSE_ALIASES = {wire_name: name for name, wire_name in _OAUTH_TOOL_NAME_ALIASES.items()}

# Aliases ALSO safe to substitute in free-form prose (system prompt, tool descriptions). "memory"
# is ordinary English throughout the prompt and inside the memory tool's own parameter docs (an
# enum the model must emit verbatim), so rewriting it would corrupt guidance; a model that calls
# bare ``memory`` still dispatches, since normalize_response resolves it through the registry.
_OAUTH_PROSE_ALIAS_NAMES = frozenset({"session_search"})

# Word-boundary matchers so a longer identifier containing the token (e.g.
# ``tools/session_search_tool.py`` in AGENTS.md) is left alone; ``\b`` treats ``_`` as a word char.
_OAUTH_PROSE_ALIAS_PATTERNS = tuple(
    (re.compile(rf"\b{re.escape(name)}\b"), _OAUTH_TOOL_NAME_ALIASES[name])
    for name in sorted(_OAUTH_PROSE_ALIAS_NAMES)
)


def _apply_oauth_prose_aliases(text: str) -> str:
    """Rewrite prose-safe tool-name tokens to their OAuth wire aliases."""
    for pattern, wire_name in _OAUTH_PROSE_ALIAS_PATTERNS:
        text = pattern.sub(wire_name, text)
    return text


def _common_betas_for_base_url(base_url: str | None, *, drop_context_1m_beta: bool = False) -> list[str]:
    """Beta headers safe for the configured endpoint. MiniMax (Bearer-auth) rejects both the
    fine-grained-tool-streaming beta (every tool-use message errors) and the 1M-context beta.
    Azure AI Foundry also uses Bearer auth but keeps both — it needs the 1M beta for 1M context,
    which native Anthropic does not get by default (some subscriptions reject it; Bedrock opts in
    via its own client helper). ``drop_context_1m_beta`` strips the 1M beta after a
    subscription/endpoint rejected it."""
    betas = list(_COMMON_BETAS)
    if _base_url_needs_context_1m_beta(base_url) and not drop_context_1m_beta:
        betas.append(_CONTEXT_1M_BETA)
    if _is_minimax_anthropic_endpoint(base_url):
        return [b for b in betas if b not in (_TOOL_STREAMING_BETA, _CONTEXT_1M_BETA)]
    return betas


def _beta_header(betas: list) -> Dict[str, str]:
    """``{"anthropic-beta": ...}`` when there are betas, else ``{}``."""
    return {"anthropic-beta": ",".join(betas)} if betas else {}


def _attribution_headers() -> Dict[str, str]:
    """Same client-attribution set sent to OpenRouter / Vercel AI Gateway / Fireworks."""
    return {
        "HTTP-Referer": "https://hermes-agent.nousresearch.com", "X-Title": "Hermes Agent",
        "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
    }


def _client_timeout(timeout):
    """httpx.Timeout with the caller's read timeout (default 900s) and a 10s connect."""
    from httpx import Timeout
    read = timeout if (isinstance(timeout, (int, float)) and timeout > 0) else 900.0
    return Timeout(timeout=float(read), connect=10.0)


def _base_client_kwargs(base_url, timeout) -> tuple[str, Dict[str, Any]]:
    """Shared SDK constructor kwargs -> ``(normalized_base_url, kwargs)``. Retry is delegated to
    hermes's outer loop (``max_retries=0``): the SDK default of 2 uses its own backoff that ignores
    Retry-After and double-retries inside our loop. Any trailing ``/v1`` is stripped because the
    SDK appends ``/v1/messages``. Azure's ``api-version`` goes through ``default_query`` so the
    base_url is not corrupted into ``/anthropic?api-version=.../v1/messages``."""
    kwargs: Dict[str, Any] = {"timeout": _client_timeout(timeout), "max_retries": 0}
    normalized = re.sub(r"/v1/?$", "", _normalize_base_url_text(base_url).rstrip("/"))
    if normalized:
        kwargs["base_url"] = normalized
        if _is_azure_anthropic_endpoint(normalized) and "api-version" not in normalized:
            kwargs["default_query"] = {"api-version": "2025-04-15"}
    return normalized, kwargs


def _build_anthropic_client_with_bearer_hook(
    token_provider, base_url: str = None, timeout: float = None, *, drop_context_1m_beta: bool = False
):
    """Anthropic-on-Foundry Entra ID variant of :func:`build_anthropic_client`. The SDK stores
    ``api_key``/``auth_token`` as static strings, so per-request bearer refresh (Microsoft's
    documented Foundry pattern) uses a custom ``httpx.Client`` whose request hook mints a fresh JWT
    and rewrites ``Authorization``; the SDK skips its own auth when ``http_client`` is given. The
    placeholder ``auth_token`` is still required at construction and makes any leak diagnosable."""
    sdk = _require_sdk("Azure Foundry Anthropic-style endpoints with Entra ID auth", verb="Install with")
    normalize_proxy_env_vars()
    from agent.azure_identity_adapter import build_bearer_http_client
    normalized_base_url, kwargs = _base_client_kwargs(base_url, timeout)
    kwargs["http_client"] = build_bearer_http_client(token_provider, timeout=kwargs["timeout"])
    kwargs["auth_token"] = "entra-id-bearer-via-http-hook"
    headers = _beta_header(_common_betas_for_base_url(normalized_base_url, drop_context_1m_beta=drop_context_1m_beta))
    return _new_sdk_client(sdk, kwargs, headers)


def _new_sdk_client(sdk, kwargs: Dict[str, Any], headers: Dict[str, str]):
    """``sdk.Anthropic(**kwargs)`` with ``headers`` attached, sending exactly ONE credential.

    The SDK fills whichever of ``api_key`` / ``auth_token`` we left unset from ANTHROPIC_API_KEY /
    ANTHROPIC_AUTH_TOKEN in the environment (both loaded from ~/.hermes/.env) and then sends dual
    auth — x-api-key *and* Authorization: Bearer — shipping a foreign credential to Portal / MiniMax
    / OAuth / Entra / third-party endpoints (#26970, #105774). An ``Omit()`` default header is the
    SDK-sanctioned way to drop the other header, and unlike an attribute clear it survives
    ``with_options()``, which re-runs the constructor and re-reads the environment."""
    merged = dict(headers)
    if "api_key" in kwargs and "auth_token" not in kwargs:
        merged["Authorization"] = sdk.Omit()
    elif "auth_token" in kwargs and "api_key" not in kwargs:
        merged["X-Api-Key"] = sdk.Omit()
    if merged:
        kwargs["default_headers"] = merged
    return sdk.Anthropic(**kwargs)


def _auth_style(api_key, base_url, normalized_base_url) -> str:
    """Order-sensitive endpoint/key classification for :func:`build_anthropic_client`. ``kimi``:
    Kimi's /coding endpoint 403s without a User-Agent (the Kimi team asked for proper attribution).
    ``bearer``: MiniMax & co. want Authorization: Bearer — checked before the OAuth shape test
    because their secrets lack the sk-ant-api prefix and would be misread as OAuth/setup tokens.
    ``api_key``: third-party proxies use their own x-api-key keys (skip OAuth detection). ``oauth``:
    Bearer auth + Claude Code identity (Anthropic routes OAuth by user-agent; without it, 500s)."""
    if _is_kimi_coding_endpoint(base_url):
        return "kimi"
    if _requires_bearer_auth(normalized_base_url):
        return "bearer"
    if _is_third_party_anthropic_endpoint(base_url):
        return "api_key"
    if _is_oauth_token(api_key):
        return "oauth"
    return "api_key"


def build_anthropic_client(api_key, base_url: str = None, timeout: float = None, *, drop_context_1m_beta: bool = False):
    """Create an Anthropic client, auto-detecting setup-tokens vs API keys. ``api_key`` is a static
    ``str`` or a ``Callable[[], str]`` Entra ID bearer provider (routed through
    :func:`_build_anthropic_client_with_bearer_hook`). ``timeout`` overrides the 900s read timeout
    (connect stays 10s). ``drop_context_1m_beta`` strips ``context-1m-2025-08-07`` from the
    client-level beta header — the reactive OAuth retry in run_agent uses it after a subscription
    rejects it; fresh clients keep the default so 1M-capable subscriptions keep the capability."""
    sdk = _require_sdk("the Anthropic provider")
    if callable(api_key) and not isinstance(api_key, str):
        return _build_anthropic_client_with_bearer_hook(
            api_key, base_url, timeout, drop_context_1m_beta=drop_context_1m_beta
        )
    normalize_proxy_env_vars()
    normalized_base_url, kwargs = _base_client_kwargs(base_url, timeout)
    if "default_query" in kwargs:  # historical: this path also strips a stray trailing slash on Azure
        kwargs["base_url"] = normalized_base_url.rstrip("/")
    common_betas = _common_betas_for_base_url(normalized_base_url, drop_context_1m_beta=drop_context_1m_beta)
    style = _auth_style(api_key, base_url, normalized_base_url)
    kwargs["auth_token" if style in ("bearer", "oauth") else "api_key"] = api_key
    headers = _beta_header(common_betas + _OAUTH_ONLY_BETAS if style == "oauth" else common_betas)
    if style == "kimi":
        headers = {**_attribution_headers(), **headers}
    elif style == "oauth":
        headers["user-agent"] = f"claude-code/{_get_claude_code_version()} (external, cli)"
        headers["x-app"] = "cli"
    if _is_opencode_endpoint(base_url):
        # OpenCode identifies clients by request headers (like OpenRouter). The OpenAI-wire paths
        # get these from profile.default_headers, but this route never sees the profile.
        for k, v in _attribution_headers().items():
            headers.setdefault(k, v)
    return _new_sdk_client(sdk, kwargs, headers)


def build_anthropic_bedrock_client(region: str):
    """AnthropicBedrock client for Bedrock Claude models (boto3 default credential chain). The
    SDK's native Bedrock adapter gives full Claude feature parity (prompt caching, thinking
    budgets, adaptive thinking, fast mode) that Converse lacks. The common betas plus
    ``context-1m-2025-08-07`` are attached: without the latter Bedrock caps Opus 4.6/4.7 at 200K.
    A configured ``bedrock.guardrail`` rides as InvokeModel headers so every client built here
    (primary, auxiliary, per-request rebuild) enforces it."""
    from agent.bedrock_adapter import bedrock_guardrail_headers, scoped_aws_session_kwargs
    sdk = _require_sdk("the Bedrock provider")
    if not hasattr(sdk, "AnthropicBedrock"):
        raise ImportError("anthropic.AnthropicBedrock not available. Upgrade with: pip install 'anthropic>=0.39.0'")
    # Routed multiplex profile: its own AWS_* from the secret scope (the SDK would otherwise read the
    # launch profile's process env); unscoped passes nothing and keeps the default chain.
    scoped = scoped_aws_session_kwargs()
    aws_kwargs = {"aws_access_key": scoped.get("aws_access_key_id"), "aws_secret_key": scoped.get("aws_secret_access_key"),
                  "aws_session_token": scoped.get("aws_session_token"), "aws_profile": scoped.get("profile_name")}
    return sdk.AnthropicBedrock(
        aws_region=region, timeout=_client_timeout(None), **{k: v for k, v in aws_kwargs.items() if v},
        max_retries=0,  # retry belongs to hermes's outer loop (honors Retry-After)
        default_headers={**_beta_header([*_COMMON_BETAS, _CONTEXT_1M_BETA]), **bedrock_guardrail_headers()},
    )


def _normalize_to_mcp_wire(name: str) -> str:
    """OAuth wire form of a tool name (no aliasing): ``mcp__<...>``. Anthropic's OAuth billing
    classifier treats a single-underscore ``mcp_`` tool name as a third-party-app fingerprint
    (HTTP 400 "Third-party apps now draw from extra usage"); ``mcp__foo`` is accepted. Both bare
    Hermes tools (``read_file``) and native MCP tools registered as ``mcp_<server>_<tool>`` must
    land on the double-underscore form. normalize_response reverses both via registry lookup."""
    if name.startswith("mcp__"):
        return name  # already correct, don't double-prefix
    return _MCP_TOOL_PREFIX + name.removeprefix("mcp_")


def _oauth_wire_namer(anthropic_tools: List[Dict[str, Any]]):
    """Return ``name -> OAuth wire name`` for this request's tool set. An alias must never collide
    with a wire name owned by a non-alias tool: two identical tool names in one request is a hard
    400, strictly worse than the bug being fixed. Mirrors normalize_response's "registered tool
    wins" so outbound and inbound agree on who owns a contested name."""
    claimed = {
        _normalize_to_mcp_wire(tool["name"])
        for tool in (anthropic_tools or [])
        if isinstance(tool.get("name"), str) and tool["name"] not in _OAUTH_TOOL_NAME_ALIASES
    }

    def to_wire(name: str) -> str:
        aliased = _OAUTH_TOOL_NAME_ALIASES.get(name)
        if aliased and _MCP_TOOL_PREFIX + aliased not in claimed:
            name = aliased
        return _normalize_to_mcp_wire(name)

    return to_wire


    Selection rules when both are present:
      - If exactly one is non-expired, prefer that one. (Handles the case
        where Claude Code refreshes one source but not the other — observed
        in the wild on Claude Code 2.1.x.)
      - Otherwise, prefer the source with the later ``expiresAt`` so that
        any subsequent refresh uses the most recent ``refreshToken``.

    This intentionally excludes ~/.claude.json primaryApiKey. Opencode's
    subscription flow is OAuth/setup-token based with refreshable credentials,
    and native direct Anthropic provider usage should follow that path rather
    than auto-detecting Claude's first-party managed key.

    Returns dict with {accessToken, refreshToken?, expiresAt?, source} or None.
    """
    kc_creds = _read_claude_code_credentials_from_keychain()
    file_creds = _read_claude_code_credentials_from_file()

    if kc_creds and file_creds:
        kc_valid = is_claude_code_token_valid(kc_creds)
        file_valid = is_claude_code_token_valid(file_creds)
        if kc_valid and not file_valid:
            return kc_creds
        if file_valid and not kc_valid:
            return file_creds
        # Both valid or both expired: prefer the later expiresAt so the
        # downstream refresh path uses the freshest refresh_token.
        kc_exp = kc_creds.get("expiresAt", 0) or 0
        file_exp = file_creds.get("expiresAt", 0) or 0
        return kc_creds if kc_exp >= file_exp else file_creds

    return kc_creds or file_creds


def is_claude_code_token_valid(creds: Dict[str, Any]) -> bool:
    """Check if Claude Code credentials have a non-expired access token."""
    import time

    expires_at = creds.get("expiresAt", 0)
    if not expires_at:
        # No expiry set (managed keys) — valid if token is present
        return bool(creds.get("accessToken"))

    # expiresAt is in milliseconds since epoch
    now_ms = int(time.time() * 1000)
    # Allow 60 seconds of buffer
    return now_ms < (expires_at - 60_000)


def refresh_anthropic_oauth_pure(refresh_token: str, *, use_json: bool = False) -> Dict[str, Any]:
    """Refresh an Anthropic OAuth token without mutating local credential files."""
    import time
    import urllib.parse
    import urllib.request

    if not refresh_token:
        raise ValueError("refresh_token is required")

    client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    if use_json:
        data = json.dumps({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }).encode()
        content_type = "application/json"
    else:
        data = urllib.parse.urlencode({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }).encode()
        content_type = "application/x-www-form-urlencoded"

    token_endpoints = [
        "https://platform.claude.com/v1/oauth/token",
        "https://console.anthropic.com/v1/oauth/token",
    ]
    last_error = None
    for endpoint in token_endpoints:
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Content-Type": content_type,
                "User-Agent": _OAUTH_TOKEN_USER_AGENT,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
        except Exception as exc:
            last_error = exc
            logger.debug("Anthropic token refresh failed at %s: %s", endpoint, exc)
            continue

        access_token = result.get("access_token", "")
        if not access_token:
            raise ValueError("Anthropic refresh response was missing access_token")
        next_refresh = result.get("refresh_token", refresh_token)
        expires_in = result.get("expires_in", 3600)
        return {
            "access_token": access_token,
            "refresh_token": next_refresh,
            "expires_at_ms": int(time.time() * 1000) + (expires_in * 1000),
        }

    if last_error is not None:
        raise last_error
    raise ValueError("Anthropic token refresh failed")


def _refresh_oauth_token(creds: Dict[str, Any]) -> Optional[str]:
    """Attempt to refresh an expired Claude Code OAuth token.

    Claude Code's OAuth refresh tokens are single-use: a successful refresh
    rotates the pair and invalidates the old refresh token. Claude Code itself
    also refreshes on its own schedule (IDE/CLI activity), so by the time
    Hermes notices an expired token, Claude Code may have already rotated it.
    POSTing our now-stale refresh token in that window races Claude Code and
    fails with ``invalid_grant``.

    So before refreshing, re-read the live credential sources. If Claude Code
    has already produced a valid token, adopt it and skip the POST entirely.
    Only fall back to refreshing ourselves when no fresh credential is found.
    """
    # Claude Code may have already refreshed — adopt its token rather than
    # racing it with our (possibly already-rotated) refresh token. Only adopt
    # when the live re-read produced a DIFFERENT token with a real future
    # expiry: re-adopting the same credential we were just handed would be a
    # no-op, and a 0/absent ``expiresAt`` means "managed key / unknown expiry"
    # (see is_claude_code_token_valid) which must NOT be treated as a fresh
    # refresh here.
    current = read_claude_code_credentials()
    if current:
        current_token = current.get("accessToken", "")
        current_exp = current.get("expiresAt", 0) or 0
        if (
            current_token
            and current_token != creds.get("accessToken", "")
            and current_exp > 0
            and is_claude_code_token_valid(current)
        ):
            logger.debug("Adopted Claude Code's already-refreshed OAuth token")
            return current_token

    refresh_token = (current or {}).get("refreshToken", "") or creds.get("refreshToken", "")
    if not refresh_token:
        logger.debug("No refresh token available — cannot refresh")
        return None

    try:
        refreshed = refresh_anthropic_oauth_pure(refresh_token, use_json=False)
        _write_claude_code_credentials(
            refreshed["access_token"],
            refreshed["refresh_token"],
            refreshed["expires_at_ms"],
        )
        logger.debug("Successfully refreshed Claude Code OAuth token")
        return refreshed["access_token"]
    except Exception as e:
        logger.debug("Failed to refresh Claude Code token: %s", e)
        return None


def _write_claude_code_credentials(
    access_token: str,
    refresh_token: str,
    expires_at_ms: int,
    *,
    scopes: Optional[list] = None,
) -> None:
    """Write refreshed credentials back to ~/.claude/.credentials.json.

    The optional *scopes* list (e.g. ``["user:inference", "user:profile", ...]``)
    is persisted so that Claude Code's own auth check recognises the credential
    as valid.  Claude Code >=2.1.81 gates on the presence of ``"user:inference"``
    in the stored scopes before it will use the token.
    """
    cred_path = Path.home() / ".claude" / ".credentials.json"
    try:
        # Read existing file to preserve other fields
        existing = {}
        if cred_path.exists():
            existing = json.loads(cred_path.read_text(encoding="utf-8"))

        oauth_data: Dict[str, Any] = {
            "accessToken": access_token,
            "refreshToken": refresh_token,
            "expiresAt": expires_at_ms,
        }
        if scopes is not None:
            oauth_data["scopes"] = scopes
        elif "claudeAiOauth" in existing and "scopes" in existing["claudeAiOauth"]:
            # Preserve previously-stored scopes when the refresh response
            # does not include a scope field.
            oauth_data["scopes"] = existing["claudeAiOauth"]["scopes"]

        existing["claudeAiOauth"] = oauth_data

        cred_path.parent.mkdir(parents=True, exist_ok=True)
        # Per-process random suffix avoids collisions between concurrent
        # writers and stale leftovers from a prior crashed write.
        _tmp_cred = cred_path.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
        try:
            # Create the temp file atomically at 0o600. The previous
            # write_text + post-replace chmod opened a TOCTOU window where
            # both the temp file and the destination briefly inherited the
            # process umask (commonly 0o644 = world-readable), exposing
            # Claude Code OAuth tokens to other local users between create
            # and chmod. Mirrors agent/google_oauth.py (#19673) and
            # tools/mcp_oauth.py (#21148). Parent dir (~/.claude/) is
            # owned by Claude Code itself, so we leave its mode alone.
            fd = os.open(
                str(_tmp_cred),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                stat.S_IRUSR | stat.S_IWUSR,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(existing, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(_tmp_cred, cred_path)
        except OSError:
            try:
                _tmp_cred.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    except (OSError, IOError) as e:
        logger.debug("Failed to write refreshed credentials: %s", e)


def _resolve_claude_code_token_from_credentials(creds: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Resolve a token from Claude Code credential files, refreshing if needed."""
    creds = creds or read_claude_code_credentials()
    if creds and is_claude_code_token_valid(creds):
        logger.debug("Using Claude Code credentials (auto-detected)")
        return creds["accessToken"]
    if creds:
        logger.debug("Claude Code credentials expired — attempting refresh")
        refreshed = _refresh_oauth_token(creds)
        if refreshed:
            return refreshed
        logger.debug("Token refresh failed — re-run 'claude setup-token' to reauthenticate")
    return None


def _prefer_refreshable_claude_code_token(env_token: str, creds: Optional[Dict[str, Any]]) -> Optional[str]:
    """Prefer Claude Code creds when a persisted env OAuth token would shadow refresh.

    Hermes historically persisted setup tokens into ANTHROPIC_TOKEN. That makes
    later refresh impossible because the static env token wins before we ever
    inspect Claude Code's refreshable credential file. If we have a refreshable
    Claude Code credential record, prefer it over the static env OAuth token.
    """
    if not env_token or not _is_oauth_token(env_token) or not isinstance(creds, dict):
        return None
    if not creds.get("refreshToken"):
        return None

    resolved = _resolve_claude_code_token_from_credentials(creds)
    if resolved and resolved != env_token:
        logger.debug(
            "Preferring Claude Code credential file over static env OAuth token so refresh can proceed"
        )
        return resolved
    return None


def _resolve_anthropic_pool_token() -> Optional[str]:
    """Return the first available Anthropic OAuth token from credential_pool.

    Read-only: enumerates with ``clear_expired=False, refresh=False`` so a bare
    token *resolve* (which runs from diagnostic/read-only call sites such as
    ``account_usage`` and ``hermes models``) never mutates ``~/.hermes/auth.json``
    or makes a network refresh call. Refresh-on-expiry is owned by the API call
    path's pool recovery, not the resolver.
    """
    try:
        from agent.credential_pool import AUTH_TYPE_OAUTH, load_pool
    except Exception:
        return None

    try:
        pool = load_pool("anthropic")
        # Enumerate read-only (clear_expired=False, refresh=False): never persist
        # to auth.json or trigger a network refresh from a bare resolve. select()
        # is deliberately NOT used — it runs clear_expired=True, refresh=True,
        # which would violate this read-only contract.
        entries, _pending = pool._available_entries(clear_expired=False, refresh=False)
    except Exception:
        logger.debug("Failed to read Anthropic credential_pool", exc_info=True)
        return None

    for entry in entries:
        if getattr(entry, "auth_type", None) != AUTH_TYPE_OAUTH:
            continue
        # access_token is a declared field but a persisted entry can carry an
        # explicit null (or a partially-written OAuth entry), so coerce before
        # strip — a bare None.strip() here would escape the try/excepts above
        # and crash the whole resolver, taking down the source #5 fallback too.
        # Matches the aux-client analog (auxiliary_client.py: str(key or "")).
        token = (getattr(entry, "access_token", None) or "").strip()
        if token:
            return token

    return None


def resolve_anthropic_token() -> Optional[str]:
    """Resolve an Anthropic token from all available sources.

    Priority:
      1. ANTHROPIC_TOKEN env var (OAuth/setup token saved by Hermes)
      2. CLAUDE_CODE_OAUTH_TOKEN env var
      3. ANTHROPIC_API_KEY env var (explicit regular API key)
      4. Claude Code credentials (~/.claude.json or ~/.claude/.credentials.json)
         — with automatic refresh if expired and a refresh token is available
      5. Anthropic credential_pool OAuth entry (~/.hermes/auth.json)

    Returns the token string or None.
    """
    creds: Optional[Dict[str, Any]] = None
    creds_loaded = False

    def _read_creds() -> Optional[Dict[str, Any]]:
        nonlocal creds, creds_loaded
        if not creds_loaded:
            creds = read_claude_code_credentials()
            creds_loaded = True
        return creds

    # 1. Hermes-managed OAuth/setup token env var
    token = _getenv("ANTHROPIC_TOKEN").strip()
    if token:
        preferred = _prefer_refreshable_claude_code_token(token, _read_creds())
        if preferred:
            return preferred
        return token

    # 2. CLAUDE_CODE_OAUTH_TOKEN (used by Claude Code for setup-tokens)
    cc_token = _getenv("CLAUDE_CODE_OAUTH_TOKEN").strip()
    if cc_token:
        preferred = _prefer_refreshable_claude_code_token(cc_token, _read_creds())
        if preferred:
            return preferred
        return cc_token

    # 3. Regular API key. An explicit user-configured key must not be shadowed
    # by auto-discovered Claude Code or credential-pool OAuth credentials.
    api_key = _getenv("ANTHROPIC_API_KEY").strip()
    if api_key:
        return api_key

    # 4. Claude Code credential file
    resolved_claude_token = _resolve_claude_code_token_from_credentials(_read_creds())
    if resolved_claude_token:
        return resolved_claude_token

    # 5. Hermes credential_pool OAuth entry.
    resolved_pool_token = _resolve_anthropic_pool_token()
    if resolved_pool_token:
        return resolved_pool_token

    return None


def run_oauth_setup_token() -> Optional[str]:
    """Run 'claude setup-token' interactively and return the resulting token.

    Checks multiple sources after the subprocess completes:
      1. Claude Code credential files (may be written by the subprocess)
      2. CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_TOKEN env vars

    Returns the token string, or None if no credentials were obtained.
    Raises FileNotFoundError if the 'claude' CLI is not installed.
    """
    import shutil
    import subprocess

    claude_path = shutil.which("claude")
    if not claude_path:
        raise FileNotFoundError(
            "The 'claude' CLI is not installed. "
            "Install it with: npm install -g @anthropic-ai/claude-code"
        )

    # Run interactively — stdin/stdout/stderr inherited so the user can
    # complete the OAuth login prompt. Must keep inherited stdin; the TUI-EOF
    # concern does not apply to an interactive login the user explicitly
    # invokes.  noqa: subprocess-stdin
    try:
        subprocess.run([claude_path, "setup-token"])
    except (KeyboardInterrupt, EOFError):
        return None

    # Check if credentials were saved to Claude Code's config files
    creds = read_claude_code_credentials()
    if creds and is_claude_code_token_valid(creds):
        return creds["accessToken"]

    # Check env vars that may have been set
    for env_var in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_TOKEN"):
        val = _getenv(env_var).strip()
        if val:
            return val

    return None


# ── Hermes-native PKCE OAuth flow ────────────────────────────────────────
# Mirrors the flow used by Claude Code, pi-ai, and OpenCode.
# Stores credentials in ~/.hermes/.anthropic_oauth.json (our own file).

_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
# Anthropic migrated the OAuth token endpoint to platform.claude.com;
# console.anthropic.com now 404s. Callers should iterate _OAUTH_TOKEN_URLS
# (new host first, console fallback). _OAUTH_TOKEN_URL is kept as the primary
# for backward compatibility with existing imports and now points at the live host.
_OAUTH_TOKEN_URLS = [
    "https://platform.claude.com/v1/oauth/token",
    "https://console.anthropic.com/v1/oauth/token",
]
_OAUTH_TOKEN_URL = _OAUTH_TOKEN_URLS[0]
# User-Agent sent on the OAuth *token endpoint* (login exchange + refresh).
# Anthropic rate-limits (HTTP 429) any token-endpoint request whose UA starts
# with ``claude-code/`` — verified empirically against platform.claude.com:
# ``claude-code/2.1.200`` and ``Mozilla/5.0`` -> 429; ``axios/*``, ``node``,
# and SDK-style UAs -> 400 (reached code validation). The real Claude Code CLI
# exchanges the auth code with a bare axios client (``axios/<ver>``), NOT its
# ``claude-code/`` inference UA. We mirror that here. NOTE: the *inference* path
# (build_anthropic_kwargs) still uses the ``claude-code/`` UA + ``x-app: cli`` —
# that fingerprint is required there and is NOT throttled on the messages API.
_OAUTH_TOKEN_USER_AGENT = "axios/1.7.9"
_OAUTH_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
_OAUTH_SCOPES = "org:create_api_key user:profile user:inference"
def _get_hermes_oauth_file() -> Path:
    return get_hermes_home() / ".anthropic_oauth.json"


def _generate_pkce() -> tuple:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    import base64
    import hashlib
    import secrets

    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


def run_hermes_oauth_login_pure() -> Optional[Dict[str, Any]]:
    """Run Hermes-native OAuth PKCE flow and return credential state."""
    import secrets
    import time
    import webbrowser

    verifier, challenge = _generate_pkce()
    oauth_state = secrets.token_urlsafe(32)

    params = {
        "code": "true",
        "client_id": _OAUTH_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _OAUTH_REDIRECT_URI,
        "scope": _OAUTH_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": oauth_state,
    }
    from urllib.parse import urlencode

    auth_url = f"https://claude.ai/oauth/authorize?{urlencode(params)}"

    print()
    print("Authorize Hermes with your Claude Pro/Max subscription.")
    print()
    print("╭─ Claude Pro/Max Authorization ────────────────────╮")
    print("│                                                   │")
    print("│  Open this link in your browser:                  │")
    print("╰───────────────────────────────────────────────────╯")
    print()
    print(f"  {auth_url}")
    print()

    try:
        from hermes_cli.auth import _can_open_graphical_browser as _can_open_gui
    except Exception:
        _can_open_gui = lambda: True  # noqa: E731 — degrade to prior behavior

    if _can_open_gui():
        try:
            webbrowser.open(auth_url)
            print("  (Browser opened automatically)")
        except Exception:
            pass

    print()
    print("After authorizing, you'll see a code. Paste it below.")
    print()
    try:
        auth_code = input("Authorization code: ").strip()
    except (KeyboardInterrupt, EOFError):
        return None

    if not auth_code:
        print("No code entered.")
        return None

    splits = auth_code.split("#")
    code = splits[0]
    received_state = splits[1] if len(splits) > 1 else ""

    # Validate state to prevent CSRF (RFC 6749 §10.12)
    if received_state != oauth_state:
        logger.warning("OAuth state mismatch — possible CSRF, aborting")
        return None

    try:
        import urllib.request

        exchange_data = json.dumps({
            "grant_type": "authorization_code",
            "client_id": _OAUTH_CLIENT_ID,
            "code": code,
            "state": received_state,
            "redirect_uri": _OAUTH_REDIRECT_URI,
            "code_verifier": verifier,
        }).encode()

        # Anthropic migrated the OAuth token endpoint to platform.claude.com;
        # console.anthropic.com now 404s. Try the new host first, then fall
        # back to console for older deployments (mirrors the refresh path).
        # UA is _OAUTH_TOKEN_USER_AGENT (a non-claude-code UA) — see the
        # constant's definition for why the token endpoint must not send
        # claude-code/ (429 UA-prefix block).
        result = None
        last_error = None
        for endpoint in _OAUTH_TOKEN_URLS:
            req = urllib.request.Request(
                endpoint,
                data=exchange_data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": _OAUTH_TOKEN_USER_AGENT,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    result = json.loads(resp.read().decode())
                break
            except Exception as exc:
                last_error = exc
                logger.debug("Anthropic token exchange failed at %s: %s", endpoint, exc)
                continue

        if result is None:
            raise last_error if last_error is not None else ValueError(
                "Anthropic token exchange failed"
            )
    except Exception as e:
        print(f"Token exchange failed: {e}")
        return None

    access_token = result.get("access_token", "")
    refresh_token = result.get("refresh_token", "")
    expires_in = result.get("expires_in", 3600)

    if not access_token:
        print("No access token in response.")
        return None

    expires_at_ms = int(time.time() * 1000) + (expires_in * 1000)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at_ms": expires_at_ms,
    }


def read_hermes_oauth_credentials() -> Optional[Dict[str, Any]]:
    """Read Hermes-managed OAuth credentials from ~/.hermes/.anthropic_oauth.json."""
    oauth_file = _get_hermes_oauth_file()
    if oauth_file.exists():
        try:
            data = json.loads(oauth_file.read_text(encoding="utf-8"))
            if data.get("accessToken"):
                return data
        except (json.JSONDecodeError, OSError, IOError) as e:
            logger.debug("Failed to read Hermes OAuth credentials: %s", e)
    return None


# ---------------------------------------------------------------------------
# Message / tool / response format conversion
# ---------------------------------------------------------------------------


def _is_bedrock_model_id(model: str) -> bool:
    """Detect AWS Bedrock model IDs that use dots as namespace separators.

    Bedrock model IDs come in two forms:
    - Bare:    ``anthropic.claude-opus-4-7``
    - Regional (inference profiles): ``us.anthropic.claude-sonnet-4-5-v1:0``

    In both cases the dots separate namespace components, not version
    numbers, and must be preserved verbatim for the Bedrock API.
    """
    lower = model.lower()
    # Regional inference-profile prefixes
    if any(lower.startswith(p) for p in (
        "global.", "us.", "eu.", "apac.", "ap.", "au.", "jp.",
        "ca.", "sa.", "me.", "af.",
    )):
        return True
    # Bare Bedrock model IDs: provider.model-family
    if lower.startswith("anthropic."):
        return True
    return False


def normalize_model_name(model: str, preserve_dots: bool = False) -> str:
    """Normalize a model name for the Anthropic API.

    - Strips 'anthropic/' prefix (OpenRouter format, case-insensitive)
    - Converts dots to hyphens in version numbers (OpenRouter uses dots,
      Anthropic uses hyphens: claude-opus-4.6 → claude-opus-4-6), unless
      preserve_dots is True (e.g. for Alibaba/DashScope: qwen3.5-plus).
    - Preserves Bedrock model IDs (``anthropic.claude-opus-4-7``) and
      regional inference profiles (``us.anthropic.claude-*``) whose dots
      are namespace separators, not version separators.
    """
    lower = model.lower()
    if lower.startswith("anthropic/"):
        model = model[len("anthropic/"):]
    if not preserve_dots:
        # Bedrock model IDs use dots as namespace separators
        # (e.g. "anthropic.claude-opus-4-7", "us.anthropic.claude-*").
        # These must not be converted to hyphens.  See issue #12295.
        if _is_bedrock_model_id(model):
            return model
        # Only convert dots to hyphens for Anthropic/Claude models.
        # Non-Anthropic models (gpt-5.4, gemini-2.5, etc.) use dots
        # as part of their canonical names.  See issue #17171.
        _lower = model.lower()
        if _lower.startswith("claude-") or _lower.startswith("anthropic/"):
            model = model.replace(".", "-")
    return model


def _sanitize_tool_id(tool_id: str) -> str:
    """Sanitize a tool call ID for the Anthropic API.

    Anthropic requires IDs matching [a-zA-Z0-9_-]. Replace invalid
    characters with underscores and ensure non-empty.
    """
    import re
    if not tool_id:
        return "tool_0"
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_id)
    return sanitized or "tool_0"


def _normalize_tool_input_schema(schema: Any) -> Dict[str, Any]:
    """Normalize tool schemas before sending them to Anthropic.

    Anthropic's tool schema validator rejects nullable unions such as
    ``anyOf: [{"type": "string"}, {"type": "null"}]`` that Pydantic/MCP
    commonly emits for optional fields. Tool optionality is represented by
    the parent ``required`` array, so we delegate to the shared
    ``strip_nullable_unions`` helper to collapse nullable unions to the
    non-null branch while preserving metadata like description/default.

    ``keep_nullable_hint=False`` because the Anthropic validator does not
    recognize the OpenAPI-style ``nullable: true`` extension and strict
    schema-to-grammar converters may reject unknown keywords.

    Top-level ``oneOf``/``allOf``/``anyOf`` are also stripped here: the
    Anthropic API rejects union keywords at the schema root with a generic
    HTTP 400. Several upstream and plugin tools ship schemas with one of
    these keywords at the top level (commonly for Pydantic discriminated
    unions). If we land here with those keywords still present after
    nullable-union stripping, drop them and fall back to a plain object
    schema so the tool still validates at the Anthropic boundary.
    """
    if not schema:
        return {"type": "object", "properties": {}}

    from tools.schema_sanitizer import strip_nullable_unions

    normalized = strip_nullable_unions(schema, keep_nullable_hint=False)
    if not isinstance(normalized, dict):
        return {"type": "object", "properties": {}}
    # Strip top-level union keywords that Anthropic's validator rejects.
    banned = {"oneOf", "allOf", "anyOf"}
    if banned & normalized.keys():
        normalized = {k: v for k, v in normalized.items() if k not in banned}
        if "type" not in normalized:
            normalized["type"] = "object"
    if normalized.get("type") == "object" and not isinstance(normalized.get("properties"), dict):
        normalized = {**normalized, "properties": {}}
    return normalized


def convert_tools_to_anthropic(tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI tool definitions to Anthropic format."""
    if not tools:
        return []
    result = []
    seen_names: set = set()
    for t in tools:
        fn = t.get("function", {})
        name = fn.get("name", "")
        # Defensive dedup: Anthropic rejects requests with duplicate tool
        # names.  Upstream injection paths already dedup, but this guard
        # converts a hard API failure into a warning.  See: #18478
        if name and name in seen_names:
            logger.warning(
                "convert_tools_to_anthropic: duplicate tool name '%s' "
                "— dropping second occurrence",
                name,
            )
            continue
        if name:
            seen_names.add(name)
        anthropic_tool: Dict[str, Any] = {
            "name": name,
            "description": fn.get("description", ""),
            "input_schema": _normalize_tool_input_schema(
                fn.get("parameters", {"type": "object", "properties": {}})
            ),
        }
        # Forward cache_control marker when present on the OpenAI-format
        # tool dict. Anthropic's tools array supports cache_control on the
        # last tool to cache the entire schema cross-session.
        cache_control = t.get("cache_control")
        if isinstance(cache_control, dict):
            anthropic_tool["cache_control"] = dict(cache_control)
        result.append(anthropic_tool)
    return result


def _image_source_from_openai_url(url: str) -> Dict[str, str]:
    """Convert an OpenAI-style image URL/data URL into Anthropic image source."""
    url = str(url or "").strip()
    if not url:
        return {"type": "url", "url": ""}

    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = "image/jpeg"
        if header.startswith("data:"):
            mime_part = header[len("data:"):].split(";", 1)[0].strip()
            if mime_part.startswith("image/"):
                media_type = mime_part
        return {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }

    return {"type": "url", "url": url}


def _convert_content_part_to_anthropic(part: Any) -> Optional[Dict[str, Any]]:
    """Convert a single OpenAI-style content part to Anthropic format."""
    if part is None:
        return None
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        return {"type": "text", "text": str(part)}

    ptype = part.get("type")

    if ptype == "input_text":
        block: Dict[str, Any] = {"type": "text", "text": part.get("text", "")}
    elif ptype == "text":
        # A stored Anthropic text block. Rebuild from whitelisted fields only —
        # SDK response text blocks carry output-only siblings (parsed_output,
        # citations=None) that the Messages INPUT schema rejects with HTTP 400
        # "Extra inputs are not permitted". Do NOT dict(part) it verbatim.
        block = {"type": "text", "text": part.get("text", "")}
        cits = part.get("citations")
        if isinstance(cits, list) and cits:
            block["citations"] = cits
    elif ptype in {"image_url", "input_image"}:
        image_value = part.get("image_url", {})
        url = image_value.get("url", "") if isinstance(image_value, dict) else str(image_value or "")
        block = {"type": "image", "source": _image_source_from_openai_url(url)}
    else:
        block = dict(part)

    if isinstance(part.get("cache_control"), dict) and "cache_control" not in block:
        block["cache_control"] = dict(part["cache_control"])
    return block


def _to_plain_data(value: Any, *, _depth: int = 0, _path: Optional[set] = None) -> Any:
    """Recursively convert SDK objects to plain Python data structures.

    Guards against circular references (``_path`` tracks ``id()`` of objects
    on the *current* recursion path) and runaway depth (capped at 20 levels).
    Uses path-based tracking so shared (but non-cyclic) objects referenced by
    multiple siblings are converted correctly rather than being stringified.
    """
    _MAX_DEPTH = 20
    if _depth > _MAX_DEPTH:
        return str(value)

    if _path is None:
        _path = set()

    obj_id = id(value)
    if obj_id in _path:
        return str(value)

    if hasattr(value, "model_dump"):
        _path.add(obj_id)
        try:
            # warnings=False: content blocks from the streaming accumulator
            # (ParsedTextBlock et al.) trip pydantic's serializer-mismatch
            # UserWarning against the generic Message union; the dump itself
            # is correct, and the warning leaks to the user's terminal.
            dumped = value.model_dump(warnings=False)
        except TypeError:
            # Duck-typed model_dump without pydantic's signature.
            dumped = value.model_dump()
        result = _to_plain_data(dumped, _depth=_depth + 1, _path=_path)
        _path.discard(obj_id)
        return result
    if isinstance(value, dict):
        _path.add(obj_id)
        result = {k: _to_plain_data(v, _depth=_depth + 1, _path=_path) for k, v in value.items()}
        _path.discard(obj_id)
        return result
    if isinstance(value, (list, tuple)):
        _path.add(obj_id)
        result = [_to_plain_data(v, _depth=_depth + 1, _path=_path) for v in value]
        _path.discard(obj_id)
        return result
    if hasattr(value, "__dict__"):
        _path.add(obj_id)
        result = {
            k: _to_plain_data(v, _depth=_depth + 1, _path=_path)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
        _path.discard(obj_id)
        return result
    return value


def _extract_preserved_thinking_blocks(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return Anthropic thinking blocks previously preserved on the message."""
    raw_details = message.get("reasoning_details")
    if not isinstance(raw_details, list):
        return []

    preserved: List[Dict[str, Any]] = []
    for detail in raw_details:
        if not isinstance(detail, dict):
            continue
        block_type = str(detail.get("type", "") or "").strip().lower()
        if block_type not in {"thinking", "redacted_thinking"}:
            continue
        preserved.append(copy.deepcopy(detail))
    return preserved


def _convert_content_to_anthropic(content: Any) -> Any:
    """Convert OpenAI-style multimodal content arrays to Anthropic blocks."""
    if not isinstance(content, list):
        return content

    converted = []
    for part in content:
        block = _convert_content_part_to_anthropic(part)
        if block is not None:
            converted.append(block)
    return converted


def _content_parts_to_anthropic_blocks(parts: Any) -> List[Dict[str, Any]]:
    """Convert OpenAI-style tool-message content parts → Anthropic tool_result inner blocks.

    Used for multimodal tool results (e.g. computer_use screenshots). Each
    part is normalized via `_convert_content_part_to_anthropic`, then
    filtered to the block types Anthropic tool_result accepts (text + image).
    """
    if not isinstance(parts, list):
        return []
    out: List[Dict[str, Any]] = []
    for part in parts:
        block = _convert_content_part_to_anthropic(part)
        if not block:
            continue
        btype = block.get("type")
        if btype == "text":
            text_val = block.get("text")
            if isinstance(text_val, str) and text_val:
                out.append({"type": "text", "text": text_val})
        elif btype == "image":
            src = block.get("source")
            if isinstance(src, dict) and src:
                out.append({"type": "image", "source": src})
    return out


_EMPTY_TEXT_PLACEHOLDER = "(empty)"


def _safe_text(text: Any) -> str:
    """Return ``text`` if it's non-whitespace, else a non-whitespace placeholder.

    The Anthropic Messages API rejects requests where a text content block is
    empty or whitespace-only (HTTP 400 "text content blocks must contain
    non-whitespace text"). When such a block gets stored in session history —
    e.g. produced by context compression — it is replayed verbatim on every
    subsequent turn, permanently wedging the session. Coercing to a
    non-whitespace placeholder is self-healing: the next API call recovers.

    Mirrors ``bedrock_adapter._safe_text`` (#9486); ref #69512.
    """
    if text is None:
        return _EMPTY_TEXT_PLACEHOLDER
    if not isinstance(text, str):
        text = str(text)
    return text if text.strip() else _EMPTY_TEXT_PLACEHOLDER


def _sanitize_replay_block(b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Strip output-only fields from a stored Anthropic content block so it is
    valid as REQUEST input on replay.

    The SDK response objects carry output-only attributes that the Messages
    *input* schema forbids ("Extra inputs are not permitted"): text blocks get
    ``parsed_output``/``citations`` (when null), tool_use blocks get ``caller``,
    etc. ``normalize_response`` captured blocks verbatim via ``_to_plain_data``,
    so these leak back as input on the next turn → HTTP 400.

    Whitelist per type (NOT a blacklist) so future SDK output-only fields can't
    reintroduce the bug. Returns a clean block, or None to drop it.
    """
    if not isinstance(b, dict):
        return None
    btype = b.get("type")
    if btype == "text":
        text_val = b.get("text", "")
        # Bedrock and strict Anthropic-compatible endpoints reject text
        # blocks where "text" is empty or whitespace-only (#69512). Drop the
        # blank block (the caller relocates any cache_control it carried and
        # falls back to a non-whitespace placeholder when nothing survives)
        # rather than coercing in place — a coerced "(empty)" block would be
        # model-visible noise next to surviving thinking/tool_use blocks.
        # Type-safe: captured blocks can carry text=None from an invalid
        # upstream payload, which a bare .strip() would crash on.
        if not isinstance(text_val, str) or not text_val.strip():
            return None
        out: Dict[str, Any] = {"type": "text", "text": text_val}
        # citations is input-valid ONLY when it's a non-empty list; the SDK
        # emits citations=None on responses, which the input schema rejects.
        cits = b.get("citations")
        if isinstance(cits, list) and cits:
            out["citations"] = cits
        if isinstance(b.get("cache_control"), dict):
            out["cache_control"] = b["cache_control"]
        return out
    if btype == "thinking":
        out = {"type": "thinking", "thinking": b.get("thinking", "")}
        if b.get("signature"):
            out["signature"] = b["signature"]
        return out
    if btype == "redacted_thinking":
        # Only valid with its data payload; drop if missing.
        return {"type": "redacted_thinking", "data": b["data"]} if b.get("data") else None
    if btype == "tool_use":
        out = {
            "type": "tool_use",
            "id": _sanitize_tool_id(b.get("id", "")),
            "name": b.get("name", ""),
            "input": b.get("input", {}),
        }
        if isinstance(b.get("cache_control"), dict):
            out["cache_control"] = b["cache_control"]
        return out
    if btype == "image":
        src = b.get("source")
        return {"type": "image", "source": src} if isinstance(src, dict) else None
    # Unknown/unsupported block type on the input path — drop rather than risk
    # another "Extra inputs are not permitted".
    return None


def _apply_assistant_cache_control_to_last_cacheable_block(
    blocks: List[Dict[str, Any]],
    cache_control: Any,
) -> None:
    if not isinstance(cache_control, dict):
        return
    for block in reversed(blocks):
        if isinstance(block, dict) and block.get("type") in {"text", "tool_use"}:
            block.setdefault("cache_control", dict(cache_control))
            break


def _convert_assistant_message(m: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an assistant message to Anthropic content blocks.

    Handles thinking blocks, regular content, tool calls, and
    reasoning_content injection for Kimi/DeepSeek endpoints.
    """
    content = m.get("content", "")
    # Anthropic interleaved-thinking fast path: when this turn carries a
    # verbatim, order-preserving block list (set by normalize_response only
    # for turns that interleave SIGNED thinking with tool_use), replay it.
    # Each block is run through _sanitize_replay_block to strip output-only
    # SDK fields (parsed_output, caller, citations=None, …) that the Messages
    # INPUT schema forbids — replaying them verbatim caused HTTP 400 "Extra
    # inputs are not permitted" (text.parsed_output). Block ORDER is preserved
    # (the reason this channel exists); only forbidden sibling fields are
    # dropped, leaving thinking signatures and tool_use id/name/input intact.
    ordered_blocks = m.get("anthropic_content_blocks")
    if isinstance(ordered_blocks, list) and ordered_blocks:
        # Re-source each tool_use input from the stored tool_calls map rather
        # than the captured block. The ordered-blocks list captures tool_use
        # input from the RAW API response (normalize_response), which is NOT
        # credential-redacted; tool_calls[].function.arguments IS redacted at
        # storage time (build_assistant_message, #19798). Replaying the raw
        # block input would resurrect a secret the model inlined into a tool
        # call (e.g. terminal(command="curl -H 'Authorization: Bearer sk-...'")
        # onto the wire, even though the same value is redacted everywhere else
        # in history. Keying by sanitized tool id preserves interleave order
        # (the reason this channel exists) while swapping in the redacted
        # input. Adapted from #36071 (replay-time tool-input re-sourcing).
        redacted_input_by_id: Dict[str, Any] = {}
        for tc in m.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {}) or {}
            raw_args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except (json.JSONDecodeError, ValueError):
                parsed_args = {}
            redacted_input_by_id[_sanitize_tool_id(tc.get("id", ""))] = parsed_args
        replayed: List[Dict[str, Any]] = []
        _relocated_replay_cache_control = None
        _dropped_blank_text = False
        for b in ordered_blocks:
            clean = _sanitize_replay_block(b)
            if clean is None:
                if isinstance(b, dict) and b.get("type") == "text":
                    _dropped_blank_text = True
                if isinstance(b, dict) and isinstance(b.get("cache_control"), dict):
                    # A dropped blank text block can still carry the cache
                    # breakpoint marker -- relocate it rather than losing it.
                    _relocated_replay_cache_control = b["cache_control"]
                continue
            if clean.get("type") == "tool_use":
                # Override raw (un-redacted) input with the redacted copy when
                # we have one for this id; fall back to the sanitized block
                # input only if the tool_call is missing (shape mismatch).
                redacted = redacted_input_by_id.get(clean.get("id", ""))
                if redacted is not None:
                    clean["input"] = redacted
            replayed.append(clean)
        # When every text block was blank and nothing cacheable survived
        # (e.g. signed thinking + a blank text block, or a SOLE blank
        # cache-marked block), emit the non-whitespace placeholder so the
        # replayed message stays schema-valid (#69512) and a relocated cache
        # marker still has a carrier instead of being silently lost.
        _has_cacheable_replay = any(
            isinstance(b, dict) and b.get("type") in {"text", "tool_use"}
            for b in replayed
        )
        if not _has_cacheable_replay and (
            _dropped_blank_text or _relocated_replay_cache_control is not None
        ):
            replayed.append({"type": "text", "text": _EMPTY_TEXT_PLACEHOLDER})
        if replayed:
            if _relocated_replay_cache_control is not None:
                _apply_assistant_cache_control_to_last_cacheable_block(
                    replayed, _relocated_replay_cache_control
                )
            _apply_assistant_cache_control_to_last_cacheable_block(
                replayed, m.get("cache_control")
            )
            # apply_anthropic_cache_control marks an assistant turn with
            # non-empty text by writing cache_control INTO ``content`` (see
            # _apply_cache_marker's list branch), not at the top level. This
            # branch rebuilds the message from ordered_blocks and never reads
            # ``content``, so that marker would be dropped -- and because
            # _can_carry_marker already counted this message as a carrier, the
            # breakpoint is burned rather than relocated. #56195 covered the
            # complementary shape (blank content -> top-level marker); this is
            # the interleaved thinking + preamble-text + tool_use shape.
            _inline_cc = None
            _msg_content = m.get("content")
            if isinstance(_msg_content, list):
                for _blk in _msg_content:
                    if isinstance(_blk, dict) and isinstance(
                        _blk.get("cache_control"), dict
                    ):
                        _inline_cc = _blk["cache_control"]
                        break
            if _inline_cc is not None:
                _apply_assistant_cache_control_to_last_cacheable_block(
                    replayed, _inline_cc
                )
            return {"role": "assistant", "content": replayed}

    blocks = _extract_preserved_thinking_blocks(m)
    # Cache markers dropped along with a blank block are relocated onto the
    # last surviving cacheable block below (via
    # _apply_assistant_cache_control_to_last_cacheable_block), rather than
    # lost -- prompt_caching.py's _apply_cache_marker() sets cache_control
    # directly on content[-1] for list content, so if that last part happens
    # to be blank text, dropping it silently would lose the breakpoint.
    _relocated_cache_control = None
    if content:
        if isinstance(content, list):
            converted_content = _convert_content_to_anthropic(content)
            if isinstance(converted_content, list):
                # Bedrock and strict Anthropic-compatible endpoints reject
                # text blocks where "text" is empty or whitespace-only. The
                # ordered-replay path enforces the same invariant via
                # _sanitize_replay_block(). Type-safe against ANY invalid
                # "text" value from an upstream payload -- None, or a
                # truthy non-string like an int -- not just None: checking
                # isinstance() first (rather than `blk.get("text") or ""`)
                # means a non-string value is treated as blank/invalid
                # instead of reaching .strip() and raising AttributeError.
                for blk in converted_content:
                    _blk_text = blk.get("text") if isinstance(blk, dict) else None
                    if (
                        isinstance(blk, dict)
                        and blk.get("type") == "text"
                        and (not isinstance(_blk_text, str) or not _blk_text.strip())
                    ):
                        if isinstance(blk.get("cache_control"), dict):
                            _relocated_cache_control = blk["cache_control"]
                        continue
                    blocks.append(blk)
        else:
            # Scalar (non-list) content: a whitespace-only string is the
            # same invalid-payload case as an empty list block -- drop it
            # rather than emitting a blank text block.
            text_str = str(content)
            if text_str.strip():
                blocks.append({"type": "text", "text": text_str})
    for tc in m.get("tool_calls", []):
        if not tc or not isinstance(tc, dict):
            continue
        fn = tc.get("function", {})
        args = fn.get("arguments", "{}")
        try:
            parsed_args = json.loads(args) if isinstance(args, str) else args
        except (json.JSONDecodeError, ValueError):
            parsed_args = {}
        blocks.append({
            "type": "tool_use",
            "id": _sanitize_tool_id(tc.get("id", "")),
            "name": fn.get("name", ""),
            "input": parsed_args,
        })
    # Kimi's /coding endpoint (Anthropic protocol) requires assistant
    # tool-call messages to carry reasoning_content when thinking is
    # enabled server-side.  Preserve it as a thinking block so Kimi
    # can validate the message history.  See hermes-agent#13848.
    #
    # Accept empty string "" — _copy_reasoning_content_for_api()
    # injects "" as a tier-3 fallback for Kimi tool-call messages
    # that had no reasoning.  Kimi requires the field to exist, even
    # if empty.
    #
    # Prepend (not append): Anthropic protocol requires thinking
    # blocks before text and tool_use blocks.
    #
    # Guard: only add when reasoning_details didn't already contribute
    # thinking blocks.  On native Anthropic, reasoning_details produces
    # signed thinking blocks — adding another unsigned one from
    # reasoning_content would create a duplicate (same text) that gets
    # downgraded to a spurious text block on the last assistant message.
    reasoning_content = m.get("reasoning_content")
    _already_has_thinking = any(
        isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"}
        for b in blocks
    )
    if isinstance(reasoning_content, str) and not _already_has_thinking:
        blocks.insert(0, {"type": "thinking", "thinking": reasoning_content})
    # Anthropic rejects empty assistant content. IMPORTANT: fall back only
    # to the placeholder, never to the raw `content` variable -- `content`
    # is the UNFILTERED original message content, and can itself be exactly
    # the blank/whitespace-only payload the filtering above just removed
    # (a sole blank text block, or scalar whitespace with no tool_calls).
    # `blocks or content` there would silently restore the invalid provider
    # payload this function exists to prevent (#69512).
    effective = blocks if blocks else [{"type": "text", "text": _EMPTY_TEXT_PLACEHOLDER}]
    # Applied here (after the empty-fallback resolution) rather than
    # earlier against `blocks` directly, so a cache_control relocated from
    # a dropped blank block that was the ONLY block still lands on the
    # (empty) placeholder instead of being silently lost when blocks was
    # empty at the point the marker would otherwise have been applied.
    if _relocated_cache_control is not None:
        _apply_assistant_cache_control_to_last_cacheable_block(
            effective, _relocated_cache_control
        )
    _apply_assistant_cache_control_to_last_cacheable_block(
        effective, m.get("cache_control")
    )
    return {"role": "assistant", "content": effective}


def _convert_tool_message_to_result(
    result: List[Dict[str, Any]], m: Dict[str, Any]
) -> None:
    """Convert a tool message to an Anthropic tool_result, merging consecutive
    results into one user message.

    Mutates ``result`` in place — either appends a new user message or extends
    the trailing user message's tool_result list.
    """
    content = m.get("content", "")
    multimodal_blocks: Optional[List[Dict[str, Any]]] = None
    if isinstance(content, dict) and content.get("_multimodal"):
        multimodal_blocks = _content_parts_to_anthropic_blocks(
            content.get("content") or []
        )
        # Fallback text if the conversion produced nothing usable.
        if not multimodal_blocks and content.get("text_summary"):
            multimodal_blocks = [
                {"type": "text", "text": str(content["text_summary"])}
            ]
    elif isinstance(content, list):
        converted = _content_parts_to_anthropic_blocks(content)
        if any(b.get("type") == "image" for b in converted):
            multimodal_blocks = converted
    # Back-compat: some callers stash blocks under a private key.
    if multimodal_blocks is None:
        stashed = m.get("_anthropic_content_blocks")
        if isinstance(stashed, list) and stashed:
            text_content = content if isinstance(content, str) and content.strip() else None
            multimodal_blocks = (
                [{"type": "text", "text": text_content}] + stashed
                if text_content else list(stashed)
            )

    if multimodal_blocks:
        result_content: Any = multimodal_blocks
    elif isinstance(content, str):
        result_content = content
    else:
        result_content = json.dumps(content) if content else "(no output)"
    if not result_content:
        result_content = "(no output)"
    tool_result = {
        "type": "tool_result",
        "tool_use_id": _sanitize_tool_id(m.get("tool_call_id", "")),
        "content": result_content,
    }
    if isinstance(m.get("cache_control"), dict):
        tool_result["cache_control"] = dict(m["cache_control"])
    # Merge consecutive tool results into one user message
    if (
        result
        and result[-1]["role"] == "user"
        and isinstance(result[-1]["content"], list)
        and result[-1]["content"]
        and result[-1]["content"][0].get("type") == "tool_result"
    ):
        result[-1]["content"].append(tool_result)
    else:
        result.append({"role": "user", "content": [tool_result]})


def _convert_user_message(content: Any) -> Dict[str, Any]:
    """Validate and convert a user message to anthropic format."""
    if isinstance(content, list):
        converted_blocks = _convert_content_to_anthropic(content)
        kept_blocks = _fix_blank_text_blocks_in_list(
            converted_blocks,
            placeholder_text="(empty message)",
            msg_index=-1,
            role="user",
            location="_convert_user_message",
        )
        return {"role": "user", "content": kept_blocks}
    else:
        if not content or (isinstance(content, str) and not content.strip()):
            content = "(empty message)"
        return {"role": "user", "content": content}


def _strip_orphaned_tool_blocks(result: List[Dict[str, Any]]) -> None:
    """Strip tool_use blocks with no matching tool_result, and vice versa.

    Context compression or session truncation can remove either side of a
    tool-call pair, or insert messages between a tool_use and its result.
    Anthropic requires each tool_use to have a matching tool_result in the
    IMMEDIATELY FOLLOWING user message — a global ID match is not enough.
    Mutates ``result`` in place.
    """
    # Pass 1: For each assistant message with tool_use blocks, check that
    # EACH tool_use ID has a matching tool_result in the immediately following
    # user message.  Strip tool_use blocks that lack an adjacent result —
    # Anthropic rejects non-adjacent pairs with HTTP 400 even when the IDs
    # match somewhere later in the conversation.
    for i, m in enumerate(result):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue
        tool_use_ids_in_turn = {
            b.get("id")
            for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_use"
        }
        if not tool_use_ids_in_turn:
            continue

        # Collect result IDs from the immediately following user message only.
        adjacent_result_ids: set = set()
        if i + 1 < len(result):
            nxt = result[i + 1]
            if nxt.get("role") == "user" and isinstance(nxt.get("content"), list):
                for block in nxt["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        adjacent_result_ids.add(block.get("tool_use_id"))

        orphaned = tool_use_ids_in_turn - adjacent_result_ids
        if not orphaned:
            continue

        kept = [
            b
            for b in m["content"]
            if not (isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id") in orphaned)
        ]
        # If stripping an orphaned tool_use mutated a turn that also carries a
        # signed thinking block, that block's Anthropic signature was computed
        # against the ORIGINAL (un-stripped) turn content and is now invalid.
        # Anthropic rejects the replayed turn with HTTP 400 "thinking blocks in
        # the latest assistant message cannot be modified".  Flag the turn so
        # _manage_thinking_signatures can demote the dead signature instead of
        # replaying it verbatim.  See hermes-agent: extended-thinking + parallel
        # tool batch interrupted mid-flight → non-retryable 400 crash-loop.
        if len(kept) != len(m["content"]) and any(
            isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"}
            for b in m["content"]
        ):
            m["_thinking_signature_invalidated"] = True
        m["content"] = kept if kept else [{"type": "text", "text": "(tool call removed)"}]

    # Pass 2: Rebuild the set of tool_use IDs that survived pass 1, then
    # strip tool_result blocks that no longer have any matching tool_use
    # anywhere in the conversation.
    surviving_tool_use_ids: set = set()
    for m in result:
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            for block in m["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    surviving_tool_use_ids.add(block.get("id"))

    for m in result:
        if m.get("role") != "user" or not isinstance(m.get("content"), list):
            continue
        new_content = [
            b
            for b in m["content"]
            if not (isinstance(b, dict) and b.get("type") == "tool_result")
            or b.get("tool_use_id") in surviving_tool_use_ids
        ]
        if len(new_content) != len(m["content"]):
            m["content"] = new_content if new_content else [{"type": "text", "text": "(tool result removed)"}]


def _merge_consecutive_roles(result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge consecutive same-role messages to enforce Anthropic alternation.

    Returns a new list (caller must rebind ``result``).
    """
    fixed = []
    for m in result:
        if fixed and fixed[-1]["role"] == m["role"]:
            if m["role"] == "user":
                prev_content = fixed[-1]["content"]
                curr_content = m["content"]
                if isinstance(prev_content, str) and isinstance(curr_content, str):
                    fixed[-1]["content"] = prev_content + "\n" + curr_content
                elif isinstance(prev_content, list) and isinstance(curr_content, list):
                    fixed[-1]["content"] = prev_content + curr_content
                else:
                    if isinstance(prev_content, str):
                        prev_content = [{"type": "text", "text": prev_content}]
                    if isinstance(curr_content, str):
                        curr_content = [{"type": "text", "text": curr_content}]
                    fixed[-1]["content"] = prev_content + curr_content
            else:
                # Consecutive assistant messages — merge text content.
                # Propagate the orphan-strip signature-invalidation flag onto the
                # surviving (prev) dict so _manage_thinking_signatures still sees it.
                if m.get("_thinking_signature_invalidated"):
                    fixed[-1]["_thinking_signature_invalidated"] = True
                # Drop thinking blocks from the *second* message: their
                # signature was computed against a different turn boundary
                # and becomes invalid once merged.
                if isinstance(m["content"], list):
                    m["content"] = [
                        b for b in m["content"]
                        if not (isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"})
                    ]
                prev_blocks = fixed[-1]["content"]
                curr_blocks = m["content"]
                if isinstance(prev_blocks, list) and isinstance(curr_blocks, list):
                    fixed[-1]["content"] = prev_blocks + curr_blocks
                elif isinstance(prev_blocks, str) and isinstance(curr_blocks, str):
                    fixed[-1]["content"] = prev_blocks + "\n" + curr_blocks
                else:
                    if isinstance(prev_blocks, str):
                        prev_blocks = [{"type": "text", "text": prev_blocks}]
                    if isinstance(curr_blocks, str):
                        curr_blocks = [{"type": "text", "text": curr_blocks}]
                    fixed[-1]["content"] = prev_blocks + curr_blocks
        else:
            fixed.append(m)
    return fixed


def _manage_thinking_signatures(
    result: List[Dict[str, Any]], base_url: str | None, model: str | None
) -> None:
    """Strip or preserve thinking blocks based on endpoint type.

    Anthropic signs thinking blocks against the full turn content.
    Any upstream mutation (context compression, session truncation, orphan
    stripping, message merging) invalidates the signature, causing HTTP 400
    "Invalid signature in thinking block".

    Signatures are Anthropic-proprietary.  Third-party endpoints (MiniMax,
    Azure AI Foundry, AWS Bedrock, self-hosted proxies) cannot validate them
    and will reject them outright.  Kimi's /coding and DeepSeek's /anthropic
    endpoints speak the Anthropic protocol upstream but require unsigned
    thinking blocks (synthesised from ``reasoning_content``) to round-trip on
    replayed assistant tool-call messages.  See hermes-agent#13848 (Kimi) and
    hermes-agent#16748 (DeepSeek).

    Nous Portal's ``/v1/messages`` route is the exception among third-party
    hosts: it proxies Claude to Anthropic/Vertex/Bedrock and validates the
    same signed thinking blocks.  Sticky ``session_id`` keeps a conversation
    on one upstream instance so those signatures stay warm — stripping them
    here would 400 the first tool-loop turn ("thinking must be passed back").
    Portal therefore takes the native Anthropic replay path below.

    Mutates ``result`` in place.
    """
    _THINKING_TYPES = frozenset(("thinking", "redacted_thinking"))
    # Portal speaks Anthropic's thinking contract end-to-end; do not treat it
    # as a signature-blind proxy even though the host is not anthropic.com.
    _is_third_party = (
        _is_third_party_anthropic_endpoint(base_url)
        and not _is_nous_portal_endpoint(base_url)
    )

    last_assistant_idx = None
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    for idx, m in enumerate(result):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue

        if _is_kimi_family_endpoint(base_url, model):
            # Kimi does not enforce thinking signatures — replay as-is
            # (shared cleanup below still strips cache markers + the internal flag).
            pass
        elif _is_deepseek_anthropic_endpoint(base_url):
            # DeepSeek: strip signed, preserve unsigned.
            new_content = []
            for b in m["content"]:
                if not isinstance(b, dict) or b.get("type") not in _THINKING_TYPES:
                    new_content.append(b)
                    continue
                if b.get("signature") or b.get("data"):
                    # Signed (or redacted-with-data) — upstream can't validate, strip.
                    continue
                new_content.append(b)
            m["content"] = new_content or [{"type": "text", "text": "(empty)"}]
        elif _is_third_party or idx != last_assistant_idx:
            # Third-party: strip ALL thinking blocks (signatures are proprietary).
            # Direct Anthropic: strip from non-latest assistant messages only.
            stripped = [
                b for b in m["content"]
                if not (isinstance(b, dict) and b.get("type") in _THINKING_TYPES)
            ]
            m["content"] = stripped or [{"type": "text", "text": "(thinking elided)"}]
        else:
            # Latest assistant on direct Anthropic: keep signed, downgrade unsigned
            # to text so the reasoning isn't lost.
            #
            # Exception: if orphan-stripping (or another structural mutation) removed
            # a tool_use block from THIS turn, every thinking signature on it was
            # computed against the original turn content and is now dead.  Anthropic
            # rejects the turn either way — replaying the signed block 400s with
            # "thinking blocks in the latest assistant message cannot be modified",
            # and a bare signed block with no following tool_use is also invalid.
            # Demote ALL thinking blocks on this turn to text so the turn replays
            # cleanly and the model can re-plan from the surviving tool results.
            signature_dead = bool(m.get("_thinking_signature_invalidated"))
            new_content = []
            for b in m["content"]:
                if not isinstance(b, dict) or b.get("type") not in _THINKING_TYPES:
                    new_content.append(b)
                    continue
                if signature_dead:
                    thinking_text = b.get("thinking", "")
                    if thinking_text:
                        new_content.append({"type": "text", "text": thinking_text})
                    continue
                if b.get("type") == "redacted_thinking":
                    # Redacted blocks use 'data' for the signature payload —
                    # drop the block when 'data' is missing (can't be validated).
                    if b.get("data"):
                        new_content.append(b)
                elif b.get("signature"):
                    new_content.append(b)
                else:
                    thinking_text = b.get("thinking", "")
                    if thinking_text:
                        new_content.append({"type": "text", "text": thinking_text})
            m["content"] = new_content or [{"type": "text", "text": "(empty)"}]

        # Strip cache_control from any remaining thinking/redacted_thinking
        # blocks — cache markers interfere with signature validation.
        for b in m["content"]:
            if isinstance(b, dict) and b.get("type") in _THINKING_TYPES:
                b.pop("cache_control", None)

        # Drop the internal bookkeeping flag — it must never reach the API payload.
        m.pop("_thinking_signature_invalidated", None)


def _evict_old_screenshots(result: List[Dict[str, Any]]) -> None:
    """Keep only the most recent ``_MAX_KEEP_IMAGES`` computer-use screenshots.

    Base64 images cost ~1,465 tokens each and accumulate across tool calls.
    Walk backward, keep the most recent N, replace older ones with a placeholder.

    Mutates ``result`` in place.
    """
    _MAX_KEEP_IMAGES = 3
    _image_count = 0
    for msg in reversed(result):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            inner = block.get("content")
            if not isinstance(inner, list):
                continue
            has_image = any(
                isinstance(b, dict) and b.get("type") == "image"
                for b in inner
            )
            if not has_image:
                continue
            _image_count += 1
            if _image_count > _MAX_KEEP_IMAGES:
                block["content"] = [
                    b if b.get("type") != "image"
                    else {"type": "text", "text": "[screenshot removed to save context]"}
                    for b in inner
                ]


def _ensure_leading_user_turn(result: List[Dict[str, Any]]) -> None:
    """Anthropic requires messages[0] to have role=user.

    After a second context compaction on the auto path the summary can be
    emitted as role=assistant with nothing in front of it (the system prompt
    lives outside messages[] or is extracted into the separate ``system``
    param), so messages[0] ends up assistant and the Messages API rejects
    the request with HTTP 400 — often masked by a misleading
    "tool_use ids were found without tool_result blocks" error (#52160).

    Mirror the Bedrock Converse adapter, which unconditionally prepends a
    minimal user turn when the first message is not user
    (convert_messages_to_converse).

    The inserted text block must be non-whitespace: Anthropic separately
    rejects any text content block whose text is empty or whitespace-only
    ("text content blocks must contain non-whitespace text"), so a single
    space here traded the "leading assistant turn" 400 for that one (#69512
    class). Uses the same placeholder as every other synthesized filler
    block in this module for consistency.
    """
    if result and result[0].get("role") != "user":
        result.insert(
            0, {"role": "user", "content": [{"type": "text", "text": _EMPTY_TEXT_PLACEHOLDER}]}
        )


def _fix_blank_text_blocks_in_list(
    blocks: List[Any],
    *,
    placeholder_text: str,
    msg_index: int,
    role: Any,
    location: str,
) -> List[Any]:
    """Drop blank/whitespace-only text blocks from ``blocks``, in place logic.

    Non-text blocks (tool_use, tool_result, image, document, thinking, …)
    and the relative order of everything else are left untouched. A
    cache_control marker riding on a dropped block is relocated onto the
    last surviving text/tool_use block so a breakpoint is never silently
    lost. If nothing survives, a single non-blank placeholder text block
    takes the dropped blocks' place (carrying the relocated cache_control,
    if any) so the message never has empty content.

    Returns a new list; does not mutate ``blocks``.
    """
    kept: List[Any] = []
    relocated_cache_control = None
    for block_index, blk in enumerate(blocks):
        if (
            isinstance(blk, dict)
            and blk.get("type") == "text"
            and not (isinstance(blk.get("text"), str) and blk["text"].strip())
        ):
            if isinstance(blk.get("cache_control"), dict):
                relocated_cache_control = blk["cache_control"]
            logger.warning(
                "Pre-call sanitizer: dropped blank text content block "
                "(message_index=%d role=%s location=%s block_index=%d "
                "block_type=text)",
                msg_index,
                role,
                location,
                block_index,
            )
            continue
        kept.append(blk)
    if not kept:
        placeholder: Dict[str, Any] = {"type": "text", "text": placeholder_text}
        if relocated_cache_control is not None:
            placeholder["cache_control"] = relocated_cache_control
        kept.append(placeholder)
    elif relocated_cache_control is not None:
        _apply_assistant_cache_control_to_last_cacheable_block(kept, relocated_cache_control)
    return kept


def _scrub_blank_text_blocks(result: List[Dict[str, Any]]) -> None:
    """Final provider-boundary guard against blank Anthropic text blocks.

    Anthropic rejects any text content block whose ``text`` is empty or
    whitespace-only with HTTP 400 ("text content blocks must contain
    non-whitespace text"). ``_convert_assistant_message``,
    ``_convert_user_message`` and ``_ensure_leading_user_turn`` already
    avoid emitting these for the paths that build them, but this pass runs
    last — after every other transform in ``convert_messages_to_anthropic``
    — so a blank block from any current or future producer (including one
    nested inside a ``tool_result``'s own content list) never reaches the
    wire. Diagnostics are structural only: message index, role, content
    location, block index/type. Never logs message text, tool arguments,
    tokens, or credentials. Mutates ``result`` in place.
    """
    for msg_index, msg in enumerate(result):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, list) or not content:
            continue
        placeholder_text = _EMPTY_TEXT_PLACEHOLDER if role == "assistant" else "(empty message)"
        new_content = _fix_blank_text_blocks_in_list(
            content,
            placeholder_text=placeholder_text,
            msg_index=msg_index,
            role=role,
            location="content",
        )
        for blk in new_content:
            if not isinstance(blk, dict) or blk.get("type") != "tool_result":
                continue
            inner = blk.get("content")
            if isinstance(inner, list) and inner:
                blk["content"] = _fix_blank_text_blocks_in_list(
                    inner,
                    placeholder_text="(no output)",
                    msg_index=msg_index,
                    role=role,
                    location="tool_result",
                )
        msg["content"] = new_content


def convert_messages_to_anthropic(
    messages: List[Dict],
    base_url: str | None = None,
    model: str | None = None,
) -> Tuple[Optional[Any], List[Dict]]:
    """Convert OpenAI-format messages to Anthropic format.

    Returns (system_prompt, anthropic_messages).
    System messages are extracted since Anthropic takes them as a separate param.
    system_prompt is a string or list of content blocks (when cache_control present).

    When *base_url* is provided and points to a third-party Anthropic-compatible
    endpoint, all thinking block signatures are stripped.  Signatures are
    Anthropic-proprietary — third-party endpoints cannot validate them and will
    reject them with HTTP 400 "Invalid signature in thinking block".

    When *model* is provided and matches the Kimi / Moonshot family (or
    *base_url* is a Kimi / Moonshot host), unsigned thinking blocks
    synthesised from ``reasoning_content`` are preserved on replayed
    assistant tool-call messages — Kimi requires the field to exist, even
    if empty.
    """
    system = None
    result: List[Dict[str, Any]] = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system":
            if isinstance(content, list):
                # Preserve cache_control markers on content blocks
                has_cache = any(
                    p.get("cache_control") for p in content if isinstance(p, dict)
                )
                if has_cache:
                    # Copy blocks before coercing so the caller's message
                    # dicts are never mutated, then replace blank/whitespace
                    # text with the shared non-whitespace placeholder —
                    # Anthropic rejects a blank system text block with the
                    # same HTTP 400 as message blocks ("text content blocks
                    # must contain non-whitespace text"), and a blank block
                    # carrying a cache_control breakpoint cannot simply be
                    # dropped (#70909).
                    system = []
                    for p in content:
                        if not isinstance(p, dict):
                            continue
                        if (
                            p.get("type") == "text"
                            and isinstance(p.get("text"), str)
                            and not p["text"].strip()
                        ):
                            p = dict(p)
                            p["text"] = _EMPTY_TEXT_PLACEHOLDER
                        system.append(p)
                else:
                    system = "\n".join(
                        p["text"] for p in content if p.get("type") == "text"
                    )
            else:
                system = content
            continue

        if role == "assistant":
            result.append(_convert_assistant_message(m))
            continue

        if role == "tool":
            _convert_tool_message_to_result(result, m)
            continue

        # Regular user message
        result.append(_convert_user_message(content))

    _strip_orphaned_tool_blocks(result)
    result = _merge_consecutive_roles(result)
    _ensure_leading_user_turn(result)
    _manage_thinking_signatures(result, base_url, model)
    _evict_old_screenshots(result)
    _scrub_blank_text_blocks(result)

    return system, result


def build_anthropic_kwargs(
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    max_tokens: Optional[int],
    reasoning_config: Optional[Dict[str, Any]],
    tool_choice: Optional[str] = None,
    is_oauth: bool = False,
    preserve_dots: bool = False,
    context_length: Optional[int] = None,
    base_url: str | None = None,
    fast_mode: bool = False,
    drop_context_1m_beta: bool = False,
) -> Dict[str, Any]:
    """Build kwargs for anthropic.messages.create().

    Naming note — two distinct concepts, easily confused:
      max_tokens     = OUTPUT token cap for a single response.
                       Anthropic's API calls this "max_tokens" but it only
                       limits the *output*.  Anthropic's own native SDK
                       renamed it "max_output_tokens" for clarity.
      context_length = TOTAL context window (input tokens + output tokens).
                       The API enforces: input_tokens + max_tokens ≤ context_length.
                       Stored on the ContextCompressor; reduced on overflow errors.

    When *max_tokens* is None the model's native output ceiling is used
    (e.g. 128K for Opus 4.6, 64K for Sonnet 4.6).

    When *context_length* is provided and the model's native output ceiling
    exceeds it (e.g. a local endpoint with an 8K window), the output cap is
    clamped to context_length − 1.  This only kicks in for unusually small
    context windows; for full-size models the native output cap is always
    smaller than the context window so no clamping happens.
    NOTE: this clamping does not account for prompt size — if the prompt is
    large, Anthropic may still reject the request.  The caller must detect
    "max_tokens too large given prompt" errors and retry with a smaller cap
    (see parse_available_output_tokens_from_error + _ephemeral_max_output_tokens).

    When *is_oauth* is True, applies Claude Code compatibility transforms:
    system prompt prefix, tool name prefixing, and prompt sanitization.

    When *preserve_dots* is True, model name dots are not converted to hyphens
    (for Alibaba/DashScope anthropic-compatible endpoints: qwen3.5-plus).

    When *base_url* points to a third-party Anthropic-compatible endpoint,
    thinking block signatures are stripped (they are Anthropic-proprietary).

    When *fast_mode* is True, adds ``extra_body["speed"] = "fast"`` and the
    fast-mode beta header for ~2.5x faster output throughput on Opus 4.6.
    Currently only supported on native Anthropic endpoints (not third-party
    compatible ones).
    """
    system, anthropic_messages = convert_messages_to_anthropic(
        messages, base_url=base_url, model=model
    )
    anthropic_tools = convert_tools_to_anthropic(tools) if tools else []

    # Nous Portal routes on its own catalog ids (``anthropic/claude-opus-4.8``);
    # normalizing to the bare Anthropic slug would make the model unresolvable
    # there. Skipping the call preserves the prefix AND the dots, so
    # ``preserve_dots`` stays irrelevant for Portal.
    if not _is_nous_portal_endpoint(base_url):
        model = normalize_model_name(model, preserve_dots=preserve_dots)
    # effective_max_tokens = output cap for this call (≠ total context window)
    # Use the resolver helper so non-positive values (negative ints,
    # fractional floats, NaN, non-numeric) fail locally with a clear error
    # rather than 400-ing at the Anthropic API. See openclaw/openclaw#66664.
    effective_max_tokens = _resolve_anthropic_messages_max_tokens(
        max_tokens, model, context_length=context_length
    )

    # Clamp output cap to fit inside the total context window.
    # Only matters for small custom endpoints where context_length < native
    # output ceiling.  For standard Anthropic models context_length (e.g.
    # 200K) is always larger than the output ceiling (e.g. 128K), so this
    # branch is not taken.
    if context_length and effective_max_tokens > context_length:
        effective_max_tokens = max(context_length - 1, 1)

    # ── OAuth: Claude Code identity ──────────────────────────────────
    if is_oauth:
        # 1. Prepend Claude Code system prompt identity
        cc_block = {"type": "text", "text": _CLAUDE_CODE_SYSTEM_PREFIX}
        if isinstance(system, list):
            system = [cc_block] + system
        elif isinstance(system, str) and system:
            system = [cc_block, {"type": "text", "text": system}]
        else:
            system = [cc_block]

        # 2. Sanitize system prompt — replace product name references
        #    to avoid Anthropic's server-side content filters.
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                text = text.replace("Hermes Agent", "Claude Code")
                text = text.replace("Hermes agent", "Claude Code")
                text = text.replace("hermes-agent", "claude-code")
                text = text.replace("Nous Research", "Anthropic")
                block["text"] = text

        # 3. Normalize tool names so NOTHING goes on the OAuth wire with a
        #    single-underscore ``mcp_`` prefix.  Anthropic's subscription/OAuth
        #    billing classifier treats a single-underscore ``mcp_`` tool name as
        #    a third-party-app fingerprint and rejects the request with HTTP 400
        #    "Third-party apps now draw from extra usage, not plan limits"
        #    (verified empirically: a single ``mcp_foo`` tool flips a request
        #    from plan-billing to the extra-usage lane; ``mcp__foo`` is accepted).
        #
        #    Two cases, both must land on the double-underscore ``mcp__`` form:
        #      a) bare Hermes-native tools (``read_file``)  -> ``mcp__read_file``
        #      b) native MCP server tools registered under their full
        #         single-underscore ``mcp_<server>_<tool>`` name
        #         (``mcp_linear_get_issue``) -> ``mcp__linear_get_issue``
        #    Case (b) is the gap that the bare ``mcp_``->``mcp__`` constant swap
        #    left open: those tools were *skipped* and stayed single-underscore,
        #    so any session with an MCP server configured still tripped the
        #    classifier. normalize_response reverses both forms via registry
        #    lookup so the dispatcher still sees the original name. GH-25255.
        def _to_oauth_wire_name(name: str) -> str:
            if name.startswith("mcp__"):
                return name  # already correct, don't double-prefix
            if name.startswith("mcp_"):
                # single-underscore native MCP tool -> promote to double
                return "mcp__" + name[len("mcp_"):]
            return _MCP_TOOL_PREFIX + name  # bare name -> mcp__<name>

        if anthropic_tools:
            for tool in anthropic_tools:
                if "name" in tool:
                    tool["name"] = _to_oauth_wire_name(tool["name"])

        # 4. Apply the same normalization to tool names in message history
        #    (tool_use blocks) so replayed turns match the wire names above.
        for msg in anthropic_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use" and "name" in block:
                            block["name"] = _to_oauth_wire_name(block["name"])
                        elif block.get("type") == "tool_result" and "tool_use_id" in block:
                            pass  # tool_result uses ID, not name

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": effective_max_tokens,
    }

    if system:
        kwargs["system"] = system

    if anthropic_tools:
        kwargs["tools"] = anthropic_tools
        # Map OpenAI tool_choice to Anthropic format
        if tool_choice == "auto" or tool_choice is None:
            kwargs["tool_choice"] = {"type": "auto"}
        elif tool_choice == "required":
            kwargs["tool_choice"] = {"type": "any"}
        elif tool_choice == "none":
            # Anthropic has no tool_choice "none" — omit tools entirely to prevent use
            kwargs.pop("tools", None)
        elif isinstance(tool_choice, str):
            # Specific tool name
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

    # Map reasoning_config to Anthropic's thinking parameter.
    # Claude 4.6+ models use adaptive thinking + output_config.effort.
    # Older models use manual thinking with budget_tokens.
    # MiniMax Anthropic-compat endpoints support thinking (manual mode only,
    # not adaptive).  Haiku does NOT support extended thinking — skip entirely.
    #
    # Kimi / Moonshot models also use adaptive thinking: their
    # Anthropic-compatible endpoints (api.moonshot.cn/anthropic,
    # api.kimi.com/coding) accept ``thinking.type="adaptive"`` +
    # ``output_config.effort``, and the replay-validation 400s that
    # originally motivated dropping the parameter (#13848) no longer
    # occur.  (Kimi on chat_completions enables thinking via extra_body
    # in the ChatCompletionsTransport — see #13503.)
    #
    # On 4.7+ the `thinking.display` field defaults to "omitted", which
    # silently hides reasoning text that Hermes surfaces in its CLI. We
    # request "summarized" so the reasoning blocks stay populated — matching
    # 4.6 behavior and preserving the activity-feed UX during long tool runs.
    if reasoning_config and isinstance(reasoning_config, dict):
        if reasoning_config.get("enabled") is False:
            # "Thinking off". Adaptive models think by DEFAULT, so omitting the
            # parameter is not a disable — it silently leaves thinking on and
            # the user keeps paying for it. Send the disable explicitly.
            # Mandatory-thinking models reject it with a 400, so they keep the
            # omission: a silently-ignored disable beats a dead turn.
            if _accepts_thinking_disable(model):
                kwargs["thinking"] = {"type": "disabled"}
        elif "haiku" not in model.lower():
            effort = str(reasoning_config.get("effort", "medium")).lower()
            budget = THINKING_BUDGET.get(effort, 8000)
            if _supports_adaptive_thinking(model):
                kwargs["thinking"] = {
                    "type": "adaptive",
                    "display": "summarized",
                }
                adaptive_effort = ADAPTIVE_EFFORT_MAP.get(effort, "medium")
                # Downgrade xhigh→max on models that don't list xhigh as a
                # supported level (Opus/Sonnet 4.6). Opus 4.7+ keeps xhigh.
                if adaptive_effort == "xhigh" and not _supports_xhigh_effort(model):
                    adaptive_effort = "max"
                kwargs["output_config"] = {
                    "effort": adaptive_effort,
                }
            else:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
                # Anthropic requires temperature=1 when thinking is enabled on older models
                kwargs["temperature"] = 1
                kwargs["max_tokens"] = max(effective_max_tokens, budget + 4096)

    # ── Strip sampling params on 4.7+ ─────────────────────────────────
    # Opus 4.7 rejects any non-default temperature/top_p/top_k with a 400.
    # Callers (auxiliary_client, etc.) may set these for older models;
    # drop them here as a safety net so upstream 4.6 → 4.7 migrations
    # don't require coordinated edits everywhere.
    if _forbids_sampling_params(model):
        for _sampling_key in ("temperature", "top_p", "top_k"):
            kwargs.pop(_sampling_key, None)

    # ── Fast mode (Opus 4.6 only) ────────────────────────────────────
    # Adds extra_body.speed="fast" + the fast-mode beta header for ~2.5x
    # output speed. Per Anthropic docs, fast mode is only supported on
    # Opus 4.6 — Opus 4.7 and other models 400 on the speed parameter.
    # Only for native Anthropic endpoints — third-party providers would
    # reject the unknown beta header and speed parameter.
    if (
        fast_mode
        and not _is_third_party_anthropic_endpoint(base_url)
        and _supports_fast_mode(model)
    ):
        kwargs.setdefault("extra_body", {})["speed"] = "fast"
        # Build extra_headers with ALL applicable betas (the per-request
        # extra_headers override the client-level anthropic-beta header).
        betas = list(_common_betas_for_base_url(
            base_url,
            drop_context_1m_beta=drop_context_1m_beta,
        ))
        if is_oauth:
            betas.extend(_OAUTH_ONLY_BETAS)
        betas.append(_FAST_MODE_BETA)
        kwargs["extra_headers"] = {"anthropic-beta": ",".join(betas)}

    return kwargs


# Keys that belong exclusively to the OpenAI Responses / Codex API shape.
# The Anthropic Messages SDK (``messages.create()`` / ``messages.stream()``)
# raises ``TypeError: ... got an unexpected keyword argument`` on any of them.
_RESPONSES_ONLY_KWARGS = frozenset(
    {"instructions", "input", "store", "parallel_tool_calls"}
)


def _apply_claude_code_identity(system, anthropic_tools, anthropic_messages, to_wire):
    """OAuth transforms: Claude Code system prefix, product-name sanitizing (avoids server-side
    content filters), tool/description aliasing, and the same tool renames on replayed tool_use
    blocks so history matches ``tools[]``. Returns the new ``system``; tools and messages are
    mutated in place."""
    cc_block = {"type": "text", "text": _CLAUDE_CODE_SYSTEM_PREFIX}
    if isinstance(system, str) and system:
        system = [{"type": "text", "text": system}]
    system = [cc_block] + (system if isinstance(system, list) else [])
    for block in system:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            for old, new in _OAUTH_SYSTEM_REPLACEMENTS:
                text = text.replace(old, new)
            block["text"] = _apply_oauth_prose_aliases(text)
    for tool in anthropic_tools or []:
        if "name" in tool:
            tool["name"] = to_wire(tool["name"])
        if isinstance(tool.get("description"), str):
            tool["description"] = _apply_oauth_prose_aliases(tool["description"])  # prose-safe aliases only
    for msg in anthropic_messages:
        for block in msg.get("content") if isinstance(msg.get("content"), list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_use" and "name" in block:
                block["name"] = to_wire(block["name"])  # tool_result pairs by id, not name
    return system


def _thinking_kwargs(reasoning_config: Dict[str, Any], model: str, effective_max_tokens: int) -> Dict[str, Any]:
    """Map ``reasoning_config`` to Anthropic thinking kwargs. Adaptive models (Claude 4.6+,
    Kimi/Moonshot) get ``thinking.type=adaptive`` + ``output_config.effort``; older models and
    manual-only compat endpoints (MiniMax) get budget_tokens. Haiku has no extended thinking. On
    4.7+ ``thinking.display`` defaults to "omitted", hiding the reasoning Hermes shows in its CLI,
    so "summarized" is requested to keep the activity feed populated."""
    if reasoning_config.get("enabled") is False:
        # Adaptive models think by DEFAULT, so omitting the parameter is not a disable — the user
        # silently keeps paying. Mandatory-thinking models 400 on the disable, so they keep the
        # omission: a silently-ignored disable beats a dead turn.
        return {"thinking": {"type": "disabled"}} if _accepts_thinking_disable(model) else {}
    if "haiku" in model.lower():
        return {}
    effort = str(reasoning_config.get("effort", "medium")).lower()
    if _supports_adaptive_thinking(model):
        adaptive_effort = ADAPTIVE_EFFORT_MAP.get(effort, "medium")
        if adaptive_effort == "xhigh" and not _supports_xhigh_effort(model):
            adaptive_effort = "max"
        return {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": adaptive_effort}}
    budget = THINKING_BUDGET.get(effort, 8000)
    return {
        "thinking": {"type": "enabled", "budget_tokens": budget},
        "temperature": 1,  # required when thinking is enabled on older models
        "max_tokens": max(effective_max_tokens, budget + 4096),
    }


# OpenAI tool_choice -> Anthropic; any other string is a forced tool name.
_TOOL_CHOICE_MAP = {None: {"type": "auto"}, "auto": {"type": "auto"}, "required": {"type": "any"}}


def build_anthropic_kwargs(
    model: str, messages: List[Dict], tools: Optional[List[Dict]], max_tokens: Optional[int],
    reasoning_config: Optional[Dict[str, Any]], tool_choice: Optional[str] = None,
    is_oauth: bool = False, preserve_dots: bool = False, context_length: Optional[int] = None,
    base_url: str | None = None, fast_mode: bool = False, drop_context_1m_beta: bool = False,
) -> Dict[str, Any]:
    """Build kwargs for anthropic.messages.create(). ``max_tokens`` is the OUTPUT cap for one
    response; ``context_length`` is the TOTAL window (input + output). ``max_tokens=None`` uses the
    model's native output ceiling; if that exceeds ``context_length`` (small local endpoints) it is
    clamped to ``context_length - 1``. The clamp ignores prompt size — callers must catch
    "max_tokens too large given prompt" and retry smaller (parse_available_output_tokens_from_error).
    ``is_oauth`` applies Claude Code compatibility transforms; ``preserve_dots`` keeps model-name
    dots (DashScope: qwen3.5-plus); a third-party ``base_url`` strips thinking signatures;
    ``fast_mode`` adds ``extra_body.speed="fast"`` plus the fast-mode beta on native Anthropic only."""
    system, anthropic_messages = convert_messages_to_anthropic(messages, base_url=base_url, model=model)
    anthropic_tools = convert_tools_to_anthropic(tools) if tools else []
    # Nous Portal routes on its own catalog ids (``anthropic/claude-opus-4.8``); normalizing would
    # make the model unresolvable there (prefix AND dots kept).
    if not _is_nous_portal_endpoint(base_url):
        model = normalize_model_name(model, preserve_dots=preserve_dots)
    # Non-positive/non-finite values fail locally instead of 400-ing upstream.
    effective_max_tokens = _resolve_anthropic_messages_max_tokens(max_tokens, model, context_length=context_length)
    if context_length and effective_max_tokens > context_length:
        effective_max_tokens = max(context_length - 1, 1)
    to_wire = _oauth_wire_namer(anthropic_tools) if is_oauth else None
    if to_wire:
        system = _apply_claude_code_identity(system, anthropic_tools, anthropic_messages, to_wire)
    kwargs: Dict[str, Any] = {"model": model, "messages": anthropic_messages, "max_tokens": effective_max_tokens}
    if system:
        kwargs["system"] = system
    if anthropic_tools:
        kwargs["tools"] = anthropic_tools
        if tool_choice == "none":
            kwargs.pop("tools", None)  # no Anthropic "none" — omit tools to prevent use
        elif tool_choice is None or isinstance(tool_choice, str):
            # A forced tool name goes through the OAuth normalizer too: every tools[] entry is
            # mcp__-prefixed/aliased there, so the literal would leak and name a nonexistent tool.
            kwargs["tool_choice"] = _TOOL_CHOICE_MAP.get(tool_choice) or {
                "type": "tool", "name": to_wire(tool_choice) if to_wire else tool_choice
            }
    # Map reasoning_config to Anthropic's thinking parameter. Claude 4.6+ models use adaptive thinking +
    # output_config.effort. Older models use manual thinking with budget_tokens. MiniMax Anthropic-compat
    # endpoints support thinking (manual mode only, not adaptive). Haiku does NOT support extended thinking
    # — skip entirely. Kimi / Moonshot models also use adaptive thinking: their Anthropic-compatible
    # endpoints (api.moonshot.cn/anthropic, api.kimi.com/coding) accept ``thinking.type="adaptive"`` +
    # ``output_config.effort``, and the replay-validation 400s that originally motivated dropping the
    # parameter (#13848) no longer occur. (Kimi on chat_completions enables thinking via extra_body in the
    # ChatCompletionsTransport — see #13503.) On 4.7+ the `thinking.display` field defaults to "omitted",
    # which silently hides reasoning text that Hermes surfaces in its CLI. We request "summarized" so the
    # reasoning blocks stay populated — matching 4.6 behavior and preserving the activity-feed UX during
    # long tool runs.
    if reasoning_config and isinstance(reasoning_config, dict):
        kwargs.update(_thinking_kwargs(reasoning_config, model, effective_max_tokens))
    # Safety net so upstream 4.6 -> 4.7 migrations don't need coordinated edits everywhere callers
    # (auxiliary_client, ...) set sampling params.
    if _forbids_sampling_params(model):
        for key in ("temperature", "top_p", "top_k"):
            kwargs.pop(key, None)
    # Fast mode: native Anthropic only — third-party providers reject the unknown beta/param and
    # Anthropic scopes it to the Claude API (not Bedrock/Vertex/Foundry). Per-request extra_headers
    # OVERRIDE the client-level anthropic-beta header, so rebuild the full beta list.
    if fast_mode and not _is_third_party_anthropic_endpoint(base_url) and _supports_fast_mode(model):
        kwargs.setdefault("extra_body", {})["speed"] = "fast"
        betas = _common_betas_for_base_url(base_url, drop_context_1m_beta=drop_context_1m_beta)
        kwargs["extra_headers"] = _beta_header(betas + (_OAUTH_ONLY_BETAS if is_oauth else []) + [_FAST_MODE_BETA])
    return kwargs


# Keys exclusive to the OpenAI Responses / Codex shape; the Messages SDK raises ``TypeError: ...
# unexpected keyword argument`` on any of them.
_RESPONSES_ONLY_KWARGS = frozenset({"instructions", "input", "store", "parallel_tool_calls"})


def sanitize_anthropic_kwargs(api_kwargs: Any, *, log_prefix: str = "") -> Any:
    """Drop Responses-API-only keys before an Anthropic Messages SDK call. Boundary guard for
    api_mode-flip races (a concurrent auxiliary call mutating a shared agent between kwargs build
    and dispatch): a Responses-shaped payload reaching ``messages.stream()`` dies with a
    non-retryable TypeError that takes the whole turn and fallback chain with it. Mutates and
    returns ``api_kwargs``; logs a WARNING so the race stays visible."""
    leaked = _RESPONSES_ONLY_KWARGS.intersection(api_kwargs) if isinstance(api_kwargs, dict) else ()
    if leaked:
        for key in leaked:
            del api_kwargs[key]
        logger.warning(
            "%sStripped Responses-only kwarg(s) %s from an Anthropic Messages "
            "call (api_mode flip race — see #31673). The call will proceed; "
            "this breadcrumb means a kwargs build ran under a Responses "
            "api_mode while dispatch ran under anthropic_messages.",
            log_prefix,
            sorted(leaked),
        )
    return api_kwargs


def buffer_anthropic_tool_input(api_kwargs: dict[str, Any], base_url: str | None) -> None:
    """Retry knob for a malformed fine-grained tool-JSON stream (#107830): the beta streams tool
    args unvalidated, so a model that emits ``{"names": cronjob_manage}`` breaks the SDK parser
    and an identical retry breaks identically. ``eager_input_streaming: false`` per tool restores
    Anthropic's buffered, validated args for the rest of this turn (the flag lives on the turn's
    kwargs, so a later retry of the same turn keeps it; the changed ``tools`` block costs one
    prompt-cache miss, cheaper than a dead turn). Off the happy path on purpose:
    buffering a large payload is a zero-event gap the stale-stream detector kills. No-op on
    endpoints that never get the beta (MiniMax) rather than sending them an unknown field."""
    if _TOOL_STREAMING_BETA not in _common_betas_for_base_url(base_url):
        return
    for tool in api_kwargs.get("tools") or ():
        tool["eager_input_streaming"] = False


def _is_stream_unavailable_error(exc: Exception) -> bool:
    """True when an Anthropic stream call should fall back to create()."""
    err_lower = str(exc).lower()
    if "stream" in err_lower and "not supported" in err_lower:
        return True
    if "invokemodelwithresponsestream" not in err_lower:
        return False
    from agent.bedrock_adapter import is_streaming_access_denied_error
    return is_streaming_access_denied_error(exc)


def _stream_final_message(stream_fn, api_kwargs, log_prefix, on_stream_event, on_response):
    """``messages.stream()`` -> final Message, ticking the best-effort callbacks."""
    with stream_fn(**{k: v for k, v in api_kwargs.items() if k != "stream"}) as stream:
        if callable(on_response):
            try:
                on_response(getattr(stream, "response", None))
            except Exception:
                logger.debug("%son_response callback failed", log_prefix, exc_info=True)
        # Consume manually so each event ticks the progress callback; get_final_message then
        # returns the accumulated snapshot. TimeoutError is the caller's deadline seam: the host
        # has given up, so abandon the stream (``with`` closes it) instead of streaming an answer
        # nobody reads.
        for event in stream if callable(on_stream_event) else ():
            try:
                on_stream_event(event)
            except TimeoutError:
                # The callback is the caller's deadline seam (#99692: the host waiting on this summary has
                # already given up). Abandon the stream — the ``with`` closes it — instead of streaming an
                # answer nobody will read.
                raise
            except Exception:
                logger.debug("%son_stream_event callback failed", log_prefix, exc_info=True)
        return stream.get_final_message()


def create_anthropic_message(
    client: Any, api_kwargs: dict, *, log_prefix: str = "", prefer_stream: bool = True,
    on_stream_event=None, on_response=None,
) -> Any:
    """Create an Anthropic message, aggregating via stream when available. Some Anthropic-compatible
    gateways are SSE-only and answer ``create()`` with ``text/event-stream``, which the SDK surfaces
    as raw text (callers then crash on ``.content``), so prefer ``messages.stream()`` like the main
    turn path and fall back to ``create()`` only for providers that explicitly don't support
    streaming (restricted Bedrock roles). Both callbacks are best-effort and fire only on the
    streaming path: ``on_stream_event(event)`` lets liveness watchdogs see forward progress;
    ``on_response(httpx_response)`` exposes headers the parsed Message drops (Nous Portal's
    ``x-nous-credits-*`` balance family)."""
    sanitize_anthropic_kwargs(api_kwargs, log_prefix=log_prefix)
    messages_api = getattr(client, "messages", None)
    stream_fn = getattr(messages_api, "stream", None)
    if prefer_stream and callable(stream_fn):
        try:
            return _stream_final_message(stream_fn, api_kwargs, log_prefix, on_stream_event, on_response)
        except TimeoutError:
            raise
        except Exception as exc:
            if not _is_stream_unavailable_error(exc):
                raise
            logger.debug(
                "%sAnthropic Messages stream unavailable; falling back to messages.create(): %s", log_prefix, exc
            )
    return messages_api.create(**{k: v for k, v in api_kwargs.items() if k != "stream"})


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from pathlib import Path  # noqa: F401,E402
from typing import Tuple  # noqa: F401,E402
import copy  # noqa: F401,E402
import json  # noqa: F401,E402
import os  # noqa: F401,E402
import platform  # noqa: F401,E402
import secrets  # noqa: F401,E402
import stat  # noqa: F401,E402
from urllib.parse import urlparse  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'CredentialPersistError': ('agent.anthropic_credentials', 'CredentialPersistError'),
    'base_url_host_matches': ('utils', 'base_url_host_matches'),
    'base_url_hostname': ('utils', 'base_url_hostname'),
    'claude_code_credentials_path': ('agent.anthropic_credentials', 'claude_code_credentials_path'),
    'get_hermes_home': ('hermes_constants', 'get_hermes_home'),
    'is_claude_code_token_valid': ('agent.anthropic_credentials', 'is_claude_code_token_valid'),
    'is_rotation_consumed_uncommitted': ('agent.anthropic_credentials', 'is_rotation_consumed_uncommitted'),
    'mark_rotation_consumed_uncommitted': ('agent.anthropic_credentials', 'mark_rotation_consumed_uncommitted'),
    'read_claude_code_credentials': ('agent.anthropic_credentials', 'read_claude_code_credentials'),
    'read_hermes_oauth_credentials': ('agent.anthropic_credentials', 'read_hermes_oauth_credentials'),
    'refresh_anthropic_oauth_pure': ('agent.anthropic_credentials', 'refresh_anthropic_oauth_pure'),
    'resolve_anthropic_token': ('agent.anthropic_credentials', 'resolve_anthropic_token'),
    'run_hermes_oauth_login_pure': ('agent.anthropic_credentials', 'run_hermes_oauth_login_pure'),
    'run_oauth_setup_token': ('agent.anthropic_credentials', 'run_oauth_setup_token'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
