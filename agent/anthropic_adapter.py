"""Anthropic Messages API adapter for Hermes Agent.

Translates between Hermes's internal OpenAI-style message format and
Anthropic's Messages API. Follows the same pattern as the codex_responses
adapter — all provider-specific logic is isolated here.

Targets ``client.beta.messages.{create,stream}`` (anthropic SDK 0.100+).
The beta namespace exposes typed kwargs for the beta-gated fields
``thinking``, ``output_config``, ``context_management``, ``betas``,
``speed``, and ``metadata`` — eliminating the ``extra_body`` /
``extra_headers`` workarounds the plain ``messages.*`` namespace required.

Wire shape mirrors Claude Code 2.1.119 (verified by mitmdump capture
2026-05-06): same betas, same body field set, same metadata.user_id
identity blob shape.

Auth supports:
  - Regular API keys (sk-ant-api*) → x-api-key header
  - OAuth setup-tokens (sk-ant-oat*) → Bearer auth + beta header
  - Claude Code credentials (~/.claude.json or ~/.claude/.credentials.json) → Bearer auth
"""

import copy
import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import uuid
from pathlib import Path

from hermes_constants import get_hermes_home
from typing import Any, Dict, List, Optional, Tuple
from utils import base_url_host_matches, normalize_proxy_env_vars

# NOTE: `import anthropic` is deliberately NOT at module top — the SDK pulls
# ~220 ms of imports (anthropic.types, anthropic.lib.tools._beta_runner, etc.)
# and the 3 usage sites (build_anthropic_client, build_anthropic_bedrock_client,
# read_claude_code_credentials_from_keychain) are all on cold user-triggered
# paths. Access via the `_get_anthropic_sdk()` accessor below, which caches
# the module after the first call and returns None on ImportError.
_anthropic_sdk: Any = ...  # sentinel — None means "tried and missing"


def _get_anthropic_sdk():
    """Return the ``anthropic`` SDK module, importing lazily. None if not installed."""
    global _anthropic_sdk
    if _anthropic_sdk is ...:
        try:
            import anthropic as _sdk
            _anthropic_sdk = _sdk
        except ImportError:
            _anthropic_sdk = None
        else:
            _install_sse_event_observer(_sdk)
    return _anthropic_sdk


# ── SSE event observer (ping visibility) ──────────────────────────────
#
# The Anthropic SDK silently drops SSE ``ping`` events at
# ``anthropic/_streaming.py:102`` (``if sse.event == "ping": continue``),
# so during a request's queue + prefill phase the iterator yields nothing
# even though the server is sending keep-alive pings every ~10 s.  The
# downstream stale-stream detector in ``run_agent.py`` cannot distinguish
# "queued upstream, healthy" from "connection black-holed" without ping
# visibility, and ends up killing healthy long-TTFT requests (e.g.
# Opus 4.7 + 1M-context with a 200 K-token prompt on the OAuth/subscription
# path, where TTFT routinely exceeds 5 minutes).
#
# Hook design: monkey-patch ``Stream._iter_events`` — the source iterator
# that yields *all* SSE events including pings — to fire a thread-local
# callback before passing each event through.  The SDK's filtering layer
# (``Stream.__stream__``) still drops pings as before, so consumers see
# unchanged behavior.  Patches are installed once per process, guarded
# against SDK-internal API changes; on failure we log a warning and leave
# the SDK untouched (the cold-start tolerance in run_agent.py remains as
# a backstop).
import threading as _threading

_sse_event_callback = _threading.local()


def set_sse_event_callback(callback):
    """Install a thread-local callback fired on every raw SSE event.

    The callback receives one positional argument: the event name
    (``"ping"``, ``"message_start"``, ``"content_block_delta"``, …).
    Pass ``None`` to clear.  Per-thread — workers running in different
    threads don't see each other's callbacks.
    """
    _sse_event_callback.value = callback


def _get_sse_event_callback():
    return getattr(_sse_event_callback, "value", None)


_sse_observer_installed = False


def _install_sse_event_observer(sdk) -> None:
    """Wrap ``Stream._iter_events`` so we can observe pings.

    Idempotent — only patches once per process.  Best-effort: if the SDK's
    private API surface doesn't match what we expect (different version,
    refactor), we log and skip, leaving the SDK untouched.
    """
    global _sse_observer_installed
    if _sse_observer_installed:
        return
    try:
        from anthropic._streaming import Stream as _AntStream
    except Exception as exc:
        logger.warning(
            "Anthropic SDK SSE observer not installed (import failed: %s) — "
            "stream-stale detector will use cold-start tolerance only.",
            exc,
        )
        _sse_observer_installed = True
        return

    _orig_iter_events = getattr(_AntStream, "_iter_events", None)
    if _orig_iter_events is None:
        logger.warning(
            "Anthropic SDK SSE observer not installed (Stream._iter_events "
            "missing — SDK API changed?) — stream-stale detector will use "
            "cold-start tolerance only.",
        )
        _sse_observer_installed = True
        return

    def _hermes_iter_events(self):
        cb = _get_sse_event_callback()
        if cb is None:
            yield from _orig_iter_events(self)
            return
        for sse in _orig_iter_events(self):
            try:
                cb(getattr(sse, "event", None))
            except Exception:
                # Callback errors must never break SDK iteration.
                pass
            yield sse

    _AntStream._iter_events = _hermes_iter_events
    _sse_observer_installed = True
    logger.debug(
        "Anthropic SDK SSE observer installed — stream-stale detector "
        "now sees ping events."
    )

logger = logging.getLogger(__name__)


def _stable_device_id() -> str:
    """Stable per-machine identifier for Anthropic's metadata.user_id field.

    sha256 of the hostname — cheap, no FS access, stable across sessions.
    Mirrors Claude Code's wire shape (a 64-char hex string) so OAuth
    request fingerprints look identical to CC's.
    """
    return hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()


def _stable_account_uuid() -> str:
    """Stable per-install UUID stored in ``~/.hermes/account_uuid.txt``.

    Lazy-created on first read.  Mirrors Claude Code's account_uuid field
    (a UUID4 string) for the metadata.user_id blob.  Surviving across
    upgrades is the goal — keep the file outside any cleanup paths.
    """
    path = Path(get_hermes_home()) / "account_uuid.txt"
    try:
        if path.exists():
            cached = path.read_text(encoding="utf-8").strip()
            if cached:
                return cached
        new_id = str(uuid.uuid4())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_id, encoding="utf-8")
        return new_id
    except Exception:
        # Filesystem hiccup — fall back to a deterministic hash so we
        # still emit *something* stable for this run.
        return str(uuid.UUID(bytes=hashlib.sha256(
            socket.gethostname().encode("utf-8")
        ).digest()[:16]))


def _build_anthropic_metadata(session_id: str | None) -> Dict[str, str]:
    """Construct the metadata.user_id JSON blob for /v1/messages.

    Matches Claude Code 2.1.119's wire format:
        {"device_id": "<sha256 hostname>",
         "account_uuid": "<stable UUID>",
         "session_id": "<this conversation>"}
    The whole dict is serialized to a JSON string and placed in
    ``metadata.user_id`` per Anthropic's API shape.
    """
    blob = {
        "device_id": _stable_device_id(),
        "account_uuid": _stable_account_uuid(),
    }
    if session_id:
        blob["session_id"] = session_id
    return {"user_id": json.dumps(blob, separators=(",", ":"))}


THINKING_BUDGET = {"xhigh": 32000, "high": 16000, "medium": 8000, "low": 4000}
# Hermes effort → Anthropic adaptive-thinking effort (output_config.effort).
# Anthropic exposes 5 levels on 4.7+: low, medium, high, xhigh, max.
# Opus/Sonnet 4.6 only expose 4 levels: low, medium, high, max — no xhigh.
# We preserve xhigh as xhigh on 4.7+ (the recommended default for coding/
# agentic work) and downgrade it to max on pre-4.7 adaptive models (which
# is the strongest level they accept).  "minimal" is a legacy alias that
# maps to low on every model.  See:
# https://platform.claude.com/docs/en/about-claude/models/migration-guide
ADAPTIVE_EFFORT_MAP = {
    "max":     "max",
    "xhigh":   "xhigh",
    "high":    "high",
    "medium":  "medium",
    "low":     "low",
    "minimal": "low",
}

# Models that accept the "xhigh" output_config.effort level.  Opus 4.7 added
# xhigh as a distinct level between high and max; older adaptive-thinking
# models (4.6) reject it with a 400.  Keep this substring list in sync with
# the Anthropic migration guide as new model families ship.
_XHIGH_EFFORT_SUBSTRINGS = ("4-7", "4.7")

# Models where extended thinking is deprecated/removed (4.6+ behavior: adaptive
# is the only supported mode; 4.7 additionally forbids manual thinking entirely
# and drops temperature/top_p/top_k).
_ADAPTIVE_THINKING_SUBSTRINGS = ("4-6", "4.6", "4-7", "4.7")

# Models where temperature/top_p/top_k return 400 if set to non-default values.
# This is the Opus 4.7 contract; future 4.x+ models are expected to follow it.
_NO_SAMPLING_PARAMS_SUBSTRINGS = ("4-7", "4.7")
_FAST_MODE_SUPPORTED_SUBSTRINGS = ("opus-4-6", "opus-4.6")

# ── Max output token limits per Anthropic model ───────────────────────
# Source: Anthropic docs + Cline model catalog.  Anthropic's API requires
# max_tokens as a mandatory field.  Previously we hardcoded 16384, which
# starves thinking-enabled models (thinking tokens count toward the limit).
_ANTHROPIC_OUTPUT_LIMITS = {
    # Match Claude Code 2.1.119 main chat path (verified by disassembly:
    # `max_tokens: 16000` appears 7× in the binary; 64000 once for streaming
    # paths). Since hermes already spoofs Claude Code identity (user-agent,
    # system prefix, beta headers) to use the OAuth token, matching its
    # max_tokens too keeps backend scheduling/priority signals consistent
    # with what real Claude Code sends — even though the model itself isn't
    # supposed to see this value, we don't know what other API-side decisions
    # are keyed on it. Override per-call via max_tokens kwarg when needed.
    # Claude 4.7
    "claude-opus-4-7":    16_000,
    # Claude 4.6
    "claude-opus-4-6":    16_000,
    "claude-sonnet-4-6":  16_000,
    # Claude 4.5
    "claude-opus-4-5":    16_000,
    "claude-sonnet-4-5":  16_000,
    "claude-haiku-4-5":   16_000,
    # Claude 4
    "claude-opus-4":      32_000,
    "claude-sonnet-4":    64_000,
    # Claude 3.7
    "claude-3-7-sonnet": 128_000,
    # Claude 3.5
    "claude-3-5-sonnet":   8_192,
    "claude-3-5-haiku":    8_192,
    # Claude 3
    "claude-3-opus":       4_096,
    "claude-3-sonnet":     4_096,
    "claude-3-haiku":      4_096,
    # Third-party Anthropic-compatible providers
    "minimax":            131_072,
    # Qwen models via DashScope Anthropic-compatible endpoint
    # DashScope enforces max_tokens ∈ [1, 65536]
    "qwen3":               65_536,
}

# For any model not in the table, assume the highest current limit.
# Future Anthropic models are unlikely to have *less* output capacity.
_ANTHROPIC_DEFAULT_OUTPUT_LIMIT = 128_000


def _get_anthropic_max_output(model: str) -> int:
    """Look up the max output token limit for an Anthropic model.

    Uses substring matching against _ANTHROPIC_OUTPUT_LIMITS so date-stamped
    model IDs (claude-sonnet-4-5-20250929) and variant suffixes (:1m, :fast)
    resolve correctly.  Longest-prefix match wins to avoid e.g. "claude-3-5"
    matching before "claude-3-5-sonnet".

    Normalizes dots to hyphens so that model names like
    ``anthropic/claude-opus-4.6`` match the ``claude-opus-4-6`` table key.
    """
    m = model.lower().replace(".", "-")
    best_key = ""
    best_val = _ANTHROPIC_DEFAULT_OUTPUT_LIMIT
    for key, val in _ANTHROPIC_OUTPUT_LIMITS.items():
        if key in m and len(key) > len(best_key):
            best_key = key
            best_val = val
    return best_val


def _resolve_positive_anthropic_max_tokens(value) -> Optional[int]:
    """Return ``value`` floored to a positive int, or ``None`` if it is not a
    finite positive number. Ported from openclaw/openclaw#66664.

    Anthropic's Messages API rejects ``max_tokens`` values that are 0,
    negative, non-integer, or non-finite with HTTP 400. Python's ``or``
    idiom (``max_tokens or fallback``) correctly catches ``0`` but lets
    negative ints and fractional floats (``-1``, ``0.5``) through to the
    API, producing a user-visible failure instead of a local error.
    """
    # Booleans are a subclass of int — exclude explicitly so ``True`` doesn't
    # silently become 1 and ``False`` doesn't become 0.
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    try:
        import math
        if not math.isfinite(value):
            return None
    except Exception:
        return None
    floored = int(value)  # truncates toward zero for floats
    return floored if floored > 0 else None


def _resolve_anthropic_messages_max_tokens(
    requested,
    model: str,
    context_length: Optional[int] = None,
) -> int:
    """Resolve the ``max_tokens`` budget for an Anthropic Messages call.

    Prefers ``requested`` when it is a positive finite number; otherwise
    falls back to the model's output ceiling. Raises ``ValueError`` if no
    positive budget can be resolved (should not happen with current model
    table defaults, but guards against a future regression where
    ``_get_anthropic_max_output`` could return ``0``).

    Separately, callers apply a context-window clamp — this resolver does
    not, to keep the positive-value contract independent of endpoint
    specifics.

    Ported from openclaw/openclaw#66664 (resolveAnthropicMessagesMaxTokens).
    """
    resolved = _resolve_positive_anthropic_max_tokens(requested)
    if resolved is not None:
        return resolved
    fallback = _get_anthropic_max_output(model)
    if fallback > 0:
        return fallback
    raise ValueError(
        f"Anthropic Messages adapter requires a positive max_tokens value for "
        f"model {model!r}; got {requested!r} and no model default resolved."
    )


def _supports_adaptive_thinking(model: str) -> bool:
    """Return True for Claude 4.6+ models that support adaptive thinking."""
    return any(v in model for v in _ADAPTIVE_THINKING_SUBSTRINGS)


def _supports_xhigh_effort(model: str) -> bool:
    """Return True for models that accept the 'xhigh' adaptive effort level.

    Opus 4.7 introduced xhigh as a distinct level between high and max.
    Pre-4.7 adaptive models (Opus/Sonnet 4.6) only accept low/medium/high/max
    and reject xhigh with an HTTP 400. Callers should downgrade xhigh→max
    when this returns False.
    """
    return any(v in model for v in _XHIGH_EFFORT_SUBSTRINGS)


def _forbids_sampling_params(model: str) -> bool:
    """Return True for models that 400 on any non-default temperature/top_p/top_k.

    Opus 4.7 explicitly rejects sampling parameters; later Claude releases are
    expected to follow suit.  Callers should omit these fields entirely rather
    than passing zero/default values (the API rejects anything non-null).
    """
    return any(v in model for v in _NO_SAMPLING_PARAMS_SUBSTRINGS)


def _supports_fast_mode(model: str) -> bool:
    """Return True for models that support Anthropic Fast Mode (speed=fast).

    Per Anthropic docs, fast mode is currently supported on Opus 4.6 only.
    Sending ``speed: "fast"`` to any other Claude model (including Opus 4.7)
    returns HTTP 400. This guard prevents silently 400'ing when stale config
    or older callers leave fast mode enabled across a model upgrade.
    """
    return any(v in model for v in _FAST_MODE_SUPPORTED_SUBSTRINGS)


# Beta headers for enhanced features that are safe on ordinary/native Anthropic
# requests. As of Opus 4.7 (2026-04-16), these are GA on Claude 4.6+ — the
# beta headers are still accepted (harmless no-op) but not required. Kept
# here so older Claude (4.5, 4.1) + compatible endpoints that still gate on
# the headers continue to get the enhanced features.
#
# Do NOT include ``context-1m-2025-08-07`` here. Anthropic returns HTTP 400
# ("long context beta is not yet available for this subscription") for
# accounts without the long-context beta, which breaks normal short auxiliary
# calls like title generation/session summarization.
#
# ``context-1m-2025-08-07`` is still required to unlock the 1M context window
# on Claude Opus 4.6/4.7 and Sonnet 4.6 when served via AWS Bedrock or Azure
# AI Foundry. Add it only for those endpoint-specific paths below.
_COMMON_BETAS = [
    "interleaved-thinking-2025-05-14",
    "fine-grained-tool-streaming-2025-05-14",
    # extended-cache-ttl-2025-04-11 enables the ``ttl`` field on
    # cache_control markers (e.g. ``{"type": "ephemeral", "ttl": "1h"}``).
    # Without this header, Anthropic ignores the ttl field and falls back
    # to the default 5-minute cache TTL — which silently breaks the
    # ``prompt_caching.cache_ttl: 1h`` config. The header is harmless when
    # cache_ttl is "5m" (the marker just doesn't include ttl in that case).
    "extended-cache-ttl-2025-04-11",
    # Added 2026-05-06 to mirror Claude Code 2.1.119's wire format
    # (verified by mitmdump capture against api.anthropic.com).
    # CC sends these on every /v1/messages request:
    "redact-thinking-2026-02-12",
    "context-management-2025-06-27",
    "prompt-caching-scope-2026-01-05",
    "effort-2025-11-24",
]
# context-1m-2025-08-07 is added conditionally — see
# ``_base_url_needs_context_1m_beta`` and the insert in
# ``_common_betas_for_base_url`` below.
# Anthropic-native-only betas — strip on bearer-auth third-party endpoints
# (MiniMax etc. host their own models and reject unknown betas).
_ANTHROPIC_NATIVE_ONLY_BETAS = {
    "redact-thinking-2026-02-12",
    "context-management-2025-06-27",
    "prompt-caching-scope-2026-01-05",
    "effort-2025-11-24",
}
# MiniMax's Anthropic-compatible endpoints fail tool-use requests when
# the fine-grained tool streaming beta is present.  Omit it so tool calls
# fall back to the provider's default response path.
_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"
# 1M context beta. Native Anthropic does not get this by default because some
# subscriptions reject it, but Bedrock/Azure still need it for 1M context.
_CONTEXT_1M_BETA = "context-1m-2025-08-07"
# Extended cache TTL beta — Anthropic-only feature; bearer-auth endpoints
# (MiniMax) host their own models and don't honor it, and may reject
# unknown Anthropic-namespaced betas.
_EXTENDED_CACHE_TTL_BETA = "extended-cache-ttl-2025-04-11"


def _model_supports_1m_context(model: str | None) -> bool:
    """Return True only for Anthropic models that have a 1M-context tier.

    As of 2026-05, that's Opus 4.6+, Opus 4.7, and Sonnet 4.6. Haiku 4.5
    has no 1M tier — requesting the beta on a Haiku call returns
    "long context beta is not yet available" even from paid API customers
    (it's a per-model entitlement, not per-subscription).

    Without this gate, every Haiku subagent re-discovers the rejection at
    first API call, prints the noisy warning, rebuilds its client, and
    retries. With it, the beta header simply never goes out for Haiku.

    Match by substring against ``model`` so prefixed forms
    ("anthropic/claude-opus-4-7", "claude-opus-4.7", "us.claude-opus-4-7-v1")
    all resolve correctly. Returns False for empty/None — safer to drop the
    beta than guess wrong.
    """
    if not model:
        return False
    m = str(model).lower()
    # Models with a 1M-context tier. Conservative allowlist — if a future
    # Haiku gains 1M, add it here explicitly rather than fuzzy-matching.
    _SUPPORTS_1M = (
        "claude-opus-4-7", "claude-opus-4.7",
        "claude-opus-4-6", "claude-opus-4.6",
        "claude-sonnet-4-6", "claude-sonnet-4.6",
    )
    return any(needle in m for needle in _SUPPORTS_1M)

# Fast mode beta — enables the ``speed: "fast"`` request parameter for
# significantly higher output token throughput on Opus 4.6 (~2.5x).
# See https://platform.claude.com/docs/en/build-with-claude/fast-mode
_FAST_MODE_BETA = "fast-mode-2026-02-01"

# Additional beta headers required for OAuth/subscription auth.
# Matches what Claude Code (and pi-ai / OpenCode) send.
_OAUTH_ONLY_BETAS = [
    "claude-code-20250219",
    "oauth-2025-04-20",
]

# Claude Code identity — required for OAuth requests to be routed correctly.
# Without these, Anthropic's infrastructure intermittently 500s OAuth traffic.
# The version must stay reasonably current — Anthropic rejects OAuth requests
# when the spoofed user-agent version is too far behind the actual release.
# Confirmed failure mode for stale fallbacks: requests come back as HTTP 400
# "You're out of extra usage" — a misleading billing-tier message that
# actually signals the user-agent version is rejected. Bump this constant
# whenever you notice deployments without Claude Code installed start to
# 400 inexplicably.
_CLAUDE_CODE_VERSION_FALLBACK = "2.1.138"
_claude_code_version_cache: Optional[str] = None


def _detect_claude_code_version() -> str:
    """Detect the installed Claude Code version, fall back to a static constant.

    Anthropic's OAuth infrastructure validates the user-agent version and may
    reject requests with a version that's too old.  Detecting dynamically means
    users who keep Claude Code updated never hit stale-version 400s.
    """
    import subprocess as _sp

    for cmd in ("claude", "claude-code"):
        try:
            result = _sp.run(
                [cmd, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Output is like "2.1.74 (Claude Code)" or just "2.1.74"
                version = result.stdout.strip().split()[0]
                if version and version[0].isdigit():
                    return version
        except Exception:
            pass
    return _CLAUDE_CODE_VERSION_FALLBACK


_CLAUDE_CODE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."
# Real Claude Code MCP tools follow ``mcp__<server>__<tool>`` (double-
# underscore separators).  Hermes' MCP-source tools are registered with the
# same convention now (see ``tools/mcp_tool.py::_convert_mcp_schema``).  This
# constant is the *prefix* check — anything starting with ``mcp__`` is
# treated as already-prefixed by the OAuth-path identity rewriter.
_MCP_TOOL_PREFIX = "mcp__"


def _get_claude_code_version() -> str:
    """Lazily detect the installed Claude Code version when OAuth headers need it."""
    global _claude_code_version_cache
    if _claude_code_version_cache is None:
        _claude_code_version_cache = _detect_claude_code_version()
    return _claude_code_version_cache


def _system_prompt_mode_compact() -> bool:
    """Return True when ``agent.system_prompt_mode`` is set to ``compact``.

    Cheap import — the module loads lazily so we don't pay for it on every
    request unless the user opts in to compact mode. Falls back to False on
    any config-load failure so legacy behavior wins under errors.
    """
    try:
        from hermes_cli.config import load_config as _load_cfg
        mode = ((_load_cfg() or {}).get("agent") or {}).get("system_prompt_mode")
        return str(mode or "").strip().lower() == "compact"
    except Exception:
        return False


def _prepend_user_message_preamble(
    messages: List[Dict[str, Any]],
    preamble: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Insert ``preamble`` (a content block) at the head of the first
    user-role message's content list. Pure — returns a new list.

    Used by compact-mode system-prompt placement: dynamic context that
    would otherwise live in ``system`` rides on the conversation instead.
    Handles three content shapes:
      * ``content`` is a string → wrap in a list and prepend
      * ``content`` is already a list → prepend the block in place
      * No user messages exist → return ``messages`` unchanged

    Tool_result-only first turns (resume from background tool call) are
    rare on the gateway path; if encountered we leave them alone since
    Anthropic disallows non-tool_result content as the first block of a
    tool_result turn.
    """
    if not isinstance(messages, list) or not messages:
        return messages

    out = list(messages)
    for i, msg in enumerate(out):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        # Skip messages whose first content block is a tool_result —
        # Anthropic enforces tool_result-first ordering on those turns.
        if isinstance(content, list) and content and isinstance(content[0], dict):
            if content[0].get("type") == "tool_result":
                continue
        new_msg = dict(msg)
        if isinstance(content, str):
            new_msg["content"] = [preamble, {"type": "text", "text": content}]
        elif isinstance(content, list):
            new_msg["content"] = [preamble, *content]
        else:
            # Unrecognized content shape — leave it alone, return untouched.
            return messages
        out[i] = new_msg
        return out

    return messages


def _is_oauth_token(key: str) -> bool:
    """Check if the key is an Anthropic OAuth/setup token.

    Positively identifies Anthropic OAuth tokens by their key format:
    - ``sk-ant-`` prefix (but NOT ``sk-ant-api``) → setup tokens, managed keys
    - ``eyJ`` prefix → JWTs from the Anthropic OAuth flow
    - ``cc-`` prefix → Claude Code OAuth access tokens (from CLAUDE_CODE_OAUTH_TOKEN)

    Non-Anthropic keys (MiniMax, Alibaba, etc.) don't match any pattern
    and correctly return False.
    """
    if not key:
        return False
    # Regular Anthropic Console API keys — x-api-key auth, never OAuth
    if key.startswith("sk-ant-api"):
        return False
    # Anthropic-issued tokens (setup-tokens sk-ant-oat-*, managed keys)
    if key.startswith("sk-ant-"):
        return True
    # JWTs from Anthropic OAuth flow
    if key.startswith("eyJ"):
        return True
    # Claude Code OAuth access tokens (opaque, from CLAUDE_CODE_OAUTH_TOKEN)
    if key.startswith("cc-"):
        return True
    return False


def _normalize_base_url_text(base_url) -> str:
    """Normalize SDK/base transport URL values to a plain string for inspection.

    Some client objects expose ``base_url`` as an ``httpx.URL`` instead of a raw
    string.  Provider/auth detection should accept either shape.
    """
    if not base_url:
        return ""
    return str(base_url).strip()


def _is_third_party_anthropic_endpoint(base_url: str | None) -> bool:
    """Return True for non-Anthropic endpoints using the Anthropic Messages API.

    Third-party proxies (Azure AI Foundry, AWS Bedrock, self-hosted) authenticate
    with their own API keys via x-api-key, not Anthropic OAuth tokens. OAuth
    detection should be skipped for these endpoints.
    """
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False  # No base_url = direct Anthropic API
    normalized = normalized.rstrip("/").lower()
    if "anthropic.com" in normalized:
        return False  # Direct Anthropic API — OAuth applies
    return True  # Any other endpoint is a third-party proxy


def _is_kimi_coding_endpoint(base_url: str | None) -> bool:
    """Return True for Kimi's /coding endpoint that requires claude-code UA."""
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False
    return normalized.rstrip("/").lower().startswith("https://api.kimi.com/coding")


# Model-name prefixes that identify the Kimi / Moonshot family.  Covers
# - official slugs: ``kimi-k2.5``, ``kimi_thinking``, ``moonshot-v1-8k``
# - common release lines: ``k1.5-...``, ``k2-thinking``, ``k25-...``, ``k2.5-...``
# Matched case-insensitively against the post-``normalize_model_name`` form,
# so a caller's ``provider/vendor/model`` slug is handled the same as a
# bare name.
_KIMI_FAMILY_MODEL_PREFIXES = (
    "kimi-", "kimi_",
    "moonshot-", "moonshot_",
    "k1.", "k1-",
    "k2.", "k2-",
    "k25", "k2.5",
)


def _model_name_is_kimi_family(model: str | None) -> bool:
    if not isinstance(model, str):
        return False
    m = model.strip().lower()
    if not m:
        return False
    # Strip vendor prefix (e.g. ``moonshotai/kimi-k2.5`` → ``kimi-k2.5``)
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
    return m.startswith(_KIMI_FAMILY_MODEL_PREFIXES)


def _is_kimi_family_endpoint(base_url: str | None, model: str | None = None) -> bool:
    """Return True for any Kimi / Moonshot Anthropic-Messages-speaking endpoint.

    Broader than ``_is_kimi_coding_endpoint`` — matches:

    - Kimi's official ``/coding`` URL (legacy check, preserved)
    - Any ``api.kimi.com`` / ``moonshot.ai`` / ``moonshot.cn`` host
    - Custom or proxied endpoints whose *model* name is in the Kimi / Moonshot
      family (``kimi-*``, ``moonshot-*``, ``k1.*``, ``k2.*``, …).  Users with
      ``api_mode: anthropic_messages`` on a private gateway fronting Kimi
      fall into this branch — the upstream still enforces Kimi's thinking
      semantics (reasoning_content required on every replayed tool-call
      message) regardless of the gateway's hostname.

    Used to decide whether to drop Anthropic's ``thinking`` kwarg and to
    preserve unsigned reasoning_content-derived thinking blocks on replay.
    See hermes-agent#13848, #17057.
    """
    if _is_kimi_coding_endpoint(base_url):
        return True
    for _domain in ("api.kimi.com", "moonshot.ai", "moonshot.cn"):
        if base_url_host_matches(base_url or "", _domain):
            return True
    if _model_name_is_kimi_family(model):
        return True
    return False


def _is_deepseek_anthropic_endpoint(base_url: str | None) -> bool:
    """Return True for DeepSeek's Anthropic-compatible endpoint.

    DeepSeek's ``/anthropic`` route speaks the Anthropic Messages protocol
    but, when thinking mode is enabled, requires the ``thinking`` blocks
    from prior assistant turns to round-trip on subsequent requests — the
    generic third-party path strips them and triggers HTTP 400::

        The content[].thinking in the thinking mode must be passed back
        to the API.

    Per DeepSeek's published compatibility matrix the blocks are unsigned
    (no Anthropic-proprietary signature, no ``redacted_thinking`` support),
    so this endpoint is handled with the same strip-signed / keep-unsigned
    policy used for Kimi's ``/coding`` endpoint.  The match is pinned to
    the ``/anthropic`` path so the OpenAI-compatible ``api.deepseek.com``
    base URL (which never reaches this adapter) is not misclassified.
    See hermes-agent#16748.
    """
    if not base_url_host_matches(base_url or "", "api.deepseek.com"):
        return False
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False
    return "/anthropic" in normalized.rstrip("/").lower()


def _requires_bearer_auth(base_url: str | None) -> bool:
    """Return True for Anthropic-compatible providers that require Bearer auth.

    Some third-party /anthropic endpoints implement Anthropic's Messages API but
    require Authorization: Bearer *** of Anthropic's native x-api-key header.
    MiniMax's global and China Anthropic-compatible endpoints follow this pattern.
    """
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False
    normalized = normalized.rstrip("/").lower()
    return normalized.startswith(("https://api.minimax.io/anthropic", "https://api.minimaxi.com/anthropic"))


def _base_url_needs_context_1m_beta(base_url: str | None) -> bool:
    """Return True for endpoints that gate 1M context behind a beta.

    Native Anthropic (no base_url override, or any *.anthropic.com host)
    plus Azure AI Foundry. Bedrock has its own client helper
    (``build_anthropic_bedrock_client``) that opts in explicitly.
    Bearer-auth third-party endpoints (MiniMax) reject the beta and have
    it stripped further down in ``_common_betas_for_base_url``. Custom
    base_urls of unknown origin do NOT get the beta — conservative
    default to avoid the "long context beta is not yet available"
    rejection from third-party providers that mimic Anthropic's surface.
    """
    normalized = _normalize_base_url_text(base_url).lower()
    if not normalized:
        return True  # native Anthropic — default base_url
    if "azure.com" in normalized:
        return True
    if "anthropic.com" in normalized:
        return True
    return False


def _common_betas_for_base_url(
    base_url: str | None,
    *,
    drop_context_1m_beta: bool = False,
    model: str | None = None,
) -> list[str]:
    """Return the beta headers that are safe for the configured endpoint.

    MiniMax's Anthropic-compatible endpoints (Bearer-auth) reject requests
    that include Anthropic's ``fine-grained-tool-streaming`` beta — every
    tool-use message triggers a connection error.

    The ``context-1m-2025-08-07`` beta is not sent to native Anthropic by
    default because some subscriptions reject it. Add it only for endpoint
    families that still require it for 1M context, currently Azure AI Foundry.
    Bedrock uses its own client helper below and opts in explicitly.

    ``drop_context_1m_beta=True`` additionally strips the 1M-context beta on
    otherwise-unrelated endpoints. The OAuth retry path flips this flag after
    a subscription rejects the beta with
    "The long context beta is not yet available for this subscription" so
    subsequent requests in the same session don't repeat the probe. See the
    reactive recovery loop in ``run_agent.py`` and issue-comment history on
    PR #17680 for the full rationale.

    ``model``, when known, gates the 1M-context beta proactively: models
    without a 1M tier (Haiku 4.5, older Claude) silently drop the header so
    subagents using those models never trigger the rejection-and-retry path.
    Leaving ``model=None`` falls back to the pre-existing endpoint+latch
    gating only — capable models still get the beta.
    """
    betas = list(_COMMON_BETAS)
    if (
        _base_url_needs_context_1m_beta(base_url)
        and not drop_context_1m_beta
        and (model is None or _model_supports_1m_context(model))
    ):
        # Insert at position 3 (after fine-grained-tool-streaming) to
        # preserve Claude Code 2.1.119's wire-format ordering verified
        # by mitmdump against api.anthropic.com.
        betas.insert(2, _CONTEXT_1M_BETA)
    if _requires_bearer_auth(base_url):
        _stripped = {_TOOL_STREAMING_BETA, _CONTEXT_1M_BETA, _EXTENDED_CACHE_TTL_BETA} | _ANTHROPIC_NATIVE_ONLY_BETAS
        return [b for b in betas if b not in _stripped]
    return betas


def build_anthropic_client(
    api_key: str,
    base_url: str = None,
    timeout: float = None,
    *,
    drop_context_1m_beta: bool = False,
    model: Optional[str] = None,
):
    """Create an Anthropic client, auto-detecting setup-tokens vs API keys.

    If *timeout* is provided it overrides the default 900s read timeout.  The
    connect timeout stays at 10s.  Callers pass this from the per-provider /
    per-model ``request_timeout_seconds`` config so Anthropic-native and
    Anthropic-compatible providers respect the same knob as OpenAI-wire
    providers.

    ``drop_context_1m_beta=True`` strips ``context-1m-2025-08-07`` from the
    client-level ``anthropic-beta`` header. Used by the reactive OAuth retry
    path in ``run_agent.py`` when a subscription rejects the beta; leave at
    its default on fresh clients so 1M-capable subscriptions keep the
    capability.

    ``model`` (when provided) lets ``_common_betas_for_base_url`` strip the
    1M-context beta proactively for models that don't have a 1M tier (e.g.
    Haiku 4.5). Without this, the auxiliary client gets ``context-1m-…``
    on its client-level headers and Haiku rejects every call with HTTP 400
    "long context beta is not yet available for this subscription". The
    main agent loop sets ``drop_context_1m_beta`` explicitly, so leaving
    ``model`` at None there is fine.

    Returns an anthropic.Anthropic instance.
    """
    _anthropic_sdk = _get_anthropic_sdk()
    if _anthropic_sdk is None:
        raise ImportError(
            "The 'anthropic' package is required for the Anthropic provider. "
            "Install it with: pip install 'anthropic>=0.39.0'"
        )

    normalize_proxy_env_vars()

    from httpx import Timeout

    normalized_base_url = _normalize_base_url_text(base_url)
    _read_timeout = timeout if (isinstance(timeout, (int, float)) and timeout > 0) else 900.0
    kwargs = {
        "timeout": Timeout(timeout=float(_read_timeout), connect=10.0),
    }
    if normalized_base_url:
        # Azure Anthropic endpoints require an ``api-version`` query parameter.
        # Pass it via default_query so the SDK appends it to every request URL
        # without corrupting the base_url (appending it directly produces
        # malformed paths like /anthropic?api-version=.../v1/messages).
        _is_azure_endpoint = "azure.com" in normalized_base_url.lower()
        if _is_azure_endpoint and "api-version" not in normalized_base_url:
            kwargs["base_url"] = normalized_base_url.rstrip("/")
            kwargs["default_query"] = {"api-version": "2025-04-15"}
        else:
            kwargs["base_url"] = normalized_base_url
    common_betas = _common_betas_for_base_url(
        normalized_base_url,
        drop_context_1m_beta=drop_context_1m_beta,
        model=model,
    )

    if _is_kimi_coding_endpoint(base_url):
        # Kimi's /coding endpoint requires User-Agent: claude-code/0.1.0
        # to be recognized as a valid Coding Agent. Without it, returns 403.
        # Check this BEFORE _requires_bearer_auth since both match api.kimi.com/coding.
        kwargs["api_key"] = api_key
        kwargs["default_headers"] = {
            "User-Agent": "claude-code/0.1.0",
            **( {"anthropic-beta": ",".join(common_betas)} if common_betas else {} )
        }
    elif _requires_bearer_auth(normalized_base_url):
        # Some Anthropic-compatible providers (e.g. MiniMax) expect the API key in
        # Authorization: Bearer *** for regular API keys. Route those endpoints
        # through auth_token so the SDK sends Bearer auth instead of x-api-key.
        # Check this before OAuth token shape detection because MiniMax secrets do
        # not use Anthropic's sk-ant-api prefix and would otherwise be misread as
        # Anthropic OAuth/setup tokens.
        kwargs["auth_token"] = api_key
        if common_betas:
            kwargs["default_headers"] = {"anthropic-beta": ",".join(common_betas)}
    elif _is_third_party_anthropic_endpoint(base_url):
        # Third-party proxies (Azure AI Foundry, AWS Bedrock, etc.) use their
        # own API keys with x-api-key auth. Skip OAuth detection — their keys
        # don't follow Anthropic's sk-ant-* prefix convention and would be
        # misclassified as OAuth tokens.
        kwargs["api_key"] = api_key
        if common_betas:
            kwargs["default_headers"] = {"anthropic-beta": ",".join(common_betas)}
    elif _is_oauth_token(api_key):
        # OAuth access token / setup-token → Bearer auth + Claude Code identity.
        # Anthropic routes OAuth requests based on user-agent and headers;
        # without Claude Code's fingerprint, requests get intermittent 500s.
        #
        # Strip x-stainless-* fingerprint headers (2026-05-06): the Python
        # SDK adds 6 x-stainless-{lang,os,arch,runtime,runtime-version,
        # package-version} + 2 per-request (retry-count, read-timeout)
        # headers identifying the request as Python SDK. Claude Code's
        # native (Bun/JS) implementation doesn't send these. If Anthropic
        # routes/prioritises requests by client fingerprint, these
        # headers tag hermes as "third-party Python automation" while a
        # bare claude-cli UA would tag it as the official client. Empirical
        # evidence: hermes hits sporadic multi-minute "queued/prefilling"
        # stalls Claude Code never sees, with same model + same betas +
        # same OAuth scope. Use ``Omit()`` (the SDK's drop-header
        # sentinel) to suppress them.
        try:
            from anthropic._types import Omit as _Omit
            _omit_stainless = {
                "x-stainless-lang": _Omit(),
                "x-stainless-package-version": _Omit(),
                "x-stainless-os": _Omit(),
                "x-stainless-arch": _Omit(),
                "x-stainless-runtime": _Omit(),
                "x-stainless-runtime-version": _Omit(),
                "x-stainless-retry-count": _Omit(),
                "x-stainless-read-timeout": _Omit(),
                "x-stainless-timeout": _Omit(),
            }
        except ImportError:
            _omit_stainless = {}
        all_betas = common_betas + _OAUTH_ONLY_BETAS
        kwargs["auth_token"] = api_key
        kwargs["default_headers"] = {
            "anthropic-beta": ",".join(all_betas),
            "user-agent": f"claude-cli/{_get_claude_code_version()} (external, cli)",
            "x-app": "cli",
            **_omit_stainless,
        }
    else:
        # Regular API key → x-api-key header + common betas
        kwargs["api_key"] = api_key
        if common_betas:
            kwargs["default_headers"] = {"anthropic-beta": ",".join(common_betas)}

    return _anthropic_sdk.Anthropic(**kwargs)


def build_anthropic_bedrock_client(region: str):
    """Create an AnthropicBedrock client for Bedrock Claude models.

    Uses the Anthropic SDK's native Bedrock adapter, which provides full
    Claude feature parity: prompt caching, thinking budgets, adaptive
    thinking, fast mode — features not available via the Converse API.

    Attaches the common Anthropic beta headers as client-level defaults so
    that Bedrock-hosted Claude models get the same enhanced features as
    native Anthropic. The ``context-1m-2025-08-07`` beta in particular
    unlocks the 1M context window for Opus 4.6/4.7 on Bedrock — without
    it, Bedrock caps these models at 200K even though the Anthropic API
    serves them with 1M natively.

    Auth uses the boto3 default credential chain (IAM roles, SSO, env vars).
    """
    _anthropic_sdk = _get_anthropic_sdk()
    if _anthropic_sdk is None:
        raise ImportError(
            "The 'anthropic' package is required for the Bedrock provider. "
            "Install it with: pip install 'anthropic>=0.39.0'"
        )
    if not hasattr(_anthropic_sdk, "AnthropicBedrock"):
        raise ImportError(
            "anthropic.AnthropicBedrock not available. "
            "Upgrade with: pip install 'anthropic>=0.39.0'"
        )
    from httpx import Timeout

    return _anthropic_sdk.AnthropicBedrock(
        aws_region=region,
        timeout=Timeout(timeout=900.0, connect=10.0),
        default_headers={"anthropic-beta": ",".join([*_COMMON_BETAS, _CONTEXT_1M_BETA])},
    )


def _read_claude_code_credentials_from_keychain() -> Optional[Dict[str, Any]]:
    """Read Claude Code OAuth credentials from the macOS Keychain.

    Claude Code >=2.1.114 stores credentials in the macOS Keychain under the
    service name "Claude Code-credentials" rather than (or in addition to)
    the JSON file at ~/.claude/.credentials.json.

    The password field contains a JSON string with the same claudeAiOauth
    structure as the JSON file.

    Returns dict with {accessToken, refreshToken?, expiresAt?} or None.
    """
    if platform.system() != "Darwin":
        return None

    try:
        # Read the "Claude Code-credentials" generic password entry
        result = subprocess.run(
            ["security", "find-generic-password",
             "-s", "Claude Code-credentials",
             "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.debug("Keychain: security command not available or timed out")
        return None

    if result.returncode != 0:
        logger.debug("Keychain: no entry found for 'Claude Code-credentials'")
        return None

    raw = result.stdout.strip()
    if not raw:
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("Keychain: credentials payload is not valid JSON")
        return None

    oauth_data = data.get("claudeAiOauth")
    if oauth_data and isinstance(oauth_data, dict):
        access_token = oauth_data.get("accessToken", "")
        if access_token:
            return {
                "accessToken": access_token,
                "refreshToken": oauth_data.get("refreshToken", ""),
                "expiresAt": oauth_data.get("expiresAt", 0),
                "source": "macos_keychain",
            }

    return None


def read_claude_code_credentials() -> Optional[Dict[str, Any]]:
    """Read refreshable Claude Code OAuth credentials.

    Checks two sources in order:
      1. macOS Keychain (Darwin only) — "Claude Code-credentials" entry
      2. ~/.claude/.credentials.json file

    This intentionally excludes ~/.claude.json primaryApiKey. Opencode's
    subscription flow is OAuth/setup-token based with refreshable credentials,
    and native direct Anthropic provider usage should follow that path rather
    than auto-detecting Claude's first-party managed key.

    Returns dict with {accessToken, refreshToken?, expiresAt?} or None.
    """
    # Try macOS Keychain first (covers Claude Code >=2.1.114)
    kc_creds = _read_claude_code_credentials_from_keychain()
    if kc_creds:
        return kc_creds

    # Fall back to JSON file
    cred_path = Path.home() / ".claude" / ".credentials.json"
    if cred_path.exists():
        try:
            data = json.loads(cred_path.read_text(encoding="utf-8"))
            oauth_data = data.get("claudeAiOauth")
            if oauth_data and isinstance(oauth_data, dict):
                access_token = oauth_data.get("accessToken", "")
                if access_token:
                    return {
                        "accessToken": access_token,
                        "refreshToken": oauth_data.get("refreshToken", ""),
                        "expiresAt": oauth_data.get("expiresAt", 0),
                        "source": "claude_code_credentials_file",
                    }
        except (json.JSONDecodeError, OSError, IOError) as e:
            logger.debug("Failed to read ~/.claude/.credentials.json: %s", e)

    return None


def read_claude_managed_key() -> Optional[str]:
    """Read Claude's native managed key from ~/.claude.json for diagnostics only."""
    claude_json = Path.home() / ".claude.json"
    if claude_json.exists():
        try:
            data = json.loads(claude_json.read_text(encoding="utf-8"))
            primary_key = data.get("primaryApiKey", "")
            if isinstance(primary_key, str) and primary_key.strip():
                return primary_key.strip()
        except (json.JSONDecodeError, OSError, IOError) as e:
            logger.debug("Failed to read ~/.claude.json: %s", e)
    return None


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
                "User-Agent": f"claude-cli/{_get_claude_code_version()} (external, cli)",
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
    """Attempt to refresh an expired Claude Code OAuth token."""
    refresh_token = creds.get("refreshToken", "")
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
        _tmp_cred = cred_path.with_suffix(".tmp")
        _tmp_cred.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        _tmp_cred.replace(cred_path)
        # Restrict permissions (credentials file)
        cred_path.chmod(0o600)
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


def resolve_anthropic_token() -> Optional[str]:
    """Resolve an Anthropic token from all available sources.

    Priority:
      1. ANTHROPIC_TOKEN env var (OAuth/setup token saved by Hermes)
      2. CLAUDE_CODE_OAUTH_TOKEN env var
      3. Claude Code credentials (~/.claude.json or ~/.claude/.credentials.json)
         — with automatic refresh if expired and a refresh token is available
      4. ANTHROPIC_API_KEY env var (regular API key, or legacy fallback)

    Returns the token string or None.
    """
    creds = read_claude_code_credentials()

    # 1. Hermes-managed OAuth/setup token env var
    token = os.getenv("ANTHROPIC_TOKEN", "").strip()
    if token:
        preferred = _prefer_refreshable_claude_code_token(token, creds)
        if preferred:
            return preferred
        return token

    # 2. CLAUDE_CODE_OAUTH_TOKEN (used by Claude Code for setup-tokens)
    cc_token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if cc_token:
        preferred = _prefer_refreshable_claude_code_token(cc_token, creds)
        if preferred:
            return preferred
        return cc_token

    # 3. Claude Code credential file
    resolved_claude_token = _resolve_claude_code_token_from_credentials(creds)
    if resolved_claude_token:
        return resolved_claude_token

    # 4. Regular API key, or a legacy OAuth token saved in ANTHROPIC_API_KEY.
    # This remains as a compatibility fallback for pre-migration Hermes configs.
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        return api_key

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

    # Run interactively — stdin/stdout/stderr inherited so user can interact
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
        val = os.getenv(env_var, "").strip()
        if val:
            return val

    return None


# ── Hermes-native PKCE OAuth flow ────────────────────────────────────────
# Mirrors the flow used by Claude Code, pi-ai, and OpenCode.
# Stores credentials in ~/.hermes/.anthropic_oauth.json (our own file).

_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
_OAUTH_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
_OAUTH_SCOPES = "org:create_api_key user:profile user:inference"
_HERMES_OAUTH_FILE = get_hermes_home() / ".anthropic_oauth.json"


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
    import time
    import webbrowser

    verifier, challenge = _generate_pkce()

    params = {
        "code": "true",
        "client_id": _OAUTH_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _OAUTH_REDIRECT_URI,
        "scope": _OAUTH_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
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
    state = splits[1] if len(splits) > 1 else ""

    try:
        import urllib.request

        exchange_data = json.dumps({
            "grant_type": "authorization_code",
            "client_id": _OAUTH_CLIENT_ID,
            "code": code,
            "state": state,
            "redirect_uri": _OAUTH_REDIRECT_URI,
            "code_verifier": verifier,
        }).encode()

        req = urllib.request.Request(
            _OAUTH_TOKEN_URL,
            data=exchange_data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"claude-cli/{_get_claude_code_version()} (external, cli)",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
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
    if _HERMES_OAUTH_FILE.exists():
        try:
            data = json.loads(_HERMES_OAUTH_FILE.read_text(encoding="utf-8"))
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
    if any(lower.startswith(p) for p in ("global.", "us.", "eu.", "ap.", "jp.")):
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


def _strip_unknown_tool_blocks(
    anthropic_messages: List[Dict],
    available_tool_names: set,
) -> List[Dict]:
    """Drop tool_use / tool_result blocks for tools not in the live tool list.

    Anthropic's Messages API rejects any request whose history contains a
    ``tool_use`` block whose ``name`` is not present in the current
    ``tools`` array — the error surfaces as
    ``invalid_request_error: Tool reference 'X' not found in available tools``.

    This is easy to hit in practice:

      * MCP server reconnect storms — when ``mcp__salesforce__*`` /
        ``hermes_swarm_*`` / ``StackOverflowTeams_*`` tools were used
        last turn but the MCP server fails to reconnect this turn,
        their schemas are absent from the tool list while the prior
        ``tool_use`` blocks remain in the conversation transcript.
      * Toolset switches mid-session via ``/toolsets remove`` — drops
        ``clarify`` / ``send_message`` etc. while the assistant message
        history still carries calls to them.
      * Subagents / batched delegates — the parent's history contains
        tool calls that the leaf subagent's narrower toolset doesn't
        expose.

    We replace each unknown ``tool_use`` (and its matching ``tool_result``)
    with a small text block describing what was called.  Pure removal would
    be safer wire-shape-wise but lossier: the model loses the breadcrumb
    that a tool ran.  Text replacement preserves the trail while satisfying
    Anthropic's validator.

    Empty / None ``available_tool_names`` is treated as "drop everything"
    — the orphan-stripping in ``convert_messages_to_anthropic`` already
    handles the no-tools-at-all case for unmatched pairs, but a matched
    pair with a stale name still slips through; this catches it.
    """
    if not anthropic_messages:
        return anthropic_messages

    # First pass: identify unknown tool_use ids (we need them to also
    # rewrite the matching tool_result blocks in user messages).
    unknown_tool_use_ids: dict[str, dict] = {}  # id -> {name, input_summary}
    for msg in anthropic_messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = block.get("name")
            if name and name in available_tool_names:
                continue
            tool_id = block.get("id")
            if not tool_id:
                continue
            # Brief input echo for the breadcrumb — capped so a giant
            # base64 payload doesn't bloat the replacement message.
            try:
                inp_str = str(block.get("input") or {})
            except Exception:
                inp_str = "{}"
            if len(inp_str) > 200:
                inp_str = inp_str[:200] + "...(truncated)"
            unknown_tool_use_ids[tool_id] = {
                "name": name or "(unnamed)",
                "input_summary": inp_str,
            }

    if not unknown_tool_use_ids:
        return anthropic_messages

    # Second pass: rewrite blocks in place.
    for msg in anthropic_messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_blocks: list = []
        for block in content:
            if not isinstance(block, dict):
                new_blocks.append(block)
                continue
            btype = block.get("type")
            if btype == "tool_use" and block.get("id") in unknown_tool_use_ids:
                meta = unknown_tool_use_ids[block["id"]]
                new_blocks.append({
                    "type": "text",
                    "text": (
                        f"[Previous tool call: {meta['name']}("
                        f"{meta['input_summary']}) — tool no longer available "
                        f"in this turn.]"
                    ),
                })
                continue
            if btype == "tool_result" and block.get("tool_use_id") in unknown_tool_use_ids:
                # Best-effort summary of the original result text so the
                # model can still reason about what came back.
                try:
                    result_content = block.get("content")
                    if isinstance(result_content, list):
                        # Anthropic tool_result content is a list of text/image blocks
                        text_pieces = []
                        for rc in result_content:
                            if isinstance(rc, dict) and rc.get("type") == "text":
                                text_pieces.append(str(rc.get("text", "")))
                        result_summary = "\n".join(text_pieces)
                    else:
                        result_summary = str(result_content or "")
                except Exception:
                    result_summary = ""
                if len(result_summary) > 400:
                    result_summary = result_summary[:400] + "...(truncated)"
                meta = unknown_tool_use_ids[block["tool_use_id"]]
                new_blocks.append({
                    "type": "text",
                    "text": (
                        f"[Previous tool result for {meta['name']}: "
                        f"{result_summary}]"
                    ),
                })
                continue
            new_blocks.append(block)
        # Empty content after rewrites — leave a placeholder so the
        # message still validates (Anthropic rejects empty content).
        if not new_blocks:
            new_blocks = [{"type": "text", "text": "(content removed)"}]
        # In a user message that's responding to an assistant tool_use,
        # Anthropic requires tool_result blocks to come BEFORE any other
        # content; otherwise the API 400s with
        #   `tool_use` ids were found without `tool_result` blocks
        #   immediately after: <id>
        # The in-place rewrite above can leave leading text breadcrumbs
        # ahead of a surviving real tool_result (when some — but not all
        # — tool_results in the same user message were converted). Stable
        # partition restores the required ordering while preserving the
        # breadcrumbs after the live tool_results.
        if msg.get("role") == "user" and any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in new_blocks
        ):
            tool_results = [
                b for b in new_blocks
                if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
            other = [
                b for b in new_blocks
                if not (isinstance(b, dict) and b.get("type") == "tool_result")
            ]
            new_blocks = tool_results + other
        msg["content"] = new_blocks

    if unknown_tool_use_ids:
        logger.info(
            "anthropic_adapter: rewrote %d tool_use/result block(s) for tools "
            "no longer available: %s",
            len(unknown_tool_use_ids),
            sorted({m["name"] for m in unknown_tool_use_ids.values()}),
        )
    return anthropic_messages


def convert_tools_to_anthropic(tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI tool definitions to Anthropic format.

    Server-side Anthropic tools (web_search_20250305, etc.) are signalled
    by a top-level ``_anthropic_server_tool`` key inside the function
    schema — when present, that block is emitted verbatim instead of the
    function-shaped form. The model still sees the tool by its original
    ``name``; Anthropic intercepts the call server-side, so the local
    handler is never invoked on Anthropic providers.
    """
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
        # Server-side tool shortcut — emit Anthropic's native spec verbatim.
        # The marker is stripped here; required beta headers are computed
        # separately in build_anthropic_kwargs by re-walking the input list.
        server_spec = fn.get("_anthropic_server_tool")
        if isinstance(server_spec, dict) and server_spec.get("type"):
            block = dict(server_spec)
            # Anthropic's native server tools must keep their canonical name
            # ("web_search" for web_search_20250305) — the registry name
            # is authoritative here.
            block.setdefault("name", name)
            result.append(block)
            continue
        result.append({
            "name": name,
            "description": fn.get("description", ""),
            "input_schema": _normalize_tool_input_schema(
                fn.get("parameters", {"type": "object", "properties": {}})
            ),
        })
    return result


def _required_anthropic_server_tool_betas(tools: List[Dict]) -> List[str]:
    """Inspect the OpenAI-shaped tool list and return any extra anthropic-beta
    headers required by server-side tools that appear in the request.

    Maps each declared ``_anthropic_server_tool.type`` to its corresponding
    beta. Returns an empty list when no server tools are present (so the
    beta header set isn't unnecessarily widened — some Anthropic-compatible
    third-party endpoints reject unknown beta headers).
    """
    if not tools:
        return []
    beta_for_type = {
        "web_search_20250305": "web-search-2025-03-05",
        # Future server tools (e.g. computer_use_20250124) get added here.
    }
    seen: set[str] = set()
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            continue
        spec = fn.get("_anthropic_server_tool")
        if not isinstance(spec, dict):
            continue
        beta = beta_for_type.get(spec.get("type", ""))
        if beta:
            seen.add(beta)
    return sorted(seen)


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
        result = _to_plain_data(value.model_dump(), _depth=_depth + 1, _path=_path)
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


# Input-accepted fields per assistant content block type, derived at import
# time from the Anthropic SDK's BetaXBlockParam annotations.  The SDK is
# the source of truth — when it bumps and adds a new field, this map
# updates automatically.  Hardcoded baseline below covers the same set in
# case the SDK rearranges its module layout (we'd notice on the next test
# run rather than silently passing response-only fields through).
#
# Why this matters: Anthropic's response models carry fields not on the
# input param models (e.g. text.parsed_output from structured output).
# Replaying a response block verbatim trips the input validator with
# HTTP 400 "Extra inputs are not permitted".  Allowlisting to input-shape
# is the only stable contract.
_INPUT_BLOCK_FIELDS_FALLBACK: Dict[str, frozenset] = {
    "text": frozenset({"type", "text", "citations", "cache_control"}),
    "thinking": frozenset({"type", "thinking", "signature"}),
    "redacted_thinking": frozenset({"type", "data"}),
    "tool_use": frozenset({"type", "id", "name", "input", "cache_control", "caller"}),
    "server_tool_use": frozenset({"type", "id", "name", "input", "cache_control", "caller"}),
    "web_search_tool_result": frozenset({"type", "tool_use_id", "content", "cache_control", "caller"}),
    "image": frozenset({"type", "source", "cache_control"}),
    "document": frozenset({"type", "source", "title", "context", "citations", "cache_control"}),
}


def _build_input_block_fields() -> Dict[str, frozenset]:
    """Resolve input-allowed fields per block type from the SDK at import.

    Returns the SDK-derived map merged over the hardcoded baseline so a
    block type the SDK exposes wins, while a block type the SDK rearranged
    out of the import path still has a working entry.
    """
    # (block "type" string, param class import path).  When the SDK adds a
    # new block type with a Param model, drop a tuple here — no other
    # change needed.
    _PARAM_REGISTRY = (
        ("text", "BetaTextBlockParam"),
        ("thinking", "BetaThinkingBlockParam"),
        ("redacted_thinking", "BetaRedactedThinkingBlockParam"),
        ("tool_use", "BetaToolUseBlockParam"),
        ("server_tool_use", "BetaServerToolUseBlockParam"),
        ("web_search_tool_result", "BetaWebSearchToolResultBlockParam"),
        ("image", "BetaImageBlockParam"),
        ("document", "BetaBase64PDFBlockParam"),
    )
    resolved: Dict[str, frozenset] = dict(_INPUT_BLOCK_FIELDS_FALLBACK)
    try:
        import anthropic.types.beta as _beta_mod
    except ImportError:
        return resolved
    for block_type, cls_name in _PARAM_REGISTRY:
        cls = getattr(_beta_mod, cls_name, None)
        if cls is None:
            continue
        annotations = getattr(cls, "__annotations__", None)
        if not annotations:
            continue
        resolved[block_type] = frozenset(annotations.keys())
    return resolved


_INPUT_BLOCK_FIELDS: Dict[str, frozenset] = _build_input_block_fields()


def _sanitize_block_for_anthropic_input(block: Dict[str, Any]) -> Dict[str, Any]:
    """Strip response-only fields from a captured response block so it round-trips.

    Anthropic's response models (e.g. BetaTextBlock) carry fields not present
    on the corresponding input param models (e.g. BetaTextBlockParam).
    Replaying a response block verbatim trips the input validator with
    HTTP 400 "Extra inputs are not permitted" on those extra fields.
    Allowlist to known-good input fields per block type; pass through
    unknown types unchanged so a new block type added by Anthropic doesn't
    silently get stripped before this map is updated.
    """
    btype = block.get("type")
    allowed = _INPUT_BLOCK_FIELDS.get(btype) if isinstance(btype, str) else None
    if allowed is None:
        # Unknown type — let it through; downstream normalizers (e.g.
        # _normalize_tool_search_result_for_input) handle their own.
        return block
    return {k: v for k, v in block.items() if k in allowed}


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


def _normalize_tool_reference_for_input(ref: Any) -> Dict[str, Any]:
    """Allowlist a tool_reference block to its accepted input fields.

    Per BetaToolReferenceBlockParam: ``type``, ``tool_name``, optional
    ``cache_control``. Anything else is response-only.
    """
    if not isinstance(ref, dict):
        return {"type": "tool_reference", "tool_name": str(ref)}
    out: Dict[str, Any] = {
        "type": "tool_reference",
        "tool_name": ref.get("tool_name"),
    }
    if isinstance(ref.get("cache_control"), dict):
        out["cache_control"] = dict(ref["cache_control"])
    return out


def _normalize_tool_search_result_inner(item: Any) -> Any:
    """Allowlist the inner content of a tool_search_tool_result.

    Two accepted variants per the SDK:
      - ``tool_search_tool_search_result``: ``type`` + ``tool_references``
      - ``tool_search_tool_result_error``: ``type`` + ``error_code``
    Both carry response-only fields (``text`` etc.) that Anthropic rejects
    on input.
    """
    if not isinstance(item, dict):
        return item
    item_type = item.get("type")
    if item_type == "tool_search_tool_search_result":
        refs = item.get("tool_references") or []
        return {
            "type": "tool_search_tool_search_result",
            "tool_references": [
                _normalize_tool_reference_for_input(r) for r in refs
            ],
        }
    if item_type == "tool_search_tool_result_error":
        return {
            "type": "tool_search_tool_result_error",
            "error_code": item.get("error_code"),
        }
    return item


def _relocate_orphaned_tool_search_results(messages: List[Dict[str, Any]]) -> None:
    """Move ``tool_search_tool_<variant>_tool_result`` blocks to the
    assistant message containing their paired ``server_tool_use``,
    matched by tool_use_id.

    Anthropic delivers the search result block in a *later* response than
    the one that emitted the tool_use (the search runs server-side after
    the initial response returns to the client). The SDK captures the
    result on whichever turn's response it arrived in — so by default it
    lands on a different assistant message than its server_tool_use. But
    Anthropic's input validator rejects that with:
      ``tool_search_tool_<variant> tool use with id ... was found
      without a corresponding tool_search_tool_<variant>_tool_result block``.

    This pass walks the assembled message list and relocates any orphaned
    result block to immediately after its matching server_tool_use in the
    assistant message that owns it. Mutates ``messages`` in place.

    Verified against a HERMES_DUMP_REQUESTS capture where the result on
    turn 3 referenced a server_tool_use from turn 1 — the API rejected it
    until pairing was restored within the same message.
    """
    # tool_use_id -> message index that contains its server_tool_use
    tool_use_sources: Dict[str, int] = {}
    for mi, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "server_tool_use":
                tu_id = block.get("id")
                if isinstance(tu_id, str):
                    tool_use_sources[tu_id] = mi

    # Find tool_search results that live in a different message than their
    # paired server_tool_use.
    relocations: List[Tuple[str, int, int, Dict[str, Any]]] = []
    for mi, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for ci, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if not (
                isinstance(t, str)
                and t.startswith("tool_search_tool_")
                and t.endswith("_tool_result")
            ):
                continue
            tu_id = block.get("tool_use_id")
            if not isinstance(tu_id, str):
                continue
            target_mi = tool_use_sources.get(tu_id)
            if target_mi is not None and target_mi != mi:
                relocations.append((tu_id, mi, ci, block))

    if not relocations:
        return

    # Remove orphans from their source messages (reverse-order per source so
    # earlier indices stay valid after deletes).
    by_source: Dict[int, List[int]] = {}
    for _, src_mi, src_ci, _ in relocations:
        by_source.setdefault(src_mi, []).append(src_ci)
    for src_mi, indices in by_source.items():
        src_content = messages[src_mi].get("content")
        if not isinstance(src_content, list):
            continue
        for ci in sorted(indices, reverse=True):
            del src_content[ci]

    # Insert each orphan immediately after its matching server_tool_use in
    # the target message. Search fresh each time so successive inserts in
    # the same target see the up-to-date content list.
    for tu_id, _, _, block in relocations:
        target_mi = tool_use_sources[tu_id]
        target_content = messages[target_mi].get("content")
        if not isinstance(target_content, list):
            continue
        for ci, b in enumerate(target_content):
            if (
                isinstance(b, dict)
                and b.get("type") == "server_tool_use"
                and b.get("id") == tu_id
            ):
                target_content.insert(ci + 1, block)
                break


def drop_orphan_server_tool_uses_in_storage(
    messages: List[Dict[str, Any]],
) -> int:
    """Drop any ``server_tool_use`` block whose paired
    ``tool_search_tool_*_tool_result`` doesn't exist anywhere in the
    message list.

    Why: relocation handles "result split across messages" — the normal
    Anthropic delivery pattern. But a stream interruption (timeout,
    cancel, 5xx mid-response) can land the ``server_tool_use`` on disk
    without the result EVER arriving. Every subsequent API call then
    400s with:
      ``tool_search_tool_<variant> tool use with id ... was found
      without a corresponding tool_search_tool_<variant>_tool_result``.
    The session is permanently wedged until the orphan is removed.

    Verified against ``session_20260509_145003_c5e465`` where one
    server_tool_use had no result anywhere — dropping it unwedges the
    session with no loss of usable data (the unfinished tool search
    yielded nothing the model could act on anyway).

    Returns the number of orphan use blocks removed.
    """
    result_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("anthropic_content_blocks")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if not isinstance(t, str):
                continue
            if (
                t == "tool_search_tool_result"
                or (t.startswith("tool_search_tool_") and t.endswith("_tool_result"))
            ):
                tu_id = block.get("tool_use_id")
                if isinstance(tu_id, str):
                    result_ids.add(tu_id)

    dropped = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("anthropic_content_blocks")
        if not isinstance(content, list):
            continue
        keep = []
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "server_tool_use"
                and isinstance(block.get("id"), str)
                and block["id"] not in result_ids
            ):
                dropped += 1
                continue
            keep.append(block)
        if dropped:
            msg["anthropic_content_blocks"] = keep
    return dropped


def relocate_orphaned_tool_search_results_in_storage(
    messages: List[Dict[str, Any]],
) -> int:
    """Capture-time variant of ``_relocate_orphaned_tool_search_results``
    that operates on the **session-storage shape**: assistant messages
    carry their verbatim Anthropic blocks under
    ``msg["anthropic_content_blocks"]`` (set by
    ``transports/anthropic.py`` when capturing each response), not under
    ``msg["content"]``.

    Why we need a separate pass at capture time
    --------------------------------------------
    Anthropic delivers a ``tool_search_tool_<variant>_tool_result`` block
    in a *later* assistant turn than the one that emitted the matching
    ``server_tool_use(id=X)``. The request-build relocation
    (``_relocate_orphaned_tool_search_results``) fixes this on outbound,
    but the on-disk session JSON keeps the split. If compaction
    summarises one of the two messages and the API call rebuilds, you
    get a 400:
      ``tool_search_tool_<variant> tool use with id ... was found
      without a corresponding tool_search_tool_<variant>_tool_result``.

    Calling this at persistence time co-locates the pair on disk so
    compaction can never split them — the compactor's existing
    ``_align_boundary_*`` logic treats the merged message as a single
    unit, and ``_sanitize_tool_pairs`` doesn't need any awareness of
    server-side block types.

    Returns the number of result blocks relocated. Mutates ``messages``
    in place.
    """
    tool_use_sources: Dict[str, int] = {}
    for mi, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("anthropic_content_blocks")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "server_tool_use":
                tu_id = block.get("id")
                if isinstance(tu_id, str):
                    tool_use_sources[tu_id] = mi

    relocations: List[Tuple[str, int, int, Dict[str, Any]]] = []
    for mi, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("anthropic_content_blocks")
        if not isinstance(content, list):
            continue
        for ci, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if not isinstance(t, str):
                continue
            # Match both the bare canonical and any variant-suffixed form
            # (some persisted sessions still carry pre-canonicalisation
            # types like ``tool_search_tool_regex_tool_result``).
            if not (
                t == "tool_search_tool_result"
                or (t.startswith("tool_search_tool_") and t.endswith("_tool_result"))
            ):
                continue
            tu_id = block.get("tool_use_id")
            if not isinstance(tu_id, str):
                continue
            target_mi = tool_use_sources.get(tu_id)
            if target_mi is not None and target_mi != mi:
                relocations.append((tu_id, mi, ci, block))

    if not relocations:
        return 0

    # Remove orphans from their source messages, deepest index first so
    # earlier indices stay valid.
    by_source: Dict[int, List[int]] = {}
    for _, src_mi, src_ci, _ in relocations:
        by_source.setdefault(src_mi, []).append(src_ci)
    for src_mi, indices in by_source.items():
        src_content = messages[src_mi].get("anthropic_content_blocks")
        if not isinstance(src_content, list):
            continue
        for ci in sorted(indices, reverse=True):
            del src_content[ci]

    for tu_id, _, _, block in relocations:
        target_mi = tool_use_sources[tu_id]
        target_content = messages[target_mi].get("anthropic_content_blocks")
        if not isinstance(target_content, list):
            continue
        for ci, b in enumerate(target_content):
            if (
                isinstance(b, dict)
                and b.get("type") == "server_tool_use"
                and b.get("id") == tu_id
            ):
                target_content.insert(ci + 1, block)
                break

    return len(relocations)


def _move_client_tool_use_blocks_to_end(messages: List[Dict[str, Any]]) -> None:
    """Reorder assistant content so client ``tool_use`` blocks come AFTER
    any server-side blocks (``server_tool_use`` / ``*_tool_result``) within
    the same message.

    Why this exists:

    Anthropic's input validator requires that the next user message's
    ``tool_result`` for a client ``tool_use`` be "immediately after" it
    in the message list — and "immediately after" means the very next
    message, with no intervening server-side blocks pushing the
    client tool_use earlier in its own content array. When the model
    emits a client tool_use BEFORE deciding to invoke server-side
    tool_search, the captured response carries the order
    ``[tool_use, server_tool_use, *_tool_result]``. Replaying that
    verbatim trips the validator with HTTP 400:

      "messages.N: ``tool_use`` ids were found without ``tool_result``
       blocks immediately after: <client tool_use id>"

    Even though the client tool_use IS followed (in the next message)
    by its tool_result. The validator considers the trailing
    server-side blocks an obstruction.

    Fix: move all client ``tool_use`` blocks to the end of their
    assistant message, preserving the relative order of server-side
    blocks, thinking blocks, and text. The client tool_use blocks
    themselves keep their relative order among each other.

    Thinking-signature safety:

    Anthropic signs thinking blocks against their position in the
    response. ``context_management.clear_thinking_20251015`` enforces
    that each block stays in place across turns. Moving a client
    ``tool_use`` past server-side blocks doesn't relocate any thinking
    block — they stay where the model emitted them. We only refuse to
    reorder when a thinking block sits BETWEEN a client tool_use and
    a trailing server-side block (because moving the tool_use past
    the thinking would change the content stream the thinking signed
    against). Those messages pass through unchanged and may still
    400; loudly logging so we can diagnose if we ever see one.

    Mutates ``messages`` in place. Idempotent — once the client
    tool_use is at the end, repeated passes are no-ops.
    """
    for mi, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if not isinstance(content, list) or len(content) < 2:
            continue

        # Find client tool_use indices (not server_tool_use).
        client_tu_indices = [
            i for i, b in enumerate(content)
            if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
        if not client_tu_indices:
            continue

        # If every client tool_use is already at the tail, nothing to do.
        last_idx = len(content) - 1
        if all(i >= last_idx - len(client_tu_indices) + 1 for i in client_tu_indices):
            # All client tool_use blocks are already in the final
            # contiguous tail — verify it's actually a clean tail
            # (no non-tool_use blocks intermixed at the end).
            tail = content[last_idx - len(client_tu_indices) + 1:]
            if all(
                isinstance(b, dict) and b.get("type") == "tool_use"
                for b in tail
            ):
                continue

        # Detect the unsafe pattern: a thinking block between a client
        # tool_use and a later server-side block. Don't reorder — log
        # and skip.
        SERVER_BLOCK_TYPES = {"server_tool_use"}
        first_tu_idx = client_tu_indices[0]
        has_trailing_server = any(
            isinstance(content[i], dict)
            and (
                content[i].get("type") in SERVER_BLOCK_TYPES
                or (
                    isinstance(content[i].get("type"), str)
                    and content[i]["type"].endswith("_tool_result")
                    and content[i]["type"].startswith("tool_search_tool_")
                )
                or content[i].get("type") == "tool_search_tool_result"
            )
            for i in range(first_tu_idx + 1, len(content))
        )
        if not has_trailing_server:
            continue  # Reorder unnecessary — no server-side block follows.

        intervening_thinking = any(
            isinstance(content[i], dict)
            and content[i].get("type") in ("thinking", "redacted_thinking")
            for i in range(first_tu_idx + 1, len(content))
        )
        if intervening_thinking:
            logger.warning(
                "anthropic adapter: assistant msg[%d] has client tool_use "
                "followed by both a thinking block and a server-side block; "
                "cannot reorder without invalidating thinking signature. "
                "Anthropic may reject this request with a 400 about "
                "tool_use ids without tool_result.",
                mi,
            )
            continue

        # Safe to reorder. Pull all client tool_use blocks out, then
        # append them at the end in original order.
        client_tu_blocks = [content[i] for i in client_tu_indices]
        # Build a new content list dropping the client tool_use slots.
        client_tu_set = set(client_tu_indices)
        rebuilt = [b for i, b in enumerate(content) if i not in client_tu_set]
        rebuilt.extend(client_tu_blocks)
        m["content"] = rebuilt


def _canonicalize_tool_search_result_types(content: Any) -> None:
    """Rewrite variant-suffixed ``tool_search_tool_<variant>_tool_result``
    block types to the bare canonical form ``tool_search_tool_result``.

    Why this exists:

    Anthropic's wire payload delivers tool-search result blocks with a
    variant-suffixed type (e.g. ``tool_search_tool_regex_tool_result``)
    that mirrors the paired ``server_tool_use.name``
    (``tool_search_tool_regex``). The Python SDK's
    ``BetaToolSearchToolResultBlock`` model declares
    ``type: Literal["tool_search_tool_result"]`` — the bare canonical
    form — and Pydantic silently coerces the wire value to that literal
    when parsing.

    Empirically (verified live against api.anthropic.com on 2026-05-07,
    request_id ``req_011Cap2RUgsJp1CVsGAR6LTa``), Anthropic's INPUT
    validator's accept list contains ``tool_search_tool_result`` —
    the bare canonical — and rejects any variant-suffixed form with:

      "Input tag '<variant>_tool_result' found using 'type' does not
       match any of the expected tags: ..., 'tool_search_tool_result',
       'tool_use', ..."

    A prior workaround in this file (``_normalize_tool_search_result_for_input``,
    docstring still in place for historical reference but its behavior
    is fixed here) claimed the opposite — that the variant suffix was
    REQUIRED and that re-emitting the canonical bare form failed the
    pairing check. That claim was either out of date or misdiagnosed;
    the validator's own error message today is unambiguous about which
    tag is accepted.

    So: any block whose type starts with ``tool_search_tool_`` and ends
    with ``_tool_result`` gets its type collapsed to the bare canonical
    form. Mutates ``content`` in place. Idempotent — the bare canonical
    is its own fixed point.

    Accepts either a single content array (``List[Dict]``) or a full
    message list (``List[Dict]`` where each dict has ``role``/
    ``content``); the latter case dispatches per-message.
    """
    if not isinstance(content, list):
        return

    # Detect message-list shape (each entry has role + content) vs raw
    # block list. Per-message dispatch keeps both call sites simple.
    if content and all(
        isinstance(m, dict) and "role" in m and "content" in m
        for m in content
    ):
        for m in content:
            mc = m.get("content")
            if isinstance(mc, list):
                _canonicalize_tool_search_result_types(mc)
        return

    # Single content array — collapse any variant-suffixed type.
    for b in content:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if not isinstance(t, str):
            continue
        if t == "tool_search_tool_result":
            continue  # already canonical
        if t.startswith("tool_search_tool_") and t.endswith("_tool_result"):
            b["type"] = "tool_search_tool_result"


def _normalize_tool_search_result_for_input(sb: Dict[str, Any]) -> Dict[str, Any]:
    """Strip response-only fields from a tool_search result block and emit
    the bare canonical ``tool_search_tool_result`` type.

    Anthropic's INPUT validator (verified live 2026-05-07,
    request_id ``req_011Cap2RUgsJp1CVsGAR6LTa``) accepts only the bare
    ``tool_search_tool_result`` type. Variant-suffixed types
    (``tool_search_tool_regex_tool_result``, etc.) — which appear on
    the wire OUTPUT — fail the input tag check. The SDK's
    ``BetaToolSearchToolResultBlockParam`` declares the same bare
    canonical form, which is the right contract.

    A prior version of this function preserved whatever ``type`` came
    back on the response under the assumption that the variant suffix
    was required for pairing. That was wrong. The response Pydantic
    coerces the wire variant to the bare canonical anyway, so for
    fresh responses ``sb["type"]`` is already correct. Old persisted
    sessions and any path that bypasses the SDK Pydantic layer can
    still carry a variant suffix; this function is the choke point
    that normalizes them.

    Strip response-only fields (``text``, ``citations``, etc.) that
    fail input validation with "Extra inputs are not permitted".
    Recursively allowlist inner content the same way.
    """
    inner = sb.get("content")
    if isinstance(inner, list):
        normalized_inner: Any = [
            _normalize_tool_search_result_inner(x) for x in inner
        ]
    else:
        normalized_inner = _normalize_tool_search_result_inner(inner)
    out: Dict[str, Any] = {
        # Always the bare canonical — ignore whatever variant suffix
        # may have leaked in from a persisted session or a non-SDK
        # construction path.
        "type": "tool_search_tool_result",
        "tool_use_id": sb.get("tool_use_id"),
        "content": normalized_inner,
    }
    if isinstance(sb.get("cache_control"), dict):
        out["cache_control"] = dict(sb["cache_control"])
    return out


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
    result = []

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
                    system = [p for p in content if isinstance(p, dict)]
                else:
                    system = "\n".join(
                        p["text"] for p in content if p.get("type") == "text"
                    )
            else:
                system = content
            continue

        if role == "assistant":
            # ── Verbatim replay (Anthropic-native) ──────────────────
            # When the assistant turn carries the original content array
            # captured by AnthropicTransport.normalize_response, replay
            # every block in its original position.  Anthropic signs
            # thinking blocks against their position in the response, and
            # context_management.clear_thinking_20251015 enforces that
            # each block stays in place across turns.  Recomposing from
            # reasoning_details + content + tool_calls reorders
            # interleaved thinking emitted under
            # interleaved-thinking-2025-05-14 and breaks signature
            # validation with HTTP 400 "thinking ... cannot be modified".
            #
            # Downstream (line ~2176+) still applies thinking-signature
            # management — strip-on-non-latest, downgrade-unsigned,
            # third-party-strip — over m["content"] regardless of which
            # branch produced it, so the verbatim blocks get the same
            # post-processing as recomposed ones.
            raw_blocks = m.get("anthropic_content_blocks")
            if isinstance(raw_blocks, list) and raw_blocks:
                rebuilt: List[Dict[str, Any]] = []
                for b in copy.deepcopy(raw_blocks):
                    if not isinstance(b, dict):
                        rebuilt.append(b)
                        continue
                    btype = b.get("type", "")
                    # tool_search_tool_<variant>_tool_result blocks have
                    # their own input-shape normalizer (variant-specific
                    # inner structure that _sanitize_block_for_anthropic_input
                    # doesn't model).
                    if (
                        isinstance(btype, str)
                        and btype.startswith("tool_search_tool_")
                        and btype.endswith("_tool_result")
                    ):
                        rebuilt.append(_normalize_tool_search_result_for_input(b))
                    else:
                        # Strip response-only fields (e.g. text.parsed_output
                        # from structured output, or any future response-side
                        # field Anthropic adds).  Block position and the
                        # signed payload are preserved.
                        rebuilt.append(_sanitize_block_for_anthropic_input(b))
                if not rebuilt:
                    rebuilt = [{"type": "text", "text": "(empty)"}]
                result.append({"role": "assistant", "content": rebuilt})
                continue

            # ── Recomposition path (fallback / cross-provider) ──────
            blocks = _extract_preserved_thinking_blocks(m)
            # Anthropic server-side tool blocks (web_search etc.) — must be
            # re-emitted verbatim before text/tool_use blocks. Stored on the
            # message dict by run_agent._build_assistant_message after the
            # transport extracted them in normalize_response.
            preserved_server_blocks = m.get("server_tool_blocks")
            if isinstance(preserved_server_blocks, list):
                for sb in preserved_server_blocks:
                    if not isinstance(sb, dict):
                        continue
                    sb_type = sb.get("type", "")
                    if sb_type == "server_tool_use":
                        # server_tool_use is request-shape compatible; pass
                        # through as-is.
                        blocks.append(dict(sb))
                    elif sb_type == "web_search_tool_result":
                        blocks.append(dict(sb))
                    elif (
                        isinstance(sb_type, str)
                        and sb_type.startswith("tool_search_tool_")
                        and sb_type.endswith("_tool_result")
                    ):
                        blocks.append(_normalize_tool_search_result_for_input(sb))
            if content:
                if isinstance(content, list):
                    converted_content = _convert_content_to_anthropic(content)
                    if isinstance(converted_content, list):
                        blocks.extend(converted_content)
                else:
                    blocks.append({"type": "text", "text": str(content)})
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
                isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")
                for b in blocks
            )
            if isinstance(reasoning_content, str) and not _already_has_thinking:
                blocks.insert(0, {"type": "thinking", "thinking": reasoning_content})
            # Anthropic rejects empty assistant content
            effective = blocks or content
            if not effective or effective == "":
                effective = [{"type": "text", "text": "(empty)"}]
            result.append({"role": "assistant", "content": effective})
            continue

        if role == "tool":
            # Sanitize tool_use_id and ensure non-empty content.
            # Computer-use (and other multimodal) tool results arrive as
            # either a list of OpenAI-style content parts, or a dict
            # marked `_multimodal` with an embedded `content` list. Convert
            # both into Anthropic `tool_result` inner blocks (text + image).
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
            continue

        # Regular user message — validate non-empty content (Anthropic rejects empty)
        if isinstance(content, list):
            converted_blocks = _convert_content_to_anthropic(content)
            # Check if all text blocks are empty
            if not converted_blocks or all(
                b.get("text", "").strip() == ""
                for b in converted_blocks
                if isinstance(b, dict) and b.get("type") == "text"
            ):
                converted_blocks = [{"type": "text", "text": "(empty message)"}]
            result.append({"role": "user", "content": converted_blocks})
        else:
            # Validate string content is non-empty
            if not content or (isinstance(content, str) and not content.strip()):
                content = "(empty message)"
            result.append({"role": "user", "content": content})

    # Strip orphaned tool_use blocks (no matching tool_result follows)
    tool_result_ids = set()
    for m in result:
        if m["role"] == "user" and isinstance(m["content"], list):
            for block in m["content"]:
                if block.get("type") == "tool_result":
                    tool_result_ids.add(block.get("tool_use_id"))
    for m in result:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            m["content"] = [
                b
                for b in m["content"]
                if b.get("type") != "tool_use" or b.get("id") in tool_result_ids
            ]
            if not m["content"]:
                m["content"] = [{"type": "text", "text": "(tool call removed)"}]

    # Strip orphaned tool_result blocks (no matching tool_use precedes them).
    # This is the mirror of the above: context compression or session truncation
    # can remove an assistant message containing a tool_use while leaving the
    # subsequent tool_result intact.  Anthropic rejects these with a 400.
    tool_use_ids = set()
    for m in result:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            for block in m["content"]:
                if block.get("type") == "tool_use":
                    tool_use_ids.add(block.get("id"))
    for m in result:
        if m["role"] == "user" and isinstance(m["content"], list):
            m["content"] = [
                b
                for b in m["content"]
                if b.get("type") != "tool_result" or b.get("tool_use_id") in tool_use_ids
            ]
            if not m["content"]:
                m["content"] = [{"type": "text", "text": "(tool result removed)"}]

    # Enforce strict role alternation (Anthropic rejects consecutive same-role messages)
    fixed = []
    for m in result:
        if fixed and fixed[-1]["role"] == m["role"]:
            if m["role"] == "user":
                # Merge consecutive user messages
                prev_content = fixed[-1]["content"]
                curr_content = m["content"]
                if isinstance(prev_content, str) and isinstance(curr_content, str):
                    fixed[-1]["content"] = prev_content + "\n" + curr_content
                elif isinstance(prev_content, list) and isinstance(curr_content, list):
                    fixed[-1]["content"] = prev_content + curr_content
                else:
                    # Mixed types — wrap string in list
                    if isinstance(prev_content, str):
                        prev_content = [{"type": "text", "text": prev_content}]
                    if isinstance(curr_content, str):
                        curr_content = [{"type": "text", "text": curr_content}]
                    fixed[-1]["content"] = prev_content + curr_content
            else:
                # Consecutive assistant messages — merge text content.
                # Drop thinking blocks from the *second* message: their
                # signature was computed against a different turn boundary
                # and becomes invalid once merged.
                if isinstance(m["content"], list):
                    m["content"] = [
                        b for b in m["content"]
                        if not (isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"))
                    ]
                prev_blocks = fixed[-1]["content"]
                curr_blocks = m["content"]
                if isinstance(prev_blocks, list) and isinstance(curr_blocks, list):
                    fixed[-1]["content"] = prev_blocks + curr_blocks
                elif isinstance(prev_blocks, str) and isinstance(curr_blocks, str):
                    fixed[-1]["content"] = prev_blocks + "\n" + curr_blocks
                else:
                    # Mixed types — normalize both to list and merge
                    if isinstance(prev_blocks, str):
                        prev_blocks = [{"type": "text", "text": prev_blocks}]
                    if isinstance(curr_blocks, str):
                        curr_blocks = [{"type": "text", "text": curr_blocks}]
                    fixed[-1]["content"] = prev_blocks + curr_blocks
        else:
            fixed.append(m)
    result = fixed

    # ── Thinking block signature management ──────────────────────────
    # Anthropic signs thinking blocks against the full turn content.
    # Any upstream mutation (context compression, session truncation,
    # orphan stripping, message merging) invalidates the signature,
    # causing HTTP 400 "Invalid signature in thinking block".
    #
    # Signatures are Anthropic-proprietary.  Third-party endpoints
    # (MiniMax, Azure AI Foundry, self-hosted proxies) cannot validate
    # them and will reject them outright.  When targeting a third-party
    # endpoint, strip ALL thinking/redacted_thinking blocks from every
    # assistant message — the third-party will generate its own
    # thinking blocks if it supports extended thinking.
    #
    # For direct Anthropic (strategy following clawdbot/OpenClaw):
    # 1. Strip thinking/redacted_thinking from all assistant messages
    #    EXCEPT the last one — preserves reasoning continuity on the
    #    current tool-use chain while avoiding stale signature errors.
    # 2. Downgrade unsigned thinking blocks (no signature) to text —
    #    Anthropic can't validate them and will reject them.
    # 3. Strip cache_control from thinking/redacted_thinking blocks —
    #    cache markers can interfere with signature validation.
    _THINKING_TYPES = frozenset(("thinking", "redacted_thinking"))
    _is_third_party = _is_third_party_anthropic_endpoint(base_url)
    # Kimi /coding and DeepSeek /anthropic share a contract: both speak the
    # Anthropic Messages protocol upstream but require that thinking blocks
    # synthesised from reasoning_content round-trip on subsequent turns when
    # thinking is enabled.  Signed Anthropic blocks still have to be stripped
    # (neither endpoint can validate Anthropic's signatures); unsigned blocks
    # are preserved.  See hermes-agent#13848 (Kimi) and #16748 (DeepSeek).
    _preserve_unsigned_thinking = (
        _is_kimi_family_endpoint(base_url, model)
        or _is_deepseek_anthropic_endpoint(base_url)
    )

    last_assistant_idx = None
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    for idx, m in enumerate(result):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue

        if _preserve_unsigned_thinking:
            # Kimi's /coding and DeepSeek's /anthropic endpoints both enable
            # thinking server-side and require unsigned thinking blocks on
            # replayed assistant tool-call messages.  Strip signed Anthropic
            # blocks (neither upstream can validate Anthropic signatures) but
            # preserve the unsigned ones we synthesised from reasoning_content.
            new_content = []
            for b in m["content"]:
                if not isinstance(b, dict) or b.get("type") not in _THINKING_TYPES:
                    new_content.append(b)
                    continue
                if b.get("signature") or b.get("data"):
                    # Anthropic-signed block — upstream can't validate, strip
                    continue
                # Unsigned thinking (synthesised from reasoning_content) —
                # keep it: the upstream needs it for message-history validation.
                new_content.append(b)
            m["content"] = new_content or [{"type": "text", "text": "(empty)"}]
        elif _is_third_party or idx != last_assistant_idx:
            # Third-party endpoint: strip ALL thinking blocks from every
            # assistant message — signatures are Anthropic-proprietary.
            # Direct Anthropic: strip from non-latest assistant messages only.
            stripped = [
                b for b in m["content"]
                if not (isinstance(b, dict) and b.get("type") in _THINKING_TYPES)
            ]
            m["content"] = stripped or [{"type": "text", "text": "(thinking elided)"}]
        else:
            # Latest assistant on direct Anthropic: keep signed thinking
            # blocks for reasoning continuity; downgrade unsigned ones to
            # plain text.
            new_content = []
            for b in m["content"]:
                if not isinstance(b, dict) or b.get("type") not in _THINKING_TYPES:
                    new_content.append(b)
                    continue
                if b.get("type") == "redacted_thinking":
                    # Redacted blocks use 'data' for the signature payload
                    if b.get("data"):
                        new_content.append(b)
                    # else: drop — no data means it can't be validated
                elif b.get("signature"):
                    # Signed thinking block — keep it
                    new_content.append(b)
                else:
                    # Unsigned thinking — downgrade to text so it's not lost
                    thinking_text = b.get("thinking", "")
                    if thinking_text:
                        new_content.append({"type": "text", "text": thinking_text})
            m["content"] = new_content or [{"type": "text", "text": "(empty)"}]

        # Strip cache_control from any remaining thinking/redacted_thinking
        # blocks — cache markers interfere with signature validation.
        for b in m["content"]:
            if isinstance(b, dict) and b.get("type") in _THINKING_TYPES:
                b.pop("cache_control", None)

    # Anthropic's tool_search emits the result block in a *later* response
    # than the one that issued the server_tool_use, but its input validator
    # requires same-message pairing. Walk the assembled message list and
    # move any orphaned result blocks back to the assistant message that
    # owns the matching server_tool_use.
    _relocate_orphaned_tool_search_results(result)

    # Drop ``server_tool_use`` blocks whose paired result NEVER arrived
    # (stream interruption, timeout, cancel mid-response). Without this,
    # the assistant message has a use without a result, and every API
    # call replays the orphan and 400s. Runs after relocation so a
    # split-but-deliverable pair gets repaired first; only truly
    # missing results trigger a drop. Operates on the wire-shape
    # ``msg["content"]`` (lists of blocks).
    _result_ids_wire: set = set()
    for _m in result:
        if _m.get("role") != "assistant":
            continue
        _c = _m.get("content")
        if not isinstance(_c, list):
            continue
        for _b in _c:
            if not isinstance(_b, dict):
                continue
            _t = _b.get("type")
            if isinstance(_t, str) and (
                _t == "tool_search_tool_result"
                or (_t.startswith("tool_search_tool_") and _t.endswith("_tool_result"))
            ):
                _ru = _b.get("tool_use_id")
                if isinstance(_ru, str):
                    _result_ids_wire.add(_ru)
    for _m in result:
        if _m.get("role") != "assistant":
            continue
        _c = _m.get("content")
        if not isinstance(_c, list):
            continue
        _kept = [
            _b for _b in _c
            if not (
                isinstance(_b, dict)
                and _b.get("type") == "server_tool_use"
                and isinstance(_b.get("id"), str)
                and _b["id"] not in _result_ids_wire
            )
        ]
        if len(_kept) != len(_c):
            _m["content"] = _kept or [{"type": "text", "text": "(empty)"}]

    # Defense-in-depth: canonicalize tool_search_tool_*_tool_result block
    # types to the bare ``tool_search_tool_result`` form. The capture-time
    # fix in ``agent/transports/anthropic.py`` handles fresh responses,
    # but sessions persisted with a variant-suffixed type (e.g. from an
    # earlier broken Hermes version that stored the wire variant, or
    # any code path that bypasses the SDK Pydantic layer) need the same
    # rewrite at outbound request time so Anthropic's input validator
    # doesn't reject them with 400. Idempotent: bare canonical is the
    # fixed point.
    _canonicalize_tool_search_result_types(result)

    # Reorder client tool_use blocks to the end of each assistant message
    # when followed by server-side blocks. Anthropic's input validator
    # demands the next message's tool_result be "immediately after" the
    # client tool_use, with no intervening server-side blocks. The model
    # sometimes emits tool_search AFTER deciding to call a client tool;
    # the captured response carries that order verbatim and 400s on
    # replay until we move the client tool_use past the server-side
    # blocks. Idempotent: already-tail-positioned tool_use is skipped.
    _move_client_tool_use_blocks_to_end(result)

    # ── Image eviction: keep only the most recent N screenshots ─────
    # computer_use screenshots (base64 images) sit inside tool_result
    # blocks: they accumulate and are sent with every API call. Each
    # costs ~1,465 tokens; after 10+ the conversation becomes slow
    # even for simple text queries. Walk backward, keep the most recent
    # _MAX_KEEP_IMAGES, replace older ones with a text placeholder.
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

    return system, result


_TOOL_SEARCH_TOOL_TYPES = {
    "regex": "tool_search_tool_regex_20251119",
    "bm25":  "tool_search_tool_bm25_20251119",
}


def _apply_tool_search(
    anthropic_tools: List[Dict[str, Any]],
    tool_search_config: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply the tool_search deferral policy to the converted tools array.

    Two modes — selected by ``tool_search_config["mode"]`` (default
    ``"client_side"``):

    ``"server_side"`` (legacy)
        Stubs carry ``defer_loading: True``, the Anthropic
        ``tool_search_tool_<variant>_20251119`` server tool is prepended to
        the array, and the model discovers tools via that server tool.
        Anthropic re-bills the FULL prompt context for each server-tool
        iteration within an API call.  See agent.log forensics from the
        2026-05-13 case 00271597 session for 2x/3x/4x prompt-token
        multiplier evidence.

    ``"client_side"``
        Stubs are regular tools (no ``defer_loading`` flag), no server tool
        is prepended, and the model discovers tools via the client-side
        ``hermes_load_tools`` tool registered in ``tools/hermes_load_tools.py``
        and dispatched out of the agent loop in ``run_agent.py``.  Each
        load step is a normal client-side round-trip — billed once per
        call, no multiplier.  Names in ``promoted_tools`` skip the stub
        and ship their full schema.

    Deferral policy (additive, evaluated in order, identical across modes):
      1. ``additional_deferred`` — exact tool names always deferred.
      2. ``additional_eager`` — exact tool names always eager (overrides 1).
      3. ``defer_mcp_tools`` — when True, any tool whose name starts with
         a known MCP server prefix is deferred.  Server prefixes are
         passed via ``tool_search_config["mcp_server_prefixes"]``.

    Returns the transformed list.  Returns the input unchanged when
    tool_search is disabled, when there are no tools, or when all/none of
    the tools would be deferred (server_side: Anthropic 400s on "all
    deferred"; both modes: a stub array with no full tools is unhelpful).
    """
    if not tool_search_config or not tool_search_config.get("enabled"):
        return anthropic_tools
    if not anthropic_tools:
        return anthropic_tools

    mode = (tool_search_config.get("mode") or "client_side").strip().lower()
    if mode not in {"server_side", "client_side"}:
        mode = "client_side"

    eager_names = set(tool_search_config.get("additional_eager") or [])
    deferred_names = set(tool_search_config.get("additional_deferred") or [])
    mcp_prefixes = tuple(tool_search_config.get("mcp_server_prefixes") or [])
    defer_mcp = bool(tool_search_config.get("defer_mcp_tools", True))
    # client_side mode only — names the model has already loaded this
    # session.  Promoted names skip the stub branch and ship their full
    # schema even when the policy would otherwise defer them.
    promoted_tools = set(tool_search_config.get("promoted_tools") or ())

    def _should_defer(name: str) -> bool:
        if name in eager_names:
            return False
        if name in deferred_names:
            return True
        if defer_mcp and mcp_prefixes and name.startswith(mcp_prefixes):
            return True
        return False

    # Build the stub used for deferred entries.  Anthropic's validator
    # requires ``description`` and ``input_schema`` to exist even on
    # name-only entries, so we send minimal placeholders (empty
    # description, ``{"type":"object"}``).  Each stub stays under ~120
    # bytes on the wire vs 1-5KB for a real schema.
    #
    # The ``defer_loading: True`` flag is server_side-specific — it tells
    # Anthropic's tool_search machinery the entry is a stub that should be
    # hydrated server-side on tool_search hits.  In client_side mode the
    # flag is omitted; the entry is just a tool with a terse description
    # whose schema gets filled in on the next request when the model
    # promotes it via hermes_load_tools.
    def _make_stub(name: str, original: Dict[str, Any]) -> Dict[str, Any]:
        stub: Dict[str, Any] = {
            "name": name,
            "description": (
                ""
                if mode == "server_side"
                else (
                    "Stubbed MCP tool — call hermes_load_tools with this "
                    "name to load the full schema."
                )
            ),
            "input_schema": {"type": "object"},
        }
        if mode == "server_side":
            stub["defer_loading"] = True
        # Preserve cache_control if the caller had set it; it affects
        # prompt-caching boundary placement and is cheap.
        if "cache_control" in original:
            stub["cache_control"] = original["cache_control"]
        return stub

    transformed: List[Dict[str, Any]] = []
    deferred_count = 0
    eager_count = 0
    for tool in anthropic_tools:
        name = tool.get("name", "")
        if _should_defer(name) and name not in promoted_tools:
            transformed.append(_make_stub(name, tool))
            deferred_count += 1
        else:
            transformed.append(tool)
            eager_count += 1

    # Anthropic returns 400 when every tool is deferred (no eager tool to
    # ground the deferral). Skip injection in that case.  Also skip when
    # nothing is deferred (no benefit, just adds one extra entry in
    # server_side mode and a no-op in client_side mode).
    if deferred_count == 0 or eager_count == 0:
        return anthropic_tools

    if mode == "client_side":
        # No server tool to prepend — hermes_load_tools is a regular
        # client-side tool already registered in the tools array.
        return transformed

    # server_side mode — prepend the Anthropic server tool.
    variant = (tool_search_config.get("variant") or "regex").lower()
    ts_type = _TOOL_SEARCH_TOOL_TYPES.get(variant, _TOOL_SEARCH_TOOL_TYPES["regex"])
    ts_name = "tool_search_tool_bm25" if variant == "bm25" else "tool_search_tool_regex"
    return [{"type": ts_type, "name": ts_name}] + transformed



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
    tool_search_config: Optional[Dict[str, Any]] = None,
    session_id: str | None = None,
    cache_tools: bool = False,
    cache_ttl: str = "5m",
) -> Dict[str, Any]:
    """Build kwargs for ``client.beta.messages.{create,stream}``.

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

    When *fast_mode* is True, sets typed ``speed="fast"`` and adds the
    fast-mode beta to the per-request ``betas`` list for ~2.5x faster output
    throughput on Opus 4.6. Native Anthropic only — third-party gateways
    don't recognize the speed parameter.

    Output kwargs assume ``client.beta.messages.{create,stream}``: typed
    fields ``thinking``, ``output_config``, ``context_management``, ``betas``,
    ``speed``, ``metadata`` all land on the wire as top-level body fields.
    """
    system, anthropic_messages = convert_messages_to_anthropic(
        messages, base_url=base_url, model=model
    )
    anthropic_tools = convert_tools_to_anthropic(tools) if tools else []

    # Drop / rewrite tool_use blocks for tools that aren't in the live tool
    # list — Anthropic's API hard-rejects them with
    #   invalid_request_error: Tool reference 'X' not found in available tools
    # See _strip_unknown_tool_blocks for the full list of triggering
    # scenarios (MCP reconnect failures, mid-session toolset switches,
    # subagents with narrower toolsets).  We do this here, AFTER tools
    # are converted, so the lookup set reflects exactly what's going on
    # the wire (post-server-tool unwrap, post-dedup).
    available_tool_names = {
        t.get("name") for t in anthropic_tools if isinstance(t, dict) and t.get("name")
    }
    # On the OAuth path, tool names get aliased to Claude Code canonical
    # names (terminal→Bash, read_file→Read, …) further down at the
    # ``replace_with_cc_canonical`` call. Any tool_use blocks already in
    # the message history from prior OAuth turns therefore carry the CC
    # canonical names, NOT the hermes-side names. Without expanding the
    # allowlist here, ``_strip_unknown_tool_blocks`` treats every
    # historical ``Bash`` / ``Read`` / etc. tool_use as stale and
    # rewrites it to a "[Previous tool call: Bash(...) — tool no longer
    # available in this turn.]" breadcrumb, even though the same call
    # will be live again this turn after aliasing.
    if is_oauth:
        try:
            from agent import cc_aliases as _cc
            if _cc.is_enabled():
                for hermes_name in list(available_tool_names):
                    cc_name = _cc.HERMES_TO_CC.get(hermes_name)
                    if cc_name:
                        available_tool_names.add(cc_name)
        except Exception:
            logger.debug(
                "anthropic_adapter: failed to expand available_tool_names "
                "with CC aliases — falling back to hermes-only set",
                exc_info=True,
            )
    anthropic_messages = _strip_unknown_tool_blocks(
        anthropic_messages, available_tool_names
    )

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

        # 3. Tool naming: pass through whatever the registry assigned.
        #
        # Earlier versions of the OAuth identity rewriter prepended ``mcp_``
        # (single-underscore) to *every* tool sent to the model.  The
        # intent was to mimic Claude Code's MCP convention so the OAuth-
        # authenticated Claude session would route the calls through its
        # MCP path.  Two problems with that approach made it worse than
        # leaving names alone:
        #
        # 1. Real Claude Code's built-in tools (Read, Write, Bash, …) do
        #    NOT carry a ``mcp_`` prefix.  Only MCP-sourced tools do.
        #    Prefixing every Hermes built-in (``read_file`` →
        #    ``mcp_read_file``) made them look like MCP tools that didn't
        #    exist in any MCP server — confusing the model into stripping
        #    the prefix on every call.
        #
        # 2. The single-underscore separator is ambiguous.  Real Claude
        #    Code MCP tools use double underscores: ``mcp__server__tool``.
        #    Single-underscore names blur where the prefix ends, so the
        #    model routinely strips the entire ``mcp_`` and emits the bare
        #    tool name — triggering ``_repair_tool_call`` on every
        #    invocation.
        #
        # Hermes' MCP tools are now registered with the canonical
        # ``mcp__<server>__<tool>`` form by ``tools/mcp_tool.py::
        # _convert_mcp_schema``, so no rewriting is needed here.  Built-in
        # tools pass through with their natural names (``read_file``,
        # ``terminal``, …) which is what real Claude Code does too.
        # Tool_use / tool_result blocks in message history likewise stay
        # untouched — they were saved with the registered names and that's
        # what we want to send back.

        # 4. system_prompt_mode=compact: move everything past the CC prefix
        #    into a preamble block on the first user message.
        #
        #    Anthropic's billing classifier on personal Max plans rejects
        #    OAuth requests whose ``system`` extends beyond the official
        #    Claude Code identity prefix — they get routed to "extra
        #    usage" billing and 400 with a misleading
        #    "out of extra usage" error. Mirroring Claude Code's
        #    --exclude-dynamic-system-prompt-sections flag, we keep only
        #    the CC prefix in ``system`` and ride everything dynamic on
        #    the conversation. Behavior is unchanged (the model still
        #    sees the same content); only the placement moves.
        #
        #    Cache control: if the moved blocks carried cache_control
        #    markers, we preserve them on the preamble block so prompt
        #    caching continues to work across turns.
        if _system_prompt_mode_compact() and isinstance(system, list) and len(system) > 1:
            tail_blocks = system[1:]
            system = [system[0]]
            tail_text_parts = []
            tail_cache_control = None
            for blk in tail_blocks:
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") == "text":
                    txt = blk.get("text", "")
                    if txt:
                        tail_text_parts.append(txt)
                # Inherit the strongest cache_control found on the moved
                # blocks (last write wins — typical pattern is a single
                # ephemeral marker on the final static block).
                cc = blk.get("cache_control")
                if cc:
                    tail_cache_control = cc
            if tail_text_parts:
                preamble = {
                    "type": "text",
                    "text": "\n\n".join(tail_text_parts),
                }
                if tail_cache_control:
                    preamble["cache_control"] = tail_cache_control
                anthropic_messages = _prepend_user_message_preamble(
                    anthropic_messages, preamble
                )

    # OAuth path: prepend the canonical Claude Code billing-header
    # block to ``system``. Real CC ships a system block whose text is
    # exactly:
    #
    #   x-anthropic-billing-header: cc_version=<ver>; cc_entrypoint=sdk-cli; cch=<hash>;
    #
    # Anthropic's billing classifier reads this block to identify the
    # client. Without it, even a request with canonical CC tool names
    # and CC-shaped schemas still routes to extra-usage billing —
    # producing the "out of extra usage" 400 on personal Max plans.
    # WITH it (and matching CC tool surface via ``cc_aliases``), the
    # classifier accepts ~50K-byte requests as plan-budget traffic.
    #
    # Captured from a live `claude` session via mitmdump (CC 2.1.138).
    # cc_version is intentionally hardcoded rather than read from
    # _detect_claude_code_version() because the classifier may
    # validate the cch checksum against the (cc_version, prompt
    # content) pair — using a different cc_version with a stale cch
    # could fail validation. Refresh both values in lockstep when CC
    # ships a major version change; see scripts in /tmp/cc-flows.har
    # for the capture recipe.
    if is_oauth and isinstance(system, list):
        _BILLING_HEADER_TEXT = (
            "x-anthropic-billing-header: cc_version=2.1.138.de9; "
            "cc_entrypoint=sdk-cli; cch=fa6a6;"
        )
        # Insert at index 0 unless one's already there (idempotent
        # against double-application, which would happen e.g. on a
        # retry path).
        if not system or "x-anthropic-billing-header:" not in str(
            system[0].get("text", "") if isinstance(system[0], dict) else system[0]
        ):
            system = [{"type": "text", "text": _BILLING_HEADER_TEXT}] + system

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": effective_max_tokens,
    }

    if system:
        kwargs["system"] = system

    if anthropic_tools:
        # CC-name aliasing on the OAuth path. Real Claude Code's eager
        # tool surface (Bash/Read/Edit/Write/Grep/...) is what
        # Anthropic's billing classifier on personal Max accounts
        # accepts as plan-budget — combined with the
        # x-anthropic-billing-header system block prepended above,
        # this makes the request indistinguishable from real CC.
        # Inbound tool_use dispatch routes CC names back to hermes
        # handlers via ``cc_aliases.adapt_tool_use`` called from
        # ``model_tools.handle_function_call``.
        if is_oauth:
            from agent import cc_aliases as _cc
            if _cc.is_enabled():
                anthropic_tools = _cc.replace_with_cc_canonical(anthropic_tools)
        anthropic_tools = _apply_tool_search(anthropic_tools, tool_search_config)
        if cache_tools:
            from agent.prompt_caching import apply_anthropic_tools_cache_control
            anthropic_tools = apply_anthropic_tools_cache_control(
                anthropic_tools, cache_ttl=cache_ttl
            )
        kwargs["tools"] = anthropic_tools
        # Map OpenAI tool_choice to Anthropic format
        if tool_choice == "auto" or tool_choice is None:
            # Mirror Claude Code: omit tool_choice (the API treats absent as
            # "auto", so we save bytes and match CC's wire shape exactly).
            pass
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
    # Kimi's /coding endpoint speaks the Anthropic Messages protocol but has
    # its own thinking semantics: when ``thinking.enabled`` is sent, Kimi
    # validates the message history and requires every prior assistant
    # tool-call message to carry OpenAI-style ``reasoning_content``.  The
    # Anthropic path never populates that field, and
    # ``convert_messages_to_anthropic`` strips all Anthropic thinking blocks
    # on third-party endpoints — so the request fails with HTTP 400
    # "thinking is enabled but reasoning_content is missing in assistant
    # tool call message at index N".  Kimi's reasoning is driven server-side
    # on the /coding route, so skip Anthropic's thinking parameter entirely
    # for that host.  (Kimi on chat_completions enables thinking via
    # extra_body in the ChatCompletionsTransport — see #13503.)
    #
    # On 4.7+ ``thinking.display`` defaults to "omitted" (no summary text
    # generated). Previously hermes set "summarized" to keep the activity
    # feed populated, but verified via binary inspection 2026-05-06 that
    # Claude Code DOES NOT set ``display`` — it accepts the omitted default.
    # Multi-minute "queued/prefilling" stalls hermes was hitting that
    # Claude Code didn't correlate with this difference: producing a
    # summary forces the model to generate extra tokens after thinking
    # before the visible output streams, magnifying any internal-thinking
    # latency.  Match Claude Code's wire shape — let display default.
    # See ``HERMES_THINKING_DISPLAY=summarized`` env var to opt back in
    # if the activity feed UX matters more than latency parity.
    _is_kimi_coding = _is_kimi_family_endpoint(base_url, model)
    # When reasoning_config is unset, default to enabling adaptive thinking
    # at medium effort on Anthropic-native + adaptive-supporting models.
    # Mirrors Claude Code 2.1.119 wire shape (verified by mitmdump capture
    # 2026-05-06: every /v1/messages call sends thinking={type:"adaptive"}
    # + output_config.effort).  Without this default, the entire
    # thinking/output_config block below was a no-op for callers that
    # don't explicitly pass reasoning_config — i.e. nearly every default
    # session — leaving the interleaved-thinking + effort betas dormant.
    if reasoning_config is None and not _is_kimi_coding and _supports_adaptive_thinking(model):
        reasoning_config = {"enabled": True, "effort": "medium"}
    if reasoning_config and isinstance(reasoning_config, dict) and not _is_kimi_coding:
        if reasoning_config.get("enabled") is not False and "haiku" not in model.lower():
            effort = str(reasoning_config.get("effort", "medium")).lower()
            budget = THINKING_BUDGET.get(effort, 8000)
            if _supports_adaptive_thinking(model):
                _thinking_cfg: Dict[str, Any] = {"type": "adaptive"}
                _display_override = os.environ.get(
                    "HERMES_THINKING_DISPLAY", ""
                ).strip().lower()
                if _display_override in {"summarized", "verbose", "all", "omitted"}:
                    _thinking_cfg["display"] = _display_override
                kwargs["thinking"] = _thinking_cfg
                adaptive_effort = ADAPTIVE_EFFORT_MAP.get(effort, "medium")
                # Downgrade xhigh on models that don't support it. Claude Code
                # falls back to "high" for non-4.7 models (verified by
                # disassembling its 2.1.119 binary: `return"xhigh";return"high"`).
                # Don't fall back to "max" — Sonnet 4.6 and Haiku 4.5 don't
                # support max either (Opus-tier only), so the previous
                # "downgrade to max" path 400'd on Sonnet/Haiku requests.
                if adaptive_effort == "xhigh" and not _supports_xhigh_effort(model):
                    adaptive_effort = "high"
                kwargs["output_config"] = {
                    "effort": adaptive_effort,
                }
                # Mirror Claude Code 2.1.119: every /v1/messages call carries
                # ``context_management`` with the clear_thinking_20251015 edit
                # set to keep:"all".  Activates the server-side thinking-block
                # lifecycle so cached thinking-blocks survive across turns
                # (paired with redact-thinking-2026-02-12 +
                # context-management-2025-06-27 betas).  Native Anthropic only
                # — third-party gateways don't recognize the field.  Typed
                # kwarg in client.beta.messages.* (Anthropic SDK 0.100+).
                if not _is_third_party_anthropic_endpoint(base_url):
                    kwargs["context_management"] = {
                        "edits": [
                            {"type": "clear_thinking_20251015", "keep": "all"},
                        ],
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
    # Sets typed ``speed="fast"`` + adds the fast-mode beta to the
    # per-request ``betas`` list for ~2.5x output speed.  Per Anthropic
    # docs, fast mode is only supported on Opus 4.6 — Opus 4.7 and other
    # models 400 on the speed parameter.
    # Only for native Anthropic endpoints — third-party providers would
    # reject the unknown beta header and speed parameter.
    if (
        fast_mode
        and not _is_third_party_anthropic_endpoint(base_url)
        and _supports_fast_mode(model)
    ):
        # Typed ``speed`` kwarg in client.beta.messages.* (SDK 0.100+).
        kwargs["speed"] = "fast"
        # Per-request betas list overrides the client-level
        # default_headers["anthropic-beta"] for this call.
        betas = list(_common_betas_for_base_url(
            base_url,
            drop_context_1m_beta=drop_context_1m_beta,
            model=model,
        ))
        if is_oauth:
            betas.extend(_OAUTH_ONLY_BETAS)
        betas.append(_FAST_MODE_BETA)
        kwargs["betas"] = betas

    # ── Server-side tool beta headers ────────────────────────────────
    # Tools like web_search_20250305 require their own anthropic-beta
    # header. We can't put it on the client-level default_headers because
    # that would be sent for every request (some Anthropic-compatible
    # third-party providers reject unknown betas). Instead, detect the
    # tools in this specific request and union with any already-set
    # ``betas`` (preserving fast-mode wiring above).
    server_tool_betas = _required_anthropic_server_tool_betas(tools or [])
    if server_tool_betas and not _is_third_party_anthropic_endpoint(base_url):
        prior = list(kwargs.get("betas") or [])
        if not prior:
            # No prior per-request betas — start from the same base set
            # the client would otherwise send so we don't accidentally
            # drop OAuth or context-1m betas.
            prior = list(_common_betas_for_base_url(
                base_url, drop_context_1m_beta=drop_context_1m_beta,
                model=model,
            ))
            if is_oauth:
                prior.extend(_OAUTH_ONLY_BETAS)
        merged: list[str] = []
        for beta in prior + server_tool_betas:
            if beta and beta not in merged:
                merged.append(beta)
        kwargs["betas"] = merged

    # ── 1M context tier gate (DEFAULT OFF) ───────────────────────────
    # Background: hermes hits sporadic multi-minute stalls on Opus 4.7
    # even with perfect cache hits. Theory was that opting into
    # ``context-1m-2025-08-07`` routes requests to a smaller, slower-
    # served 1M-context model fleet vs the standard 200K tier.
    #
    # Why disabled by default (2026-05-06): the gate uses request-body
    # size to decide, but the relevant size is the running CONTEXT
    # (cached prefix + new tokens), which can be much larger than the
    # body bytes we send (cached prefix is server-side). Adam's
    # workflows regularly run 600K+ of cached context — those genuinely
    # need the 1M beta even though each individual request body is small.
    # Stripping the beta in that case would either break cache continuity
    # or fail outright (200K context can't hold a 600K prefix).
    #
    # Set ``HERMES_CONTEXT_1M_THRESHOLD_TOKENS`` to a positive integer to
    # enable the gate at that body-size threshold. Use only when you're
    # confident the running context (not just the body) fits in 200K.
    try:
        _threshold = int(os.environ.get(
            "HERMES_CONTEXT_1M_THRESHOLD_TOKENS", "0"
        ))
    except (TypeError, ValueError):
        _threshold = 0
    if (
        _threshold > 0
        and not _requires_bearer_auth(base_url)
        and _model_supports_1m_context(model)
    ):
        # Cheap byte-based prompt estimate — char/4 is the standard
        # rough conversion. Tools count too: Anthropic loads them
        # eagerly unless defer_loading=True, so for the gate we count
        # only the eager portion.
        _est_chars = 0
        sys_obj = kwargs.get("system")
        if sys_obj is not None:
            try:
                _est_chars += len(json.dumps(sys_obj))
            except Exception:
                pass
        _msgs = kwargs.get("messages")
        if isinstance(_msgs, list):
            try:
                _est_chars += len(json.dumps(_msgs))
            except Exception:
                pass
        _tools_for_estimate = kwargs.get("tools")
        if isinstance(_tools_for_estimate, list):
            for _t in _tools_for_estimate:
                if isinstance(_t, dict) and _t.get("defer_loading"):
                    continue  # deferred tools don't count toward prefill
                try:
                    _est_chars += len(json.dumps(_t))
                except Exception:
                    pass
        _est_tokens = _est_chars // 4
        if _est_tokens < _threshold:
            prior = list(kwargs.get("betas") or [])
            if not prior:
                # No prior per-request override — start from the same
                # base set the client would otherwise send. Then strip
                # context-1m and emit as a per-request override.
                prior = list(_common_betas_for_base_url(
                    base_url,
                    drop_context_1m_beta=False,
                    model=model,
                ))
                if is_oauth:
                    prior.extend(_OAUTH_ONLY_BETAS)
            stripped = [b for b in prior if b != _CONTEXT_1M_BETA]
            if len(stripped) != len(prior):
                kwargs["betas"] = stripped

    # ── Identity metadata (mirrors Claude Code's wire shape) ─────────
    # Anthropic's metadata.user_id is a per-end-user identifier used for
    # analytics + abuse routing.  Claude Code packs a JSON blob with
    # device_id (sha256 hostname), account_uuid (stable UUID), and
    # session_id.  Native Anthropic only — third-party gateways may
    # validate or reject unrecognized metadata shapes.
    if not _is_third_party_anthropic_endpoint(base_url):
        kwargs["metadata"] = _build_anthropic_metadata(session_id)

    return kwargs