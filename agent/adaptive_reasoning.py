"""Classify per-message reasoning effort for ``agent.reasoning_effort: adaptive``.

Runs synchronously *before* the real response, unlike title_generator's
fire-and-forget pattern — the classification result becomes the actual
``reasoning_effort`` used for this turn's real API call, so it must
complete first. Kept deliberately cheap and small (single word out,
minimal effort on the classification call itself) so it doesn't reproduce
the exact problem it's solving.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Keep in sync with hermes_constants.VALID_REASONING_EFFORTS. Duplicated
# (rather than imported) to avoid a hard import-time dependency on
# hermes_constants from this small, focused module — mirrors how
# title_generator.py stays decoupled from callers' internals.
_VALID_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
_FALLBACK_EFFORT = "medium"

# Common placeholder API keys that local/self-hosted OpenAI-compatible
# servers (llama.cpp, vLLM, LM Studio, etc.) don't actually validate — many
# setup guides for these tell users to put a dummy string here since the
# OpenAI SDK/clients require *some* non-empty value. Recognizing them lets
# this module skip sending a meaningless Authorization header rather than
# forwarding a placeholder string as if it were a real credential.
_PLACEHOLDER_API_KEYS = frozenset({"not-needed", "none", "no-key", "sk-noauth", "local"})

# Providers known to be OAuth-gated, where the caller's plain ``api_key``
# kwarg (e.g. ``self.api_key`` on the agent) is NOT the live credential —
# the real bearer token lives in Hermes's own credential pool / auth store
# and must be resolved fresh (it rotates). Sending no/stale auth to these
# doesn't fail fast — several hang until the client-side timeout instead of
# rejecting quickly, which is what silently made every adaptive-reasoning
# classification against xai-oauth eat the full 25s and fall back to
# 'medium' (confirmed via ~2 weeks of unbroken "Read timed out" warnings in
# the gateway log — this was never a one-off blip).
_OAUTH_GATED_PROVIDERS = frozenset({"xai-oauth", "openai-codex", "nous"})

# Providers where the classification call should behave like talking to a
# real hosted OpenAI-compatible API (real auth required, no local-only
# chat-template quirks) rather than a bare local llama.cpp server. Anything
# not in this set, or whose base_url looks like localhost, is treated as
# local-style.
_CLOUD_PROVIDER_HINTS = frozenset({
    "xai", "xai-oauth", "openai", "openai-codex", "anthropic", "nous",
    "openrouter", "kimi", "github-models",
})

# API modes that adaptive classification can speak natively. Bedrock and
# other specialty transports fall back to medium (with a log) rather than
# inventing a half-working Converse client here.
_SUPPORTED_API_MODES = frozenset({
    "chat_completions",
    "anthropic_messages",
    "codex_responses",
})

_CLASSIFY_PROMPT = (
    "You will be shown a single user message. Decide how much reasoning "
    "effort a response to it deserves, choosing exactly one of: "
    "minimal, low, medium, high, xhigh.\n"
    "- minimal: greetings, small talk, trivial lookups\n"
    "- low: simple factual questions, short direct requests\n"
    "- medium: multi-step but routine tasks\n"
    "- high: non-trivial coding, debugging, multi-step reasoning, OR any "
    "request emphasizing correctness/rigor (\"properly\", \"make sure\", "
    "\"verify\", \"double-check\", \"understand it\", \"read carefully\") even "
    "if the message itself is short — short phrasing does not mean low "
    "effort when the intent is verification or deep understanding, not a "
    "quick answer\n"
    "- xhigh: hard math/proofs, complex architecture/design, deep multi-step analysis\n"
    "Reply with EXACTLY ONE WORD from that list and nothing else — "
    "no punctuation, no explanation."
)


def _resolve_oauth_credentials(provider: str) -> Optional[Tuple[str, str]]:
    """Resolve a fresh (api_key, base_url) for an OAuth-gated provider.

    Returns None if resolution fails for any reason (never raises) — the
    caller falls back to whatever was explicitly passed in, and ultimately
    to the fixed default effort if that's unusable too.

    Wired for all three Hermes OAuth-gated providers:
    ``xai-oauth``, ``openai-codex``, ``nous``. Each uses the same runtime
    helpers the rest of the agent uses for non-primary calls.
    """
    if provider == "xai-oauth":
        try:
            # Mirrors agent.auxiliary_client._resolve_xai_oauth_for_aux(),
            # the same helper the rest of Hermes uses to get a live xAI
            # OAuth token for non-primary calls (compression, titles, etc.)
            from agent.credential_pool import load_pool
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
                        str(getattr(entry, "runtime_base_url", None) or "").strip().rstrip("/")
                        or str(getattr(entry, "base_url", None) or "").strip().rstrip("/"),
                        fallback=DEFAULT_XAI_OAUTH_BASE_URL,
                    )
                    if api_key and base_url:
                        return api_key, base_url
        except Exception as exc:
            logger.debug("Adaptive reasoning: xAI OAuth pool credential resolution failed: %s", exc)

        try:
            from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

            creds = resolve_xai_oauth_runtime_credentials()
            api_key = str(creds.get("api_key") or "").strip()
            base_url = str(creds.get("base_url") or "").strip().rstrip("/")
            if api_key and base_url:
                return api_key, base_url
        except Exception as exc:
            logger.debug("Adaptive reasoning: xAI OAuth runtime credential resolution failed: %s", exc)
        return None

    if provider == "openai-codex":
        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials()
            api_key = str(creds.get("api_key") or "").strip()
            base_url = str(creds.get("base_url") or "").strip().rstrip("/")
            if api_key and base_url:
                return api_key, base_url
        except Exception as exc:
            logger.debug("Adaptive reasoning: openai-codex credential resolution failed: %s", exc)
        return None

    if provider == "nous":
        try:
            from hermes_cli.auth import resolve_nous_runtime_credentials

            creds = resolve_nous_runtime_credentials()
            api_key = str(creds.get("api_key") or "").strip()
            base_url = str(creds.get("base_url") or "").strip().rstrip("/")
            if api_key and base_url:
                return api_key, base_url
        except Exception as exc:
            logger.debug("Adaptive reasoning: nous credential resolution failed: %s", exc)
        return None

    return None


def _is_cloud_provider(provider: str, base_url: str) -> bool:
    if provider in _CLOUD_PROVIDER_HINTS:
        return True
    if "://" in (base_url or "") and not any(
        local_hint in base_url for local_hint in ("127.0.0.1", "localhost", "0.0.0.0")
    ):
        return True
    return False


def _parse_effort_from_text(content: str) -> str:
    """Map free-form model text down to one of ``_VALID_EFFORTS`` or fallback."""
    try:
        from agent.agent_runtime_helpers import strip_think_blocks
        content = strip_think_blocks(None, content or "")
    except Exception:
        content = content or ""

    cleaned = content.strip().strip("\"'.,!").lower()
    # Some models answer in a short phrase ("I'd say high.") despite the
    # single-word instruction — take the first token that matches a known
    # level rather than requiring an exact whole-string match.
    for token in cleaned.split():
        token = token.strip("\"'.,!:;")
        if token in _VALID_EFFORTS:
            if token != cleaned:
                logger.debug(
                    "Adaptive reasoning classification parsed %r out of full reply %r",
                    token, content,
                )
            return token

    logger.warning(
        "Adaptive reasoning classification returned unparseable effort %r, defaulting to %r",
        content, _FALLBACK_EFFORT,
    )
    return _FALLBACK_EFFORT


def _extract_chat_completions_content(data: Dict[str, Any]) -> str:
    return (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""


def _extract_anthropic_content(data: Dict[str, Any]) -> str:
    # Anthropic Messages: content is a list of blocks; take first text block.
    blocks = data.get("content") or []
    parts = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
        elif isinstance(block, str):
            parts.append(block)
    return " ".join(parts).strip()


def _extract_responses_content(data: Dict[str, Any]) -> str:
    # OpenAI Responses API: prefer output_text, else walk output[].content[].text
    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"]
    parts = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"message", "output_message"} or "content" in item:
            for block in item.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in {"output_text", "text"} and block.get("text"):
                    parts.append(block["text"])
    return " ".join(parts).strip()


def _anthropic_messages_url(base_url: str) -> str:
    """Build the Anthropic Messages endpoint from a Hermes base_url.

    Hermes stores Anthropic base URLs both with and without a trailing
    ``/v1`` (the SDK normally appends ``/v1/messages``). Normalize so we
    never produce ``/v1/v1/messages``.
    """
    root = (base_url or "").rstrip("/")
    if root.endswith("/v1"):
        return root + "/messages"
    return root + "/v1/messages"


def _responses_url(base_url: str) -> str:
    root = (base_url or "").rstrip("/")
    if root.endswith("/v1"):
        return root + "/responses"
    # Codex backend URLs are often path-specific already; append /responses
    # only when the path doesn't already end with it.
    if root.endswith("/responses"):
        return root
    return root + "/responses"


def _post_chat_completions(
    *,
    base_url: str,
    api_key: Optional[str],
    model: Optional[str],
    snippet: str,
    timeout: float,
    is_cloud: bool,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip().lower() not in _PLACEHOLDER_API_KEYS:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": model or "default",
        "messages": [
            {"role": "system", "content": _CLASSIFY_PROMPT},
            {"role": "user", "content": snippet},
        ],
        "max_tokens": 20,
        "temperature": 0,
    }
    # chat_template_kwargs.enable_thinking=False is a llama.cpp-specific
    # extension (Qwen and other hybrid think/no-think models honor it to
    # skip the reasoning phase). Only send it for local-style deployments.
    if not is_cloud:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return _extract_chat_completions_content(resp.json())


def _post_anthropic_messages(
    *,
    base_url: str,
    api_key: Optional[str],
    model: Optional[str],
    snippet: str,
    timeout: float,
) -> str:
    url = _anthropic_messages_url(base_url)
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    if api_key and api_key.strip().lower() not in _PLACEHOLDER_API_KEYS:
        # Anthropic native uses x-api-key; some gateways also accept Bearer.
        headers["x-api-key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model or "claude-sonnet-4-20250514",
        "max_tokens": 20,
        "temperature": 0,
        "system": _CLASSIFY_PROMPT,
        "messages": [
            {"role": "user", "content": snippet},
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return _extract_anthropic_content(resp.json())


def _post_codex_responses(
    *,
    base_url: str,
    api_key: Optional[str],
    model: Optional[str],
    snippet: str,
    timeout: float,
) -> str:
    url = _responses_url(base_url)
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip().lower() not in _PLACEHOLDER_API_KEYS:
        headers["Authorization"] = f"Bearer {api_key}"

    # Minimal Responses API shape: system as instructions, user as input.
    payload = {
        "model": model or "default",
        "instructions": _CLASSIFY_PROMPT,
        "input": snippet,
        "max_output_tokens": 20,
        "temperature": 0,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return _extract_responses_content(resp.json())


def classify_reasoning_effort(
    user_message: str,
    timeout: float = 25.0,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_mode: Optional[str] = None,
) -> str:
    """Ask the model to self-select a reasoning effort level for one message.

    Returns one of ``_VALID_EFFORTS``. Falls back to ``_FALLBACK_EFFORT``
    (\"medium\") on any failure, timeout, empty response, or output that
    doesn't parse to a known level — this function never raises, so a
    classification hiccup can never break the real response that follows.

    Provider-aware: for OAuth-gated providers (``xai-oauth``,
    ``openai-codex``, ``nous``), resolves a fresh live bearer token from
    Hermes's own credential pool/auth store instead of trusting the
    caller's ``api_key`` kwarg, which is only ever a static credential and
    is None/stale for OAuth providers.

    Mode-aware: routes the classification HTTP call through the transport
    matching ``api_mode``:

    - ``chat_completions`` (default) → ``POST …/chat/completions``
    - ``anthropic_messages`` → ``POST …/v1/messages``
    - ``codex_responses`` → ``POST …/responses``
    - anything else (e.g. ``bedrock_converse``) → fall back to medium

    Deliberately makes a **raw HTTP call directly to base_url** instead of
    going through auxiliary_client.call_llm()'s task/provider/config
    resolution + client-cache machinery — see git history for why (an
    earlier attempt through call_llm() intermittently produced spurious
    401s from unrelated cache state).

    Default timeout is deliberately generous (25s): on single-slot local
    backends (e.g. llama.cpp ``--parallel 1``) this call can queue behind
    the previous turn's still-finishing response rather than running
    instantly. For real cloud providers this should complete in ~1-2s once
    auth is actually valid.
    """
    snippet = (user_message or "")[:2000]
    if not snippet.strip():
        return _FALLBACK_EFFORT

    resolved_provider = (provider or "").strip().lower()
    mode = (api_mode or "chat_completions").strip().lower() or "chat_completions"

    if mode not in _SUPPORTED_API_MODES:
        logger.warning(
            "Adaptive reasoning: api_mode %r has no classification transport; "
            "defaulting to %r",
            mode, _FALLBACK_EFFORT,
        )
        return _FALLBACK_EFFORT

    # Auto-detect: if this provider is OAuth-gated, resolve a live
    # credential/base_url pair fresh rather than trusting what was passed
    # in. Falls back to the passed-in api_key/base_url if resolution fails.
    if resolved_provider in _OAUTH_GATED_PROVIDERS:
        resolved = _resolve_oauth_credentials(resolved_provider)
        if resolved is not None:
            api_key, base_url = resolved
        else:
            logger.warning(
                "Adaptive reasoning: could not resolve live OAuth credentials for "
                "provider %r, falling back to passed-in api_key (likely stale/absent)",
                resolved_provider,
            )

    if not base_url:
        logger.warning(
            "Adaptive reasoning classification skipped (no base_url), defaulting to %r",
            _FALLBACK_EFFORT,
        )
        return _FALLBACK_EFFORT

    is_cloud = _is_cloud_provider(resolved_provider, base_url)

    try:
        if mode == "anthropic_messages":
            content = _post_anthropic_messages(
                base_url=base_url,
                api_key=api_key,
                model=model,
                snippet=snippet,
                timeout=timeout,
            )
        elif mode == "codex_responses":
            content = _post_codex_responses(
                base_url=base_url,
                api_key=api_key,
                model=model,
                snippet=snippet,
                timeout=timeout,
            )
        else:
            content = _post_chat_completions(
                base_url=base_url,
                api_key=api_key,
                model=model,
                snippet=snippet,
                timeout=timeout,
                is_cloud=is_cloud,
            )
    except Exception as e:
        logger.warning(
            "Adaptive reasoning classification failed (mode=%r), defaulting to %r: %s",
            mode, _FALLBACK_EFFORT, e,
        )
        logger.debug("Adaptive reasoning classification traceback", exc_info=True)
        return _FALLBACK_EFFORT

    if not (content or "").strip():
        logger.warning(
            "Adaptive reasoning classification empty response (mode=%r), defaulting to %r",
            mode, _FALLBACK_EFFORT,
        )
        return _FALLBACK_EFFORT

    return _parse_effort_from_text(content)


def apply_adaptive_reasoning_intent(agent: Any, reasoning_config: Optional[dict]) -> None:
    """Keep ``agent._adaptive_reasoning`` in sync with a live reasoning_config.

    Adaptive intent is a separate flag from the concrete ``reasoning_config``
    that gets overwritten per-turn with a resolved effort. Call this whenever
    a path (TUI ``/reasoning``, gateway override, model menu) writes a new
    reasoning_config onto a live agent so fixed overrides actually stick and
    re-enabling adaptive works without rebuilding the agent.
    """
    intent = bool(
        reasoning_config
        and isinstance(reasoning_config, dict)
        and reasoning_config.get("effort") == "adaptive"
    )
    agent._adaptive_reasoning = intent
    # Drop any cached classification so the next API build re-classifies
    # (or stops classifying) under the new intent.
    if hasattr(agent, "_adaptive_reasoning_cache"):
        agent._adaptive_reasoning_cache = None
