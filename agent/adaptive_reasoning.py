"""Classify per-message reasoning effort for ``agent.reasoning_effort: adaptive``.

Runs synchronously *before* the real response, unlike title_generator's
fire-and-forget pattern — the classification result becomes the actual
``reasoning_effort`` used for this turn's real API call, so it must
complete first. Kept deliberately cheap and small (single word out,
minimal effort on the classification call itself) so it doesn't reproduce
the exact problem it's solving.
"""

import logging
from typing import Optional, Tuple

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
    """
    if provider == "xai-oauth":
        try:
            # Mirrors agent.auxiliary_client._resolve_xai_oauth_for_aux(),
            # the same helper the rest of Hermes uses to get a live xAI
            # OAuth token for non-primary calls (compression, titles, etc.)
            # — adaptive-reasoning classification is exactly that shape of
            # call, so it should resolve credentials the same way instead
            # of assuming a static self.api_key.
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

    # openai-codex / nous OAuth resolution would follow the same pattern if
    # adaptive reasoning is ever used with those as the main provider — not
    # wired yet since only xai-oauth has been observed to actually need it
    # (the other providers in this deployment used plain API keys).
    return None


def classify_reasoning_effort(
    user_message: str,
    timeout: float = 25.0,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Ask the model to self-select a reasoning effort level for one message.

    Returns one of ``_VALID_EFFORTS``. Falls back to ``_FALLBACK_EFFORT``
    ("medium") on any failure, timeout, empty response, or output that
    doesn't parse to a known level — this function never raises, so a
    classification hiccup can never break the real response that follows.

    Provider-aware: for OAuth-gated providers (currently ``xai-oauth``),
    resolves a fresh live bearer token from Hermes's own credential
    pool/auth store instead of trusting the caller's ``api_key`` kwarg,
    which is only ever a static credential and is None/stale for OAuth
    providers. This was the actual root cause of adaptive reasoning always
    timing out and silently defaulting to 'medium' after switching the
    main provider to xai-oauth (and, before that, it was untested against
    any authenticated provider at all — the original local-llama.cpp-only
    version never needed this path).

    Deliberately makes a **raw HTTP call directly to base_url** instead of
    going through auxiliary_client.call_llm()'s task/provider/config
    resolution + client-cache machinery — see git history for why (an
    earlier attempt through call_llm() intermittently produced spurious
    401s from unrelated cache state). This is a narrow, single-purpose
    call, so a plain ``requests.post`` with explicitly-resolved credentials
    is the more appropriate tool here, not a workaround.

    Default timeout is deliberately generous (25s): on single-slot local
    backends (e.g. llama.cpp ``--parallel 1``) this call can queue behind
    the previous turn's still-finishing response rather than running
    instantly. For real cloud providers this should complete in ~1-2s once
    auth is actually valid — a full 25s timeout there means something is
    still wrong (bad credentials, wrong endpoint), not just "busy."
    """
    snippet = (user_message or "")[:2000]
    if not snippet.strip():
        return _FALLBACK_EFFORT
    if not base_url:
        logger.warning("Adaptive reasoning classification skipped (no base_url), defaulting to %r",
                        _FALLBACK_EFFORT)
        return _FALLBACK_EFFORT

    resolved_provider = (provider or "").strip().lower()

    # Auto-detect: if this provider is OAuth-gated, resolve a live
    # credential/base_url pair fresh rather than trusting what was passed
    # in — this is the fix. Falls back to the passed-in api_key/base_url
    # if resolution fails, so behavior degrades gracefully rather than
    # hard-failing differently from before.
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

    messages = [
        {"role": "system", "content": _CLASSIFY_PROMPT},
        {"role": "user", "content": snippet},
    ]

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip().lower() not in _PLACEHOLDER_API_KEYS:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model or "default",
        "messages": messages,
        "max_tokens": 20,
        "temperature": 0,
    }

    # chat_template_kwargs.enable_thinking=False is a llama.cpp-specific
    # extension (Qwen and other hybrid think/no-think models honor it to
    # skip the reasoning phase). It's meaningless — and potentially an
    # unrecognized-field risk — against a real hosted API, so only send it
    # for local-style deployments: an explicit local provider name, or any
    # base_url that isn't clearly a real cloud host.
    is_cloud_provider = resolved_provider in _CLOUD_PROVIDER_HINTS or (
        "://" in base_url and not any(
            local_hint in base_url for local_hint in ("127.0.0.1", "localhost", "0.0.0.0")
        )
    )
    if not is_cloud_provider:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"].get("content") or ""
    except Exception as e:
        logger.warning("Adaptive reasoning classification failed, defaulting to %r: %s",
                        _FALLBACK_EFFORT, e)
        logger.debug("Adaptive reasoning classification traceback", exc_info=True)
        return _FALLBACK_EFFORT

    try:
        from agent.agent_runtime_helpers import strip_think_blocks
        content = strip_think_blocks(None, content)
    except Exception:
        pass

    cleaned = content.strip().strip('"\'.,!').lower()
    # Some models answer in a short phrase ("I'd say high.") despite the
    # single-word instruction — take the first token that matches a known
    # level rather than requiring an exact whole-string match.
    for token in cleaned.split():
        token = token.strip('"\'.,!:;')
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
