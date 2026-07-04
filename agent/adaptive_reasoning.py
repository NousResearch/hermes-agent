"""Classify per-message reasoning effort for ``agent.reasoning_effort: adaptive``.

Runs synchronously *before* the real response, unlike title_generator's
fire-and-forget pattern — the classification result becomes the actual
``reasoning_effort`` used for this turn's real API call, so it must
complete first. Kept deliberately cheap and small (single word out,
minimal effort on the classification call itself) so it doesn't reproduce
the exact problem it's solving.
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Keep in sync with hermes_constants.VALID_REASONING_EFFORTS. Duplicated
# (rather than imported) to avoid a hard import-time dependency on
# hermes_constants from this small, focused module — mirrors how
# title_generator.py stays decoupled from callers' internals.
_VALID_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
_FALLBACK_EFFORT = "medium"

_CLASSIFY_PROMPT = (
    "You will be shown a single user message. Decide how much reasoning "
    "effort a response to it deserves, choosing exactly one of: "
    "minimal, low, medium, high, xhigh.\n"
    "- minimal: greetings, small talk, trivial lookups\n"
    "- low: simple factual questions, short direct requests\n"
    "- medium: multi-step but routine tasks\n"
    "- high: non-trivial coding, debugging, multi-step reasoning\n"
    "- xhigh: hard math/proofs, complex architecture/design, deep multi-step analysis\n"
    "Reply with EXACTLY ONE WORD from that list and nothing else — "
    "no punctuation, no explanation."
)


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

    Deliberately makes a **raw HTTP call directly to base_url** instead of
    going through auxiliary_client.call_llm()'s task/provider/config
    resolution + client-cache machinery. Two earlier attempts routed
    through call_llm() (first via main_runtime=, then via explicit
    provider/model/base_url/api_key kwargs) both intermittently produced a
    401 "User not found" from an unrelated authenticated backend and took
    tens of seconds to fail — root cause not fully isolated (suspected
    stale/miskeyed entry in call_llm's internal client cache), but not
    worth chasing further: this is a narrow, single-purpose call against a
    known local endpoint, so a plain ``requests.post`` is not just a
    workaround but the more appropriate tool — zero shared state, zero
    chance of resolving to any endpoint other than the one explicitly
    passed in.

    Default timeout is deliberately generous (25s, not a token-budget-sized
    handful of seconds): on single-slot local backends (e.g. llama.cpp
    ``--parallel 1``) this call can queue behind the previous turn's still-
    finishing response rather than running instantly.
    """
    snippet = (user_message or "")[:2000]
    if not snippet.strip():
        return _FALLBACK_EFFORT
    if not base_url:
        logger.warning("Adaptive reasoning classification skipped (no base_url), defaulting to %r",
                        _FALLBACK_EFFORT)
        return _FALLBACK_EFFORT

    messages = [
        {"role": "system", "content": _CLASSIFY_PROMPT},
        {"role": "user", "content": snippet},
    ]

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "not-needed":
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model or "default",
        "messages": messages,
        "max_tokens": 20,
        "temperature": 0,
        # The classification call itself must not recurse into deep
        # thinking — Qwen (and other hybrid think/no-think models served
        # via llama.cpp) honor chat_template_kwargs.enable_thinking=False
        # to skip the reasoning phase entirely, unlike the OpenRouter-style
        # "reasoning": {"effort": ...} field, which llama.cpp's local
        # server silently ignores (confirmed: it still burns the whole
        # max_tokens budget on a <think> block and never reaches content).
        # Servers/models that don't recognize chat_template_kwargs simply
        # ignore it, so this is safe to send unconditionally.
        "chat_template_kwargs": {"enable_thinking": False},
    }

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
