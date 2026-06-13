"""Perplexity Sonar "ask" tool — synthesized, citation-grounded answers.

Where ``web_search`` returns ranked links and ``web_extract`` returns page
content, ``perplexity_ask`` returns a *finished answer* with its sources in a
single call by hitting Perplexity's OpenAI-compatible Chat Completions API
(``POST /chat/completions``) with a Sonar model. This collapses the usual
``web_search`` → ``web_extract`` → summarize loop into one tool round-trip,
saving turn-loop iterations and tokens.

Modes map to Sonar models::

    "sonar"     -> sonar                 (fast, cheap, grounded answer)
    "pro"       -> sonar-pro             (deeper, more sources)
    "reasoning" -> sonar-reasoning       (chain-of-thought + search)
    "deep"      -> sonar-deep-research   (long multi-step research report)

Env vars::

    PERPLEXITY_API_KEY=...     # required — https://www.perplexity.ai/account/api
    PERPLEXITY_BASE_URL=...    # optional override of https://api.perplexity.ai

Cost note: Sonar calls are billed (tokens + a per-request search fee). A flat
per-request estimate is registered in :mod:`agent.usage_pricing` so the cost
line / ``/cost`` reflect each call. The estimate is conservative, not
token-exact — see ``TOOL_REQUEST_PRICING`` keys ``perplexity-sonar*``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# mode -> (Sonar model id, request timeout seconds)
_MODE_TO_MODEL: Dict[str, tuple[str, float]] = {
    "sonar": ("sonar", 60.0),
    "pro": ("sonar-pro", 90.0),
    "reasoning": ("sonar-reasoning", 120.0),
    "deep": ("sonar-deep-research", 300.0),
}

# mode -> usage_pricing backend key (see agent/usage_pricing.py:TOOL_REQUEST_PRICING)
MODE_TO_BILLING_KEY: Dict[str, str] = {
    "sonar": "perplexity-sonar",
    "pro": "perplexity-sonar-pro",
    "reasoning": "perplexity-sonar-reasoning",
    "deep": "perplexity-sonar-deep-research",
}


def billing_key_for_args(function_args: Dict[str, Any] | None) -> str:
    """Resolve the usage_pricing backend key for a ``perplexity_ask`` call."""
    mode = str((function_args or {}).get("mode", "sonar")).strip().lower()
    return MODE_TO_BILLING_KEY.get(mode, "perplexity-sonar")


def _format_answer(content: str, response: Dict[str, Any]) -> str:
    """Render the Sonar answer plus a numbered Sources list for the agent."""
    lines: List[str] = [content.strip() or "(empty answer)"]

    # Prefer the richer ``search_results`` (title + url); fall back to the
    # legacy ``citations`` list of bare URLs.
    sources: List[str] = []
    search_results = response.get("search_results")
    if isinstance(search_results, list) and search_results:
        for i, sr in enumerate(search_results, 1):
            if not isinstance(sr, dict):
                continue
            title = (sr.get("title") or "").strip()
            url = (sr.get("url") or "").strip()
            date = (sr.get("date") or "").strip()
            label = title or url or "source"
            suffix = f" ({date})" if date else ""
            sources.append(f"[{i}] {label}{suffix} — {url}" if url else f"[{i}] {label}{suffix}")
    else:
        citations = response.get("citations")
        if isinstance(citations, list):
            for i, url in enumerate(citations, 1):
                if isinstance(url, str) and url.strip():
                    sources.append(f"[{i}] {url.strip()}")

    if sources:
        lines.append("")
        lines.append("Sources:")
        lines.extend(sources)
    return "\n".join(lines)


def perplexity_ask_tool(
    question: str,
    mode: str = "sonar",
    recency: str | None = None,
    domains: List[str] | None = None,
) -> str:
    """Ask Perplexity Sonar a question and return a cited answer string.

    Returns a human/agent-readable string: the answer followed by a numbered
    ``Sources:`` list. On failure returns an ``Error: ...`` string (the tool
    contract is a plain string, matching ``web_search``).
    """
    question = (question or "").strip()
    if not question:
        return "Error: 'question' must be a non-empty string."

    try:
        from tools.interrupt import is_interrupted

        if is_interrupted():
            return "Error: Interrupted"
    except Exception:  # noqa: BLE001 — interrupt module optional
        pass

    api_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    if not api_key:
        return (
            "Error: PERPLEXITY_API_KEY environment variable not set. "
            "Get your API key at https://www.perplexity.ai/account/api"
        )

    model, timeout = _MODE_TO_MODEL.get(mode.strip().lower(), _MODE_TO_MODEL["sonar"])
    base_url = os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")
    url = f"{base_url.rstrip('/')}/chat/completions"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
    }
    if recency and recency.strip():
        # Perplexity accepts: hour | day | week | month | year
        payload["search_recency_filter"] = recency.strip().lower()
    if domains:
        # Perplexity domain allow/deny list (prefix a domain with '-' to block).
        payload["search_domain_filter"] = [d for d in domains if isinstance(d, str) and d.strip()][:10]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        import httpx

        logger.info("Perplexity ask (%s): '%.80s'", model, question)
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001 — including httpx errors
        logger.warning("Perplexity ask error: %s", exc)
        return f"Error: Perplexity ask failed: {exc}"

    try:
        content = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return f"Error: Unexpected Perplexity response shape: {str(data)[:300]}"

    return _format_answer(content, data)


PERPLEXITY_ASK_SCHEMA = {
    "name": "perplexity_ask",
    "description": (
        "Ask Perplexity Sonar a question and get a finished, citation-grounded "
        "answer in one call (instead of web_search + web_extract + summarize). "
        "Best for current-events / factual / research questions where you want a "
        "synthesized answer with sources. Returns the answer followed by a "
        "numbered Sources list. NOTE: this is a paid call billed per request."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask, in natural language.",
            },
            "mode": {
                "type": "string",
                "enum": ["sonar", "pro", "reasoning", "deep"],
                "description": (
                    "sonar = fast cheap grounded answer (default); "
                    "pro = deeper, more sources; "
                    "reasoning = chain-of-thought + search; "
                    "deep = long multi-step research report (slow, most expensive)."
                ),
                "default": "sonar",
            },
            "recency": {
                "type": "string",
                "enum": ["hour", "day", "week", "month", "year"],
                "description": "Optional: restrict sources to this recency window.",
            },
            "domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional allow/deny list of domains (max 10). Prefix a domain "
                    "with '-' to exclude it, e.g. ['arxiv.org', '-reddit.com']."
                ),
                "maxItems": 10,
            },
        },
        "required": ["question"],
    },
}


def _perplexity_ask_available() -> bool:
    return bool(os.getenv("PERPLEXITY_API_KEY", "").strip())


def register_ask_tool(ctx) -> None:
    """Register the ``perplexity_ask`` tool with the plugin context."""
    ctx.register_tool(
        name="perplexity_ask",
        toolset="web",
        schema=PERPLEXITY_ASK_SCHEMA,
        handler=lambda args, **kw: perplexity_ask_tool(
            args.get("question", ""),
            mode=args.get("mode", "sonar"),
            recency=args.get("recency"),
            domains=args.get("domains"),
        ),
        check_fn=_perplexity_ask_available,
        requires_env=["PERPLEXITY_API_KEY"],
        is_async=False,
        description="Ask Perplexity Sonar for a cited answer in one call.",
        emoji="🪄",
    )
