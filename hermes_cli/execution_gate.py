"""Conservative pre-agent execution gate for one-shot Hermes requests.

A direct turn is an OpenAI-compatible *single request*, not an agent. It cannot
perform effects, tool calls, retrieval or fallback. Bounded mode retains the
existing Hermes gateway with a small step budget. Default is unchanged agent.
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request


_EXACT_REPLY = re.compile(
    r"^\s*(?:reply|respond|return|say|output|print)\s+(?:(?:with|only|exactly)\s+){0,3}(?:the\s+)?(?:text\s+)?[\"'`]?[^\n\r]{1,120}[\"'`]?\s*[.!]?\s*$",
    re.IGNORECASE,
)
_ALLOWED_MODEL = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,180}$")


def _single_request(req, timeout):
    """Use Hermes' HTTP transport; reject redirects to avoid leaking credentials.

    Some providers reject Python-urllib clients at the edge before authentication;
    a regular HTTPX request preserves standard Hermes-compatible headers.
    """
    import httpx

    try:
        response = httpx.request(
            req.get_method(),
            req.full_url,
            content=req.data,
            headers={
                **dict(req.header_items()),
                "User-Agent": "Hermes-ExecutionGate/1.0",
            },
            timeout=timeout,
            follow_redirects=False,
        )
    except httpx.RequestError as exc:
        raise URLError("provider request failed") from exc
    if not 200 <= response.status_code < 300:
        raise HTTPError(
            req.full_url, response.status_code, "provider rejected request", None, None
        )
    return io.BytesIO(response.content)


@dataclass(frozen=True)
class Decision:
    mode: str
    reason: str


def choose_execution_mode(
    prompt: str, requested: str = "agent", *, direct_available: bool = False
) -> Decision:
    """Conservative selection; ambiguity always promotes to the full agent."""
    if requested not in {"auto", "direct", "bounded", "agent"}:
        raise ValueError("EXECUTION_MODE_INVALID")
    if requested != "auto":
        return Decision(requested, "explicit")
    if not isinstance(prompt, str) or not direct_available:
        return Decision("agent", "unsupported_direct_route")
    if (
        len(prompt) <= 150
        and re.search(r"\b(?:exactly|only)\b", prompt, re.IGNORECASE)
        and _EXACT_REPLY.fullmatch(prompt)
    ):
        return Decision("direct", "bounded_literal_response")
    return Decision("agent", "requires_agent_or_unknown")


def resolve_direct_route(
    config: dict, *, provider: str | None = None, model: str | None = None
) -> dict | None:
    """Only named OpenAI-chat-compatible custom providers; never guess endpoints/keys."""
    selected = provider or config.get("model", {}).get("provider", "")
    if not isinstance(selected, str) or not selected.startswith("custom:"):
        return None
    name = selected.removeprefix("custom:")
    entry = (config.get("providers") or {}).get(name)
    if not isinstance(entry, dict):
        return None
    mode = entry.get("api_mode")
    if mode not in ("chat_completions", "openai", "openai_chat"):
        return None
    url = str(entry.get("base_url") or entry.get("api") or "").rstrip("/")
    parts = urlparse(url)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        return None
    key_env = entry.get("key_env") or entry.get("api_key_env")
    if not isinstance(key_env, str) or not re.fullmatch(
        r"[A-Z][A-Z0-9_]{1,100}", key_env
    ):
        return None
    selected_model = model or entry.get("default_model") or entry.get("model")
    if not isinstance(selected_model, str) or not _ALLOWED_MODEL.fullmatch(
        selected_model
    ):
        return None
    return {
        "model": selected_model,
        "url": url + "/chat/completions",
        "key_env": key_env,
        "provider": selected,
    }


def direct_completion(
    prompt: str,
    route: dict,
    api_key: str,
    *,
    timeout: int = 20,
    max_tokens: int = 64,
    opener=_single_request,
    clock=time.monotonic,
) -> dict:
    """Exactly one network request; no retries, tools, or automatic model fallback."""
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12000:
        raise ValueError("EXECUTION_PROMPT_INVALID")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("EXECUTION_CREDENTIAL_MISSING")
    if not 1 <= max_tokens <= 256 or not 1 <= timeout <= 120:
        raise ValueError("EXECUTION_BUDGET_INVALID")
    body = json.dumps(
        {
            "model": route["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode()
    req = Request(
        route["url"],
        data=body,
        method="POST",
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
    )
    started = clock()
    try:
        with opener(req, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"EXECUTION_HTTP_{exc.code}") from None
    except (URLError, TimeoutError):
        raise RuntimeError("EXECUTION_NETWORK_FAILED") from None
    output = ((result.get("choices") or [{}])[0].get("message") or {}).get("content")
    if not isinstance(output, str) or not output.strip():
        raise RuntimeError("EXECUTION_EMPTY_RESPONSE")
    usage = result.get("usage") or {}
    return {
        "output": output,
        "model": result.get("model") or route["model"],
        "provider": route["provider"],
        "api_calls": 1,
        "input_tokens": usage.get("prompt_tokens")
        if isinstance(usage.get("prompt_tokens"), int)
        else None,
        "output_tokens": usage.get("completion_tokens")
        if isinstance(usage.get("completion_tokens"), int)
        else None,
        "elapsed_ms": int((clock() - started) * 1000),
        "completed": True,
        "cost_status": "unknown",
        "estimated_cost_usd": None,
        "execution_mode": "direct",
    }


def run_pre_agent_oneshot(
    args, *, load_config=None, get_secret=None, opener=_single_request
) -> int | None:
    """Return None for legacy gateway, otherwise exit code. No silent direct→agent fallback."""
    selected = getattr(args, "execution_mode", "agent")
    if selected in ("direct", "auto") and any(
        getattr(args, flag, None)
        for flag in ("resume", "continue_last", "skills", "worktree")
    ):
        if selected == "direct":
            print("EXECUTION_DIRECT_CONTEXT_UNSUPPORTED", file=sys.stderr)
            return 2
        return None
    if selected == "agent":
        return None
    if selected == "bounded":
        # Require an explicit narrow toolset; otherwise this is just the full agent
        # with a short counter, which is precisely the costly failure we observed.
        toolsets = getattr(args, "toolsets", None)
        names = (
            [v.strip() for v in toolsets.split(",") if v.strip()]
            if isinstance(toolsets, str)
            else list(toolsets or [])
        )
        if (
            not names
            or len(names) > 3
            or any(name in ("all", "*", "hermes-cli") for name in names)
        ):
            print("EXECUTION_BOUNDED_TOOLSETS_REQUIRED", file=sys.stderr)
            return 2
        current = getattr(args, "max_turns", None)
        args.max_turns = (
            min(2, current) if isinstance(current, int) and current > 0 else 2
        )
        return None
    if load_config is None:
        from hermes_cli.config import load_config as load_config_impl

        load_config = load_config_impl
    config = load_config()
    route = resolve_direct_route(
        config,
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
    )
    decision = choose_execution_mode(
        args.oneshot, selected, direct_available=route is not None
    )
    if decision.mode == "agent":
        return None
    if route is None:
        print("EXECUTION_DIRECT_ROUTE_UNAVAILABLE", file=sys.stderr)
        return 2
    if get_secret is None:
        from agent.secret_scope import get_secret_str

        get_secret = get_secret_str
    try:
        result = direct_completion(
            args.oneshot, route, get_secret(route["key_env"], ""), opener=opener
        )
        print(result["output"])
        status = 0
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        result = {
            "completed": False,
            "failed": True,
            "execution_mode": "direct",
            "cost_status": "unknown",
            "estimated_cost_usd": None,
            "api_calls": 0 if isinstance(exc, ValueError) else 1,
        }
        status = 2
    usage_path = getattr(args, "usage_file", None)
    if usage_path:
        report = {key: value for key, value in result.items() if key != "output"}
        Path(usage_path).expanduser().write_text(json.dumps(report, indent=2) + "\n")
    return status
