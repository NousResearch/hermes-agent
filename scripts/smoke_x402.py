#!/usr/bin/env python3
"""Manual smoke test for the Hermes x402 provider path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from hermes_cli.config import load_config
from hermes_cli.runtime_provider import resolve_runtime_provider


def _default_model() -> str:
    cfg = load_config()
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        return str(model_cfg.get("default") or "")
    return str(model_cfg or "")


def _wrap_resolver(resolver):
    state = {"calls": 0, "refreshes": 0, "headers_emitted": 0}

    def wrapped(**kwargs):
        state["calls"] += 1
        if kwargs.get("force_refresh"):
            state["refreshes"] += 1
            print("[x402] refresh requested")
        headers = resolver(**kwargs)
        if headers:
            state["headers_emitted"] += 1
            print(f"[x402] emitted header(s): {', '.join(sorted(headers.keys()))}")
        else:
            print("[x402] emitted no headers")
        return headers

    return wrapped, state


def _extract_text(response: Any) -> str:
    try:
        choice = response.choices[0]
        return str(choice.message.content or "")
    except Exception:
        pass
    try:
        content = getattr(response, "content", None)
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
            return "".join(parts)
    except Exception:
        pass
    try:
        if hasattr(response, "model_dump"):
            dumped = response.model_dump()
            content = dumped.get("content")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                        parts.append(str(item["text"]))
                return "".join(parts)
    except Exception:
        pass
    return ""


def _response_debug_payload(response: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    try:
        payload["model"] = getattr(response, "model", None)
        payload["id"] = getattr(response, "id", None)
        payload["usage"] = getattr(response, "usage", None)
        if hasattr(response, "model_dump"):
            dumped = response.model_dump()
            payload["raw"] = dumped
        else:
            choice = response.choices[0]
            payload["finish_reason"] = getattr(choice, "finish_reason", None)
            payload["message"] = {
                "role": getattr(choice.message, "role", None),
                "content": getattr(choice.message, "content", None),
                "tool_calls": getattr(choice.message, "tool_calls", None),
            }
    except Exception as exc:
        payload["debug_error"] = str(exc)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test Hermes x402 provider flow.")
    parser.add_argument("--model", default=_default_model(), help="Model slug to call")
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: smoke ok",
        help="Prompt to send to the provider",
    )
    parser.add_argument(
        "--provider",
        default="x402",
        help="Provider id to resolve (defaults to x402)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="max_tokens value to send with the smoke request",
    )
    args = parser.parse_args()

    runtime = resolve_runtime_provider(requested=args.provider)
    resolver = runtime.get("request_headers_resolver")
    if not callable(resolver):
        print("Resolved provider did not return a request_headers_resolver.", file=sys.stderr)
        return 2

    wrapped_resolver, state = _wrap_resolver(resolver)
    client = OpenAI(
        api_key=runtime["api_key"],
        base_url=runtime["base_url"],
    )
    api_kwargs: Dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
    }

    print(json.dumps({
        "provider": runtime.get("provider"),
        "base_url": runtime.get("base_url"),
        "auth_source": runtime.get("source"),
        "model": args.model,
        "max_tokens": args.max_tokens,
    }, indent=2))

    try:
        headers = wrapped_resolver(force_refresh=False, api_kwargs=api_kwargs, error=None)
        response = client.chat.completions.create(**api_kwargs, extra_headers=headers or None)
    except Exception as exc:
        print(f"[x402] initial request failed: {type(exc).__name__}: {exc}")
        retry_headers = wrapped_resolver(force_refresh=True, api_kwargs=api_kwargs, error=exc)
        if not retry_headers:
            print("[x402] resolver did not produce retry headers", file=sys.stderr)
            return 1
        response = client.chat.completions.create(**api_kwargs, extra_headers=retry_headers)

    text = _extract_text(response)
    print("\n=== response ===\n")
    print(text)
    if not text.strip():
        print("\n=== raw-response ===\n")
        print(json.dumps(_response_debug_payload(response), indent=2, default=str))
    print("\n=== stats ===\n")
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
