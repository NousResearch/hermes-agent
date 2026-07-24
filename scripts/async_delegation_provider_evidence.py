"""Capture redacted Anthropic prompt-cache evidence for async steering docs."""

from __future__ import annotations

import json
import time
from pathlib import Path

from agent.anthropic_adapter import (
    _CLAUDE_CODE_SYSTEM_PREFIX,
    _is_oauth_token,
    build_anthropic_client,
    resolve_anthropic_token,
)


MODEL = "claude-haiku-4-5-20251001"
OUTPUT = Path("async-delegation-provider-evidence.json")
MAX_ESTIMATED_USD = 0.05


def usage_dict(response) -> dict[str, int]:
    usage = response.usage
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0)),
        "cache_creation_input_tokens": int(
            getattr(usage, "cache_creation_input_tokens", 0)
        ),
        "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0)),
        "output_tokens": int(getattr(usage, "output_tokens", 0)),
    }


def estimated_cost_usd(usage: dict[str, int]) -> float:
    # Claude Haiku 4.5 public rates, USD per million tokens.
    return (
        usage["input_tokens"] * 1.0
        + usage["cache_creation_input_tokens"] * 1.25
        + usage["cache_read_input_tokens"] * 0.10
        + usage["output_tokens"] * 5.0
    ) / 1_000_000


def call(client, *, system, messages) -> dict:
    started = time.perf_counter()
    response = client.messages.create(
        model=MODEL,
        max_tokens=8,
        temperature=0,
        system=system,
        messages=messages,
    )
    usage = usage_dict(response)
    return {
        "status": "ok",
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": usage,
        "estimated_cost_usd": round(estimated_cost_usd(usage), 8),
        "response_id_redacted": f"{response.id[:10]}…",
    }


def main() -> None:
    token = resolve_anthropic_token()
    if not token:
        raise SystemExit("Anthropic credential unavailable")

    client = build_anthropic_client(token, timeout=60)
    cache_text = " ".join(
        f"Stable Hermes delegation invariant {index}: preserve prior message bytes."
        for index in range(430)
    )
    system_text = (
        f"{_CLAUDE_CODE_SYSTEM_PREFIX}\n\n{cache_text}"
        if _is_oauth_token(token)
        else cache_text
    )
    system = [
        {
            "type": "text",
            "text": system_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    # Conservative preflight: ~8K uncached input tokens across three calls.
    estimated_worst_case = (8_000 * 1.25 + 24 * 5.0) / 1_000_000
    if estimated_worst_case > MAX_ESTIMATED_USD:
        raise SystemExit("Preflight cost cap exceeded")

    evidence = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": "Anthropic direct",
        "model": MODEL,
        "credential_kind": "OAuth" if _is_oauth_token(token) else "API key",
        "cost_cap_usd": MAX_ESTIMATED_USD,
        "preflight_worst_case_usd": round(estimated_worst_case, 8),
        "calls": {},
    }

    evidence["calls"]["cache_warm"] = call(
        client,
        system=system,
        messages=[{"role": "user", "content": "Reply only WARM."}],
    )
    evidence["calls"]["cache_read"] = call(
        client,
        system=system,
        messages=[{"role": "user", "content": "Reply only HIT."}],
    )
    evidence["calls"]["cache_safe_steering"] = call(
        client,
        system=system,
        messages=[
            {"role": "user", "content": "Inspect repository."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_demo",
                        "name": "terminal",
                        "input": {"command": "scan"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_demo",
                        "content": "scan complete\n[USER STEER]: inspect only *.py files",
                    }
                ],
            },
        ],
    )

    try:
        client.messages.create(
            model=MODEL,
            max_tokens=8,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "missing_tool_call",
                            "content": "invalid orphan result",
                        }
                    ],
                }
            ],
        )
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        evidence["negative_control"] = {
            "status": "rejected",
            "http_status": status_code,
            "error_type": type(exc).__name__,
            "secret_free_summary": "orphan tool_result rejected by provider",
        }
    else:
        evidence["negative_control"] = {"status": "unexpectedly accepted"}

    total_cost = sum(
        item["estimated_cost_usd"] for item in evidence["calls"].values()
    )
    evidence["total_estimated_cost_usd"] = round(total_cost, 8)
    if total_cost > MAX_ESTIMATED_USD:
        raise RuntimeError("Measured estimated cost exceeded internal cap")

    OUTPUT.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
