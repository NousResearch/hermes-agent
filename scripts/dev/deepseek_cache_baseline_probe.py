#!/usr/bin/env python3
"""Deterministic DeepSeek context-cache baseline probe.

Dry-run mode is the default and never touches the network. Live mode uses the
OpenAI-compatible DeepSeek endpoint with an API key read from an environment
variable, but never prints the key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, cast

MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
PRICE_PER_1M = {
    "input_cache_hit": 0.0028,
    "input_cache_miss": 0.14,
    "output": 0.28,
}
DOCS = {
    "context_caching": "https://api-docs.deepseek.com/guides/kv_cache",
    "pricing": "https://api-docs.deepseek.com/quick_start/pricing",
}

SYSTEM_PROMPT = "You are a precise cache-baseline probe. Answer in one short sentence."
TAIL_ASKS = [
    "Return the three section ids that discuss cache persistence.",
    "Return the two usage field names that report cache hit status.",
    "Return the pricing ratio between cache-miss and cache-hit input for flash.",
]


@dataclass
class ProbeRow:
    request_index: int
    mode: str
    model: str
    prompt_tokens: int
    prompt_cache_hit_tokens: int
    prompt_cache_miss_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_hit_rate: float
    estimated_usd: float
    usage_fields_present: bool
    elapsed_seconds: float | None = None
    response_preview: str | None = None


def stable_prefix(repetitions: int = 96) -> str:
    """Return a byte-stable prefix large enough to make cache behavior visible."""
    base = [
        "DEEPSEEK CACHE BASELINE PREFIX v1",
        "Source: official DeepSeek context caching and pricing docs.",
        "Invariant: this prefix must remain byte-identical across all probe requests.",
        "The variable question is appended after CACHE_PROBE_TAIL_START.",
        "DeepSeek documents cache hits for fully matched persisted cache prefix units.",
        "DeepSeek exposes usage.prompt_cache_hit_tokens and usage.prompt_cache_miss_tokens.",
        "deepseek-v4-flash pricing snapshot: hit=$0.0028/M, miss=$0.14/M, output=$0.28/M.",
    ]
    for i in range(repetitions):
        base.append(
            f"stable-record-{i:03d}: prompt-cache engineering keeps identity, rules, "
            "references, and examples before volatile task asks."
        )
    return "\n".join(base)


def build_messages(tail_ask: str, prefix: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{prefix}\n\nCACHE_PROBE_TAIL_START\n{tail_ask}"},
    ]


def _usage_value(usage: Any, name: str, default: int = 0) -> int:
    if isinstance(usage, dict):
        value = usage.get(name, default)
    else:
        value = getattr(usage, name, default)
    return int(value or 0)


def estimate_usd(hit_tokens: int, miss_tokens: int, completion_tokens: int) -> float:
    return round(
        (
            hit_tokens * PRICE_PER_1M["input_cache_hit"]
            + miss_tokens * PRICE_PER_1M["input_cache_miss"]
            + completion_tokens * PRICE_PER_1M["output"]
        )
        / 1_000_000,
        10,
    )


def row_from_usage(
    *,
    request_index: int,
    mode: str,
    model: str,
    usage: Any,
    elapsed_seconds: float | None = None,
    response_preview: str | None = None,
) -> ProbeRow:
    prompt_tokens = _usage_value(usage, "prompt_tokens")
    hit_tokens = _usage_value(usage, "prompt_cache_hit_tokens")
    miss_tokens = _usage_value(usage, "prompt_cache_miss_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens", prompt_tokens + completion_tokens)
    usage_fields_present = _has_usage_field(usage, "prompt_cache_hit_tokens") and _has_usage_field(
        usage, "prompt_cache_miss_tokens"
    )
    denom = hit_tokens + miss_tokens
    cache_hit_rate = round(hit_tokens / denom, 6) if denom else 0.0
    return ProbeRow(
        request_index=request_index,
        mode=mode,
        model=model,
        prompt_tokens=prompt_tokens,
        prompt_cache_hit_tokens=hit_tokens,
        prompt_cache_miss_tokens=miss_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cache_hit_rate=cache_hit_rate,
        estimated_usd=estimate_usd(hit_tokens, miss_tokens, completion_tokens),
        usage_fields_present=usage_fields_present,
        elapsed_seconds=round(elapsed_seconds, 3) if elapsed_seconds is not None else None,
        response_preview=response_preview,
    )


def _has_usage_field(usage: Any, name: str) -> bool:
    if isinstance(usage, dict):
        return name in usage
    return hasattr(usage, name)


def fake_usages() -> Iterable[dict[str, int]]:
    """Deterministic dry-run usages shaped like DeepSeek usage objects."""
    yield {
        "prompt_tokens": 1680,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 1680,
        "completion_tokens": 18,
        "total_tokens": 1698,
    }
    yield {
        "prompt_tokens": 1684,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 1684,
        "completion_tokens": 20,
        "total_tokens": 1704,
    }
    yield {
        "prompt_tokens": 1688,
        "prompt_cache_hit_tokens": 1500,
        "prompt_cache_miss_tokens": 188,
        "completion_tokens": 19,
        "total_tokens": 1707,
    }


def run_dry(model: str) -> list[ProbeRow]:
    return [
        row_from_usage(request_index=i, mode="dry_run", model=model, usage=usage)
        for i, usage in enumerate(fake_usages(), start=1)
    ]


def run_live(*, model: str, base_url: str, api_key_env: str, delay_seconds: float) -> list[ProbeRow]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key env var: {api_key_env}")

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - core dependency in Hermes, safety for direct copies
        raise RuntimeError("openai package is required for --live") from exc

    client = OpenAI(api_key=api_key, base_url=base_url)
    prefix = stable_prefix()
    rows: list[ProbeRow] = []
    for i, tail_ask in enumerate(TAIL_ASKS, start=1):
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=cast(Any, build_messages(tail_ask, prefix)),
            temperature=0,
            max_tokens=64,
        )
        elapsed = time.perf_counter() - start
        usage = response.usage
        content = response.choices[0].message.content if response.choices else ""
        rows.append(
            row_from_usage(
                request_index=i,
                mode="live",
                model=model,
                usage=usage,
                elapsed_seconds=elapsed,
                response_preview=(content or "")[:160],
            )
        )
        if delay_seconds and i < len(TAIL_ASKS):
            time.sleep(delay_seconds)
    return rows


def summarize(rows: list[ProbeRow]) -> dict[str, Any]:
    total_hit = sum(row.prompt_cache_hit_tokens for row in rows)
    total_miss = sum(row.prompt_cache_miss_tokens for row in rows)
    denom = total_hit + total_miss
    return {
        "requests": len(rows),
        "prompt_cache_hit_tokens": total_hit,
        "prompt_cache_miss_tokens": total_miss,
        "cache_hit_rate": round(total_hit / denom, 6) if denom else 0.0,
        "prompt_tokens": sum(row.prompt_tokens for row in rows),
        "completion_tokens": sum(row.completion_tokens for row in rows),
        "total_tokens": sum(row.total_tokens for row in rows),
        "estimated_usd": round(sum(row.estimated_usd for row in rows), 10),
        "all_usage_fields_present": all(row.usage_fields_present for row in rows),
    }


def build_output(*, rows: list[ProbeRow], args: argparse.Namespace, credential_available: bool) -> dict[str, Any]:
    return {
        "probe": "deepseek_cache_baseline",
        "schema_version": 1,
        "mode": "live" if args.live else "dry_run",
        "model": args.model,
        "base_url": args.base_url,
        "credential_env": args.api_key_env,
        "credential_available": credential_available,
        "docs": DOCS,
        "prices_per_1m_usd": PRICE_PER_1M,
        "prefix_sha256": _sha256_text(stable_prefix()),
        "variable_tail_count": len(TAIL_ASKS),
        "summary": summarize(rows),
        "rows": [asdict(row) for row in rows],
    }


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="call the DeepSeek API; default is deterministic dry-run")
    parser.add_argument("--model", default=MODEL, help=f"DeepSeek model name (default: {MODEL})")
    parser.add_argument("--base-url", default=BASE_URL, help=f"OpenAI-compatible base URL (default: {BASE_URL})")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY", help="environment variable holding the API key")
    parser.add_argument("--delay-seconds", type=float, default=3.0, help="delay between live requests so cache persistence can settle")
    parser.add_argument("--output", type=Path, help="write JSON output to this path instead of stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    credential_available = bool(os.environ.get(args.api_key_env))
    try:
        rows = (
            run_live(
                model=args.model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                delay_seconds=args.delay_seconds,
            )
            if args.live
            else run_dry(args.model)
        )
        output = build_output(rows=rows, args=args, credential_available=credential_available)
        exit_code = 0
    except Exception as exc:
        output = {
            "probe": "deepseek_cache_baseline",
            "schema_version": 1,
            "mode": "live" if args.live else "dry_run",
            "model": args.model,
            "credential_env": args.api_key_env,
            "credential_available": credential_available,
            "error": str(exc),
        }
        exit_code = 2

    text = json.dumps(output, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
