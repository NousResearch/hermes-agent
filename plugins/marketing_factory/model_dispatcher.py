"""Model router for Marketing Agent Factory.

Implements the cheap/mid/premium routing declared in `state.json.model_routing_policy`:

  cheap   → local Ollama qwen2.5:14b  (research, classification, dedup, safety review)
  mid     → local Ollama qwen3:30b    (rewrites, channel variants, mid-value copy)
  premium → Claude CLI (OAuth)         (strategy, final review, high-value copy)

All routes return the same envelope so callers don't branch on provider:

    {
        "text": str | None,         # response text (None if everything failed)
        "tokens_used": int,         # prompt + completion tokens (estimated for premium)
        "model": str,               # actual model used
        "route": str,               # "cheap" | "mid" | "premium"
        "fallback_used": bool,      # True if route call failed
        "error": str | None,        # error string if fallback_used
        "elapsed_ms": int,
    }

Callers MUST be prepared for `fallback_used=True` and fall back to deterministic
templates — this keeps dry-run pipelines unbreakable when models are unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT_SEC = float(os.environ.get("MF_OLLAMA_TIMEOUT", "120"))
CLAUDE_TIMEOUT_SEC = float(os.environ.get("MF_CLAUDE_TIMEOUT", "120"))

ROUTE_MODELS: Dict[str, str] = {
    "cheap": os.environ.get("MF_CHEAP_MODEL", "qwen2.5:14b"),
    # `mid` defaults to qwen2.5:14b on M1 Pro 32GB — qwen3:30b technically loads
    # but generation is so slow (KV cache + weights exceed available RAM, swap kills it)
    # that timeouts dominate. Override via `MF_MID_MODEL=qwen3:30b` on bigger hardware.
    "mid": os.environ.get("MF_MID_MODEL", "qwen2.5:14b"),
    "premium": os.environ.get("MF_PREMIUM_MODEL", "claude"),
}


def dispatch(
    route: str,
    prompt: str,
    *,
    system: Optional[str] = None,
    json_mode: bool = False,
    json_schema: Optional[Dict[str, Any]] = None,
    max_tokens: int = 1500,
    temperature: float = 0.4,
) -> Dict[str, Any]:
    """Route a single prompt to the cheap/mid/premium tier and return a normalized envelope."""
    if route not in ROUTE_MODELS:
        raise ValueError(f"Unknown route: {route!r} (expected cheap|mid|premium)")

    start = time.monotonic()
    try:
        if route == "premium":
            envelope = _call_claude_cli(
                prompt,
                system=system,
                json_mode=json_mode,
                json_schema=json_schema,
                max_tokens=max_tokens,
            )
        else:
            envelope = _call_ollama(
                model=ROUTE_MODELS[route],
                prompt=prompt,
                system=system,
                json_mode=json_mode,
                max_tokens=max_tokens,
                temperature=temperature,
            )
    except Exception as exc:  # noqa: BLE001 — caller handles fallback
        logger.warning("marketing_factory.dispatch route=%s failed: %s", route, exc)
        envelope = {
            "text": None,
            "tokens_used": 0,
            "model": ROUTE_MODELS[route],
            "fallback_used": True,
            "error": str(exc),
        }
    envelope["route"] = route
    envelope.setdefault("fallback_used", False)
    envelope.setdefault("error", None)
    envelope["elapsed_ms"] = int((time.monotonic() - start) * 1000)
    return envelope


def dispatch_json(
    route: str,
    prompt: str,
    *,
    system: Optional[str] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    max_tokens: int = 1500,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """Convenience wrapper that requests JSON-mode and parses the response.

    On parse failure, returns an envelope with `parsed=None, fallback_used=True`.
    """
    env = dispatch(
        route,
        prompt,
        system=system,
        json_mode=True,
        json_schema=json_schema,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = env.get("text") or ""
    parsed: Any = None
    if text:
        try:
            parsed = json.loads(_extract_json_blob(text))
        except (ValueError, TypeError) as exc:
            logger.info("dispatch_json parse failed route=%s err=%s text=%s...", route, exc, text[:200])
            env["fallback_used"] = True
            env["error"] = f"json parse failed: {exc}"
    env["parsed"] = parsed
    return env


def _call_ollama(
    *,
    model: str,
    prompt: str,
    system: Optional[str],
    json_mode: bool,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if json_mode:
        payload["format"] = "json"

    with httpx.Client(timeout=OLLAMA_TIMEOUT_SEC) as client:
        response = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

    text = (data.get("message") or {}).get("content", "") or ""
    tokens_used = int(data.get("prompt_eval_count") or 0) + int(data.get("eval_count") or 0)
    return {
        "text": text.strip() or None,
        "tokens_used": tokens_used,
        "model": data.get("model") or model,
        "fallback_used": not bool(text.strip()),
        "error": None if text.strip() else "ollama returned empty content",
    }


def _call_claude_cli(
    prompt: str,
    *,
    system: Optional[str],
    json_mode: bool,
    json_schema: Optional[Dict[str, Any]],
    max_tokens: int,
) -> Dict[str, Any]:
    cli = shutil.which("claude")
    if not cli:
        raise RuntimeError("claude CLI not on PATH")

    combined = prompt if not system else f"<system>\n{system}\n</system>\n\n{prompt}"
    args: List[str] = [cli, "-p", "--output-format", "text"]
    if json_schema:
        args.extend(["--json-schema", json.dumps(json_schema)])
    elif json_mode:
        args.extend(
            [
                "--json-schema",
                json.dumps(
                    {"type": "object", "additionalProperties": True},
                    sort_keys=True,
                ),
            ]
        )

    proc = subprocess.run(
        args,
        input=combined,
        capture_output=True,
        text=True,
        timeout=CLAUDE_TIMEOUT_SEC,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p exit={proc.returncode} stderr={proc.stderr.strip()[:300]}")
    text = (proc.stdout or "").strip()
    if not text:
        raise RuntimeError("claude -p returned empty stdout")
    estimated_tokens = max(1, (len(combined) + len(text)) // 4)
    return {
        "text": text,
        "tokens_used": estimated_tokens,
        "model": ROUTE_MODELS["premium"],
        "fallback_used": False,
        "error": None,
    }


def _extract_json_blob(text: str) -> str:
    """Best-effort: lift the first {...} or [...] blob out of a model response.

    Models often wrap JSON in prose or markdown fences even when asked not to.
    """
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[: -3]
    text = text.strip()
    start = -1
    for idx, ch in enumerate(text):
        if ch in "{[":
            start = idx
            break
    if start == -1:
        return text
    return text[start:]
