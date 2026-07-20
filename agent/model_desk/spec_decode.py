"""L08 — Speculative / draft model pair advice with live backend detection."""

from __future__ import annotations

import shutil
from typing import Any, Dict, List, Optional


def _detect_backends() -> List[Dict[str, Any]]:
    """Detect local inference backends that may support speculative decoding."""
    found: List[Dict[str, Any]] = []

    # llama.cpp / llama-server
    for name in ("llama-server", "llama-cli", "llama.cpp"):
        path = shutil.which(name)
        if path:
            found.append(
                {
                    "backend": "llama.cpp",
                    "binary": name,
                    "path": path,
                    "spec_decode": "supported (--model-draft / --draft when built with it)",
                }
            )
            break

    # ollama
    ollama = shutil.which("ollama")
    if ollama:
        found.append(
            {
                "backend": "ollama",
                "binary": "ollama",
                "path": ollama,
                "spec_decode": "experimental / model-dependent",
            }
        )

    # vllm
    vllm = shutil.which("vllm")
    if vllm:
        found.append(
            {
                "backend": "vllm",
                "binary": "vllm",
                "path": vllm,
                "spec_decode": "supported (speculative_config)",
            }
        )

    return found


def speculative_decode_advice(main_model: str | None = None) -> dict[str, Any]:
    """Advise on draft/speculative decoding pairs for a main model."""
    main = str(main_model or "").strip() or "(main)"
    backends = _detect_backends()

    suggestion: Dict[str, Any] = {
        "draft": "small instruct GGUF (1–3B) on same llama-server --draft if supported",
        "note": "Only enable when fork supports speculative decoding; verify with doctor",
    }
    if any(b.get("backend") == "llama.cpp" for b in backends):
        suggestion = {
            "draft": f"Pair {main} with a 1–3B instruct draft GGUF on the same llama-server",
            "flags": "--model-draft <draft.gguf> --draft-max 16",
            "note": "Requires llama-server build with speculative decoding enabled",
        }
    elif any(b.get("backend") == "vllm" for b in backends):
        suggestion = {
            "draft": f"Use vLLM speculative_config with a smaller draft for {main}",
            "note": "See vLLM docs for speculative_config.method / num_speculative_tokens",
        }
    elif any(b.get("backend") == "ollama" for b in backends):
        suggestion = {
            "draft": "Ollama speculative decoding is experimental — prefer llama-server for draft pairs",
            "note": "Keep main model on ollama; use desk serve-plan for llama.cpp draft setup",
        }

    return {
        "ok": True,
        "main_model": main,
        "backends_detected": backends,
        "suggestion": suggestion,
        "live_detection": bool(backends),
    }
