"""C06 — Streaming / tool-call parity matrix with live capability probes."""

from __future__ import annotations

from typing import Any, Dict, Optional

_MATRIX: Dict[str, Dict[str, Any]] = {
    "openrouter": {"stream": True, "tools": True, "vision": "model-dependent"},
    "anthropic": {"stream": True, "tools": True, "vision": True},
    "openai": {"stream": True, "tools": True, "vision": True},
    "openai-codex": {"stream": True, "tools": True, "vision": False},
    "gemini": {"stream": True, "tools": True, "vision": True},
    "deepseek": {"stream": True, "tools": True, "vision": False},
    "custom": {
        "stream": "server-dependent",
        "tools": "server-dependent",
        "vision": "server-dependent",
    },
    "lmstudio": {
        "stream": True,
        "tools": "model-dependent",
        "vision": "model-dependent",
    },
    "ollama": {
        "stream": True,
        "tools": "model-dependent",
        "vision": "model-dependent",
    },
    "local-gguf": {"stream": True, "tools": "prompt-dependent", "vision": False},
    "ravenx-local": {"stream": True, "tools": "prompt-dependent", "vision": False},
}


def _probe_local_openai(base_url: str, timeout: float = 1.5) -> Dict[str, Any]:
    """Soft probe of a local OpenAI-compatible /v1/models endpoint."""
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        # Accept either http://host:port or .../v1
        if "/v1" not in url:
            url = url + "/v1"
    models_url = url.rstrip("/") + "/models"
    try:
        req = urllib.request.Request(models_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(2048)
            return {
                "reachable": True,
                "status": getattr(resp, "status", 200),
                "bytes": len(body),
            }
    except Exception as exc:
        return {"reachable": False, "error": str(exc)[:160]}


def parity_matrix(
    provider: str | None = None,
    *,
    probe: bool = False,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Return static parity knowledge, optionally enriched with a live probe."""
    if provider:
        key = str(provider).strip().lower()
        row = dict(_MATRIX.get(key) or _MATRIX.get("custom") or {})
        result: Dict[str, Any] = {"ok": True, "provider": key, "parity": row}
        if probe:
            url = (base_url or "").strip()
            if not url and key in ("ollama", "lmstudio", "local-gguf", "custom"):
                # Common local defaults
                defaults = {
                    "ollama": "http://127.0.0.1:11434/v1",
                    "lmstudio": "http://127.0.0.1:1234/v1",
                    "local-gguf": "http://127.0.0.1:8080/v1",
                    "custom": "http://127.0.0.1:8080/v1",
                }
                url = defaults.get(key, "")
            if url:
                result["probe"] = _probe_local_openai(url)
        return result
    return {"ok": True, "matrix": {k: dict(v) for k, v in _MATRIX.items()}}
