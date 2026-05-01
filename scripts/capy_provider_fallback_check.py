#!/usr/bin/env python3
"""Non-destructive provider fallback readiness check for Capy."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


def _http_ok(url: str, timeout: float = 1.0) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "capy-provider-check/1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local health check
            return 200 <= getattr(resp, "status", 0) < 300
    except Exception:
        return False


def _config_model() -> dict[str, str]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        if isinstance(model_cfg, dict):
            return {
                "provider": str(model_cfg.get("provider") or ""),
                "model": str(model_cfg.get("default") or model_cfg.get("model") or ""),
            }
        if isinstance(model_cfg, str):
            return {"provider": "", "model": model_cfg}
    except Exception as exc:
        return {"provider": "", "model": "", "error": str(exc)}
    return {"provider": "", "model": ""}


def collect() -> dict[str, Any]:
    model = _config_model()
    return {
        "primary": model,
        "primary_expected": {
            "provider": "openai-codex",
            "model": "gpt-5.5",
            "matches": model.get("provider") == "openai-codex" and model.get("model") == "gpt-5.5",
        },
        "lmstudio": {
            "base_url": "http://127.0.0.1:1234/v1",
            "models_endpoint_ok": _http_ok("http://127.0.0.1:1234/v1/models"),
            "profile_alias": "lmstudio",
            "known_model": "qwen3.6-35b-a3b-ud-mlx",
        },
        "optional_credentials": {
            name: bool((os.environ.get(name) or "").strip())
            for name in ("OPENROUTER_API_KEY", "NOUS_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
        },
        "note": "Credential values are never printed. This checker does not change config.",
    }


def main() -> int:
    data = collect()
    print(json.dumps(data, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
