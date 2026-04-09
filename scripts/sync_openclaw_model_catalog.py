#!/usr/bin/env python3
"""Sync OpenClaw model catalog into ~/.hermes/models.json.

CATALOG ONLY.
This script updates only the Hermes model catalog file used for picker/display.
It NEVER changes the active model, session override, or default model in either
Hermes or OpenClaw. It does NOT modify ~/.hermes/config.yaml model settings.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"
HERMES_MODELS = Path.home() / ".hermes" / "models.json"

DEFAULT_PROVIDER_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "zai": "https://api.z.ai/api/coding/paas/v4",
    "ollama": "http://localhost:11434/v1",
    "openai-codex": "https://chatgpt.com/backend-api/codex",
    "opencode-go": "https://opencode.ai/zen/go/v1",
    "opencode": "https://opencode.ai/zen/v1",
    "opencode-zen": "https://opencode.ai/zen/v1",
}

DEFAULT_PROVIDER_API_MODES: dict[str, str] = {
    "openai-codex": "responses",
    "opencode-go": "chat_completions",
    "opencode": "chat_completions",
    "opencode-zen": "chat_completions",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_api_mode(provider_cfg: dict[str, Any] | None) -> str:
    api = str((provider_cfg or {}).get("api") or "openai-completions").strip().lower()
    if api in {"openai-completions", "chat_completions", "chat-completions"}:
        return "chat_completions"
    if api in {"responses", "openai-responses"}:
        return "responses"
    if api in {"anthropic", "anthropic-messages", "messages", "anthropic_messages"}:
        return "messages"
    return api.replace("-", "_") or "chat_completions"


def build_alias_map(cfg: dict[str, Any]) -> dict[str, str]:
    raw = (((cfg.get("agents") or {}).get("defaults") or {}).get("models") or {})
    aliases: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            alias = value.get("alias")
            if isinstance(alias, str) and alias.strip():
                aliases[key] = alias.strip()
    return aliases


def build_existing_index(entries: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if provider and model:
            out[(provider, model)] = entry
    return out


def build_alias_only_provider_models(cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    raw = (((cfg.get("agents") or {}).get("defaults") or {}).get("models") or {})
    out: dict[str, list[dict[str, Any]]] = {}
    for fqid in raw.keys():
        if not isinstance(fqid, str) or "/" not in fqid:
            continue
        provider, model_id = fqid.split("/", 1)
        provider = provider.strip()
        model_id = model_id.strip()
        if not provider or not model_id:
            continue
        out.setdefault(provider, []).append({"id": model_id})

    deduped: dict[str, list[dict[str, Any]]] = {}
    for provider, models in out.items():
        seen: set[str] = set()
        deduped[provider] = []
        for model_cfg in models:
            model_id = str(model_cfg.get("id") or "").strip()
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            deduped[provider].append(model_cfg)
    return deduped


def main() -> int:
    if not OPENCLAW_CONFIG.exists():
        raise SystemExit(f"OpenClaw config not found: {OPENCLAW_CONFIG}")

    cfg = load_json(OPENCLAW_CONFIG)
    aliases = build_alias_map(cfg)
    providers = dict(((cfg.get("models") or {}).get("providers") or {}))

    alias_only_providers = build_alias_only_provider_models(cfg)
    for provider_slug, models in alias_only_providers.items():
        if provider_slug in providers:
            continue
        providers[provider_slug] = {
            "baseUrl": DEFAULT_PROVIDER_BASE_URLS.get(provider_slug, ""),
            "api": DEFAULT_PROVIDER_API_MODES.get(provider_slug, "chat_completions"),
            "models": models,
        }

    existing_entries: list[dict[str, Any]] = []
    if HERMES_MODELS.exists():
        try:
            loaded = load_json(HERMES_MODELS)
            if isinstance(loaded, list):
                existing_entries = loaded
        except Exception:
            existing_entries = []
    existing_index = build_existing_index(existing_entries)

    now_ms = int(time.time() * 1000)
    result: list[dict[str, Any]] = []

    for provider_slug, provider_cfg in providers.items():
        if not isinstance(provider_cfg, dict):
            continue
        base_url = provider_cfg.get("baseUrl")
        api_mode = normalize_api_mode(provider_cfg)
        models = provider_cfg.get("models") or []
        if not isinstance(models, list):
            continue

        for model_cfg in models:
            if not isinstance(model_cfg, dict):
                continue
            model_id = str(model_cfg.get("id") or "").strip()
            if not model_id:
                continue

            fqid = f"{provider_slug}/{model_id}"
            existing = existing_index.get((provider_slug, model_id), {})

            entry = {
                "id": existing.get("id") or str(uuid.uuid4()),
                "name": aliases.get(fqid) or model_cfg.get("name") or fqid,
                "provider": provider_slug,
                "model": model_id,
                "baseUrl": base_url or existing.get("baseUrl") or "",
                "api_mode": api_mode,
                "contextWindow": model_cfg.get("contextWindow"),
                "maxTokens": model_cfg.get("maxTokens"),
                "reasoning": bool(model_cfg.get("reasoning", False)),
                "input": model_cfg.get("input") or ["text"],
                "cost": model_cfg.get("cost") or {"input": 0, "output": 0},
                "createdAt": existing.get("createdAt") or now_ms,
            }
            result.append(entry)

    result.sort(key=lambda item: (str(item.get("provider") or ""), str(item.get("name") or ""), str(item.get("model") or "")))

    HERMES_MODELS.parent.mkdir(parents=True, exist_ok=True)
    with HERMES_MODELS.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    providers_count = len({entry["provider"] for entry in result})
    print(f"Synced {len(result)} models across {providers_count} providers to {HERMES_MODELS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
