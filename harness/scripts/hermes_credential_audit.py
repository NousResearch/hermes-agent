#!/usr/bin/env python3
"""Audita credential pool e auth providers sem expor tokens.

Somente leitura. Saída JSON no stdout — validada antes de emitir.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Fields never emitted in output (values stripped at source).
_FORBIDDEN_KEYS = frozenset({
    "access_token",
    "refresh_token",
    "agent_key",
    "token",
    "api_key",
    "secret",
    "password",
    "client_secret",
    "private_key",
    "id_token",
    "device_code",
    "code_verifier",
})

# Substrings that must not appear in serialized output.
_FORBIDDEN_OUTPUT_PATTERNS = [
    re.compile(r"\bsk-[a-zA-Z0-9]{8,}"),
    re.compile(r"\bghp_[a-zA-Z0-9]{20,}"),
    re.compile(r"\beyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9._-]+"),
    re.compile(r'"(access_token|refresh_token|agent_key)"\s*:\s*"[^"]{4,}"'),
]

_ENV_KEY_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")


def _default_hermes_home() -> Path:
    env = os.environ.get("HERMES_HOME", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / "AppData" / "Local" / "hermes"


def _profiles_root() -> Path:
    return Path.home() / "AppData" / "Local" / "hermes" / "profiles"


def _emit_json(payload: dict) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    _validate_output(text)
    try:
        sys.stdout.buffer.write(text.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()
    except (AttributeError, OSError):
        print(json.dumps(payload, indent=2, ensure_ascii=True))


def _validate_output(text: str) -> None:
    for pattern in _FORBIDDEN_OUTPUT_PATTERNS:
        if pattern.search(text):
            raise RuntimeError("audit output failed secret validation — aborting")


def _has_secret_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (int, float, bool)):
        return bool(value)
    return bool(value)


def _audit_pool_entry(provider: str, entry: dict[str, Any]) -> dict[str, Any]:
    access = entry.get("access_token")
    agent = entry.get("agent_key")
    refresh = entry.get("refresh_token")
    return {
        "provider": provider,
        "id": entry.get("id"),
        "label": entry.get("label"),
        "source": entry.get("source"),
        "auth_type": entry.get("auth_type"),
        "priority": entry.get("priority"),
        "has_token": _has_secret_value(access) or _has_secret_value(agent),
        "has_refresh_token": _has_secret_value(refresh),
        "last_status": entry.get("last_status"),
        "last_error_code": entry.get("last_error_code"),
        "last_error_reason": entry.get("last_error_reason"),
        "base_url": entry.get("base_url") or entry.get("inference_base_url"),
    }


def _audit_oauth_provider(provider_id: str, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": provider_id,
        "kind": "oauth_singleton",
        "source": state.get("source") or state.get("auth_path") or state.get("device_code_source"),
        "has_token": _has_secret_value(state.get("access_token")) or _has_secret_value(state.get("agent_key")),
        "has_refresh_token": _has_secret_value(state.get("refresh_token")),
        "logged_in": _has_secret_value(state.get("access_token"))
        or _has_secret_value(state.get("refresh_token"))
        or _has_secret_value(state.get("agent_key")),
        "expires_at": state.get("expires_at") or state.get("agent_key_expires_at"),
    }


def _load_auth_store(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _audit_auth_file(auth_path: Path, *, provider_filter: str | None) -> dict[str, Any]:
    store = _load_auth_store(auth_path)
    if store is None:
        return {
            "auth_path": str(auth_path),
            "exists": auth_path.is_file(),
            "entries": [],
            "active_provider": None,
            "pool_providers": [],
            "oauth_providers": [],
        }

    entries: list[dict[str, Any]] = []
    pool = store.get("credential_pool")
    if isinstance(pool, dict):
        for pool_key, raw_entries in pool.items():
            if provider_filter and pool_key != provider_filter:
                continue
            if not isinstance(raw_entries, list):
                continue
            for item in raw_entries:
                if isinstance(item, dict):
                    entries.append(_audit_pool_entry(str(pool_key), item))

    oauth_providers: list[dict[str, Any]] = []
    providers = store.get("providers")
    if isinstance(providers, dict):
        for pid, state in providers.items():
            if provider_filter and pid != provider_filter:
                continue
            if isinstance(state, dict) and state:
                oauth_providers.append(_audit_oauth_provider(str(pid), state))

    return {
        "auth_path": str(auth_path),
        "exists": True,
        "active_provider": store.get("active_provider"),
        "entries": entries,
        "pool_providers": sorted({e["provider"] for e in entries}),
        "oauth_providers": oauth_providers,
    }


def _audit_env_keys(env_path: Path, *, provider_filter: str | None) -> list[dict[str, Any]]:
    """Report env var names present in .env — never values."""
    if not env_path.is_file():
        return []

    # Common provider env hints when filtering.
    filter_hints: dict[str, tuple[str, ...]] = {
        "openrouter": ("OPENROUTER_API_KEY",),
        "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "openai-codex": ("OPENAI_API_KEY",),
        "deepseek": ("DEEPSEEK_API_KEY",),
        "xai": ("XAI_API_KEY",),
        "xai-oauth": ("XAI_API_KEY",),
    }

    allowed: set[str] | None = None
    if provider_filter:
        allowed = set(filter_hints.get(provider_filter, ()))

    found: list[dict[str, Any]] = []
    try:
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _ENV_KEY_LINE.match(line)
            if not m:
                continue
            key = m.group(1)
            if allowed is not None and key not in allowed:
                continue
            if key.upper().endswith(("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")) or "API" in key.upper():
                found.append({"env_var": key, "present": True, "source": ".env"})
    except OSError:
        return []
    return found


def _discover_scopes(all_profiles: bool) -> list[tuple[str, Path]]:
    scopes: list[tuple[str, Path]] = [("default", _default_hermes_home())]
    if not all_profiles:
        return scopes

    root = _profiles_root()
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "auth.json").exists():
                scopes.append((child.name, child))
    return scopes


def main() -> int:
    parser = argparse.ArgumentParser(description="Hermes credential audit (no secrets)")
    parser.add_argument("--provider", default="", help="Filter by provider id (optional)")
    parser.add_argument(
        "--hermes-home",
        default="",
        help="HERMES_HOME override (default scope only)",
    )
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Audit default HERMES_HOME plus every profile with auth.json",
    )
    args = parser.parse_args()

    provider_filter = args.provider.strip().lower() or None
    scopes = _discover_scopes(args.all_profiles)
    if args.hermes_home:
        scopes = [("custom", Path(args.hermes_home).expanduser())]

    results: list[dict[str, Any]] = []
    total_entries = 0

    for scope_name, home in scopes:
        auth_path = home / "auth.json"
        env_path = home / ".env"
        audit = _audit_auth_file(auth_path, provider_filter=provider_filter)
        env_keys = _audit_env_keys(env_path, provider_filter=provider_filter)
        total_entries += len(audit["entries"]) + len(audit["oauth_providers"])
        results.append(
            {
                "scope": scope_name,
                "hermes_home": str(home),
                **audit,
                "env_keys": env_keys,
            }
        )

    ok = any(r.get("exists") for r in results) or any(r.get("env_keys") for r in results)
    out = {
        "ok": ok,
        "provider_filter": provider_filter,
        "scopes": results,
        "summary": {
            "scope_count": len(results),
            "pool_entry_count": sum(len(r["entries"]) for r in results),
            "oauth_provider_count": sum(len(r["oauth_providers"]) for r in results),
            "env_key_count": sum(len(r["env_keys"]) for r in results),
        },
    }
    _emit_json(out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
