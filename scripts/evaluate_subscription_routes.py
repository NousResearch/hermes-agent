#!/usr/bin/env python
"""Negligible-quota subscription-route evaluation harness.

Default mode performs read-only installation/auth/model/quota probes and makes
no model calls.  ``--smoke`` is opt-in and runs exactly one tiny, tool-disabled
request; it never performs a broad comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


POOL_ORDER = (
    "ollama-cloud",
    "openai-subscription",
    "google-antigravity",
    "anthropic-subscription",
)


def _command(command: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        encoding="utf-8",
        errors="replace",
    )


def _usage(provider: str) -> dict[str, Any]:
    from agent.account_usage import fetch_account_usage

    snapshot = fetch_account_usage(provider)
    if snapshot is None:
        return {"visible": False, "state": "unknown", "windows": []}
    windows = [
        {"label": item.label, "used_percent": float(item.used_percent)}
        for item in snapshot.windows
    ]
    used = max((item["used_percent"] for item in windows), default=None)
    return {
        "visible": snapshot.available and used is not None,
        "state": "exhausted" if used is not None and used >= 100.0 else "available",
        "plan": snapshot.plan,
        "source": snapshot.source,
        "windows": windows,
        "unavailable_reason": snapshot.unavailable_reason,
    }


def _openai_paid_usage_possible() -> bool | None:
    try:
        import httpx
        from agent.account_usage import _resolve_codex_usage_credentials, _resolve_codex_usage_url

        token, base_url, account_id = _resolve_codex_usage_credentials(None, None)
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "codex-cli",
        }
        if account_id:
            headers["ChatGPT-Account-Id"] = account_id
        response = httpx.get(_resolve_codex_usage_url(base_url), headers=headers, timeout=15.0)
        response.raise_for_status()
        credits = (response.json() or {}).get("credits") or {}
        return bool(credits.get("has_credits"))
    except Exception:
        return None


def _anthropic_paid_usage_possible() -> bool | None:
    try:
        import httpx
        from agent.anthropic_adapter import resolve_anthropic_token

        token = (resolve_anthropic_token() or "").strip()
        response = httpx.get(
            "https://api.anthropic.com/api/oauth/usage",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "anthropic-beta": "oauth-2025-04-20",
                "User-Agent": "claude-code/2.1.0",
            },
            timeout=15.0,
        )
        response.raise_for_status()
        return bool(((response.json() or {}).get("extra_usage") or {}).get("is_enabled"))
    except Exception:
        return None


def _configured_ollama_subscription_pool() -> dict[str, Any]:
    """Read the configured API-key pool association without resolving secrets."""
    from hermes_cli.config import load_config

    config = load_config()
    routing = config.get("task_routing") if isinstance(config.get("task_routing"), dict) else {}
    pools = routing.get("pools") if isinstance(routing.get("pools"), list) else []
    for pool in pools:
        if not isinstance(pool, dict):
            continue
        if str(pool.get("provider") or "").strip().lower() not in {"ollama", "ollama-cloud"}:
            continue
        if pool.get("enabled", True) is not True:
            continue
        if str(pool.get("billing_mode") or "").strip().lower() != "subscription":
            continue
        if str(pool.get("subscription_pool") or "").strip():
            return dict(pool)
    return {}


def _ollama_probe() -> dict[str, Any]:
    from agent.subscription_pool import is_approved_ollama_cloud_endpoint
    from hermes_cli.models import fetch_api_models_strict
    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(requested="ollama-cloud")
    base_url = str(runtime.get("base_url") or "")
    api_key = runtime.get("api_key")
    pool = _configured_ollama_subscription_pool()
    subscription_pool = str(pool.get("subscription_pool") or "").strip()
    cloud_verified = is_approved_ollama_cloud_endpoint(base_url)
    # models.dev is discovery data, not proof that this API key authenticated
    # or that a model is available on the configured Cloud subscription.
    live_models = fetch_api_models_strict(api_key, base_url) if api_key and cloud_verified else None
    authenticated = bool(api_key and live_models is not None)
    models = live_models or []
    cloud_model_verified = bool(models)
    paid_usage_possible = pool.get("paid_usage_possible")
    eligible = all((
        authenticated,
        cloud_verified,
        cloud_model_verified,
        bool(subscription_pool),
        paid_usage_possible is False,
    ))
    blocked: list[str] = []
    if not api_key:
        blocked.append("Ollama Cloud API key is not configured")
    elif cloud_verified and live_models is None:
        blocked.append("Ollama Cloud authentication or model-roster probe failed")
    if not cloud_verified:
        blocked.append("endpoint is not an approved Ollama Cloud endpoint")
    if not cloud_model_verified:
        blocked.append("Cloud model roster is unavailable")
    if not subscription_pool:
        blocked.append("configured API key has no explicit subscription-pool association")
    if paid_usage_possible is not False:
        blocked.append("paid overage must be explicitly disabled")
    return {
        "pool": "ollama-cloud",
        "provider": "ollama-cloud",
        "execution_surface": "hermes",
        "installed": True,
        "authenticated": authenticated,
        "headless": True,
        "workspace": True,
        "tools": True,
        "cloud_verified": cloud_verified,
        "cloud_model_verified": cloud_model_verified,
        "model_roster": models,
        "model_roster_source": "authenticated live /v1/models",
        "quota": {"visible": False, "state": "unknown"},
        "subscription_pool": subscription_pool,
        "billing_mode": "subscription" if subscription_pool else "unverified",
        "paid_usage_possible": paid_usage_possible,
        "execution_supported": True,
        # Unknown quota is observable but not a gate for a configured,
        # subscription-backed Cloud pool.
        "eligible": eligible,
        "blocked_by": blocked,
    }


def _antigravity_probe() -> dict[str, Any]:
    executable = shutil.which("agy")
    models = _command([executable, "models"]) if executable else None
    help_result = _command([executable, "--help"]) if executable else None
    help_text = help_result.stdout if help_result else ""
    roster = [line.strip() for line in (models.stdout if models else "").splitlines() if line.strip()]
    return {
        "pool": "google-antigravity",
        "provider": "google-antigravity-cli",
        "execution_surface": "cli",
        "installed": executable is not None,
        "authenticated": bool(models and models.returncode == 0 and roster),
        "headless": "--print" in help_text,
        "workspace": "--add-dir" in help_text,
        "tools": "--sandbox" in help_text,
        "model_roster": roster,
        "model_roster_source": "agy models",
        "quota": {"visible": False, "state": "unknown"},
        "billing_mode": "unverified",
        "paid_usage_possible": None,
        "execution_supported": False,
        "eligible": False,
        "blocked_by": [
            "Google AI Pro billing not exposed by CLI",
            "quota not exposed headlessly",
            "Hermes CLI-worker bridge not implemented",
        ],
    }


def _openai_probe() -> dict[str, Any]:
    from hermes_cli.codex_models import get_codex_model_ids
    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(requested="openai-codex")
    usage = _usage("openai-codex")
    paid = _openai_paid_usage_possible()
    models = get_codex_model_ids(runtime.get("api_key"))
    eligible = bool(runtime.get("api_key") and models and usage["visible"] and paid is False)
    blocked = []
    if not models:
        blocked.append("runtime model roster unavailable")
    if not usage["visible"]:
        blocked.append("quota unavailable")
    if paid is not False:
        blocked.append("paid credits or overage cannot be ruled out")
    return {
        "pool": "openai-subscription",
        "provider": "openai-codex",
        "execution_surface": "hermes",
        "installed": shutil.which("codex") is not None,
        "authenticated": bool(runtime.get("api_key")),
        "headless": True,
        "workspace": True,
        "tools": True,
        "model_roster": models,
        "model_roster_source": "Codex runtime models endpoint",
        "quota": usage,
        "billing_mode": "subscription" if usage.get("plan") else "unverified",
        "paid_usage_possible": paid,
        "execution_supported": True,
        "eligible": eligible,
        "blocked_by": blocked,
    }


def _anthropic_probe() -> dict[str, Any]:
    executable = shutil.which("claude")
    auth = _command([executable, "auth", "status"]) if executable else None
    auth_payload: dict[str, Any] = {}
    try:
        auth_payload = json.loads(auth.stdout) if auth else {}
    except json.JSONDecodeError:
        pass
    usage = _usage("anthropic")
    paid = _anthropic_paid_usage_possible()
    return {
        "pool": "anthropic-subscription",
        "provider": "anthropic-cli",
        "execution_surface": "cli",
        "installed": executable is not None,
        "authenticated": bool(auth_payload.get("loggedIn")),
        "headless": True,
        "workspace": True,
        "tools": True,
        "model_roster": [],
        "model_roster_source": "requires one explicit --smoke anthropic run",
        "model_preference": {"family": "sonnet", "minimum_generation": 5},
        "quota": usage,
        "billing_mode": "subscription" if auth_payload.get("subscriptionType") else "unverified",
        "paid_usage_possible": paid,
        "execution_supported": False,
        "eligible": False,
        "blocked_by": ["Hermes CLI-worker bridge not implemented", "runtime roster needs explicit smoke"],
    }


def _smoke(name: str) -> dict[str, Any]:
    if name == "anthropic":
        result = _command([
            "claude", "-p", "Reply exactly: ROUTE_SMOKE_OK. Do not use tools.",
            "--model", "sonnet", "--tools", "", "--permission-mode", "dontAsk",
            "--output-format", "json", "--no-session-persistence",
        ], timeout=90)
        payload = json.loads(result.stdout) if result.returncode == 0 else {}
        return {
            "route": name,
            "ok": result.returncode == 0 and payload.get("result") == "ROUTE_SMOKE_OK",
            "models": sorted((payload.get("modelUsage") or {}).keys()),
            "turns": payload.get("num_turns"),
        }
    raise ValueError("Only the Anthropic roster requires an opt-in smoke; other rosters use read-only APIs.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", choices=("anthropic",), help="run one tiny tool-disabled request")
    args = parser.parse_args()

    probes = [_ollama_probe(), _openai_probe(), _antigravity_probe(), _anthropic_probe()]
    report: dict[str, Any] = {
        "pool_order": list(POOL_ORDER),
        "probes": probes,
        "automatic_metered_usage": False,
        "automatic_paid_overage": False,
        "smoke": _smoke(args.smoke) if args.smoke else None,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
