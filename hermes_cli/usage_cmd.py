"""Official OAuth quota surfaces: ``hermes usage`` and compact status lines.

Only providers with a first-party usage endpoint are fetched. Fail-open:
network/auth errors become ``unavailable`` and never crash the CLI.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from agent.account_usage import (
    OFFICIAL_USAGE_PROVIDERS,
    UNSUPPORTED_USAGE_REASONS,
    compact_account_usage_line,
    fetch_account_usage,
    normalize_usage_provider,
    snapshot_to_dict,
)

DISPLAY_NAMES = {
    "openai-codex": "Codex",
    "anthropic": "Claude",
    "openrouter": "OpenRouter",
    "nous": "Nous",
    "xai-oauth": "Grok",
    "xai": "Grok",
}


def collect_usage_report(provider: str) -> dict[str, Any]:
    """Return a JSON-safe report. Never raises."""
    normalized = normalize_usage_provider(provider)
    if not normalized:
        return {"provider": "", "status": "unavailable", "reason": "provider required"}
    if normalized in UNSUPPORTED_USAGE_REASONS:
        return {
            "provider": normalized,
            "status": "unsupported",
            "reason": UNSUPPORTED_USAGE_REASONS[normalized],
        }
    if normalized not in OFFICIAL_USAGE_PROVIDERS:
        return {
            "provider": normalized,
            "status": "unknown",
            "reason": "no official quota source",
        }
    try:
        snapshot = fetch_account_usage(normalized)
    except Exception:
        snapshot = None
    if snapshot is None or not snapshot.available:
        return {
            "provider": normalized,
            "status": "unavailable",
            "reason": "quota endpoint unavailable",
        }
    return snapshot_to_dict(snapshot)


def compact_usage_line(provider: str) -> Optional[str]:
    """One-line summary for ``hermes status`` / ``auth status``. None if nothing to show."""
    normalized = normalize_usage_provider(provider)
    if normalized in UNSUPPORTED_USAGE_REASONS or normalized not in OFFICIAL_USAGE_PROVIDERS:
        return None
    try:
        snapshot = fetch_account_usage(normalized)
    except Exception:
        return None
    if snapshot is None or not snapshot.available:
        return None
    line = compact_account_usage_line(snapshot)
    return line or None


def format_usage_text(report: dict[str, Any]) -> str:
    provider = str(report.get("provider") or "")
    name = DISPLAY_NAMES.get(provider, provider or "provider")
    status = report.get("status")
    if status == "unsupported":
        return f"{name}      unsupported ({report.get('reason') or 'no official quota API'})"
    if status == "unknown":
        return f"{name}      unknown ({report.get('reason') or 'no official quota source'})"
    if status != "ok":
        return f"{name}      unavailable"
    header = "📈 Account limits"
    plan = report.get("plan")
    lines = [header, f"Provider: {provider} ({plan})" if plan else f"Provider: {provider}"]
    for window in report.get("windows") or []:
        remaining = window.get("remaining_percent")
        used = window.get("used_percent")
        if remaining is None and used is not None:
            remaining = max(0, round(100.0 - float(used)))
            used = max(0, round(float(used)))
        if remaining is None:
            base = f"{window.get('name')}: unavailable"
        else:
            used_i = max(0, round(float(used))) if used is not None else max(0, 100 - int(remaining))
            base = f"{window.get('name')}: {int(remaining)}% remaining ({used_i}% used)"
        reset_at = window.get("reset_at")
        if reset_at:
            base += f" • resets {reset_at}"
        elif window.get("detail"):
            base += f" • {window['detail']}"
        lines.append(base)
    for detail in report.get("details") or []:
        lines.append(str(detail))
    return "\n".join(lines)


def usage_command(args) -> int:
    as_json = bool(getattr(args, "json", False))
    requested = str(getattr(args, "provider", "") or "").strip()
    if requested:
        report = collect_usage_report(requested)
        if as_json:
            print(json.dumps(report, indent=2))
        else:
            print(format_usage_text(report))
        return 2 if report.get("status") == "unknown" else 0

    official = [collect_usage_report(provider) for provider in OFFICIAL_USAGE_PROVIDERS]
    grok = {
        "provider": "xai-oauth",
        "status": "unsupported",
        "reason": UNSUPPORTED_USAGE_REASONS["xai-oauth"],
    }
    if as_json:
        print(json.dumps({"providers": official + [grok]}, indent=2))
        return 0
    ok_reports = [report for report in official if report.get("status") == "ok"]
    if not ok_reports:
        print("No official quota sources are signed in.")
        for report in official:
            print(format_usage_text(report))
        print(format_usage_text(grok))
        return 0
    for report in ok_reports:
        print(format_usage_text(report))
    print(format_usage_text(grok))
    return 0
